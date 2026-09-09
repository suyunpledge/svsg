"""v0 登录机制：SQLite 用户库 + PBKDF2 口令散列 + 不透明 Bearer 令牌。

设计取舍（初步版，保留平滑升级空间）：
- 仅用标准库（sqlite3 / hashlib / hmac / secrets），不新增依赖；
- 令牌为服务端签发的不透明随机串（256-bit），库内只存 SHA-256 摘要，
  可随时吊销（logout 即删）；有效期由 ``SVSG_AUTH_TOKEN_TTL_S`` 控制；
- 口令使用 PBKDF2-HMAC-SHA256（迭代次数 ``SVSG_AUTH_PBKDF2_ITERS``），
  校验走恒时比较，防时序侧信道；
- 启用方式：``SVSG_AUTH_ENABLED=1``。启用后 /v1/* 由 X-API-Key 切换为
  Bearer 令牌鉴权，/healthz 与 /auth/* 保持无鉴权（登录前入口）；
  首个注册用户自动成为管理员；
- 已知边界（v0 接受）：SQLite 适合单实例低并发，多 worker 部署时把
  存储层换掉即可，路由与散列格式不变；同步短连接在事件循环内执行，
  低并发场景可接受。
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from svsg.config import Settings

__all__ = [
    "AuthStore",
    "DuplicateUserError",
    "build_auth_router",
    "hash_password",
    "require_bearer",
    "verify_password",
]

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_MIN_PASSWORD_LEN = 8
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


class DuplicateUserError(Exception):
    """用户名已存在（大小写不敏感）。"""


# ---------------------------------------------------------------------- 口令散列


def hash_password(password: str, iters: int) -> str:
    """PBKDF2-HMAC-SHA256 散列，存储格式 ``pbkdf2_sha256$iters$salt$digest``。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return f"pbkdf2_sha256${iters}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """按存储格式校验口令；格式异常一律视为不匹配（恒时比较）。"""
    try:
        algo, iters_s, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters_s)
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def _token_digest(token: str) -> str:
    """令牌入库摘要：库中不存明文令牌，泄露库文件也无法伪造有效令牌。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- 存储层


class AuthStore:
    """SQLite 用户/令牌存储（每次操作独立短连接，事务自动提交）。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # 事务提交/回滚
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    pw_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def count_users(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password: str, *, iters: int) -> sqlite3.Row:
        """创建用户；首个用户自动成为管理员，重名抛 DuplicateUserError。"""
        is_admin = 1 if self.count_users() == 0 else 0
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users (username, pw_hash, is_admin, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (username, hash_password(password, iters), is_admin, _now_iso()),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUserError(username) from exc
        return self.get_user_by_name(username)  # type: ignore[return-value]

    def get_user_by_name(self, username: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()

    def verify_login(self, username: str, password: str) -> sqlite3.Row | None:
        """口令校验失败/用户不存在统一返回 None，避免用户名枚举。"""
        user = self.get_user_by_name(username)
        if user is None or not verify_password(password, user["pw_hash"]):
            return None
        return user

    def issue_token(self, user_id: int, ttl_s: int) -> tuple[str, float]:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC).timestamp() + ttl_s
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tokens (token_hash, user_id, expires_at, created_at)"
                " VALUES (?, ?, ?, ?)",
                (_token_digest(token), user_id, expires_at, _now_iso()),
            )
        return token, expires_at

    def resolve_token(self, token: str) -> sqlite3.Row | None:
        """返回令牌对应的用户行；过期令牌顺带清理。"""
        digest = _token_digest(token)
        now = datetime.now(UTC).timestamp()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.is_admin, u.created_at, t.expires_at
                FROM tokens t JOIN users u ON u.id = t.user_id
                WHERE t.token_hash = ?
                """,
                (digest,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] < now:
                conn.execute("DELETE FROM tokens WHERE token_hash = ?", (digest,))
                return None
            return row

    def revoke_token(self, token: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM tokens WHERE token_hash = ?", (_token_digest(token),)
            )
            return cur.rowcount > 0


# ---------------------------------------------------------------------- 请求模型


class Credentials(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MeResponse(BaseModel):
    username: str
    is_admin: bool
    created_at: str


def _validate_credentials(username: str, password: str) -> None:
    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=422, detail="用户名需为 3-32 位字母/数字/_.-")
    if len(password) < _MIN_PASSWORD_LEN:
        raise HTTPException(status_code=422, detail=f"密码至少 {_MIN_PASSWORD_LEN} 位")


def _bearer_dependency(store: AuthStore):
    """共享的 Bearer 校验依赖：通过后把用户信息挂到 request.state.user。"""

    async def require_bearer_token(request: Request):
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=401,
                detail="缺少 Bearer 令牌（先 POST /auth/login 获取）",
                headers=_UNAUTHORIZED_HEADERS,
            )
        user = store.resolve_token(token)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="令牌无效或已过期",
                headers=_UNAUTHORIZED_HEADERS,
            )
        request.state.user = {
            "user_id": user["id"],
            "username": user["username"],
            "is_admin": bool(user["is_admin"]),
        }
        return user

    return require_bearer_token


def build_auth_router(settings: Settings) -> APIRouter:
    """/auth/* 路由：register / login / me / logout（应用装配时按需挂载）。"""
    store = AuthStore(settings.auth_db)
    require_bearer_token = _bearer_dependency(store)
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/register", response_model=MeResponse, status_code=201)
    async def register(req: Credentials) -> MeResponse:
        _validate_credentials(req.username, req.password)
        try:
            user = store.create_user(
                req.username, req.password, iters=settings.auth_pbkdf2_iters
            )
        except DuplicateUserError as exc:
            raise HTTPException(status_code=409, detail="用户名已存在") from exc
        return MeResponse(
            username=user["username"],
            is_admin=bool(user["is_admin"]),
            created_at=user["created_at"],
        )

    @router.post("/login", response_model=TokenResponse)
    async def login(req: Credentials) -> TokenResponse:
        user = store.verify_login(req.username, req.password)
        if user is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token, _ = store.issue_token(user["id"], settings.auth_token_ttl_s)
        return TokenResponse(access_token=token, expires_in=settings.auth_token_ttl_s)

    @router.get("/me", response_model=MeResponse)
    async def me(user=Depends(require_bearer_token)) -> MeResponse:  # noqa: B008
        return MeResponse(
            username=user["username"],
            is_admin=bool(user["is_admin"]),
            created_at=user["created_at"],
        )

    @router.post("/logout")
    async def logout(
        request: Request, user=Depends(require_bearer_token)  # noqa: B008
    ) -> dict[str, bool]:
        token = request.headers.get("Authorization", "").partition(" ")[2].strip()
        return {"revoked": store.revoke_token(token)}

    return router


def require_bearer(settings: Settings):
    """/v1/* 鉴权依赖工厂：与 /auth/* 共用同一存储与校验逻辑。"""
    return _bearer_dependency(AuthStore(settings.auth_db))
