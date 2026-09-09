"""登录机制 v0 测试：注册/登录/鉴权/注销全链路 + 受保护端点 401。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from svsg.api import create_app
from svsg.config import Settings


def make_client(tmp_path) -> TestClient:
    settings = Settings(
        auth_enabled=True,
        auth_db=str(tmp_path / "auth.db"),
        _env_file=None,  # 隔离项目 .env，固定 stub 流水线，测试离线可跑
    )
    return TestClient(create_app(settings=settings))


def test_full_flow(tmp_path):
    client = make_client(tmp_path)

    # /healthz 保持无鉴权
    assert client.get("/healthz").status_code == 200

    # 未带令牌访问受保护端点 → 401（依赖先于请求体校验触发）
    assert client.post("/v1/analyze", json={"query": "测试", "ir": {}}).status_code == 401

    # 注册首个用户 → 自动成为管理员
    r = client.post("/auth/register", json={"username": "admin", "password": "secret123"})
    assert r.status_code == 201, r.text
    assert r.json()["is_admin"] is True

    # 重名注册（大小写不敏感）→ 409
    assert (
        client.post(
            "/auth/register", json={"username": "ADMIN", "password": "secret123"}
        ).status_code
        == 409
    )

    # 登录 → Bearer 令牌
    r = client.post("/auth/login", json={"username": "admin", "password": "secret123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] > 0
    headers = {"Authorization": f"Bearer {body['access_token']}"}

    # /auth/me 返回当前用户
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200 and r.json()["username"] == "admin"

    # 带令牌访问受保护端点：鉴权通过（不再 401；业务层按 IR 合法性分流）
    r = client.post("/v1/analyze", json={"query": "测试", "ir": {}}, headers=headers)
    assert r.status_code != 401, r.text

    # 注销后令牌立即失效
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_invalid_credentials(tmp_path):
    client = make_client(tmp_path)

    # 用户名格式不合法 / 密码过短
    assert (
        client.post(
            "/auth/register", json={"username": "a", "password": "secret123"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/auth/register", json={"username": "alice", "password": "123"}
        ).status_code
        == 422
    )

    # 正常注册后：错误密码 / 不存在的用户 → 401
    client.post("/auth/register", json={"username": "alice", "password": "secret123"})
    assert (
        client.post(
            "/auth/login", json={"username": "alice", "password": "wrong-pass"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/auth/login", json={"username": "ghost", "password": "secret123"}
        ).status_code
        == 401
    )
