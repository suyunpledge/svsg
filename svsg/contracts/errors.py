"""SVSG 错误码表与结构化异常。

错误码与 FSM 状态（S0~S6）一一映射，L2 各组件统一抛出 SVSGError，
由状态机决定后续转移（硬拒绝 / 软重试 / 冲突解决 / 降级转人工）。
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Any

from pydantic import ValidationError

from .enums import StateCode


@unique
class ErrorCode(StrEnum):
    """错误码分段：E1xxx=S2 硬拒绝，E2xxx=S3 依赖缺失，E3xxx=S4 冲突，
    E4xxx=S5 超时，E5xxx=S6 人工复审，E9xxx=内部错误。"""

    # -- S2: Invalid Parameter（硬拒绝）--
    SCHEMA_INVALID = "E1001"        # IR/报文不符合 Schema
    BBOX_OUT_OF_RANGE = "E1002"     # bbox 越界或退化
    FIELD_MISSING = "E1003"         # 必填字段缺失

    # -- S3: Missing Dependency --
    UNKNOWN_INSTANCE_REF = "E2001"  # relations 引用未知 instance_id
    RETRY_EXHAUSTED = "E2002"       # 软重试（最多 1 次）后仍失败
    VERIFICATION_SKIPPED = "E2003"  # 应验证但未调用 VerifyRegion（打回 S3 重试）

    # -- S4: Semantic Conflict --
    SEMANTIC_CONFLICT = "E3001"     # Detector 与 L1.5 验证结果不一致且无法解决
    EVIDENCE_MISMATCH = "E3002"     # 证据锚定校验否决（claims 与 structured_report 不一致）

    # -- S5: Timeout --
    L1_TIMEOUT = "E4001"
    L15_TIMEOUT = "E4002"           # 触发静默降级：conf_level -> lowest，转 S6
    L3_TIMEOUT = "E4003"

    # -- S6: Human Review --
    HUMAN_REVIEW_REQUIRED = "E5001"

    # -- 其他 --
    INTERNAL_ERROR = "E9999"


#: 错误码 -> FSM 状态的权威映射
ERROR_STATE_MAP: dict[ErrorCode, StateCode] = {
    ErrorCode.SCHEMA_INVALID: StateCode.S2,
    ErrorCode.BBOX_OUT_OF_RANGE: StateCode.S2,
    ErrorCode.FIELD_MISSING: StateCode.S2,
    ErrorCode.UNKNOWN_INSTANCE_REF: StateCode.S3,
    ErrorCode.RETRY_EXHAUSTED: StateCode.S6,
    ErrorCode.VERIFICATION_SKIPPED: StateCode.S3,
    ErrorCode.SEMANTIC_CONFLICT: StateCode.S4,
    ErrorCode.EVIDENCE_MISMATCH: StateCode.S4,
    ErrorCode.L1_TIMEOUT: StateCode.S5,
    ErrorCode.L15_TIMEOUT: StateCode.S5,
    ErrorCode.L3_TIMEOUT: StateCode.S5,
    ErrorCode.HUMAN_REVIEW_REQUIRED: StateCode.S6,
    ErrorCode.INTERNAL_ERROR: StateCode.S6,
}


class SVSGError(Exception):
    """携带 FSM 状态语义的结构化异常，L2 各组件抛出/捕获的唯一异常类型。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.state = ERROR_STATE_MAP.get(code, StateCode.S6)
        self.details = details or {}
        super().__init__(f"[{code.value}/{code.name}] {message}")

    def to_dict(self) -> dict[str, Any]:
        """输出可 JSON 序列化的结构化错误载荷（用于 API 响应与审计日志）。"""
        return {
            "code": self.code.value,
            "name": self.code.name,
            "message": self.message,
            "state": self.state.value,
            "details": self.details,
        }


def from_validation_error(exc: ValidationError) -> SVSGError:
    """将 Pydantic ValidationError 映射为 S2 硬拒绝异常。

    细分规则（best-effort）：
    - loc 含 bbox_px 或 msg 提到边界 -> BBOX_OUT_OF_RANGE
    - 错误类型为 missing -> FIELD_MISSING
    - 其余 -> SCHEMA_INVALID
    """
    errors = exc.errors()
    for err in errors:
        loc = ".".join(str(seg) for seg in err.get("loc", ()))
        msg = err.get("msg", "")
        if "bbox_px" in loc or "边界" in msg:
            return SVSGError(
                ErrorCode.BBOX_OUT_OF_RANGE,
                f"bbox 校验失败: {msg}",
                details={"loc": loc},
            )
        if err.get("type") == "missing":
            return SVSGError(
                ErrorCode.FIELD_MISSING, f"必填字段缺失: {loc or '<root>'}"
            )
    first_msg = errors[0].get("msg", "") if errors else ""
    return SVSGError(
        ErrorCode.SCHEMA_INVALID,
        f"报文不符合 IR Schema: {first_msg}",
        details={"error_count": len(errors)},
    )
