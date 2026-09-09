"""SVSG contracts：全系统共享的数据契约层。

- enums：跨层枚举（ConfLevel / SceneType / IntentType / StateCode ...）
- ir_schema：L1 输出的结构化 IR（S2 硬校验的唯一裁判）
- verification_schema：L1.5 验证报告 + L3 双通道答案契约
- errors：错误码表与 SVSGError 结构化异常
"""

from .enums import (
    DIRECTIONAL_RELATIONS,
    DISTANCE_RELATIONS,
    ConfLevel,
    IntentType,
    MetricType,
    RelationType,
    SceneType,
    StateCode,
    VerificationStatus,
)
from .errors import ERROR_STATE_MAP, ErrorCode, SVSGError, from_validation_error
from .ir_schema import (
    IR,
    IR_SCHEMA_VERSION,
    CameraIntrinsics,
    Detection,
    Relation,
    Uncertainty,
    check_relation_references,
)
from .verification_schema import (
    Claim,
    ClaimValue,
    L3Answer,
    VerificationReport,
    VerificationResult,
)

__all__ = [
    "DIRECTIONAL_RELATIONS",
    "DISTANCE_RELATIONS",
    "ConfLevel",
    "IntentType",
    "MetricType",
    "RelationType",
    "SceneType",
    "StateCode",
    "VerificationStatus",
    "ERROR_STATE_MAP",
    "ErrorCode",
    "SVSGError",
    "from_validation_error",
    "IR_SCHEMA_VERSION",
    "IR",
    "CameraIntrinsics",
    "Detection",
    "Relation",
    "Uncertainty",
    "check_relation_references",
    "Claim",
    "ClaimValue",
    "L3Answer",
    "VerificationReport",
    "VerificationResult",
]
