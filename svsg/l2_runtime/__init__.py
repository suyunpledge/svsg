"""L2 语义运行时：意图路由、参数校验、有限状态机、证据锚定校验。

全部为纯逻辑（零 IO、零模型、零 LLM 依赖），是 SVSG 的核心资产：
- intent_router：7 类意图词库路由（Phase 2.5 评测 F1，可换实现）
- param_validator：S0 校验管道（S2 硬拒绝 / S3 软检查）
- fsm：S0~S6 查表驱动状态机 + 重试守卫 + 静默降级记录
- evidence_anchor：V5.2.1 强制后置证据锚定校验（claims × 结构化证据）
- conflict_resolver：S4 冲突解决策略（阈值参数化注入）
"""

from .conflict_resolver import (
    ConflictResolution,
    ConflictThresholds,
    ResolutionOutcome,
    resolve_conflict,
)
from .evidence_anchor import (
    AnchorVerdict,
    Violation,
    anchor_claims,
    missing_verifications,
    nl_claims_coverage,
    run_anchor,
)
from .fsm import TERMINAL_STATES, Event, RuntimeSession, StateCode
from .intent_router import route_intent
from .param_validator import check_dependencies, validate_ir_payload

__all__ = [
    "ConflictResolution",
    "ConflictThresholds",
    "ResolutionOutcome",
    "resolve_conflict",
    "AnchorVerdict",
    "Violation",
    "anchor_claims",
    "missing_verifications",
    "nl_claims_coverage",
    "run_anchor",
    "Event",
    "RuntimeSession",
    "StateCode",
    "TERMINAL_STATES",
    "route_intent",
    "check_dependencies",
    "validate_ir_payload",
]
