"""L3 编排层：纯文本 LLM 主控 + 证据锚定后置 hook。

- prompts：system prompt（透视免责声明、降级标注、ambiguous 禁令、
  claims 双通道词表）+ 结构化上下文构建
- agent_loop：Orchestrator 编排循环（S0 校验 → 验证 → L3 生成 →
  锚定校验 → 交付/否决/降级），是 Phase 1 MVP 闭环的顶层入口
"""

from .agent_loop import (
    LLMProtocol,
    Orchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    StubLLM,
)
from .prompts import build_context, build_system_prompt

__all__ = [
    "LLMProtocol",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    "StubLLM",
    "build_context",
    "build_system_prompt",
]
