"""有限状态机（FSM S0~S6，含 S4a）：查表驱动，非 if-else 堆砌。

状态生命周期（V5.2.1 §4.3 状态转移表的机器可执行化）：

    S0 ──VALIDATION_PASSED──▶ S1 ──VERIFICATION_REQUESTED──▶ S4a
    S0 ──VALIDATION_FAILED──▶ S2 (终态, 硬拒绝)
    S1 ──DEPENDENCY_MISSING/SKIPPED──▶ S3 (守卫: 最多 1 次重试)
    S1 ──ANCHOR_PASSED──▶ S1 (delivered=true, 正常终态)
    S1 ──ANCHOR_FAILED──▶ S4 ──CONFLICT_RESOLVED──▶ S1
    S4a ──VERIFICATION_CONSISTENT──▶ S1
    S4a ──VERIFICATION_CONFLICT──▶ S4
    S4a ──VERIFICATION_TIMEOUT──▶ S6 (静默降级: conf_level→lowest)
    S4 ──CONFLICT_UNRESOLVED──▶ S6
    S3 ──RETRY_SUCCEEDED──▶ S1 / RETRY_FAILED──▶ S6
    活动态 ──L1/L3_TIMEOUT──▶ S5 (终态, 请求中止)

S5 超时策略按来源分流（V5.2.1 §5.1）：
- L1.5 超时：不进 S5，直接 S4a ──VERIFICATION_TIMEOUT──▶ S6，
  受影响实例记入 degraded_instances（conf_level 视为 lowest），
  其余高置信实例继续处理（部分可用性）。
- L1/L3 超时：进入 S5 终态，请求整体中止。

非法事件（表中不存在的转移）抛 SVSGError(INTERNAL_ERROR)，
用于在开发期捕获编排器（Orchestrator）的逻辑 bug。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from svsg.contracts import IR, ErrorCode, StateCode, SVSGError

MAX_RETRIES = 1  # S3 软重试上限（V5.2.1 状态转移表）


class Event:
    """触发状态转移的事件（编排器产生，FSM 只认事件不认来源）。"""

    # S0 出口
    VALIDATION_PASSED = "VALIDATION_PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    # S1 出口
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"        # S3 语义（软重试）
    VERIFICATION_REQUESTED = "VERIFICATION_REQUESTED"  # → S4a
    VERIFICATION_SKIPPED = "VERIFICATION_SKIPPED"    # E2003：锚定校验发现漏验证 → 打回 S3
    ANCHOR_PASSED = "ANCHOR_PASSED"                  # 证据锚定通过 → 交付
    ANCHOR_FAILED = "ANCHOR_FAILED"                  # 证据锚定否决 → S4
    L1_TIMEOUT = "L1_TIMEOUT"                        # → S5 终态
    L3_TIMEOUT = "L3_TIMEOUT"                        # → S5 终态
    # S3 出口
    RETRY_SUCCEEDED = "RETRY_SUCCEEDED"
    RETRY_FAILED = "RETRY_FAILED"
    # S4a 出口
    VERIFICATION_CONSISTENT = "VERIFICATION_CONSISTENT"
    VERIFICATION_CONFLICT = "VERIFICATION_CONFLICT"
    VERIFICATION_TIMEOUT = "VERIFICATION_TIMEOUT"    # L1.5 超时 → S6 静默降级
    # S4 出口
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"
    CONFLICT_UNRESOLVED = "CONFLICT_UNRESOLVED"


#: 终态集合：S2 硬拒绝 / S5 请求中止 / S6 人工复审
TERMINAL_STATES: frozenset[StateCode] = frozenset(
    {StateCode.S2, StateCode.S5, StateCode.S6}
)

#: 需要重试守卫的事件：retries_used >= MAX_RETRIES 时直接进 S6
_RETRY_GUARDED_EVENTS = frozenset({Event.DEPENDENCY_MISSING, Event.VERIFICATION_SKIPPED})

#: 查表转移表：(当前状态, 事件) → 目标状态。守卫事件的目标在 transition() 中裁决
_TRANSITIONS: dict[tuple[StateCode, str], StateCode] = {
    (StateCode.S0, Event.VALIDATION_PASSED): StateCode.S1,
    (StateCode.S0, Event.VALIDATION_FAILED): StateCode.S2,
    (StateCode.S1, Event.DEPENDENCY_MISSING): StateCode.S3,
    (StateCode.S1, Event.VERIFICATION_REQUESTED): StateCode.S4A,
    (StateCode.S1, Event.VERIFICATION_SKIPPED): StateCode.S3,
    (StateCode.S1, Event.ANCHOR_PASSED): StateCode.S1,  # delivered 置位
    (StateCode.S1, Event.ANCHOR_FAILED): StateCode.S4,
    (StateCode.S1, Event.L1_TIMEOUT): StateCode.S5,
    (StateCode.S1, Event.L3_TIMEOUT): StateCode.S5,
    (StateCode.S3, Event.RETRY_SUCCEEDED): StateCode.S1,
    (StateCode.S3, Event.RETRY_FAILED): StateCode.S6,
    (StateCode.S3, Event.L1_TIMEOUT): StateCode.S5,
    (StateCode.S3, Event.L3_TIMEOUT): StateCode.S5,
    (StateCode.S4A, Event.VERIFICATION_CONSISTENT): StateCode.S1,
    (StateCode.S4A, Event.VERIFICATION_CONFLICT): StateCode.S4,
    (StateCode.S4A, Event.VERIFICATION_TIMEOUT): StateCode.S6,
    (StateCode.S4A, Event.L1_TIMEOUT): StateCode.S5,
    (StateCode.S4A, Event.L3_TIMEOUT): StateCode.S5,
    (StateCode.S4, Event.CONFLICT_RESOLVED): StateCode.S1,
    (StateCode.S4, Event.CONFLICT_UNRESOLVED): StateCode.S6,
    (StateCode.S4, Event.L1_TIMEOUT): StateCode.S5,
    (StateCode.S4, Event.L3_TIMEOUT): StateCode.S5,
}


@dataclass(frozen=True)
class TransitionRecord:
    """审计用转移记录。"""

    from_state: StateCode
    event: str
    to_state: StateCode
    at: datetime
    note: str | None = None


@dataclass
class RuntimeSession:
    """单次请求的 FSM 会话（L2 运行时的最小状态载体）。

    编排器（l3/后续层）持有本对象，按事件驱动转移；
    本对象不做任何 IO，全部行为可穷举单测。
    """

    session_id: str
    ir: IR | None = None
    state: StateCode = StateCode.S0
    retries_used: int = 0
    delivered: bool = False
    #: VERIFICATION_TIMEOUT 时静默降级的实例集合（conf_level 视为 lowest）
    degraded_instances: set[int] = field(default_factory=set)
    history: list[TransitionRecord] = field(default_factory=list)

    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def effective_conf_level(self, instance_id: int) -> str | None:
        """实例的有效置信度档位：被降级实例强制视为 lowest。"""
        if instance_id in self.degraded_instances:
            return "lowest"
        if self.ir is None:
            return None
        for det in self.ir.detections:
            if det.instance_id == instance_id:
                return det.conf_level.value
        return None

    def transition(
        self,
        event: str,
        *,
        degraded: set[int] | None = None,
        note: str | None = None,
    ) -> StateCode:
        """执行一次状态转移，返回新状态。

        - 非法转移（表项不存在/终态再触发）→ SVSGError(INTERNAL_ERROR)
        - 重试守卫：DEPENDENCY_MISSING / VERIFICATION_SKIPPED 且
          retries_used >= MAX_RETRIES → 直接 S6（RETRY_EXHAUSTED）
        - 进入 S3 时 retries_used += 1
        - ANCHOR_PASSED 时 delivered 置位
        - VERIFICATION_TIMEOUT 时合并 degraded 实例集合
        """
        if self.is_terminal:
            raise SVSGError(
                ErrorCode.INTERNAL_ERROR,
                f"终态 {self.state.value} 不接受事件 {event}",
                details={"session_id": self.session_id},
            )

        target = _TRANSITIONS.get((self.state, event))
        guard_note: str | None = None

        if event in _RETRY_GUARDED_EVENTS:
            if self.retries_used >= MAX_RETRIES:
                target = StateCode.S6
                guard_note = f"RETRY_EXHAUSTED: 已用尽 {MAX_RETRIES} 次软重试"
            else:
                self.retries_used += 1

        if target is None:
            raise SVSGError(
                ErrorCode.INTERNAL_ERROR,
                f"非法转移: {self.state.value} + {event}",
                details={"session_id": self.session_id},
            )

        if event == Event.ANCHOR_PASSED:
            self.delivered = True
        if event == Event.VERIFICATION_TIMEOUT and degraded:
            self.degraded_instances |= set(degraded)
            guard_note = f"静默降级实例: {sorted(degraded)}"

        record = TransitionRecord(
            from_state=self.state,
            event=event,
            to_state=target,
            at=datetime.now(UTC),
            note=note or guard_note,
        )
        self.history.append(record)
        self.state = target
        return target

    # ------------------------------------------------------------------

    def bind_ir(self, ir: IR) -> None:
        """S0 阶段绑定 L1 产出的 IR。"""
        if self.state is not StateCode.S0:
            raise SVSGError(
                ErrorCode.INTERNAL_ERROR,
                f"仅 S0 可绑定 IR，当前状态 {self.state.value}",
            )
        self.ir = ir

    def summary(self) -> dict[str, str | int | list[int] | list[dict]]:
        """输出审计摘要（可观测性平面的结构化载荷）。"""
        return {
            "session_id": self.session_id,
            "final_state": self.state.value,
            "delivered": self.delivered,
            "retries_used": self.retries_used,
            "degraded_instances": sorted(self.degraded_instances),
            "transitions": [
                {
                    "from": r.from_state.value,
                    "event": r.event,
                    "to": r.to_state.value,
                    "note": r.note,
                }
                for r in self.history
            ],
        }
