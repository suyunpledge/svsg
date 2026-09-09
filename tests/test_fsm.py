"""FSM 全路径测试：合法路径、重试守卫、静默降级、非法转移。"""

from __future__ import annotations

import sys

from svsg.contracts import ErrorCode, StateCode, SVSGError
from svsg.l2_runtime import TERMINAL_STATES, Event, RuntimeSession


def _session() -> RuntimeSession:
    return RuntimeSession(session_id="sess_test_001")


def test_happy_path_with_verification():
    s = _session()
    assert s.transition(Event.VALIDATION_PASSED) == StateCode.S1
    assert s.transition(Event.VERIFICATION_REQUESTED) == StateCode.S4A
    assert s.transition(Event.VERIFICATION_CONSISTENT) == StateCode.S1
    assert s.transition(Event.ANCHOR_PASSED) == StateCode.S1
    assert s.delivered is True
    assert not s.is_terminal  # S1 + delivered 是正常终态，非错误终态
    assert [(r.from_state, r.event, r.to_state) for r in s.history] == [
        (StateCode.S0, Event.VALIDATION_PASSED, StateCode.S1),
        (StateCode.S1, Event.VERIFICATION_REQUESTED, StateCode.S4A),
        (StateCode.S4A, Event.VERIFICATION_CONSISTENT, StateCode.S1),
        (StateCode.S1, Event.ANCHOR_PASSED, StateCode.S1),
    ]


def test_hard_rejection_path():
    s = _session()
    assert s.transition(Event.VALIDATION_FAILED) == StateCode.S2
    assert s.is_terminal
    try:
        s.transition(Event.VALIDATION_PASSED)
    except SVSGError as exc:
        assert exc.code == ErrorCode.INTERNAL_ERROR
    else:
        raise AssertionError("终态再转移应抛 INTERNAL_ERROR")


def test_illegal_transition_raises():
    s = _session()
    try:
        s.transition(Event.ANCHOR_PASSED)  # S0 不接受 ANCHOR_PASSED
    except SVSGError as exc:
        assert exc.code == ErrorCode.INTERNAL_ERROR
        assert "非法转移" in exc.message
    else:
        raise AssertionError("非法转移应抛 INTERNAL_ERROR")
    assert s.state == StateCode.S0  # 抛异常不改变状态


def test_dependency_retry_then_recover():
    s = _session()
    s.transition(Event.VALIDATION_PASSED)
    assert s.transition(Event.DEPENDENCY_MISSING) == StateCode.S3
    assert s.retries_used == 1
    assert s.transition(Event.RETRY_SUCCEEDED) == StateCode.S1
    # 重试已用尽：再次缺依赖 → 直接 S6（RETRY_EXHAUSTED）
    assert s.transition(Event.DEPENDENCY_MISSING) == StateCode.S6
    assert s.is_terminal
    assert any("RETRY_EXHAUSTED" in (r.note or "") for r in s.history)


def test_verification_skipped_uses_same_retry_budget():
    s = _session()
    s.transition(Event.VALIDATION_PASSED)
    s.transition(Event.DEPENDENCY_MISSING)  # 用掉唯一一次重试
    s.transition(Event.RETRY_SUCCEEDED)
    # 锚定发现漏验证 → E3003，但重试已用尽 → S6 而非 S3
    assert s.transition(Event.VERIFICATION_SKIPPED) == StateCode.S6


def test_l15_timeout_silent_degradation():
    """V5.2.1 §5.1：L1.5 超时 → S4a 直接转 S6，实例降级为 lowest。"""
    s = _session()
    s.transition(Event.VALIDATION_PASSED)
    s.transition(Event.VERIFICATION_REQUESTED)
    assert (
        s.transition(Event.VERIFICATION_TIMEOUT, degraded={3, 5}) == StateCode.S6
    )
    assert s.degraded_instances == {3, 5}
    assert s.effective_conf_level(3) == "lowest"
    assert any("静默降级" in (r.note or "") for r in s.history)


def test_l1_l3_timeout_abort_to_s5():
    s = _session()
    s.transition(Event.VALIDATION_PASSED)
    assert s.transition(Event.L1_TIMEOUT) == StateCode.S5
    assert s.is_terminal

    s2 = _session()
    s2.transition(Event.VALIDATION_PASSED)
    s2.transition(Event.VERIFICATION_REQUESTED)
    s2.transition(Event.VERIFICATION_CONFLICT)
    assert s2.transition(Event.L3_TIMEOUT) == StateCode.S5


def test_anchor_fail_goes_to_s4_then_two_exits():
    # 冲突可解决 → 回 S1 重新锚定
    s = _session()
    s.transition(Event.VALIDATION_PASSED)
    assert s.transition(Event.ANCHOR_FAILED) == StateCode.S4
    assert s.transition(Event.CONFLICT_RESOLVED) == StateCode.S1
    assert s.transition(Event.ANCHOR_PASSED) == StateCode.S1
    assert s.delivered

    # 冲突不可解决 → S6
    s2 = _session()
    s2.transition(Event.VALIDATION_PASSED)
    s2.transition(Event.ANCHOR_FAILED)
    assert s2.transition(Event.CONFLICT_UNRESOLVED) == StateCode.S6


def test_verification_conflict_roundtrip():
    s = _session()
    s.transition(Event.VALIDATION_PASSED)
    s.transition(Event.VERIFICATION_REQUESTED)
    assert s.transition(Event.VERIFICATION_CONFLICT) == StateCode.S4
    assert s.transition(Event.CONFLICT_RESOLVED) == StateCode.S1


def test_terminal_states_definition():
    assert TERMINAL_STATES == {StateCode.S2, StateCode.S5, StateCode.S6}


if __name__ == "__main__":
    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
