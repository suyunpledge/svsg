"""编排器端到端测试：覆盖 FSM 全部出口的 MVP 闭环场景。

场景清单（对应 V5.2.1 三个修订点的验收演示）：
1. 无验证需求 → 直接交付；
2. 验证一致 → 交付；
3. L1.5 超时 → 静默降级 + 部分可用交付（delivered_with_review）；
4. ambiguous 幻觉 → 锚定否决 → 修正反馈重生成 → 交付；
5. 幻觉持续 → 预算耗尽 → 人工复审（答案扣留）；
6. 验证高置信冲突 → 采纳验证结论（class_overrides）→ 交付；
7. 冲突不可解决 → S6 人工复审（不调用 L3）；
8. L3 超时 → S5 中止；
9. 非法 IR → S2 硬拒绝；
10. 悬空依赖且无重建器 → S3 → S6；
11. 漏验证（E3003）→ 打回 S3 → 重验证 → 重生成 → 交付。
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys

from svsg.contracts import L3Answer, StateCode
from svsg.l3_orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    StubLLM,
)
from svsg.l15_verifier import L15Verifier, StubVerifierBackend, VerifierConfig


def _payload(*, sharp2: float = 0.3, verify2: bool = True, dangling: bool = False):
    base = {
        "schema_version": "5.2.1",
        "image_id": "img_orch_001",
        "image_width": 1920,
        "image_height": 1080,
        "timestamp": "2026-09-02T10:00:00Z",
        "scene_type": "orthographic",
        "detector_model": "stub_detector_v0",
        "detections": [
            {
                "instance_id": 1,
                "class": "screw",
                "bbox_px": [420, 150, 480, 210],
                "detection_score": 0.92,
                "conf_level": "high",
                "verification_required": False,
                "uncertainty": {
                    "classification_entropy": 0.08,
                    "local_edge_sharpness": 0.73,
                    "tta_variance": None,
                },
                "relations": [],
            },
            {
                "instance_id": 2,
                "class": "screw",
                "bbox_px": [900, 400, 1000, 500],
                "detection_score": 0.55,
                "conf_level": "low",
                "verification_required": verify2,
                "uncertainty": {
                    "classification_entropy": 0.6,
                    "local_edge_sharpness": sharp2,
                    "tta_variance": None,
                },
                "relations": [],
            },
            {
                "instance_id": 3,
                "class": "washer",
                "bbox_px": [1100, 400, 1160, 460],
                "detection_score": 0.88,
                "conf_level": "high",
                "verification_required": False,
                "uncertainty": {
                    "classification_entropy": 0.1,
                    "local_edge_sharpness": 0.8,
                    "tta_variance": None,
                },
                "relations": [],
            },
        ],
    }
    if dangling:
        base["detections"][0]["relations"] = [
            {
                "type": "left_of",
                "target_id": 99,
                "spatial_consistency": 1.0,
                "metric_accuracy": None,
            }
        ]
    return base


def _answer(claims, text="答案文本"):
    return L3Answer.model_validate({"claims": claims, "final_answer": text})


def _orch(verifier_backend, llm_answers, *, llm_timeout=3.0, llm_delay=0.0,
          verifier_timeout=0.05):
    verifier = L15Verifier(verifier_backend, VerifierConfig(timeout_s=verifier_timeout))
    llm = StubLLM(answers=llm_answers, delay_s=llm_delay)
    orch = Orchestrator(verifier, llm, OrchestratorConfig(llm_timeout_s=llm_timeout))
    return orch, llm


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ 场景 1


def test_no_verification_needed_delivers():
    orch, llm = _orch(
        StubVerifierBackend(),
        [_answer([{"instance_id": 1, "field": "class", "value": "screw"},
                  {"field": "count", "value": 3}])],
    )
    result = _run(orch.run("图里有什么", _payload(verify2=False)))
    assert result.status == "delivered"
    assert result.delivered and not result.human_review_required
    assert result.answer is not None
    assert result.session_summary["final_state"] == StateCode.S1.value
    assert result.session_summary["delivered"] is True
    assert len(llm.calls) == 1


# ------------------------------------------------------------------ 场景 2


def test_verification_consistent_delivers():
    backend = StubVerifierBackend(
        batches=[{2: {"class": "screw", "exists": True, "confidence": 0.85}}]
    )
    orch, llm = _orch(
        backend,
        [_answer([{"instance_id": 2, "field": "class", "value": "screw"},
                  {"field": "count:screw", "value": 2}])],
    )
    result = _run(orch.run("第二个零件是什么", _payload()))
    assert result.status == "delivered"
    assert backend.calls == [[2]]
    states = [t["to"] for t in result.session_summary["transitions"]]
    assert "S4a" in states


# ------------------------------------------------------------------ 场景 3


def test_l15_timeout_silent_degrade_partial_availability():
    """V5.2.1 修订 #3 验收场景：L1.5 超时 → S6 + 降级实例 + L3 继续交付。"""
    backend = StubVerifierBackend(batches=[{2: {"class": "screw"}}], delay_s=0.3)
    orch, llm = _orch(
        backend,
        # 降级实例（2）仅锚定 IR：class 断言与 IR 一致即可
        [_answer([{"instance_id": 2, "field": "class", "value": "screw"},
                  {"field": "count", "value": 3}])],
        llm_timeout=0.5,
    )
    result = _run(orch.run("描述一下", _payload()))
    assert result.status == "delivered_with_review"
    assert result.delivered and result.human_review_required
    assert result.degraded_instances == {2}
    assert result.session_summary["final_state"] == StateCode.S6.value
    assert result.session_summary["degraded_instances"] == [2]
    assert len(llm.calls) == 1  # 部分可用性：L3 仍被调用
    # Prompt 中必须携带降级标注与降级实例的 lowest 档位
    ctx = json.loads(llm.calls[0]["context"])
    assert ctx["degraded_instances"] == [2]
    assert ctx["detections"][1]["conf_level"] == "lowest"


# ------------------------------------------------------------------ 场景 4


def test_ambiguous_hallucination_vetoed_then_regen():
    """V5.2.1 修订 #1 验收场景：ambiguous 报告 vs "螺丝"断言 → 否决 → 修正。"""
    backend = StubVerifierBackend(
        batches=[{2: {"class": "ambiguous", "exists": True, "confidence": 0.55}}]
    )
    bad = _answer([{"instance_id": 2, "field": "class", "value": "screw"},
                   {"field": "count", "value": 3}])
    good = _answer([{"field": "count", "value": 3}], text="共 3 个零件，其一无法确定类别")
    orch, llm = _orch(backend, [bad, good])
    result = _run(orch.run("第二个零件是什么", _payload()))
    assert result.status == "delivered"
    assert len(llm.calls) == 2
    # 第二轮上下文必须包含修正反馈
    assert "corrective_feedback" in llm.calls[1]["context"]
    events = [t["event"] for t in result.session_summary["transitions"]]
    assert "ANCHOR_FAILED" in events and "CONFLICT_RESOLVED" in events
    assert result.session_summary["final_state"] == StateCode.S1.value


# ------------------------------------------------------------------ 场景 5


def test_persistent_hallucination_withholds_answer():
    backend = StubVerifierBackend(
        batches=[{2: {"class": "ambiguous", "confidence": 0.55}}]
    )
    bad = _answer([{"instance_id": 2, "field": "class", "value": "screw"}])
    orch, llm = _orch(backend, [bad, copy.deepcopy(bad)])
    result = _run(orch.run("q", _payload()))
    assert result.status == "human_review"
    assert result.answer is None  # 答案扣留，绝不透传
    assert len(llm.calls) == 2  # 首答 + 1 次重生成，预算封顶
    assert result.anchor_violations  # 违规记录在案
    assert result.session_summary["final_state"] == StateCode.S6.value


# ------------------------------------------------------------------ 场景 6


def test_confident_verification_adopted():
    """清晰 + 验证置信度差值 > 0.2 → 采纳验证类别，claims 按新类别锚定。"""
    backend = StubVerifierBackend(
        batches=[{2: {"class": "bolt", "exists": True, "confidence": 0.95}}]
    )
    orch, llm = _orch(
        backend,
        [_answer([{"instance_id": 2, "field": "class", "value": "bolt"},
                  {"field": "count", "value": 3}])],
    )
    result = _run(orch.run("第二个零件是什么", _payload(sharp2=0.8)))
    assert result.status == "delivered"
    assert result.class_overrides == {2: "bolt"}
    # Prompt 中实例 2 的类别已是采纳后的 bolt，且含 class_overrides 语义的规则
    assert '"class": "bolt"' in llm.calls[0]["context"]
    events = [t["event"] for t in result.session_summary["transitions"]]
    assert "VERIFICATION_CONFLICT" in events and "CONFLICT_RESOLVED" in events


# ------------------------------------------------------------------ 场景 7


def test_unresolvable_conflict_goes_human_review_without_llm():
    backend = StubVerifierBackend(
        batches=[{2: {"class": "bolt", "confidence": 0.6}}]  # 差值 0.05 < 0.2
    )
    orch, llm = _orch(backend, [])
    result = _run(orch.run("q", _payload(sharp2=0.8)))
    assert result.status == "human_review"
    assert result.answer is None
    assert len(llm.calls) == 0  # 冲突未解决前不消耗 L3
    assert result.session_summary["final_state"] == StateCode.S6.value


# ------------------------------------------------------------------ 场景 8


def test_l3_timeout_aborts():
    backend = StubVerifierBackend(batches=[{2: {"class": "screw"}}])
    orch, llm = _orch(
        backend, [_answer([{"field": "count", "value": 3}])],
        llm_timeout=0.15, llm_delay=0.3, verifier_timeout=0.04,
    )
    result = _run(orch.run("q", _payload()))
    assert result.status == "aborted"
    assert result.session_summary["final_state"] == StateCode.S5.value


# ------------------------------------------------------------------ 场景 9


def test_invalid_ir_rejected():
    orch, _ = _orch(StubVerifierBackend(), [])
    bad = _payload()
    bad["detections"][0]["bbox_px"] = [10, 10, 50, 5000]
    result = _run(orch.run("q", bad))
    assert result.status == "rejected"
    assert result.error["code"] == "E1002"
    assert result.session_summary["final_state"] == StateCode.S2.value


# ------------------------------------------------------------------ 场景 10


def test_dangling_dependency_without_rebuilder():
    orch, _ = _orch(StubVerifierBackend(), [])
    result = _run(orch.run("q", _payload(dangling=True)))
    assert result.status == "human_review"
    summary = result.session_summary
    assert summary["final_state"] == StateCode.S6.value
    events = [t["event"] for t in summary["transitions"]]
    assert "DEPENDENCY_MISSING" in events and "RETRY_FAILED" in events


def test_dangling_dependency_recovered_by_rebuilder():
    """S3 软重试成功：重建器返回无悬空引用的 IR，后续采用新 IR 并正常交付。"""
    orch, llm = _orch(
        StubVerifierBackend(),
        [_answer([{"field": "count", "value": 3}])],
    )

    def rebuilder():
        return _payload(verify2=False)  # 重建后无悬空引用

    result = _run(orch.run("q", _payload(dangling=True), ir_rebuilder=rebuilder))
    assert result.status == "delivered"
    events = [t["event"] for t in result.session_summary["transitions"]]
    assert "DEPENDENCY_MISSING" in events
    assert "RETRY_SUCCEEDED" in events
    assert "RETRY_FAILED" not in events
    assert result.session_summary["retries_used"] == 1
    assert len(llm.calls) == 1  # 重建后直接 L3 交付，无验证、无重生成


def test_rebuilder_returning_still_dangling_fails():
    """重建器仍返回悬空引用 → 依赖复检失败 → S6。"""
    orch, _ = _orch(StubVerifierBackend(), [])

    def rebuilder():
        return _payload(dangling=True)  # 仍悬空

    result = _run(orch.run("q", _payload(dangling=True), ir_rebuilder=rebuilder))
    assert result.status == "human_review"
    events = [t["event"] for t in result.session_summary["transitions"]]
    assert "RETRY_FAILED" in events
    assert result.session_summary["final_state"] == StateCode.S6.value


# ------------------------------------------------------------------ 场景 11


def test_partial_verification_result_degrades_missing_instances():
    """L1.5 部分验证结果缺失时，缺失实例按 E2003 路径降级不阻塞"""
    payload = _payload()
    payload["detections"][0]["verification_required"] = True
    backend = StubVerifierBackend(
        batches=[{1: {"class": "screw", "confidence": 0.95}}]
    )
    claims = [{"field": "count", "value": 3}]
    orch, llm = _orch(backend, [_answer(claims)])
    result = _run(orch.run("q", payload))
    assert result.status == "delivered_with_review"
    assert result.degraded_instances == {2}
    assert backend.calls == [[1, 2]]
    assert len(llm.calls) == 1
    events = [t["event"] for t in result.session_summary["transitions"]]
    assert "VERIFICATION_TIMEOUT" in events
    assert "VERIFICATION_SKIPPED" not in events


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


def test_empty_successful_verification_degrades_instead_of_retry_loop():
    """真实占位后端返回 success+空结果时，必须降级交付，不能耗尽 S3 重试进 S6 空答。"""
    backend = StubVerifierBackend(batches=[{}])
    orch, llm = _orch(
        backend,
        [_answer([{"field": "count", "value": 3}], text="检测到 3 个目标，低置信目标待复审")],
    )
    result = _run(orch.run("图里有几个目标", _payload()))
    assert result.status == "delivered_with_review"
    assert result.answer is not None
    assert result.degraded_instances == {2}
    assert backend.calls == [[2]]
    assert len(llm.calls) == 1
    events = [t["event"] for t in result.session_summary["transitions"]]
    assert events == [
        "VALIDATION_PASSED",
        "VERIFICATION_REQUESTED",
        "VERIFICATION_TIMEOUT",
    ]
