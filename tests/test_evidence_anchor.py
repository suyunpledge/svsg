"""证据锚定校验对抗测试：完整性 + 忠实度（V5.2.1 修订 #1 的验收核心）。

IR 场景：两个螺丝（1=清晰 high，2=模糊 low 需验证），一个垫片（3）。
"""

from __future__ import annotations

import sys

from svsg.contracts import (
    IR,
    ErrorCode,
    L3Answer,
    VerificationReport,
)
from svsg.l2_runtime import (
    anchor_claims,
    missing_verifications,
    run_anchor,
    validate_ir_payload,
)

IR_PAYLOAD = {
    "schema_version": "5.2.1",
    "image_id": "img_anchor_001",
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
            "verification_required": True,  # 模糊低分件 → 必须 L1.5 验证
            "uncertainty": {
                "classification_entropy": 0.6,
                "local_edge_sharpness": 0.3,
                "tta_variance": None,
            },
            "relations": [],
        },
        {
            "instance_id": 3,
            "class": "washer",
            "bbox_px": [1000, 400, 1060, 460],
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


def _ir() -> IR:
    return validate_ir_payload(IR_PAYLOAD)


def _answer(claims: list[dict], text: str = "答案") -> L3Answer:
    return L3Answer.model_validate({"claims": claims, "final_answer": text})


def _report(results: list[dict], status: str = "success") -> VerificationReport:
    return VerificationReport.model_validate(
        {
            "report_id": "vr_anchor_001",
            "image_id": "img_anchor_001",
            "status": status,
            "results": results,
        }
    )


# ---------------------------------------------------------------- 完整性


def test_missing_verification_detected():
    ir = _ir()
    assert missing_verifications(ir, executed={2}) == set()
    assert missing_verifications(ir, executed=set()) == {2}
    # 非 required 实例不需要出现在执行集合中
    assert missing_verifications(ir, executed=set()) != {1, 2, 3}


def test_completeness_violation_maps_to_e3003():
    verdict = run_anchor(_ir(), _answer([]), executed_verifications=set())
    assert verdict.passed is False
    assert verdict.primary_code == ErrorCode.VERIFICATION_SKIPPED
    assert any(v.code == ErrorCode.VERIFICATION_SKIPPED for v in verdict.violations)


# ---------------------------------------------------------------- 忠实度


def test_consistent_claims_pass():
    answer = _answer(
        [
            {"instance_id": 1, "field": "class", "value": "screw"},
            {"field": "count", "value": 3},
            {"field": "count:screw", "value": 2},
            {"instance_id": 3, "field": "exists", "value": True},
            # 实例1(左) 在 实例3(右) 左边：bbox 480 < 1000 成立
            {"instance_id": 1, "field": "relation:left_of", "value": 3},
        ]
    )
    verdict = run_anchor(
        _ir(), answer, executed_verifications={2}, report=_report(
            [{"instance_id": 2, "class": "screw", "exists": True, "confidence": 0.8}]
        )
    )
    assert verdict.passed, [v.to_dict() for v in verdict.violations]


def test_ambiguous_report_vetoes_concrete_claim():
    """V5.2.1 原文场景：L1.5 报 ambiguous，L3 输出"螺丝" → 无条件否决。"""
    answer = _answer(
        [{"instance_id": 2, "field": "class", "value": "screw"}]
    )
    report = _report(
        [{"instance_id": 2, "class": "ambiguous", "exists": True, "confidence": 0.55}]
    )
    verdict = run_anchor(_ir(), answer, executed_verifications={2}, report=report)
    assert verdict.passed is False
    assert verdict.primary_code == ErrorCode.EVIDENCE_MISMATCH
    assert any("ambiguous" in v.reason for v in verdict.violations)


def test_class_claim_against_ir_anchor():
    # 断言类别与 IR 检测结果不一致 → 否决
    answer = _answer([{"instance_id": 1, "field": "class", "value": "bolt"}])
    verdict = run_anchor(_ir(), answer, executed_verifications={2})
    assert verdict.passed is False
    assert any("IR 检测结果" in v.reason for v in verdict.violations)


def test_verification_report_overrides_detector():
    # 断言与验证报告的类别不一致（即使与 IR 一致）→ 否决
    answer = _answer([{"instance_id": 2, "field": "class", "value": "screw"}])
    report = _report(
        [{"instance_id": 2, "class": "bolt", "exists": True, "confidence": 0.9}]
    )
    verdict = run_anchor(_ir(), answer, executed_verifications={2}, report=report)
    assert verdict.passed is False
    assert any("验证报告" in v.reason for v in verdict.violations)


def test_count_claims():
    ir = _ir()
    # 总数错误
    v1 = anchor_claims(ir, _answer([{"field": "count", "value": 5}]))
    assert len(v1) == 1 and "实例总数" in v1[0].reason
    # 分类数错误
    v2 = anchor_claims(ir, _answer([{"field": "count:screw", "value": 3}]))
    assert len(v2) == 1 and "count:screw" in v2[0].reason
    # 分类数正确
    v3 = anchor_claims(ir, _answer([{"field": "count:washer", "value": 1}]))
    assert v3 == []


def test_exists_claims():
    ir = _ir()
    report = _report([{"instance_id": 2, "class": "screw", "exists": False}])
    # 验证说不存在，L3 说存在 → 否决
    v1 = anchor_claims(ir, _answer([{"instance_id": 2, "field": "exists", "value": True}]), report)
    assert len(v1) == 1 and "exists=False" in v1[0].reason
    # IR 中存在的实例，L3 断言不存在 → 否决
    v2 = anchor_claims(ir, _answer([{"instance_id": 1, "field": "exists", "value": False}]))
    assert len(v2) == 1


def test_relation_claim_recomputed_by_geometry_engine():
    ir = _ir()
    # 实例2(左) 声称在实例3(右)左边：900~1000 < 1000~1060，x2=1000 不小于 1000 → 不成立
    v = anchor_claims(ir, _answer([{"instance_id": 2, "field": "relation:left_of", "value": 3}]))
    assert len(v) == 1 and "几何引擎" in v[0].reason
    # 实例3 声称在实例2 右边：bbox 1000 > 1000？ x1=1000 不大于 source.x2=1000 → 不成立（相切不算）
    v2 = anchor_claims(ir, _answer([{"instance_id": 3, "field": "relation:right_of", "value": 2}]))
    assert len(v2) == 1
    # 实例1 在实例2 左边：480 < 900 成立
    v3 = anchor_claims(ir, _answer([{"instance_id": 1, "field": "relation:left_of", "value": 2}]))
    assert v3 == []


def test_degraded_instance_anchor_ir_only():
    """S5 降级实例（无验证结果）：claims 仅锚定 IR，检测器结论仍可用。"""
    ir = _ir()
    # 实例2 被降级 → 报告中无其结果，class 断言只与 IR 比对
    report = _report([{"instance_id": 3, "class": "washer", "exists": True}])
    verdict = run_anchor(
        ir,
        _answer([{"instance_id": 2, "field": "class", "value": "screw"}]),
        executed_verifications={2},  # 曾尝试验证（超时前已计入执行集合）
        report=report,
    )
    assert verdict.passed, [v.to_dict() for v in verdict.violations]


def test_claim_referencing_unknown_instance_rejected():
    ir = _ir()
    v = anchor_claims(ir, _answer([{"instance_id": 99, "field": "class", "value": "screw"}]))
    assert len(v) == 1 and "不存在" in v[0].reason


def test_timeout_report_not_used_as_evidence():
    """status=timeout 的报告不作为比对基准（静默降级语义）。"""
    ir = _ir()
    answer = _answer([{"instance_id": 2, "field": "class", "value": "screw"}])
    report = _report(
        [{"instance_id": 2, "class": "ambiguous", "confidence": 0.4}], status="timeout"
    )
    verdict = run_anchor(ir, answer, executed_verifications={2}, report=report)
    assert verdict.passed  # 超时报告不参与锚定，仅 IR 比对通过


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
