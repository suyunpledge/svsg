"""冲突解决策略 + 意图路由 + S0 校验管道测试。"""

from __future__ import annotations

import sys

from svsg.contracts import (
    Detection,
    ErrorCode,
    IntentType,
    SVSGError,
    Uncertainty,
    VerificationResult,
)
from svsg.l2_runtime import (
    ConflictThresholds,
    ResolutionOutcome,
    check_dependencies,
    resolve_conflict,
    route_intent,
    validate_ir_payload,
)


def _det(
    *, score: float, sharpness: float, cls: str = "screw", iid: int = 1
) -> Detection:
    return Detection(
        instance_id=iid,
        **{"class": cls},
        bbox_px=(100, 100, 200, 200),
        detection_score=score,
        conf_level="high",
        verification_required=False,
        uncertainty=Uncertainty(
            classification_entropy=0.1,
            local_edge_sharpness=sharpness,
        ),
    )


def _result(label: str | None, confidence: float | None) -> VerificationResult:
    return VerificationResult(
        instance_id=1, **{"class": label}, confidence=confidence
    )


# ------------------------------------------------------------- 冲突解决


def test_ambiguous_verification_unresolved():
    res = resolve_conflict(_det(score=0.6, sharpness=0.8), _result("ambiguous", 0.5))
    assert res.outcome == ResolutionOutcome.UNRESOLVED
    res2 = resolve_conflict(_det(score=0.6, sharpness=0.8), _result(None, 0.9))
    assert res2.outcome == ResolutionOutcome.UNRESOLVED


def test_blurry_image_degrades_and_disputes():
    res = resolve_conflict(
        _det(score=0.6, sharpness=0.3),  # 锐度 < 0.4 → 模糊
        _result("bolt", 0.95),
    )
    assert res.outcome == ResolutionOutcome.KEEP_DETECTOR_DISPUTED
    assert res.degraded_conf_level == "lowest"
    assert res.adopted_class is None
    assert "模糊" in res.reason


def test_confident_verification_adopted():
    res = resolve_conflict(
        _det(score=0.6, sharpness=0.8),
        _result("bolt", 0.95),  # 差值 0.35 > 0.2
    )
    assert res.outcome == ResolutionOutcome.ADOPT_VERIFICATION
    assert res.adopted_class == "bolt"


def test_small_confidence_gap_unresolved():
    res = resolve_conflict(
        _det(score=0.6, sharpness=0.8),
        _result("bolt", 0.7),  # 差值 0.1 < 0.2
    )
    assert res.outcome == ResolutionOutcome.UNRESOLVED


def test_thresholds_injectable():
    # 阈值收紧到 0.05 后，0.1 的差值即可采纳（Phase 2 标定场景）
    res = resolve_conflict(
        _det(score=0.6, sharpness=0.8),
        _result("bolt", 0.7),
        ConflictThresholds(confidence_diff_threshold=0.05),
    )
    assert res.outcome == ResolutionOutcome.ADOPT_VERIFICATION


# ------------------------------------------------------------- 意图路由


def test_intent_routing_priority():
    cases = {
        "图里有几个螺丝？": IntentType.COUNT,
        "左边那个是什么零件": IntentType.SPATIAL_RELATION,
        "有没有人在画面里": IntentType.EXISTENCE_CHECK,
        "那个螺丝是什么颜色的": IntentType.ATTRIBUTE_QUERY,
        "左上角区域描述一下": IntentType.REGION_QUERY,
        "检测一下图中所有物体": IntentType.OBJECT_DETECTION,
        "这张图和之前那张有什么变化": IntentType.TEMPORAL_RELATION,
        "how many screws are there": IntentType.COUNT,
    }
    for query, expected in cases.items():
        intent, score = route_intent(query)
        assert intent == expected, f"{query!r} → {intent}, 期望 {expected}"
        assert score >= 1.0


def test_intent_fallback():
    intent, score = route_intent("你好")
    assert intent == IntentType.OBJECT_DETECTION
    assert score == 0.0


# ------------------------------------------------------------- S0 管道


VALID = {
    "schema_version": "5.2.1",
    "image_id": "img_s0_001",
    "image_width": 640,
    "image_height": 480,
    "timestamp": "2026-09-02T10:00:00Z",
    "scene_type": "orthographic",
    "detector_model": "stub",
    "detections": [
        {
            "instance_id": 1,
            "class": "screw",
            "bbox_px": [10, 10, 50, 50],
            "detection_score": 0.9,
            "conf_level": "high",
            "verification_required": False,
            "uncertainty": {
                "classification_entropy": 0.1,
                "local_edge_sharpness": 0.8,
                "tta_variance": None,
            },
            "relations": [
                {
                    "type": "left_of",
                    "target_id": 7,  # 悬空引用
                    "spatial_consistency": 1.0,
                    "metric_accuracy": None,
                }
            ],
        }
    ],
}


def test_s0_pipeline_soft_vs_hard():
    # 悬空引用是 S3 软语义：校验通过，软检查捕获
    ir = validate_ir_payload(VALID)
    deps = check_dependencies(ir)
    assert deps == [(1, "left_of", 7)]

    # bbox 越界是 S2 硬语义：直接抛 SVSGError
    bad = dict(VALID)
    bad["detections"] = [
        dict(VALID["detections"][0], bbox_px=[10, 10, 50, 5000])
    ]
    try:
        validate_ir_payload(bad)
    except SVSGError as exc:
        assert exc.code == ErrorCode.BBOX_OUT_OF_RANGE
        assert exc.state.value == "S2"
    else:
        raise AssertionError("预期 S2 硬拒绝")


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
