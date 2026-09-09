"""contracts 契约测试：黄金 IR fixture + 关键负向用例。

覆盖点：
1. 黄金 IR fixture（方案附录示例）通过校验并可 round-trip；
2. V5.2.1 修订 #2：未标定透视/未知场景硬拒绝距离类关系，标定后放行；
3. V5.2.1 修订 #2：spatial_consistency 仅允许 0.0/1.0；
4. 版本锁定：旧版遗留 relations.confidence 字段被 extra="forbid" 拒绝；
5. S2/S3 责任划分：bbox 越界映射 E1002，未知引用走软检查而非解析失败；
6. L1.5 验证报告与 L3 双通道答案契约可解析。
"""

from __future__ import annotations

import copy
import json
import sys

from pydantic import ValidationError

from svsg.contracts import (
    IR,
    ErrorCode,
    L3Answer,
    RelationType,
    VerificationReport,
    VerificationStatus,
    check_relation_references,
    from_validation_error,
)

# 方案附录：IR Level 2 JSON Schema 示例（V5.2.1）原样作为黄金 fixture
GOLDEN_IR = {
    "schema_version": "5.2.1",
    "image_id": "img_20260902_001",
    "image_width": 1920,
    "image_height": 1080,
    "timestamp": "2026-09-02T10:00:00Z",
    "scene_type": "orthographic",
    "detector_model": "yolov8n_v2",
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
            "relations": [
                {
                    "type": "left_of",
                    "target_id": 2,
                    "spatial_consistency": 1.0,
                    "metric_accuracy": None,
                    "metric_type": "image_plane_px",
                }
            ],
        }
    ],
}


def _must_fail_validation(payload: dict) -> ValidationError:
    try:
        IR.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("预期校验失败，但通过了")


def test_golden_ir_parses_and_roundtrips():
    ir = IR.model_validate(GOLDEN_IR)
    assert ir.schema_version == "5.2.1"
    assert ir.detections[0].object_class == "screw"
    assert ir.detections[0].relations[0].spatial_consistency == 1.0
    # 序列化保持 "class" 别名，round-trip 幂等
    ir2 = IR.model_validate(json.loads(ir.model_dump_json(by_alias=True)))
    assert ir2 == ir


def test_golden_ir_target_reference_is_soft_checked():
    # 附录示例中 target_id=2 的实例未在 detections 中列出（示例截断），
    # Schema 应放行（S3 语义），由软检查捕获。
    ir = IR.model_validate(GOLDEN_IR)
    broken = check_relation_references(ir)
    assert broken == [(1, RelationType.LEFT_OF, 2)]


def test_perspective_uncalibrated_rejects_distance_relation():
    bad = copy.deepcopy(GOLDEN_IR)
    bad["scene_type"] = "perspective"
    bad["detections"][0]["relations"][0].update(
        {"type": "distance_to", "distance_px": 200.0, "metric_type": "image_plane_px"}
    )
    err = from_validation_error(_must_fail_validation(bad))
    assert err.code == ErrorCode.SCHEMA_INVALID
    assert err.state.value == "S2"


def test_unknown_scene_rejects_distance_relation():
    bad = copy.deepcopy(GOLDEN_IR)
    bad["scene_type"] = "unknown"
    bad["detections"][0]["relations"][0].update(
        {"type": "nearest_to", "distance_px": 120.0, "metric_type": "image_plane_px"}
    )
    err = from_validation_error(_must_fail_validation(bad))
    assert err.state.value == "S2"


def test_calibrated_perspective_allows_physical_metric():
    good = copy.deepcopy(GOLDEN_IR)
    good["scene_type"] = "perspective"
    good["camera_intrinsics"] = {"fx": 800.0, "fy": 800.0, "cx": 960.0, "cy": 540.0}
    good["detections"][0]["relations"][0].update(
        {
            "type": "distance_to",
            "distance_px": 200.0,
            "metric_type": "physical_unit",
            "metric_accuracy": 0.85,
        }
    )
    ir = IR.model_validate(good)
    assert ir.detections[0].relations[0].metric_accuracy == 0.85


def test_spatial_consistency_must_be_binary():
    bad = copy.deepcopy(GOLDEN_IR)
    bad["detections"][0]["relations"][0]["spatial_consistency"] = 0.7
    _must_fail_validation(bad)


def test_legacy_confidence_field_rejected():
    # V5.2 旧版字段残留应被版本锁定（extra="forbid"）捕获
    bad = copy.deepcopy(GOLDEN_IR)
    bad["detections"][0]["relations"][0]["confidence"] = 1.0
    _must_fail_validation(bad)


def test_bbox_out_of_range_maps_to_e1002():
    bad = copy.deepcopy(GOLDEN_IR)
    bad["detections"][0]["bbox_px"] = [420, 150, 480, 2100]  # y2 > image_height
    err = from_validation_error(_must_fail_validation(bad))
    assert err.code == ErrorCode.BBOX_OUT_OF_RANGE
    assert err.state.value == "S2"


def test_verification_report_contract():
    report = VerificationReport.model_validate(
        {
            "report_id": "vr_001",
            "image_id": "img_20260902_001",
            "status": "success",
            "results": [
                {"instance_id": 1, "class": "ambiguous", "exists": True, "confidence": 0.55}
            ],
        }
    )
    assert report.status == VerificationStatus.SUCCESS
    assert report.results[0].class_label == "ambiguous"


def test_l3_dual_channel_answer_contract():
    answer = L3Answer.model_validate(
        {
            "claims": [
                {"instance_id": 1, "field": "class", "value": "screw"},
                {"instance_id": None, "field": "count", "value": 3},
            ],
            "final_answer": "图中检测到 3 颗螺丝。",
        }
    )
    assert answer.claims[0].value == "screw"
    assert answer.claims[1].value == 3


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
