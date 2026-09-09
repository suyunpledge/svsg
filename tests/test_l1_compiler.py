"""L1 编译器管线端到端测试：Stub 检测器 → 不确定性估计 → IR 组装。"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime

from svsg.contracts import ConfLevel, ErrorCode, RelationType, SceneType, SVSGError
from svsg.l1_compiler import (
    ImageMeta,
    L1Compiler,
    L1CompilerConfig,
    RawDetection,
    StubDetector,
    binary_entropy,
    estimate_conf_level,
)
from svsg.l1_compiler.compiler import CameraIntrinsics

META = ImageMeta(
    image_id="img_test_001",
    width=1920,
    height=1080,
    timestamp=datetime(2026, 9, 2, 10, 0, 0, tzinfo=UTC),
)

# 高置信螺丝（清晰、高分）+ 低置信模糊件
RAWS = [
    RawDetection(
        label="screw",
        bbox=(420, 150, 480, 210),
        detection_score=0.92,
        local_edge_sharpness=0.73,
    ),
    RawDetection(
        label="part",
        bbox=(900, 400, 1000, 500),
        detection_score=0.55,
        local_edge_sharpness=0.3,  # 模糊
    ),
]


def test_binary_entropy_bounds():
    assert math.isclose(binary_entropy(0.5), math.log(2.0), rel_tol=1e-9)
    assert binary_entropy(0.99) < 0.1
    assert binary_entropy(0.5) > binary_entropy(0.8)


def test_conf_level_mapping():
    # 高分 + 清晰 + 低熵 → high
    assert (
        estimate_conf_level(
            classification_entropy=0.08,
            local_edge_sharpness=0.73,
            detection_score=0.92,
        )
        == ConfLevel.HIGH
    )
    # 模糊 → 降一级
    assert (
        estimate_conf_level(
            classification_entropy=0.08,
            local_edge_sharpness=0.3,
            detection_score=0.92,
        )
        == ConfLevel.MEDIUM
    )
    # 模糊 + 低分 → 连降两级到 low
    assert (
        estimate_conf_level(
            classification_entropy=0.08,
            local_edge_sharpness=0.3,
            detection_score=0.55,
        )
        == ConfLevel.LOW
    )
    # 高熵 → 基准即 low，不再下降
    assert (
        estimate_conf_level(
            classification_entropy=0.9,
            local_edge_sharpness=0.9,
            detection_score=0.99,
        )
        == ConfLevel.LOW
    )
    # 估计器无权产出 lowest（该档位专属 S5 超时降级）
    for entropy in (0.0, 0.4, 0.6, 0.9):
        for sharpness in (0.0, 0.3, 0.9):
            for score in (0.1, 0.5, 0.9):
                assert (
                    estimate_conf_level(
                        classification_entropy=entropy,
                        local_edge_sharpness=sharpness,
                        detection_score=score,
                    )
                    != ConfLevel.LOWEST
                )


def test_compile_full_pipeline_orthographic():
    compiler = L1Compiler(
        StubDetector(RAWS), L1CompilerConfig(scene_type=SceneType.ORTHOGRAPHIC)
    )
    ir = compiler.compile(b"<fake-image>", META)

    assert ir.schema_version == "5.2.1"
    assert ir.scene_type == SceneType.ORTHOGRAPHIC
    assert len(ir.detections) == 2

    d1, d2 = ir.detections
    # 高分清晰件 → high，不触发验证
    assert d1.conf_level == ConfLevel.HIGH
    assert d1.verification_required is False
    # 模糊低分件 → low，默认策略触发 L1.5 验证（S4a 入口）
    assert d2.conf_level == ConfLevel.LOW
    assert d2.verification_required is True

    # 熵由 score 推导（Stub 未显式提供）
    assert math.isclose(
        d1.uncertainty.classification_entropy, binary_entropy(0.92), rel_tol=1e-9
    )

    # 正视场景：距离关系放行，且两实例互为最近邻
    rels_1 = {r.type for r in d1.relations}
    assert RelationType.LEFT_OF in rels_1
    assert RelationType.NEAREST_TO in rels_1
    assert d1.relations[0].spatial_consistency in (0.0, 1.0)


def test_compile_unknown_scene_suppresses_distance():
    compiler = L1Compiler(
        StubDetector(RAWS), L1CompilerConfig(scene_type=SceneType.UNKNOWN)
    )
    ir = compiler.compile(b"<fake-image>", META)
    for det in ir.detections:
        assert all(
            r.type not in (RelationType.NEAREST_TO, RelationType.DISTANCE_TO)
            for r in det.relations
        )


def test_compile_perspective_with_intrinsics_allows_distance():
    compiler = L1Compiler(
        StubDetector(RAWS),
        L1CompilerConfig(
            scene_type=SceneType.PERSPECTIVE,
            camera_intrinsics=CameraIntrinsics(fx=800.0, fy=800.0, cx=960.0, cy=540.0),
        ),
    )
    ir = compiler.compile(b"<fake-image>", META)
    assert any(
        r.type == RelationType.NEAREST_TO for det in ir.detections for r in det.relations
    )


def test_compile_bbox_out_of_range_raises_s2():
    bad = [
        RawDetection(
            label="screw",
            bbox=(420, 150, 480, 2100),  # y2 超出图像高度
            detection_score=0.9,
        )
    ]
    compiler = L1Compiler(StubDetector(bad))
    try:
        compiler.compile(b"<fake-image>", META)
    except SVSGError as exc:
        assert exc.code == ErrorCode.BBOX_OUT_OF_RANGE
        assert exc.state.value == "S2"
    else:
        raise AssertionError("预期 SVSGError(E1002)")


def test_verification_policy_override():
    # 质检场景：MEDIUM 也触发验证
    policy = {
        ConfLevel.HIGH: False,
        ConfLevel.MEDIUM: True,
        ConfLevel.LOW: True,
    }
    raws = [
        RawDetection(
            label="part",
            bbox=(100, 100, 200, 200),
            detection_score=0.99,  # 低熵 → 基准 high
            local_edge_sharpness=0.3,  # 模糊 → 降一级为 medium
        )
    ]
    compiler = L1Compiler(
        StubDetector(raws), L1CompilerConfig(verification_policy=policy)
    )
    ir = compiler.compile(b"<fake-image>", META)
    assert ir.detections[0].conf_level == ConfLevel.MEDIUM
    assert ir.detections[0].verification_required is True


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
