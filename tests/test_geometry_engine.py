"""geometry_engine 纯函数穷举测试。"""

from __future__ import annotations

import math
import sys

from svsg.contracts import RelationType, SceneType
from svsg.l1_compiler.geometry_engine import (
    center_distance_px,
    compute_spatial_consistency,
    distance_between,
    distance_relations_allowed,
    nearest_neighbor,
    relation_holds,
)

# 两个水平分离的框：A 在左，B 在右
A = (100, 100, 200, 200)
B = (400, 150, 500, 250)
# 嵌套框：OUT 完全包含 IN
OUT = (50, 50, 600, 600)
IN = (200, 200, 300, 300)
# 部分重叠
C = (150, 150, 250, 250)


def test_directional_predicates():
    assert relation_holds(RelationType.LEFT_OF, A, B)
    assert relation_holds(RelationType.RIGHT_OF, B, A)
    assert not relation_holds(RelationType.LEFT_OF, B, A)
    assert relation_holds(RelationType.ABOVE, A, (400, 400, 500, 500))
    assert relation_holds(RelationType.BELOW, (400, 400, 500, 500), A)
    assert relation_holds(RelationType.CONTAINS, OUT, IN)
    assert not relation_holds(RelationType.CONTAINS, IN, OUT)
    assert relation_holds(RelationType.OVERLAPS, A, C)
    assert not relation_holds(RelationType.OVERLAPS, A, B)
    # 相离的框不满足 contains，边界相切视为包含（文档化语义）
    assert relation_holds(RelationType.CONTAINS, OUT, (50, 50, 60, 60))


def test_distance_relations_reject_boolean_predicate():
    for rel in (RelationType.NEAREST_TO, RelationType.DISTANCE_TO):
        try:
            relation_holds(rel, A, B)
        except ValueError:
            continue
        raise AssertionError(f"{rel} 应拒绝布尔判定")


def test_spatial_consistency_is_binary():
    assert compute_spatial_consistency(RelationType.LEFT_OF, A, B) == 1.0
    assert compute_spatial_consistency(RelationType.LEFT_OF, B, A) == 0.0
    # 距离类关系：几何逻辑恒确定
    assert compute_spatial_consistency(RelationType.DISTANCE_TO, A, B) == 1.0
    assert compute_spatial_consistency(RelationType.NEAREST_TO, A, B) == 1.0


def test_center_distance():
    # A 中心 (150,150)，B 中心 (450,200)
    d = center_distance_px(A, B)
    assert math.isclose(d, math.hypot(300.0, 50.0), rel_tol=1e-9)


def test_scene_gating():
    assert distance_relations_allowed(SceneType.ORTHOGRAPHIC, False)
    assert not distance_relations_allowed(SceneType.PERSPECTIVE, False)
    assert distance_relations_allowed(SceneType.PERSPECTIVE, True)
    assert not distance_relations_allowed(SceneType.UNKNOWN, True)  # unknown 保守


def test_nearest_neighbor():
    boxes = [A, B, (1000, 1000, 1100, 1100)]
    idx, dist = nearest_neighbor(0, boxes)
    assert idx == 1  # B 距 A 最近
    assert math.isclose(dist, center_distance_px(A, B))
    assert nearest_neighbor(0, [A]) is None


def test_distance_between_relation():
    rel = distance_between(1, 2, A, B)
    assert rel.type == RelationType.DISTANCE_TO
    assert rel.target_id == 2
    assert rel.spatial_consistency == 1.0
    assert math.isclose(rel.distance_px, center_distance_px(A, B))


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
