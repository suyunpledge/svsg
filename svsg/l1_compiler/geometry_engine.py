"""几何关系计算器：基于 bbox 坐标的确定性纯函数。

V5.2.1 §3.3 的机器可执行化：
- 方向性关系（left_of/right_of/above/below/contains/overlaps）由坐标
  序严格判定，spatial_consistency 输出 0.0/1.0 二值，L1 与 L2 共用
  同一实现（compute_spatial_consistency），保证校验口径一致。
- 距离类关系（nearest_to/distance_to）受场景门控：
  正视恒可用；透视需相机内参；unknown 按透视未标定保守处理
  （与 IR Schema 的硬校验规则同源，双重保险）。
- metric_accuracy：像素→物理距离的映射置信度。本引擎默认输出 null
  （image_plane_px 度量），物理度量需深度校正，属 Phase 2+ 范畴，
  由调用方通过 override_metric_accuracy 显式注入。
"""

from __future__ import annotations

import math

from svsg.contracts import MetricType, Relation, RelationType, SceneType

BBox = tuple[int, int, int, int]


def center(bbox: BBox) -> tuple[float, float]:
    """bbox 中心点。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance_px(a: BBox, b: BBox) -> float:
    """两 bbox 中心的欧氏距离（图像平面像素，非物理距离）。"""
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(ax - bx, ay - by)


def relation_holds(rel: RelationType, a: BBox, b: BBox) -> bool:
    """方向性关系的严格几何判定（a 为源实例，b 为目标实例）。

    距离类关系不是布尔谓词，传入将抛 ValueError（调用方应走
    center_distance_px / compute_spatial_consistency 路径）。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if rel == RelationType.LEFT_OF:
        return ax2 < bx1
    if rel == RelationType.RIGHT_OF:
        return bx2 < ax1
    if rel == RelationType.ABOVE:
        return ay2 < by1
    if rel == RelationType.BELOW:
        return by2 < ay1
    if rel == RelationType.CONTAINS:
        # a 完全包含 b（含边界相切的退化情形）
        return ax1 <= bx1 and ay1 <= by1 and bx2 <= ax2 and by2 <= ay2
    if rel == RelationType.OVERLAPS:
        return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2
    raise ValueError(f"{rel.value} 是距离类关系，不支持布尔判定")


def compute_spatial_consistency(rel: RelationType, a: BBox, b: BBox) -> float:
    """空间逻辑一致性：L1 生成与 L2 复核共用的唯一裁判（纯函数）。

    - 方向性关系：谓词成立 → 1.0，否则 0.0。
    - 距离类关系：中心距离恒可计算，几何逻辑本身确定 → 恒 1.0；
      其可信度由 metric_accuracy 表达，与本字段正交（V5.2.1 拆分语义）。
    """
    if rel in (RelationType.NEAREST_TO, RelationType.DISTANCE_TO):
        return 1.0
    return 1.0 if relation_holds(rel, a, b) else 0.0


def distance_relations_allowed(
    scene_type: SceneType, has_intrinsics: bool
) -> bool:
    """距离类关系的场景门控（与 IR Schema 硬校验同源的规则）。"""
    if scene_type == SceneType.ORTHOGRAPHIC:
        return True
    if scene_type == SceneType.PERSPECTIVE:
        return has_intrinsics
    return False  # unknown 按透视未标定保守处理


def nearest_neighbor(
    source_idx: int, bboxes: list[BBox]
) -> tuple[int, float] | None:
    """返回 source 的最近邻 (索引, 中心像素距离)；无其他实例时为 None。"""
    best: tuple[int, float] | None = None
    for j, other in enumerate(bboxes):
        if j == source_idx:
            continue
        d = center_distance_px(bboxes[source_idx], other)
        if best is None or d < best[1]:
            best = (j, d)
    return best


def build_directional_relations(
    instance_ids: list[int], bboxes: list[BBox]
) -> dict[int, list[Relation]]:
    """为每个实例生成所有成立的方向性关系（O(n²) 全对扫描）。

    互斥对（left_of/right_of、above/below）会双向各出一条，
    contains 与 overlaps 仅在几何定义成立时输出。
    """
    result: dict[int, list[Relation]] = {iid: [] for iid in instance_ids}
    for i, iid_a in enumerate(instance_ids):
        for j, iid_b in enumerate(instance_ids):
            if i == j:
                continue
            for rel_type in (
                RelationType.LEFT_OF,
                RelationType.RIGHT_OF,
                RelationType.ABOVE,
                RelationType.BELOW,
                RelationType.CONTAINS,
                RelationType.OVERLAPS,
            ):
                if relation_holds(rel_type, bboxes[i], bboxes[j]):
                    result[iid_a].append(
                        Relation(
                            type=rel_type,
                            target_id=iid_b,
                            spatial_consistency=1.0,
                            metric_accuracy=None,
                        )
                    )
    return result


def build_nearest_relations(
    instance_ids: list[int],
    bboxes: list[BBox],
    *,
    scene_type: SceneType,
    has_intrinsics: bool,
    override_metric_accuracy: float | None = None,
) -> dict[int, list[Relation]]:
    """为每个实例生成 nearest_to 关系（受场景门控约束）。

    门控不通过时返回空映射（不产出距离类关系），与 IR Schema 的
    硬校验形成双保险：即使调用方绕过门控，Schema 也会以 S2 拒绝。
    """
    result: dict[int, list[Relation]] = {iid: [] for iid in instance_ids}
    if not distance_relations_allowed(scene_type, has_intrinsics):
        return result
    metric_type = (
        MetricType.PHYSICAL_UNIT if has_intrinsics else MetricType.IMAGE_PLANE_PX
    )
    for i, iid in enumerate(instance_ids):
        nb = nearest_neighbor(i, bboxes)
        if nb is None:
            continue
        j, dist = nb
        result[iid].append(
            Relation(
                type=RelationType.NEAREST_TO,
                target_id=instance_ids[j],
                spatial_consistency=1.0,
                metric_accuracy=override_metric_accuracy,
                metric_type=metric_type,
                distance_px=dist,
            )
        )
    return result


def distance_between(
    source_id: int, target_id: int, a: BBox, b: BBox
) -> Relation:
    """按需构造 distance_to 关系（显式请求时使用，避免 O(n²) 全量输出）。"""
    return Relation(
        type=RelationType.DISTANCE_TO,
        target_id=target_id,
        spatial_consistency=1.0,
        metric_accuracy=None,
        metric_type=MetricType.IMAGE_PLANE_PX,
        distance_px=center_distance_px(a, b),
    )
