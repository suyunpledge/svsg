"""L1 视觉编译器输出的结构化 IR（Intermediate Representation）Schema。

对应《SVSG V5.2.1 技术方案》§6 数据协议规范（IR Level 2 JSON Schema 示例）。

校验责任划分（重要设计决策）：
- 本 Schema 承担 S2（Invalid Parameter）硬拒绝校验：字段缺失、类型错误、
  bbox 越界/退化、instance_id 重复、未标定场景输出距离类关系等。
  解析抛出的 ValidationError 由 errors.from_validation_error() 映射为 S2。
- 跨实例引用（relations.target_id 指向不存在的 instance_id）刻意不做
  硬校验（否则无法与 S2 区分），由 check_relation_references() 提供
  软检查，供 L2 路由到 S3（Missing Dependency，软重试最多 1 次）。
- 所有模型 extra="forbid"：生产锁定版本要求契约严格，Schema 漂移
  （如 V5.2 旧版残留的 relations.confidence 字段）应立即暴露为 S2 错误。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import DISTANCE_RELATIONS, ConfLevel, MetricType, RelationType, SceneType

IR_SCHEMA_VERSION: Literal["5.2.1"] = "5.2.1"

_STRICT = ConfigDict(populate_by_name=True, extra="forbid")


class Uncertainty(BaseModel):
    """不确定性估计（L1 不确定性估计器输出）。"""

    model_config = _STRICT

    classification_entropy: float = Field(
        ..., ge=0.0, description="分类熵，越低越确定"
    )
    local_edge_sharpness: float = Field(
        ..., ge=0.0, le=1.0, description="局部边缘锐度，低于阈值(默认0.4)判定模糊"
    )
    tta_variance: float | None = Field(
        default=None, ge=0.0, description="测试时增强方差（实验性），未启用时为 null"
    )


class Relation(BaseModel):
    """几何关系（V5.2.1：置信度已拆分为两个独立字段）。

    - spatial_consistency：空间逻辑一致性，由 L2 基于 bbox 坐标确定性
      计算（如 left_of 需满足 x2 < target.x1），仅允许 0.0 / 1.0，
      不依赖 LLM。
    - metric_accuracy：像素距离映射物理距离的置信度；仅在已启用度量的
      距离类关系上允许非 null（透视未标定场景强制为 null）。
    """

    model_config = _STRICT

    type: RelationType
    target_id: int = Field(..., ge=1, description="关系目标实例的 instance_id")
    spatial_consistency: float
    metric_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    metric_type: MetricType | None = None
    distance_px: float | None = Field(
        default=None, ge=0.0, description="图像平面像素距离，distance_to/nearest_to 时必填"
    )

    @field_validator("spatial_consistency")
    @classmethod
    def _spatial_consistency_is_binary(cls, v: float) -> float:
        if v not in (0.0, 1.0):
            raise ValueError("spatial_consistency 为确定性布尔语义，仅允许 0.0 或 1.0")
        return v


class Detection(BaseModel):
    """单个检测实例。"""

    model_config = _STRICT

    instance_id: int = Field(..., ge=1)
    object_class: str = Field(..., alias="class", min_length=1)
    bbox_px: tuple[int, int, int, int] = Field(
        ..., description="绝对整数像素坐标 [x1, y1, x2, y2]"
    )
    detection_score: float = Field(..., ge=0.0, le=1.0)
    conf_level: ConfLevel
    verification_required: bool = False
    uncertainty: Uncertainty
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("bbox_px")
    @classmethod
    def _bbox_well_formed(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = v
        if x1 < 0 or y1 < 0:
            raise ValueError("bbox 坐标不能为负")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox 需满足 x1 < x2 且 y1 < y2，退化框非法")
        return v


class CameraIntrinsics(BaseModel):
    """透视场景相机内参。提供后才允许输出物理度量（metric_accuracy / physical_unit）。"""

    model_config = _STRICT

    fx: float = Field(..., gt=0.0)
    fy: float = Field(..., gt=0.0)
    cx: float
    cy: float
    distortion_coeffs: list[float] | None = None


class IR(BaseModel):
    """L1 输出的顶层 IR 文档（Level 2）。"""

    model_config = _STRICT

    schema_version: Literal["5.2.1"] = IR_SCHEMA_VERSION
    image_id: str = Field(..., min_length=1)
    image_width: int = Field(..., ge=1)
    image_height: int = Field(..., ge=1)
    timestamp: datetime
    scene_type: SceneType
    detector_model: str = Field(..., min_length=1)
    camera_intrinsics: CameraIntrinsics | None = None
    detections: list[Detection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_checks(self) -> IR:
        # 1) instance_id 唯一
        ids = [d.instance_id for d in self.detections]
        if len(set(ids)) != len(ids):
            raise ValueError("instance_id 存在重复")

        # 2) bbox 落在图像边界内（S2 硬校验）
        for det in self.detections:
            _, _, x2, y2 = det.bbox_px
            if x2 > self.image_width or y2 > self.image_height:
                raise ValueError(
                    f"实例 {det.instance_id} 的 bbox 越出图像边界 "
                    f"({self.image_width}x{self.image_height})"
                )

        # 3) 距离类关系适用边界（V5.2.1 §3.3）
        distance_allowed = self._distance_relations_allowed()
        for det in self.detections:
            for rel in det.relations:
                if rel.type in DISTANCE_RELATIONS:
                    if not distance_allowed:
                        raise ValueError(
                            f"实例 {det.instance_id} 的距离类关系 {rel.type.value} "
                            f"在 scene_type={self.scene_type.value} 且未标定的场景中禁止输出"
                        )
                    if rel.metric_type is None:
                        raise ValueError(
                            f"实例 {det.instance_id} 的距离类关系必须声明 metric_type"
                        )
                    if rel.distance_px is None:
                        raise ValueError(
                            f"实例 {det.instance_id} 的距离类关系必须携带 distance_px"
                        )
                if rel.metric_accuracy is not None and not (
                    distance_allowed and rel.type in DISTANCE_RELATIONS
                ):
                    raise ValueError(
                        "metric_accuracy 仅允许出现在已启用度量的距离类关系上；"
                        "方向性关系及未标定场景必须为 null"
                    )
        return self

    def _distance_relations_allowed(self) -> bool:
        """正视/俯视恒可用；透视需提供相机内参；unknown 按透视未标定保守处理。"""
        if self.scene_type == SceneType.ORTHOGRAPHIC:
            return True
        if self.scene_type == SceneType.PERSPECTIVE:
            return self.camera_intrinsics is not None
        return False


def check_relation_references(ir: IR) -> list[tuple[int, RelationType, int]]:
    """软检查：返回引用了不存在实例的关系元组 (source_id, relation_type, target_id)。

    供 L2 路由 S3（Missing Dependency，软重试最多 1 次，失败进 S6），
    刻意不放入 Schema 硬校验，以区分 S2 与 S3 的语义。
    """
    known = {d.instance_id for d in ir.detections}
    broken: list[tuple[int, RelationType, int]] = []
    for det in ir.detections:
        for rel in det.relations:
            if rel.target_id not in known:
                broken.append((det.instance_id, rel.type, rel.target_id))
    return broken
