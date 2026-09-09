"""L1 编译器管线：检测器原始输出 → 经 Schema 硬校验的 IR。

组装顺序（对应 V5.2.1 §3）：
1. detect：检测器适配层输出 RawDetection 列表；
2. uncertainty：熵（模型提供或由 score 推导）+ 锐度 + score → conf_level；
3. verification 标记：默认策略 LOW → verification_required=true
   （S4a 触发条件），档位策略可注入覆盖；
4. relations：几何引擎生成方向性 + 最近邻关系（距离类受场景门控）；
5. IR 组装：IR.model_validate 触发全部 S2 硬校验（bbox 越界、
   距离关系门控等），失败经 from_validation_error 映射为 SVSGError。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import ValidationError

from svsg.contracts import (
    IR,
    CameraIntrinsics,
    ConfLevel,
    Detection,
    Relation,
    SceneType,
    Uncertainty,
    from_validation_error,
)

from .detector_adapter import DetectorProtocol, ImageSource, RawDetection
from .geometry_engine import (
    build_directional_relations,
    build_nearest_relations,
)
from .uncertainty_estimator import Thresholds, binary_entropy, estimate_conf_level


@dataclass(frozen=True)
class ImageMeta:
    """图像元数据（L1 编译的输入之一，坐标协议的参照系）。"""

    image_id: str
    width: int
    height: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class L1CompilerConfig:
    """编译器配置。

    verification_policy：conf_level → 是否需要 L1.5 验证。
    默认仅 LOW 触发（控制验证成本），可按场景收紧（如质检台
    MEDIUM 也触发）。HIGH 恒不触发。
    """

    scene_type: SceneType = SceneType.UNKNOWN
    camera_intrinsics: CameraIntrinsics | None = None
    thresholds: Thresholds = Thresholds()
    verification_policy: dict[ConfLevel, bool] = field(
        default_factory=lambda: {
            ConfLevel.HIGH: False,
            ConfLevel.MEDIUM: False,
            ConfLevel.LOW: True,
        }
    )


class L1Compiler:
    """像素 → IR 的编译入口（无状态，依赖全部注入）。"""

    def __init__(
        self,
        detector: DetectorProtocol,
        config: L1CompilerConfig | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or L1CompilerConfig()

    @property
    def detector_model(self) -> str:
        return self._detector.model_name

    def compile(self, image: ImageSource, meta: ImageMeta) -> IR:
        """执行完整编译管线；任何 S2 级问题以 SVSGError 抛出。"""
        raw = self._detector.detect(image)
        return self._assemble(raw, meta)

    # ------------------------------------------------------------------

    def _assemble(self, raw: list[RawDetection], meta: ImageMeta) -> IR:
        cfg = self._config
        n = len(raw)
        instance_ids = list(range(1, n + 1))
        bboxes = [r.bbox for r in raw]

        # 几何关系（方向性恒生成；距离类受场景门控）
        directional = build_directional_relations(instance_ids, bboxes)
        nearest = build_nearest_relations(
            instance_ids,
            bboxes,
            scene_type=cfg.scene_type,
            has_intrinsics=cfg.camera_intrinsics is not None,
        )

        detections: list[Detection] = []
        for iid, r in zip(instance_ids, raw, strict=True):
            entropy = (
                r.classification_entropy
                if r.classification_entropy is not None
                else binary_entropy(r.detection_score)
            )
            conf = estimate_conf_level(
                classification_entropy=entropy,
                local_edge_sharpness=r.local_edge_sharpness,
                detection_score=r.detection_score,
                thresholds=cfg.thresholds,
            )
            relations: list[Relation] = directional[iid] + nearest[iid]
            # 经 model_validate 构造：字段别名 "class" 是 Python 关键字，
            # pydantic mypy 合成的 __init__ 签名无法以字段名传参
            detections.append(
                Detection.model_validate(
                    {
                        "instance_id": iid,
                        "object_class": r.label,
                        "bbox_px": r.bbox,
                        "detection_score": r.detection_score,
                        "conf_level": conf,
                        "verification_required": cfg.verification_policy.get(conf, False),
                        "uncertainty": Uncertainty(
                            classification_entropy=entropy,
                            local_edge_sharpness=r.local_edge_sharpness,
                            tta_variance=None,  # 实验性，默认未启用
                        ),
                        "relations": relations,
                    }
                )
            )

        payload = {
            "schema_version": "5.2.1",
            "image_id": meta.image_id,
            "image_width": meta.width,
            "image_height": meta.height,
            "timestamp": meta.timestamp,
            "scene_type": cfg.scene_type,
            "detector_model": self._detector.model_name,
            "detections": [d.model_dump(by_alias=True) for d in detections],
        }
        if cfg.camera_intrinsics is not None:
            payload["camera_intrinsics"] = cfg.camera_intrinsics.model_dump()

        try:
            return IR.model_validate(payload)
        except ValidationError as exc:  # S2 硬校验失败
            raise from_validation_error(exc) from exc
