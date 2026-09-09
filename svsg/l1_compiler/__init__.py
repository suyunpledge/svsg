"""L1 视觉编译器：像素 → 结构化 IR。

模块划分（全部零外部依赖，除 detector_adapter 的真实模型实现外）：
- detector_adapter：检测器 Protocol + Stub/YOLO 实现（模型可热替换）
- uncertainty_estimator：熵/锐度/检测分 → conf_level 纯映射
- geometry_engine：8 种空间关系的纯函数计算与场景门控
- compiler：组装管线，产出经 Schema 硬校验的 IR
"""

from .compiler import ImageMeta, L1Compiler, L1CompilerConfig
from .detector_adapter import (
    DemoDetector,
    DetectorProtocol,
    RawDetection,
    StubDetector,
    YoloDetector,
    image_dimensions,
)
from .geometry_engine import (
    BBox,
    center_distance_px,
    compute_spatial_consistency,
    distance_relations_allowed,
    relation_holds,
)
from .uncertainty_estimator import Thresholds, binary_entropy, estimate_conf_level

__all__ = [
    "ImageMeta",
    "L1Compiler",
    "L1CompilerConfig",
    "DemoDetector",
    "DetectorProtocol",
    "RawDetection",
    "StubDetector",
    "YoloDetector",
    "image_dimensions",
    "BBox",
    "center_distance_px",
    "compute_spatial_consistency",
    "distance_relations_allowed",
    "relation_holds",
    "Thresholds",
    "binary_entropy",
    "estimate_conf_level",
]
