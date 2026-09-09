"""检测器适配层：模型热替换的接缝。

- DetectorProtocol：L1 编译器依赖的唯一接口（结构化子类型），
  任何检测模型（YOLOv8 / Grounding DINO / ...）实现该协议即可接入，
  IR 中记录 model_name 用于版本溯源。
- StubDetector：确定性桩实现，供无 GPU 环境开发与全链路测试。
- YoloDetector：ultralytics YOLOv8 的薄适配（惰性导入，未安装时
  给出明确指引），生产环境可替换为 Grounding DINO 等开放词表模型。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from .geometry_engine import BBox

ImageSource = bytes | Path


def image_dimensions(image: ImageSource) -> tuple[int, int]:
    """读取图像 (宽, 高)。Pillow 惰性导入；仅演示/API 路径依赖。"""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("图像尺寸解析需要 Pillow：pip install pillow") from exc
    # with 确保句柄及时释放(Path 输入时避免文件描述符泄漏)
    with (
        Image.open(io.BytesIO(image)) if isinstance(image, bytes) else Image.open(image)
    ) as img:
        return img.width, img.height


@dataclass(frozen=True)
class RawDetection:
    """检测器原始输出（未经 L1 语义加工）。"""

    label: str
    bbox: BBox
    detection_score: float
    # 模型可直接提供分类熵（如开放词表模型的 logits 熵）；缺省时
    # 由 uncertainty_estimator 从 detection_score 推导二元熵
    classification_entropy: float | None = None
    # 局部边缘锐度 [0,1]：由适配层基于裁剪区域计算（真实适配用
    # OpenCV Laplacian 方差归一化；Stub 直接注入已知值）
    local_edge_sharpness: float = 1.0


class DetectorProtocol(Protocol):
    """L1 编译器依赖的检测器接口。"""

    @property
    def model_name(self) -> str: ...

    def detect(self, image: ImageSource) -> list[RawDetection]: ...


@dataclass
class StubDetector:
    """确定性桩检测器：返回预配置的检测列表。

    用途：契约测试、FSM 全路径测试、无 GPU 环境的端到端联调。
    可通过 fail_on_detect 抛出指定异常，模拟 L1 故障注入。
    """

    detections: list[RawDetection] = field(default_factory=list)
    model_name: str = "stub_detector_v0"
    fail_with: Exception | None = None

    def detect(self, image: ImageSource) -> list[RawDetection]:
        if self.fail_with is not None:
            raise self.fail_with
        return list(self.detections)


class DemoDetector:
    """演示用确定性检测器：按图像实际尺寸缩放相对坐标的检测框。

    用途：无 GPU 环境的端到端联调 / Web 服务冒烟演示。检测框以
    (x1, y1, x2, y2) 相对比例描述，detect 时按真实图像宽高换算为
    像素坐标，保证任意尺寸输入都产出合法（不越界）的 IR。

    默认输出 2 个高置信实例（screw / washer，均不触发 L1.5 验证），
    使演示走「编译 → 生成 → 锚定 → delivered」的完整快乐路径。
    """

    model_name: str = "demo_detector_v0"

    def __init__(self) -> None:
        self._template: tuple[tuple[str, float, float, float, float, float, float], ...] = (
            ("screw", 0.20, 0.12, 0.30, 0.24, 0.93, 0.80),
            ("washer", 0.65, 0.40, 0.78, 0.55, 0.88, 0.76),
        )

    def detect(self, image: ImageSource) -> list[RawDetection]:
        w, h = image_dimensions(image)
        raw: list[RawDetection] = []
        for label, x1, y1, x2, y2, score, sharp in self._template:
            raw.append(
                RawDetection(
                    label=label,
                    bbox=(int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                    detection_score=score,
                    local_edge_sharpness=sharp,
                )
            )
        return raw


class YoloDetector:
    """ultralytics YOLOv8 适配器（惰性导入）。

    仅做结果映射，不引入训练逻辑；local_edge_sharpness 若本机装有
    OpenCV 则基于裁剪区域计算，否则保守取 1.0（并在 IR 审计中可知
    模型来源）。权重路径即版本标识，写入 IR.detector_model。
    """

    def __init__(self, weights: str = "yolov8n.pt") -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "YoloDetector 需要 ultralytics：pip install ultralytics"
            ) from exc
        self._model = YOLO(weights)
        self._weights = weights

    @property
    def model_name(self) -> str:
        return f"yolov8::{self._weights}"

    def detect(self, image: ImageSource) -> list[RawDetection]:
        # ultralytics 8.x 不接受 bytes 输入(仅支持路径/PIL/numpy/URL),
        # 这里把 bytes 转成 PIL Image 再送入推理(复用 Pillow,惰性导入)。
        src: Any
        if isinstance(image, bytes):
            from PIL import Image
            src = Image.open(io.BytesIO(image))
        else:
            src = image
        results = cast(Any, self._model.predict(src, verbose=False))
        raw: list[RawDetection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                raw.append(
                    RawDetection(
                        label=str(names[int(box.cls[0])]),
                        bbox=(x1, y1, x2, y2),
                        detection_score=float(box.conf[0]),
                        local_edge_sharpness=self._edge_sharpness(image, (x1, y1, x2, y2)),
                    )
                )
        return raw

    @staticmethod
    def _edge_sharpness(image: ImageSource, bbox: BBox) -> float:
        """裁剪区域的 Laplacian 方差归一化锐度；无 OpenCV 时保守返回 1.0。"""
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            return 1.0
        buf = np.frombuffer(image, dtype=np.uint8) if isinstance(image, bytes) else None
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf is not None else cv2.imread(
            str(image), cv2.IMREAD_GRAYSCALE
        )
        if img is None:  # pragma: no cover
            return 1.0
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:  # pragma: no cover
            return 1.0
        # 经验归一化：Laplacian 方差 500+ 视为完全清晰
        return float(min(1.0, cv2.Laplacian(crop, cv2.CV_64F).var() / 500.0))
