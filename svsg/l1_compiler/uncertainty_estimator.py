"""不确定性估计器：原始信号 → conf_level 的纯映射。

V5.2.1 §3.2：四个阈值（0.7 / 0.5 / 0.4 / 0.6）均为初始默认值，
Phase 2 标定后通过 Thresholds 注入调整，本模块不持有全局可变状态。

映射规则（文档化，作为契约测试的依据）：
1. 基准档位由分类熵决定：entropy < 0.5 → high；< 0.7 → medium；否则 low。
2. 局部锐度 < 0.4 → 判定图像模糊，基准降一级。
3. detection_score < 0.6 → 检测本身不可靠，基准降一级。
4. 降级下限为 low：lowest 仅由 S5 超时静默降级路径使用（见 enums.ConfLevel），
   估计器无权产出 lowest，保证该档位语义的单一来源。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from svsg.contracts import ConfLevel

_LEVEL_ORDER: list[ConfLevel] = [ConfLevel.HIGH, ConfLevel.MEDIUM, ConfLevel.LOW]


@dataclass(frozen=True)
class Thresholds:
    """conf_level 映射阈值（V5.2.1 初始默认值，Phase 2 标定）。"""

    entropy_high: float = 0.5   # 熵低于此值 → 基准 high
    entropy_medium: float = 0.7  # 熵低于此值 → 基准 medium，否则 low
    sharpness_blur: float = 0.4  # 锐度低于此值 → 判定模糊，降一级
    score_floor: float = 0.6     # 检测分低于此值 → 降一级


#: 默认阈值单例（V5.2.1 初始值；Phase 2 标定后经参数注入替换）
DEFAULT_THRESHOLDS = Thresholds()


def binary_entropy(p: float) -> float:
    """二元分类熵（自然对数底）。

    检测器只输出单一置信度时，用 H(p) = -p·ln p - (1-p)·ln(1-p) 近似
    分类不确定性；p 会向 [eps, 1-eps] 裁剪以避免 log(0)。
    """
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _demote(level: ConfLevel, steps: int = 1) -> ConfLevel:
    idx = _LEVEL_ORDER.index(level)
    return _LEVEL_ORDER[min(idx + steps, len(_LEVEL_ORDER) - 1)]


def estimate_conf_level(
    *,
    classification_entropy: float,
    local_edge_sharpness: float,
    detection_score: float,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ConfLevel:
    """由三个原始信号估计实例置信度档位（纯函数）。

    classification_entropy 通常来自模型 logits；只有单一置信度时
    可先经 binary_entropy(detection_score) 推导。
    """
    if classification_entropy < thresholds.entropy_high:
        base = ConfLevel.HIGH
    elif classification_entropy < thresholds.entropy_medium:
        base = ConfLevel.MEDIUM
    else:
        base = ConfLevel.LOW

    if local_edge_sharpness < thresholds.sharpness_blur:
        base = _demote(base)
    if detection_score < thresholds.score_floor:
        base = _demote(base)
    return base
