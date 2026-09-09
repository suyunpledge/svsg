"""S4 语义冲突解决策略（V5.2.1 §5.2，阈值参数化）。

Detector 结论与 L1.5 验证结果不一致时的裁决顺序：
1. 验证结果不可用（ambiguous / 无置信度）→ UNRESOLVED，进 S6；
2. 图像模糊（edge_sharpness < 阈值 0.4）→ KEEP_DETECTOR_DISPUTED：
   结果降权、标记争议，进 S6（模糊场景下双方证据都不可靠）；
3. 验证置信度显著高于 Detector（差值 > 0.2）→ ADOPT_VERIFICATION：
   采纳验证结论，返回 S1 继续；
4. 其余 → UNRESOLVED，进 S6。

阈值均为初始默认值，Phase 2 标定后经 ConflictThresholds 注入调整。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from svsg.contracts import Detection, VerificationResult


@dataclass(frozen=True)
class ConflictThresholds:
    """冲突解决阈值（V5.2.1 初始默认值，Phase 2 标定）。"""

    edge_sharpness_threshold: float = 0.4  # 低于此值判定图像模糊
    confidence_diff_threshold: float = 0.2  # 验证分需高于检测分的最小差值


@unique
class ResolutionOutcome(StrEnum):
    ADOPT_VERIFICATION = "adopt_verification"      # 采纳 L1.5 结论 → S1
    KEEP_DETECTOR_DISPUTED = "keep_detector_disputed"  # 降权+标记争议 → S6
    UNRESOLVED = "unresolved"                      # 无法裁决 → S6


#: 默认冲突阈值单例（V5.2.1 初始值；Phase 2 标定后经参数注入替换）
DEFAULT_CONFLICT_THRESHOLDS = ConflictThresholds()


@dataclass(frozen=True)
class ConflictResolution:
    """冲突裁决结果。

    adopted_class 仅在 ADOPT_VERIFICATION 时非 None；
    degraded_conf_level：降权后的档位（disputed 时 lowest）。
    """

    outcome: ResolutionOutcome
    adopted_class: str | None = None
    degraded_conf_level: str | None = None
    reason: str = ""


def resolve_conflict(
    detection: Detection,
    result: VerificationResult,
    thresholds: ConflictThresholds = DEFAULT_CONFLICT_THRESHOLDS,
) -> ConflictResolution:
    """裁决单实例的 Detector × L1.5 冲突（纯函数）。"""
    label = result.class_label
    if label is None or label == "ambiguous" or result.confidence is None:
        return ConflictResolution(
            outcome=ResolutionOutcome.UNRESOLVED,
            reason="验证结果不可用（ambiguous 或缺失置信度），无法裁决",
        )

    sharpness = detection.uncertainty.local_edge_sharpness
    if sharpness < thresholds.edge_sharpness_threshold:
        return ConflictResolution(
            outcome=ResolutionOutcome.KEEP_DETECTOR_DISPUTED,
            degraded_conf_level="lowest",
            reason=(
                f"局部锐度 {sharpness:.2f} < {thresholds.edge_sharpness_threshold}，"
                "图像模糊：结果降权并标记争议"
            ),
        )

    diff = result.confidence - detection.detection_score
    if diff > thresholds.confidence_diff_threshold:
        return ConflictResolution(
            outcome=ResolutionOutcome.ADOPT_VERIFICATION,
            adopted_class=label,
            reason=f"验证置信度高于检测分 {diff:.2f} > {thresholds.confidence_diff_threshold}",
        )

    return ConflictResolution(
        outcome=ResolutionOutcome.UNRESOLVED,
        reason=f"置信度差值 {diff:.2f} 未达采纳阈值，标记争议",
    )
