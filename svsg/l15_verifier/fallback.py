"""静默降级策略：超时预算断言 + 降级实例集合推导。

V5.2.1 §5.1 / §9：
- 超时预算规则：L1.5 超时阈值必须严格小于 L3 总超时的 1/3。
- 静默降级：超时实例不经过 S4a 冲突解决，直接标记 conf_level=lowest
  并转 S6 人工复审；主控 L3 继续处理其余高置信实例（部分可用性）。
"""

from __future__ import annotations

from collections.abc import Iterable

from svsg.contracts import VerificationReport, VerificationStatus


def assert_timeout_budget(l15_timeout_s: float, l3_budget_s: float) -> None:
    """装配期断言：L1.5 超时必须严格小于 L3 总预算的 1/3。

    违反时抛 ValueError（配置错误应在部署前暴露，而非运行时劣化）。
    """
    limit = l3_budget_s / 3.0
    if not (l15_timeout_s < limit):
        raise ValueError(
            f"L1.5 超时阈值 {l15_timeout_s}s 违反 V5.2.1 预算规则："
            f"必须严格小于 L3 总超时 {l3_budget_s}s 的 1/3（即 < {limit:.4f}s）"
        )


def silent_degrade(
    requested: Iterable[int], report: VerificationReport
) -> set[int]:
    """推导应静默降级的实例集合。

    - 报告非成功（timeout/error）：全部请求实例降级（保守：部分输出不采信）；
    - 报告成功：无降级（个别实例缺失结果由锚定校验的 E2003 路径处理）。
    """
    if report.status != VerificationStatus.SUCCESS:
        return set(requested)
    return set()
