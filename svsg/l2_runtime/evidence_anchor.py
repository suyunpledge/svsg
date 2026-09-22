"""V5.2.1 强制后置证据锚定校验器（Critical）。

在 L3 生成最终答案后、返回给用户前执行两道硬校验：

1. 工具调用完整性：IR 中所有 verification_required=true 的实例，
   是否均出现在已执行的 VerifyRegion 集合中。缺失 → E3003
   （VERIFICATION_SKIPPED），打回 S3 重试。
2. 证据忠实度锚定：提取 L3 claims（结构化断言，非自然语言解析），
   与 IR（检测器锚）及 L1.5 structured_report（验证锚）做严格比对。
   不一致 → E3002（EVIDENCE_MISMATCH），无条件否决，触发 S4。

比对规则（全部确定性，无模糊匹配）：
- class 断言：必须与 IR 的 object_class 严格相等；若该实例有验证结果，
  还须与 result.class_label 严格相等，且后者为 "ambiguous" 时一律否决
  （V5.2.1 示例：报告 ambiguous 而答案写"螺丝"→ 否决）。
- count 断言：field="count" 比对 IR 总实例数；field="count:<class>"
  比对该类别实例数。
- exists 断言：True 断言与 result.exists=False 冲突 → 否决；
  False 断言与 IR 中实际存在该实例冲突 → 否决。
- attribute:<name> 断言：验证结果携带同名属性且不等 → 否决。
- relation:<type> 断言：value 为目标 instance_id，用几何引擎
  （与 L1 共用的同一裁判）基于 IR bbox 重算，不成立 → 否决。
- 被静默降级（S5 路径）的实例无验证结果：claims 仅锚定 IR，
  检测器结论仍可用，但 conf_level 按 lowest 参与表述约束（编排器职责）。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from svsg.contracts import (
    IR,
    Claim,
    ErrorCode,
    L3Answer,
    RelationType,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from svsg.l1_compiler.geometry_engine import (
    center_distance_px,
    distance_relations_allowed,
    relation_holds,
)

AMBIGUOUS_LABEL = "ambiguous"


@dataclass(frozen=True)
class Violation:
    """单条锚定违规。"""

    code: ErrorCode
    field: str
    reason: str
    instance_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "field": self.field,
            "instance_id": self.instance_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnchorVerdict:
    """锚定校验裁决。"""

    passed: bool
    violations: list[Violation]

    @property
    def primary_code(self) -> ErrorCode | None:
        """映射 FSM 事件用：E3003 → VERIFICATION_SKIPPED，否则 E3002。"""
        if self.passed:
            return None
        if any(v.code == ErrorCode.VERIFICATION_SKIPPED for v in self.violations):
            return ErrorCode.VERIFICATION_SKIPPED
        return ErrorCode.EVIDENCE_MISMATCH


def missing_verifications(
    ir: IR, executed: Iterable[int], exempt: Iterable[int] = ()
) -> set[int]:
    """工具调用完整性：应验证而未验证的实例集合。

    exempt：S5 静默降级实例（已请求验证但 L1.5 超时），不参与完整性
    检查——其 claims 仅锚定 IR（见 anchor_claims 的降级语义）。
    """
    executed_set = set(executed)
    exempt_set = set(exempt)
    return {
        d.instance_id
        for d in ir.detections
        if d.verification_required
        and d.instance_id not in executed_set
        and d.instance_id not in exempt_set
    }


def _claim_shape_error(claim: Claim) -> str | None:
    """Reject unsupported claims before comparing values (bool is an int in Python)."""
    field, value = claim.field, claim.value
    if field == "count" or field.startswith("count:"):
        if field == "count:" or claim.instance_id is not None:
            return "计数断言须为全局断言，类别名不能为空"
        if type(value) is not int or value < 0:
            return "计数断言须使用非负整数（不能使用布尔值或浮点数）"
        return None
    if field not in ("class", "exists") and not field.startswith(("attribute:", "relation:")):
        return f"不支持的断言字段: {field}"
    if claim.instance_id is None:
        return "实例级断言必须提供 instance_id"
    if field == "class" and (not isinstance(value, str) or not value.strip()):
        return "class 断言须使用非空字符串"
    if field == "exists" and type(value) is not bool:
        return "exists 断言须使用布尔值"
    if field.startswith("attribute:"):
        if not field.split(":", 1)[1] or not isinstance(value, str):
            return "属性断言须提供属性名和字符串值"
    if field.startswith("relation:"):
        if type(value) is not int or value < 1 or value == claim.instance_id:
            return "关系目标须为不同实例的正整数 ID"
        try:
            relation = RelationType(field.split(":", 1)[1])
        except ValueError:
            return "不支持的关系类型"
        if relation == RelationType.DISTANCE_TO:
            return "distance_to 数值断言尚无可校验的数值/单位契约，禁止交付"
    return None


def anchor_claims(
    ir: IR,
    answer: L3Answer,
    report: VerificationReport | None = None,
) -> list[Violation]:
    """证据忠实度锚定：claims × (IR ∪ structured_report) 严格比对。"""
    by_id = {d.instance_id: d for d in ir.detections}
    results: dict[int, VerificationResult] = {}
    if report is not None and report.status == VerificationStatus.SUCCESS:
        ids = [r.instance_id for r in report.results]
        if (report.image_id != ir.image_id or len(ids) != len(set(ids))
                or not set(ids) <= set(by_id)):
            return [Violation(
                code=ErrorCode.EVIDENCE_MISMATCH,
                field="<report>",
                reason="验证报告图像不匹配或包含重复/未知实例",
            )]
        results = {r.instance_id: r for r in report.results}

    violations: list[Violation] = []
    for claim in answer.claims:
        shape_error = _claim_shape_error(claim)
        if shape_error is not None:
            violations.append(Violation(
                code=ErrorCode.EVIDENCE_MISMATCH,
                field=claim.field,
                instance_id=claim.instance_id,
                reason=shape_error,
            ))
            continue
        # 引用不存在的实例 → 直接否决
        if claim.instance_id is not None and claim.instance_id not in by_id:
            violations.append(
                Violation(
                    code=ErrorCode.EVIDENCE_MISMATCH,
                    field=claim.field,
                    instance_id=claim.instance_id,
                    reason=f"断言引用了 IR 中不存在的实例 {claim.instance_id}",
                )
            )
            continue
        det = by_id.get(claim.instance_id) if claim.instance_id is not None else None
        result = results.get(claim.instance_id) if claim.instance_id is not None else None
        field = claim.field

        # Aggregate and non-existence fields must not bypass negative verifier evidence.
        if result is not None and result.exists is False and field != "exists":
            violations.append(Violation(
                code=ErrorCode.EVIDENCE_MISMATCH, field=field,
                instance_id=claim.instance_id,
                reason="验证报告 exists=False，不能断言该实例的类别、属性或关系",
            ))
            continue
        if field == "count" or field.startswith("count:"):
            cls = field.split(":", 1)[1] if ":" in field else None
            conflicting = any(
                r.exists is False
                or (cls is not None and r.class_label is not None
                    and r.class_label != by_id[iid].object_class
                    and (r.class_label == AMBIGUOUS_LABEL
                         or cls in (r.class_label, by_id[iid].object_class)))
                for iid, r in results.items()
            )
            if conflicting:
                violations.append(Violation(
                    code=ErrorCode.EVIDENCE_MISMATCH, field=field,
                    reason="计数断言涉及验证报告中的存在性或类别冲突",
                ))
                continue

        if field == "class":
            if det is not None:
                if claim.value != det.object_class:
                    violations.append(
                        Violation(
                            code=ErrorCode.EVIDENCE_MISMATCH,
                            field=field,
                            instance_id=claim.instance_id,
                            reason=(
                                f"class 断言 {claim.value!r} 与 IR 检测结果 "
                                f"{det.object_class!r} 不一致"
                            ),
                        )
                    )
                    continue
                if result is not None:
                    label = result.class_label
                    if label == AMBIGUOUS_LABEL:
                        violations.append(
                            Violation(
                                code=ErrorCode.EVIDENCE_MISMATCH,
                                field=field,
                                instance_id=claim.instance_id,
                                reason="验证报告判定 ambiguous，禁止输出具体类别",
                            )
                        )
                    elif label is not None and label != claim.value:
                        violations.append(
                            Violation(
                                code=ErrorCode.EVIDENCE_MISMATCH,
                                field=field,
                                instance_id=claim.instance_id,
                                reason=(
                                    f"class 断言 {claim.value!r} 与验证报告 "
                                    f"{label!r} 不一致"
                                ),
                            )
                        )

        elif field == "count":
            if claim.value != len(ir.detections):
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        reason=f"count 断言 {claim.value} != IR 实例总数 {len(ir.detections)}",
                    )
                )

        elif field.startswith("count:"):
            cls = field.split(":", 1)[1]
            actual = sum(1 for d in ir.detections if d.object_class == cls)
            if claim.value != actual:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        reason=f"count:{cls} 断言 {claim.value} != IR 实际 {actual}",
                    )
                )

        elif field == "exists":
            if det is None:
                continue
            if claim.value is True and result is not None and result.exists is False:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason="exists=True 断言与验证报告 exists=False 冲突",
                    )
                )
            elif claim.value is False:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason="exists=False 断言与 IR 中实际存在的实例冲突",
                    )
                )

        elif field.startswith("attribute:"):
            # V5.2.4 修复1（attribute 验证盲区）：attribute 断言的唯一合法
            # 证据来源是 L1.5 验证报告（IR 的 attributes 默认为空）。
            # 无报告 → 证据链断裂，一律否决；不再"无报告即静默放行"。
            name = field.split(":", 1)[1]
            if result is None:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason=(
                            f"attribute 断言缺少 L1.5 验证报告"
                            f"（instance {claim.instance_id}），证据链断裂"
                        ),
                    )
                )
            elif name not in result.attributes:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason=f"验证报告中不存在属性 {name!r}",
                    )
                )
            elif result.attributes[name] != str(claim.value):
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason=(
                            f"属性 {name}={claim.value!r} 与验证报告 "
                            f"{result.attributes[name]!r} 不一致"
                        ),
                    )
                )

        elif field.startswith("relation:"):
            rel_name = field.split(":", 1)[1]
            assert det is not None and isinstance(claim.value, int)
            rel = RelationType(rel_name)
            target = by_id.get(claim.value)
            if target is None:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason=f"关系断言引用了不存在的目标实例 {claim.value}",
                    )
                )
                continue
            target_result = results.get(target.instance_id)
            if target_result is not None and target_result.exists is False:
                violations.append(Violation(
                    code=ErrorCode.EVIDENCE_MISMATCH, field=field,
                    instance_id=claim.instance_id,
                    reason="关系目标在验证报告中 exists=False",
                ))
                continue
            if rel == RelationType.NEAREST_TO:
                allowed = distance_relations_allowed(
                    ir.scene_type, ir.camera_intrinsics is not None
                )
                distance = center_distance_px(det.bbox_px, target.bbox_px)
                holds = allowed and all(
                    distance <= center_distance_px(det.bbox_px, other.bbox_px)
                    for other in ir.detections if other.instance_id != det.instance_id
                )
            else:
                holds = relation_holds(rel, det.bbox_px, target.bbox_px)
            if not holds:
                violations.append(
                    Violation(
                        code=ErrorCode.EVIDENCE_MISMATCH,
                        field=field,
                        instance_id=claim.instance_id,
                        reason=f"关系断言 {rel_name}(→{claim.value}) 未通过几何引擎复核",
                    )
                )

    return violations


def nl_claims_coverage(
    ir: IR,
    answer: L3Answer,
    threshold: float = 0.8,
) -> float:
    """NL-claims 关键词覆盖率（V5.2.4 §4.5.2 第 5 步，Phase 1 简化实现）。

    提取自然语言中的视觉关键词（IR 类别词 + claim 值词），检查每个
    关键词是否至少出现在一个 claim 的 value 中。返回覆盖率 [0,1]。

    Phase 1 语义：覆盖率 < threshold 时由调用方记录告警日志，不触发
    S7（已知风险敞口，Phase 2 关闭：完整 NL-claims 结构化对齐）。
    """
    # 关键词集 = IR 中出现过的类别名（检测器词汇表）
    keywords: set[str] = {d.object_class.lower() for d in ir.detections}
    if not keywords:
        return 1.0  # 无检测实例时无关键词可覆盖
    # claims 值文本（小写化）
    claim_text = " ".join(
        str(cl.value).lower() for cl in answer.claims if cl.value is not None
    )
    hit = sum(1 for kw in keywords if kw in claim_text)
    return hit / len(keywords)


def run_anchor(
    ir: IR,
    answer: L3Answer,
    executed_verifications: Iterable[int],
    report: VerificationReport | None = None,
    exempt: Iterable[int] = (),
) -> AnchorVerdict:
    """执行完整锚定校验（完整性 → 忠实度），输出裁决。

    编排器根据 primary_code 映射 FSM 事件：
    None → ANCHOR_PASSED；E2003 → VERIFICATION_SKIPPED；E3002 → ANCHOR_FAILED。
    exempt：S5 静默降级实例（L1.5 超时），豁免完整性检查。
    """
    violations: list[Violation] = []

    missing = missing_verifications(ir, executed_verifications, exempt=exempt)
    if missing:
        violations.append(
            Violation(
                code=ErrorCode.VERIFICATION_SKIPPED,
                field="<completeness>",
                reason=f"应验证而未调用 VerifyRegion 的实例: {sorted(missing)}",
            )
        )

    violations.extend(anchor_claims(ir, answer, report))
    return AnchorVerdict(passed=not violations, violations=violations)
