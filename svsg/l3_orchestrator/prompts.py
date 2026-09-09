"""L3 主控的 Prompt 资产。

system prompt 将 V5.2.1 的三条强制语义写入 LLM 规则：
1. 透视免责（修订 #2）：metric_accuracy 不可用时，距离表述必须附带
   "图像平面内约 X 像素，非物理距离" 的免责声明；
2. 降级标注（修订 #3）：lowest/degraded 实例的结论必须附带
   "已降级处理，待人工复审"；
3. ambiguous 禁令（修订 #1 的前端）：验证结论为 ambiguous 的实例
   禁止给出具体类别（后端由锚定校验兜底否决）。

claims 双通道词表与 evidence_anchor 的比对规则一一对应——这是
"断言提取零 NLP 解析"得以成立的前提。
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from svsg.contracts import IR, VerificationReport, VerificationStatus

SYSTEM_PROMPT_TEMPLATE = (
    """你是 SVSG 视觉问答主控引擎（L3）。你看不到图像，
只能基于 L2 提供的结构化证据作答。

硬性规则（违反任何一条，答案会被证据锚定校验无条件否决）：
1. 只允许使用【结构化证据】中的事实作答；禁止推测、编造或用外部知识补充视觉细节。
2. 必须以双通道输出：claims（结构化断言数组）与 final_answer（面向用户的自然语言回答）。
   final_answer 中的所有核心视觉事实必须与 claims 一一对应，不得出现 claims 之外的视觉断言。
3. claims 字段词表（全局断言省略 instance_id）：
   - {{"instance_id": <id>, "field": "class", "value": "<类别>"}}
   - {{"field": "count", "value": <实例总数>}}
   - {{"field": "count:<类别>", "value": <该类别实例数>}}
   - {{"instance_id": <id>, "field": "exists", "value": true 或 false}}
   - {{"instance_id": <id>, "field": "attribute:<名称>", "value": <属性值>}}
   - {{"instance_id": <id>, "field": "relation:<类型>", "value": <目标 instance_id>}}
   relation 类型词表：left_of / right_of / above / below / contains /
   overlaps / nearest_to / distance_to。
4. {distance_rule}
5. conf_level 为 lowest、或列于 degraded_instances 的实例：final_answer 涉及该实例时，
   必须附带说明"该结论已经降级处理，待人工复审"。
6. 验证结论（verification_results）中 class 为 ambiguous 的实例：
   禁止在 claims 和 final_answer 中为其给出具体类别。
7. class_overrides 中给出的类别是冲突解决后的最终结论，claims 必须采用该类别而非检测器原始类别。
"""
)

UNSAFE_DISTANCE_RULE = (
    "涉及距离、远近表述时，必须附带免责声明（如"
    "\"图像平面内约 200 像素，非物理距离\"）；"
    "当前场景未标定（metric_accuracy 不可用），严禁将像素距离表述为物理距离。"
)

SAFE_DISTANCE_RULE = (
    "本场景物理度量已标定：距离按结构化证据中的数值与单位表述，不得改动数值。"
)


def build_system_prompt(physical_metrics_available: bool) -> str:
    """构建 system prompt。

    physical_metrics_available：正视场景，或透视场景已提供相机内参。
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        distance_rule=(
            SAFE_DISTANCE_RULE if physical_metrics_available else UNSAFE_DISTANCE_RULE
        )
    )


def build_context(
    query: str,
    ir: IR,
    report: VerificationReport | None = None,
    *,
    degraded: Iterable[int] = (),
    disputed: Iterable[int] = (),
    corrective_feedback: list[dict] | None = None,
) -> str:
    """构建单轮用户上下文：结构化证据 JSON。

    corrective_feedback：上一轮被锚定否决的断言及原因（重生成时注入），
    LLM 必须修正后重新作答。
    """
    degraded_set = set(degraded)
    disputed_set = set(disputed)

    detections = []
    for d in ir.detections:
        digest = {
            "instance_id": d.instance_id,
            "class": d.object_class,
            "bbox_px": list(d.bbox_px),
            "conf_level": "lowest" if d.instance_id in degraded_set else d.conf_level.value,
            "relations": [
                {
                    "type": r.type.value,
                    "target_id": r.target_id,
                    **({"distance_px": r.distance_px} if r.distance_px is not None else {}),
                }
                for r in d.relations
            ],
        }
        detections.append(digest)

    payload: dict = {
        "query": query,
        "scene_type": ir.scene_type.value,
        "image": {"width": ir.image_width, "height": ir.image_height},
        "detections": detections,
        "degraded_instances": sorted(degraded_set),
        "disputed_instances": sorted(disputed_set),
    }
    if report is not None and report.status == VerificationStatus.SUCCESS:
        payload["verification_results"] = [
            {
                "instance_id": r.instance_id,
                "class": r.class_label,
                "attributes": r.attributes,
                "exists": r.exists,
                "confidence": r.confidence,
            }
            for r in report.results
        ]
    if corrective_feedback:
        payload["corrective_feedback"] = {
            "notice": "上一轮答案被证据锚定校验否决，以下断言与结构化证据不一致，必须修正",
            "violations": corrective_feedback,
        }

    return json.dumps(payload, ensure_ascii=False, indent=1)
