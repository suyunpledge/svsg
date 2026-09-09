"""意图路由器：7 类意图的词库路由（V5.2.1 §4.1）。

Phase 1 采用可扩展词库 + 优先级裁决（最具体的意图优先，最泛化的
OBJECT_DETECTION 兜底）；Phase 2.5 评测分类 F1（验收线 90%，
不达标则扩展词库），可整体替换为向量/模型路由而不影响调用方。
"""

from __future__ import annotations

from svsg.contracts import IntentType

#: 词库按优先级排列：越靠前越具体，OBJECT_DETECTION 兜底
_LEXICON: list[tuple[IntentType, tuple[str, ...]]] = [
    (
        IntentType.TEMPORAL_RELATION,
        ("之前", "之后", "变化", "先后", "previous", "later", "change", "before", "after"),
    ),
    (
        IntentType.SPATIAL_RELATION,
        (
            "左边", "右边", "上方", "下方", "左边", "旁边", "最近", "挨着",
            "里面", "外面", "left of", "right of", "above", "below",
            "nearest", "inside", "next to",
        ),
    ),
    (IntentType.COUNT, ("几个", "多少", "数量", "几条", "how many", "count", "number of")),
    (
        IntentType.EXISTENCE_CHECK,
        ("有没有", "是否存在", "是不是有", "is there", "are there", "exists"),
    ),
    (
        IntentType.ATTRIBUTE_QUERY,
        ("什么颜色", "颜色", "多大", "大小", "材质", "形状", "color", "size", "material", "shape"),
    ),
    (
        IntentType.REGION_QUERY,
        ("区域", "这块", "左上角", "右下角", "标记的地方", "region", "this part", "area"),
    ),
    (
        IntentType.OBJECT_DETECTION,
        ("检测", "识别", "有哪些", "都是什么", "detect", "identify", "objects", "what is in"),
    ),
]

_FALLBACK = IntentType.OBJECT_DETECTION


def route_intent(query: str) -> tuple[IntentType, float]:
    """路由用户查询到意图类别。

    返回 (意图, 匹配得分)。得分 = 命中该意图的关键词数（无命中时
    兜底意图得分为 0.0，调用方可据此决定是否转 L3 自行判断）。
    优先级保证"图里有几个螺丝"命中 COUNT 而非 OBJECT_DETECTION。
    """
    q = query.lower()
    for intent, keywords in _LEXICON:
        hits = sum(1 for kw in keywords if kw in q)
        if hits > 0:
            return intent, float(hits)
    return _FALLBACK, 0.0


def route_intent_or_none(query: str) -> IntentType | None:
    """无命中时返回 None（供需要显式兜底信号的调用方）。"""
    intent, score = route_intent(query)
    return intent if score > 0.0 else None
