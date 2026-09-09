"""OpenAI 兼容适配器的纯解析函数测试（不涉及网络）。"""

from __future__ import annotations

import sys

from svsg.contracts import L3Answer
from svsg.l3_orchestrator.openai_adapter import parse_l3_answer


def test_parse_plain_json():
    answer = parse_l3_answer(
        '{"claims": [{"instance_id": 1, "field": "class", "value": "screw"}],'
        ' "final_answer": "图中有一颗螺丝"}'
    )
    assert isinstance(answer, L3Answer)
    assert answer.claims[0].value == "screw"
    assert answer.final_answer == "图中有一颗螺丝"


def test_parse_with_markdown_fences_and_prose():
    content = (
        "好的，以下是分析结果：\n"
        "```json\n"
        '{"claims": [{"field": "count", "value": 3}], "final_answer": "共 3 个"}\n'
        "```\n"
        "如需进一步分析请告知。"
    )
    answer = parse_l3_answer(content)
    assert answer.claims[0].value == 3


def test_parse_nested_braces_inside_strings():
    # 字符串字面量内的花括号不应破坏平衡提取
    content = '{"claims": [], "final_answer": "配置 {a: 1} 已生效"} 尾部噪声'
    answer = parse_l3_answer(content)
    assert answer.final_answer == "配置 {a: 1} 已生效"


def test_parse_failure_raises_value_error():
    for bad in ("完全不是 JSON", "{'single': 'quote'}"):
        try:
            parse_l3_answer(bad)
        except ValueError:
            continue
        raise AssertionError(f"应抛 ValueError: {bad!r}")


if __name__ == "__main__":
    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
