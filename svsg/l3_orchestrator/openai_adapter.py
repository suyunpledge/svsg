"""OpenAI 兼容 LLM 适配器（实现 LLMProtocol）。

仅依赖 httpx（惰性导入，核心包保持零第三方依赖）。请求侧约定：
- system prompt 承载硬性规则（透视免责 / 降级标注 / ambiguous 禁令）；
- user 消息为结构化证据 JSON 上下文；
- 低温度 + 禁用流式，保证双通道输出可解析。

解析策略：从回复文本提取首个平衡 JSON 对象（容忍 markdown 代码栅栏
与前后缀文本），反序列化为 L3Answer。解析失败抛 ValueError，由
Orchestrator 的 L3 异常路径保守中止（S5 语义）。
"""

from __future__ import annotations

import json

from svsg.contracts import L3Answer


class OpenAICompatibleLLM:
    """chat/completions 端点的薄适配（任何 OpenAI 兼容网关可用）。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 3.0,
        temperature: float = 0.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._temperature = temperature

    async def complete(self, system_prompt: str, context: str) -> L3Answer:
        import httpx  # 惰性导入：核心包不强制该依赖

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return parse_l3_answer(content)


def parse_l3_answer(content: str) -> L3Answer:
    """从 LLM 回复文本提取双通道 JSON 并构造 L3Answer。"""
    extracted = _extract_json_object(content)
    if extracted is None:
        raise ValueError(f"L3 回复中未找到 JSON 对象: {content[:120]!r}")
    return L3Answer.model_validate(json.loads(extracted))


def _extract_json_object(text: str) -> str | None:
    """提取首个平衡的 {...}（跳过字符串字面量内部的花括号）。"""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
