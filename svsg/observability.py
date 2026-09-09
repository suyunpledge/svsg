"""可观测性平面：三档动态采样存储 + 全链路结构化日志（V5.2.1 §7）。

采样策略（三档动态）：
- dev    ：100% 采样（开发期全量留痕）；
- warmup ：10% 采样（预热期）；
- steady ：<1% 采样 + 异常触发全采（稳态）。

"异常" 判定（任一命中即恒采样，不受档位费率约束）：
- 会话终态为 S2 / S5 / S6；
- 证据锚定校验存在违规记录；
- 存在静默降级实例（L1.5 超时路径）。

trace_id 经 contextvars 贯穿单次请求内 L1→L2→L3→L1.5 的全部日志与
采样记录（§2 "全链路 Trace 埋点"）。存储为内存环形缓冲 + 可选 JSONL
文件 sink，Phase 4 可整体替换为外部后端而不影响调用方。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import sys
import uuid
from collections import deque
from enum import StrEnum, unique
from pathlib import Path
from typing import Any

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "svsg_trace_id", default="-"
)

#: 判定为异常的 FSM 终态（错误拒绝 / 超时中止 / 人工复审）
ANOMALY_FINAL_STATES = frozenset({"S2", "S5", "S6"})


def new_trace_id() -> str:
    """生成并绑定新 trace_id，返回生成值。"""
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    _trace_id.set(tid)


@unique
class SampleMode(StrEnum):
    DEV = "dev"
    WARMUP = "warmup"
    STEADY = "steady"


def sample_mode_from_env(default: SampleMode = SampleMode.DEV) -> SampleMode:
    """读取 SVSG_SAMPLE_MODE（dev|warmup|steady），非法值回落 default。"""
    raw = os.environ.get("SVSG_SAMPLE_MODE", "").strip().lower()
    try:
        return SampleMode(raw)
    except ValueError:
        return default


def is_anomaly(
    session_summary: dict[str, Any],
    *,
    anchor_violations: list[Any] | None = None,
) -> bool:
    """判定一次编排结果是否属于"异常"（稳态下仍恒采样）。"""
    if session_summary.get("final_state") in ANOMALY_FINAL_STATES:
        return True
    if session_summary.get("degraded_instances"):
        return True
    if anchor_violations:
        return True
    return False


class SamplingPolicy:
    """三档动态采样（费率为 V5.2.1 默认值，可经构造参数调整）。"""

    def __init__(
        self,
        mode: SampleMode = SampleMode.DEV,
        *,
        warmup_rate: float = 0.1,
        steady_rate: float = 0.01,
        rng: random.Random | None = None,
    ) -> None:
        self.mode = mode
        self.warmup_rate = warmup_rate
        self.steady_rate = steady_rate
        self._rng = rng or random.Random()

    def should_sample(self, *, anomaly: bool = False) -> bool:
        if anomaly:
            return True  # 异常触发：不受档位费率约束
        if self.mode is SampleMode.DEV:
            return True
        rate = self.warmup_rate if self.mode is SampleMode.WARMUP else self.steady_rate
        return self._rng.random() < rate


class TraceStore:
    """采样留痕：内存环形缓冲 + 可选 JSONL 文件 sink（append-only）。"""

    def __init__(
        self,
        policy: SamplingPolicy | None = None,
        *,
        capacity: int = 1000,
        sink_path: str | Path | None = None,
    ) -> None:
        self.policy = policy or SamplingPolicy()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._sink: Path | None = Path(sink_path) if sink_path else None
        if self._sink is not None:
            self._sink.parent.mkdir(parents=True, exist_ok=True)
        self.sampled = 0
        self.dropped = 0

    def record(self, payload: dict[str, Any], *, anomaly: bool = False) -> bool:
        """按策略决定是否留痕；被采样的记录附 trace_id 与异常标记。"""
        if not self.policy.should_sample(anomaly=anomaly):
            self.dropped += 1
            return False
        entry = {"trace_id": get_trace_id(), "anomaly": anomaly, **payload}
        self._buffer.append(entry)
        self.sampled += 1
        if self._sink is not None:
            try:
                with self._sink.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            except OSError:
                # 留痕 sink 写失败不应影响请求响应(磁盘满/路径不可写)
                pass
        return True

    def retained(self) -> list[dict[str, Any]]:
        return list(self._buffer)


# ---------------------------------------------------------------------- 日志

#: LogRecord 标准属性（不进入 JSON 附加字段）
_RESERVED = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志：ts / level / layer / trace_id / msg / 附加字段。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "layer": getattr(record, "layer", "-"),
            "trace_id": get_trace_id(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _LayerFilter(logging.Filter):
    """向记录注入 layer 字段（logger 名即层标识）。"""

    def __init__(self, layer: str) -> None:
        super().__init__()
        self.layer = layer

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "layer"):
            record.layer = self.layer
        return True


def configure_logging(
    level: int = logging.INFO, stream: Any | None = None
) -> logging.Logger:
    """配置 "svsg" 日志树（幂等）：JSON 格式，默认输出 stderr。"""
    root = logging.getLogger("svsg")
    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        root.setLevel(level)
    return root


def get_logger(layer: str) -> logging.Logger:
    """获取带 layer 标记的子 logger（如 get_logger("l2") → svsg.l2）。"""
    logger = logging.getLogger(f"svsg.{layer}")
    if not any(getattr(f, "layer", None) == layer for f in logger.filters):
        logger.addFilter(_LayerFilter(layer))
    return logger
