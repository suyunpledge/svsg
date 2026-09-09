"""可观测性平面测试：三档采样、异常触发、trace_id、JSON 日志格式。"""

from __future__ import annotations

import io
import json
import logging
import random
import sys
from pathlib import Path

from svsg.observability import (
    ANOMALY_FINAL_STATES,
    JsonFormatter,
    SampleMode,
    SamplingPolicy,
    TraceStore,
    configure_logging,
    get_logger,
    get_trace_id,
    is_anomaly,
    new_trace_id,
    sample_mode_from_env,
    set_trace_id,
)


def test_dev_mode_samples_everything():
    policy = SamplingPolicy(SampleMode.DEV, rng=random.Random(0))
    assert all(policy.should_sample() for _ in range(100))


def test_warmup_sampling_rate():
    policy = SamplingPolicy(SampleMode.WARMUP, rng=random.Random(42))
    hits = sum(1 for _ in range(200) if policy.should_sample())
    # 种子固定 → 结果确定；10% 费率下 200 次约 20 次，取宽容区间
    assert 5 <= hits <= 60, hits


def test_anomaly_always_sampled_in_steady():
    policy = SamplingPolicy(SampleMode.STEADY, steady_rate=0.0, rng=random.Random(0))
    assert policy.should_sample() is False  # 费率 0：正常流量全丢
    assert policy.should_sample(anomaly=True) is True  # 异常恒采样


def test_trace_store_record_and_drop_counters():
    store = TraceStore(SamplingPolicy(SampleMode.STEADY, steady_rate=0.0,
                                      rng=random.Random(0)))
    assert store.record({"k": 1}) is False  # 未采样
    assert store.dropped == 1 and store.sampled == 0
    assert store.record({"k": 2}, anomaly=True) is True  # 异常采样
    assert store.sampled == 1
    retained = store.retained()
    assert len(retained) == 1
    assert retained[0]["k"] == 2 and retained[0]["anomaly"] is True
    assert retained[0]["trace_id"] == get_trace_id()  # 附 trace_id


def test_trace_store_ring_capacity():
    store = TraceStore(SamplingPolicy(SampleMode.DEV), capacity=3)
    for i in range(5):
        store.record({"seq": i})
    assert [e["seq"] for e in store.retained()] == [2, 3, 4]  # 只留最近 3 条


def test_trace_store_creates_sink_parent_dirs(tmp_path: Path):
    sink = tmp_path / "nested" / "trace.jsonl"
    store = TraceStore(SamplingPolicy(SampleMode.DEV), sink_path=sink)
    store.record({"event": "demo"})
    assert sink.exists()


def test_trace_id_contextvar():
    tid = new_trace_id()
    assert get_trace_id() == tid and len(tid) == 16
    set_trace_id("manual_tid")
    assert get_trace_id() == "manual_tid"


def test_is_anomaly_rules():
    assert is_anomaly({"final_state": "S1", "degraded_instances": []}) is False
    for state in ANOMALY_FINAL_STATES:
        assert is_anomaly({"final_state": state}) is True
    assert is_anomaly({"final_state": "S1", "degraded_instances": [3]}) is True
    assert is_anomaly(
        {"final_state": "S1"}, anchor_violations=[{"code": "E3002"}]
    ) is True


def test_sample_mode_from_env():
    import os

    old = os.environ.pop("SVSG_SAMPLE_MODE", None)
    try:
        assert sample_mode_from_env() is SampleMode.DEV
        os.environ["SVSG_SAMPLE_MODE"] = "steady"
        assert sample_mode_from_env() is SampleMode.STEADY
        os.environ["SVSG_SAMPLE_MODE"] = "nonsense"
        assert sample_mode_from_env() is SampleMode.DEV  # 非法值回落
    finally:
        if old is None:
            os.environ.pop("SVSG_SAMPLE_MODE", None)
        else:
            os.environ["SVSG_SAMPLE_MODE"] = old


def test_json_formatter_fields():
    stream = io.StringIO()
    logger = get_logger("l2")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        logger.setLevel(logging.INFO)
        set_trace_id("traceabc123")
        logger.info("fsm.transition", extra={"event": "ANCHOR_PASSED"})
    finally:
        logger.removeHandler(handler)
    payload = json.loads(stream.getvalue().strip())
    assert payload["layer"] == "l2"
    assert payload["trace_id"] == "traceabc123"
    assert payload["msg"] == "fsm.transition"
    assert payload["event"] == "ANCHOR_PASSED"
    assert payload["level"] == "INFO"
    assert "ts" in payload


def test_configure_logging_idempotent():
    a = configure_logging()
    b = configure_logging()
    assert a is b
    assert len(a.handlers) == 1


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
