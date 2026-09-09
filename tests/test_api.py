"""FastAPI 入口测试：请求封装、错误映射、trace 头、三档采样留痕。"""

from __future__ import annotations

import random
import sys

from fastapi.testclient import TestClient

from svsg.api import create_app
from svsg.contracts import L3Answer
from svsg.l3_orchestrator import Orchestrator, OrchestratorConfig, StubLLM
from svsg.l15_verifier import L15Verifier, StubVerifierBackend, VerifierConfig
from svsg.observability import SampleMode, SamplingPolicy, TraceStore


def _payload(*, verify2: bool = True):
    return {
        "schema_version": "5.2.1",
        "image_id": "img_api_001",
        "image_width": 1920,
        "image_height": 1080,
        "timestamp": "2026-09-02T10:00:00Z",
        "scene_type": "orthographic",
        "detector_model": "stub_detector_v0",
        "detections": [
            {
                "instance_id": 1,
                "class": "screw",
                "bbox_px": [420, 150, 480, 210],
                "detection_score": 0.92,
                "conf_level": "high",
                "verification_required": False,
                "uncertainty": {
                    "classification_entropy": 0.08,
                    "local_edge_sharpness": 0.73,
                    "tta_variance": None,
                },
                "relations": [],
            },
            {
                "instance_id": 2,
                "class": "screw",
                "bbox_px": [900, 400, 1000, 500],
                "detection_score": 0.55,
                "conf_level": "low",
                "verification_required": verify2,
                "uncertainty": {
                    "classification_entropy": 0.6,
                    "local_edge_sharpness": 0.3,
                    "tta_variance": None,
                },
                "relations": [],
            },
            {
                "instance_id": 3,
                "class": "washer",
                "bbox_px": [1100, 400, 1160, 460],
                "detection_score": 0.88,
                "conf_level": "high",
                "verification_required": False,
                "uncertainty": {
                    "classification_entropy": 0.1,
                    "local_edge_sharpness": 0.8,
                    "tta_variance": None,
                },
                "relations": [],
            },
        ],
    }


def _answer(claims, text="答案文本"):
    return L3Answer.model_validate({"claims": claims, "final_answer": text})


def _client(answers, backend=None):
    verifier = L15Verifier(
        backend or StubVerifierBackend(), VerifierConfig(timeout_s=0.05)
    )
    llm = StubLLM(answers=answers)
    orch = Orchestrator(verifier, llm, OrchestratorConfig(llm_timeout_s=0.5))
    app = create_app(orchestrator=orch)
    return TestClient(app), app


def test_analyze_delivers_with_trace_header():
    claims = [{"instance_id": 1, "field": "class", "value": "screw"},
              {"field": "count", "value": 3}]
    client, _ = _client([_answer(claims)])
    resp = client.post(
        "/v1/analyze", json={"query": "图里有什么", "ir": _payload(verify2=False)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "delivered"
    assert body["final_answer"] == "答案文本"
    assert body["human_review_required"] is False
    assert body["claims"][0]["value"] == "screw"
    assert body["trace_id"]
    # 响应头回传 trace_id，供调用方关联日志与采样记录
    assert resp.headers["X-SVSG-Trace-Id"] == body["trace_id"]


def test_analyze_rejected_is_business_level_200():
    client, _ = _client([])
    bad = _payload(verify2=False)
    bad["detections"][0]["bbox_px"] = [10, 10, 50, 5000]
    resp = client.post("/v1/analyze", json={"query": "q", "ir": bad})
    assert resp.status_code == 200  # 业务级结果，非 HTTP 错误
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["error"]["code"] == "E1002"
    assert body["final_answer"] is None


def test_analyze_degraded_marks_human_review():
    backend = StubVerifierBackend(batches=[{2: {"class": "screw"}}], delay_s=0.3)
    claims = [{"field": "count", "value": 3}]
    client, _ = _client([_answer(claims)], backend)
    resp = client.post(
        "/v1/analyze", json={"query": "描述一下", "ir": _payload()}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "delivered_with_review"
    assert body["human_review_required"] is True
    assert body["degraded_instances"] == [2]


def test_anomaly_always_recorded_in_steady_mode():
    backend = StubVerifierBackend(batches=[{2: {"class": "screw"}}], delay_s=0.3)
    client, app = _client([_answer([{"field": "count", "value": 3}])], backend)
    # 稳态 + 费率 0：正常流量全丢，异常（降级）恒采样
    app.state.trace_store = TraceStore(
        SamplingPolicy(SampleMode.STEADY, steady_rate=0.0, rng=random.Random(0))
    )
    client.post("/v1/analyze", json={"query": "q", "ir": _payload()})
    retained = app.state.trace_store.retained()
    assert len(retained) == 1
    assert retained[0]["anomaly"] is True
    assert retained[0]["session"]["final_state"] == "S6"
    assert retained[0]["violations"] == []


def test_normal_traffic_dropped_in_steady_mode():
    claims = [{"field": "count", "value": 3}]
    client, app = _client([_answer(claims)])
    app.state.trace_store = TraceStore(
        SamplingPolicy(SampleMode.STEADY, rng=random.Random(42))
    )
    client.post(
        "/v1/analyze", json={"query": "q", "ir": _payload(verify2=False)}
    )
    assert app.state.trace_store.retained() == []
    assert app.state.trace_store.dropped == 1


def test_request_validation_returns_422():
    client, _ = _client([])
    resp = client.post("/v1/analyze", json={"query": "", "ir": {}})
    assert resp.status_code == 422  # FastAPI 请求模型校验


def test_healthz():
    client, _ = _client([])
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["version"] == "5.2.1"


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
