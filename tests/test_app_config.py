"""应用化装配测试：集中配置 + 图片上传端到端 + 可选鉴权。

覆盖「如何让系统应用化」的机器可执行验收：
- Settings ?????????? 127.0.0.1:3002?
- ??? AI Platform 3001 ???openai ? base_url ??????
- POST /v1/analyze-image 走 image → L1 编译 → 编排 → 答案 完整闭环；
- 可选 X-API-Key 鉴权与不可读图片的 422。
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from svsg.api import create_app
from svsg.config import Settings
from svsg.contracts import L3Answer
from svsg.l1_compiler import DemoDetector, L1Compiler
from svsg.l3_orchestrator import Orchestrator, OrchestratorConfig, StubLLM
from svsg.l15_verifier import L15Verifier, StubVerifierBackend, VerifierConfig


def _png_bytes(size: tuple[int, int] = (640, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


def _answer() -> L3Answer:
    return L3Answer.model_validate(
        {
            "claims": [
                {"field": "count", "value": 2},
                {"instance_id": 1, "field": "class", "value": "screw"},
                {"instance_id": 2, "field": "class", "value": "washer"},
            ],
            "final_answer": "检测到 1 个螺丝和 1 个垫圈",
        }
    )


def _client(settings: Settings | None = None) -> TestClient:
    verifier = L15Verifier(StubVerifierBackend(), VerifierConfig(timeout_s=0.9))
    llm = StubLLM(answers=[_answer()])
    orch = Orchestrator(verifier, llm, OrchestratorConfig(llm_timeout_s=3.0))
    app = create_app(
        orchestrator=orch, compiler=L1Compiler(DemoDetector()), settings=settings
    )
    return TestClient(app)


# ------------------------------------------------------------------ 配置


def test_settings_default_port_3002():
    assert Settings.model_fields["port"].default == 3002


def test_settings_default_host_loopback():
    assert Settings.model_fields["host"].default == "127.0.0.1"


def test_settings_openai_requires_base_url():
    with pytest.raises(ValidationError):
        Settings(llm_provider="openai", _env_file=None)


def test_settings_openai_requires_api_key():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_provider="openai",
            llm_base_url="https://api.example.com/v1",
            llm_api_key="   ",
        )


def test_settings_openai_accepts_full_config():
    s = Settings(
        _env_file=None,
        port=4321,
        llm_provider="openai",
        llm_base_url="https://api.example.com/v1",
        llm_api_key="sk-test",
    )
    assert s.port == 4321
    assert s.llm_base_url == "https://api.example.com/v1"


def test_blank_api_key_is_treated_as_disabled_auth():
    s = Settings(_env_file=None, api_key="   ")
    assert s.api_key is None


# ------------------------------------------------------------------ 图片上传


def test_analyze_image_delivers():
    client = _client()
    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("demo.png", _png_bytes(), "image/png")},
        data={"query": "图里有什么"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "delivered"
    assert body["final_answer"] == "检测到 1 个螺丝和 1 个垫圈"
    assert body["trace_id"]
    assert resp.headers["X-SVSG-Trace-Id"] == body["trace_id"]


def test_analyze_image_unreadable_returns_422():
    client = _client()
    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
        data={"query": "q"},
    )
    assert resp.status_code == 422


def test_analyze_image_enforces_api_key():
    client = _client(settings=Settings(api_key="secret", _env_file=None))
    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("demo.png", _png_bytes(), "image/png")},
        data={"query": "q"},
    )
    assert resp.status_code == 401

    ok = client.post(
        "/v1/analyze-image",
        files={"file": ("demo.png", _png_bytes(), "image/png")},
        data={"query": "q"},
        headers={"X-API-Key": "secret"},
    )
    assert ok.status_code == 200



def test_settings_default_cors_is_closed():
    assert Settings.model_fields["cors_origins"].default == []


def test_analyze_image_rejects_oversized_upload():
    client = _client()
    oversized = b"x" * (16 * 1024 * 1024 + 1)
    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("too-large.png", oversized, "image/png")},
        data={"query": "q"},
    )
    assert resp.status_code == 413
