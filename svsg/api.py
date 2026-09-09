"""SVSG HTTP 入口：依赖装配 + 请求封装 + 错误映射 + 访问留痕。

两种请求形态：
- ``POST /v1/analyze``：传入 L1 已产出的 IR 载荷（供 L1 与网关解耦部署）；
- ``POST /v1/analyze-image``：直接上传图片 + 查询文本，网关内部走
  ``image → L1 编译 → IR → 编排 → 答案`` 的完整闭环（应用化主形态）。

装配全部由 svsg.config.Settings 驱动（环境变量前缀 ``SVSG_`` + ``.env``）：
    SVSG_PORT=3001
    SVSG_LLM_PROVIDER=openai
    SVSG_LLM_BASE_URL / SVSG_LLM_API_KEY / SVSG_LLM_MODEL
    SVSG_DETECTOR=stub|yolo   SVSG_YOLO_WEIGHTS=yolov8n.pt
    SVSG_SAMPLE_MODE=steady   SVSG_TRACE_FILE=traces.jsonl
    SVSG_API_KEY=...          # 设置后要求请求头 X-API-Key

语义约定：
- 业务级结果（rejected / human_review / delivered_with_review）以
  HTTP 200 + status 字段返回，由调用方按 status 分流；
- SVSGError 逃逸到 API 层视为编排器未消化的内部错误：
  E1xxx（参数类）→ 422，其余 → 500；
- 每个请求分配 trace_id（响应头 X-SVSG-Trace-Id），并在响应后
  按三档采样策略留痕（异常恒采样）。
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from svsg import __version__
from svsg.auth import build_auth_router, require_bearer
from svsg.config import Settings, get_settings
from svsg.contracts import SVSGError
from svsg.l1_compiler import DemoDetector, ImageMeta, L1Compiler, YoloDetector, image_dimensions
from svsg.l3_orchestrator import (
    LLMProtocol,
    Orchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    StubLLM,
)
from svsg.l15_verifier import L15Verifier, StubVerifierBackend, VerifierConfig
from svsg.observability import (
    SampleMode,
    SamplingPolicy,
    TraceStore,
    configure_logging,
    get_logger,
    get_trace_id,
    is_anomaly,
    new_trace_id,
)

logger = get_logger("api")

#: 上传图片字节上限(16MB):Starlette 不限制 multipart 大小,需自行设防
MAX_IMAGE_BYTES = 16 * 1024 * 1024


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    ir: dict[str, Any] = Field(..., description="L1 产出的 IR 载荷（JSON）")


class AnalyzeResponse(BaseModel):
    status: str
    final_answer: str | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)
    class_overrides: dict[str, str] = Field(default_factory=dict)
    degraded_instances: list[int] = Field(default_factory=list)
    disputed_instances: list[int] = Field(default_factory=list)
    human_review_required: bool = False
    anchor_violations: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    session_summary: dict[str, Any] = Field(default_factory=dict)
    trace_id: str


# ---------------------------------------------------------------------- 装配


def build_llm(settings: Settings) -> LLMProtocol:
    """按配置装配 L3 后端；openai 模式缺 base_url 已在 Settings 校验期失败。"""
    if settings.llm_provider == "openai":
        from .l3_orchestrator.openai_adapter import OpenAICompatibleLLM

        assert settings.llm_base_url is not None
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    # Stub 后端无脚本答案 → complete 抛异常 → 编排器按 S5 中止。
    # 仅用于冒烟联调；生产必须切换 openai provider。
    return StubLLM()


def build_verifier(settings: Settings) -> L15Verifier:
    """L1.5 验证门面（当前后端为确定性桩，真实视觉验证后端为 Phase 2 项）。"""
    return L15Verifier(
        StubVerifierBackend(), VerifierConfig(timeout_s=settings.l15_timeout_s)
    )


def build_compiler(settings: Settings) -> L1Compiler:
    """装配 L1 编译器：stub → 演示桩（无模型）；yolo → YOLOv8 适配器。"""
    if settings.detector == "yolo":
        return L1Compiler(YoloDetector(settings.yolo_weights))
    return L1Compiler(DemoDetector())


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """从配置装配编排器（L1.5/L3 超时预算由 Orchestrator 装配期断言）。"""
    settings = settings or get_settings()
    return Orchestrator(
        build_verifier(settings),
        build_llm(settings),
        OrchestratorConfig(llm_timeout_s=settings.llm_timeout_s),
    )


def _to_response(result: OrchestratorResult) -> AnalyzeResponse:
    """编排结果 → 统一响应体（analyze / analyze-image 共用）。"""
    return AnalyzeResponse(
        status=result.status,
        final_answer=result.answer.final_answer if result.answer else None,
        claims=[c.model_dump() for c in result.answer.claims] if result.answer else [],
        class_overrides={str(k): v for k, v in result.class_overrides.items()},
        degraded_instances=sorted(result.degraded_instances),
        disputed_instances=sorted(result.disputed_instances),
        human_review_required=result.human_review_required,
        anchor_violations=result.anchor_violations,
        error=result.error,
        session_summary=result.session_summary,
        trace_id=get_trace_id(),
    )


def _image_id(filename: str | None) -> str:
    """从上传文件名派生稳定的 image_id（未知文件名回落 uuid）。"""
    stem = Path(filename or "").stem.strip() or "img"
    return f"{stem}_{uuid4().hex[:8]}"


def _api_key_dependency(settings: Settings):
    """可选 API Key 鉴权依赖：settings.api_key 未设置时放行。"""

    async def require_api_key(request: Request) -> None:
        if settings.api_key is None:
            return
        provided = request.headers.get("X-API-Key")
        # 恒时比较:消除逐字符匹配的时序侧信道
        if not provided or not hmac.compare_digest(provided, settings.api_key):
            raise HTTPException(status_code=401, detail="无效或缺失的 API Key")

    return require_api_key


# ---------------------------------------------------------------------- 应用工厂


def create_app(
    orchestrator: Orchestrator | None = None,
    *,
    settings: Settings | None = None,
    compiler: L1Compiler | None = None,
) -> FastAPI:
    """应用工厂（测试注入编排器/编译器，生产走配置装配）。"""
    settings = settings or get_settings()
    configure_logging(level=getattr(logging, settings.log_level, logging.INFO))

    app = FastAPI(title="SVSG", version=__version__)
    app.state.settings = settings
    app.state.orchestrator = orchestrator or build_orchestrator(settings)
    app.state.compiler = compiler or build_compiler(settings)
    app.state.trace_store = TraceStore(
        SamplingPolicy(mode=SampleMode(settings.sample_mode)),
        sink_path=settings.trace_file,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    require_api_key = _api_key_dependency(settings)
    if settings.auth_enabled:  # 登录机制 v0：挂载 /auth/* 并切换为 Bearer 鉴权
        app.include_router(build_auth_router(settings))
        require_api_key = require_bearer(settings)

    @app.middleware("http")
    async def trace_and_sample(request: Request, call_next):
        trace_id = new_trace_id()
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-SVSG-Trace-Id"] = trace_id
        summary = getattr(request.state, "session_summary", None)
        if summary is not None:
            violations = getattr(request.state, "anchor_violations", None) or []
            app.state.trace_store.record(
                {"path": request.url.path, "session": summary, "violations": violations},
                anomaly=is_anomaly(summary, anchor_violations=violations),
            )
        return response

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def analyze(req: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        result = await app.state.orchestrator.run(req.query, req.ir)
        request.state.session_summary = result.session_summary
        request.state.anchor_violations = result.anchor_violations
        logger.info(
            "analyze.done", extra={"status": result.status, "query_len": len(req.query)}
        )
        return _to_response(result)

    @app.post(
        "/v1/analyze-image",
        response_model=AnalyzeResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def analyze_image(
        request: Request,
        query: Annotated[str, Form(min_length=1, max_length=2000)],
        file: Annotated[UploadFile, File()],
    ) -> AnalyzeResponse:
        chunks: list[bytes] = []
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="????(?? 16MB)")
            chunks.append(chunk)
        data = b"".join(chunks)
        try:
            width, height = await asyncio.to_thread(image_dimensions, data)
        except ImportError as exc:  # pragma: no cover - Pillow 缺失属部署配置错误
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"无法解析图片: {exc}") from exc

        meta = ImageMeta(image_id=_image_id(file.filename), width=width, height=height)
        try:
            ir = await asyncio.to_thread(app.state.compiler.compile, data, meta)
        except SVSGError as err:
            # L1 编译后的 IR 未通过 S2 硬校验 → 业务级拒绝（与 analyze 语义一致）
            return AnalyzeResponse(
                status="rejected", error=err.to_dict(), trace_id=get_trace_id()
            )

        result = await app.state.orchestrator.run(query, ir)
        request.state.session_summary = result.session_summary
        request.state.anchor_violations = result.anchor_violations
        logger.info(
            "analyze_image.done",
            extra={"status": result.status, "image_id": meta.image_id},
        )
        return _to_response(result)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.exception_handler(SVSGError)
    async def svsg_error_handler(request: Request, exc: SVSGError) -> JSONResponse:
        # SVSGError 逃逸到 API 层 = 编排器未消化的内部错误
        status_code = 422 if exc.code.value.startswith("E1") else 500
        logger.error("svsg_error", extra={"code": exc.code.value})
        return JSONResponse(status_code=status_code, content={"error": exc.to_dict()})

    return app


app = create_app()
