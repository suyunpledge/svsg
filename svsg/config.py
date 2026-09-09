"""集中配置：环境变量 + .env 驱动的单一事实源（工程化部署底座）。

所有可调项经 pydantic-settings 从环境变量（前缀 ``SVSG_``）与 ``.env``
加载，非法值在应用装配期即失败（fail-fast），避免运行时劣化。默认监听
``0.0.0.0:3001``（可通过 ``SVSG_HOST`` / ``SVSG_PORT`` 覆盖）。字段名
经前缀拼接后与历史环境变量命名（SVSG_LLM_* / SVSG_SAMPLE_MODE 等）完全
兼容：``llm_provider`` → ``SVSG_LLM_PROVIDER``。
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
SampleModeStr = Literal["dev", "warmup", "steady"]


class Settings(BaseSettings):
    """SVSG 服务配置（进程级单例，见 get_settings）。"""

    model_config = SettingsConfigDict(
        env_prefix="SVSG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- 服务监听（默认仅本机回环，避免无鉴权服务暴露到局域网/公网；
    #     端口与 AI Platform 的 3001 错开，实际部署形态为 127.0.0.1:3002）--
    host: str = "127.0.0.1"
    port: int = 3002

    # -- L3 主控 LLM --
    llm_provider: Literal["stub", "openai"] = "stub"
    llm_base_url: str | None = None
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 3.0

    # -- L1.5 视觉验证（必须严格小于 llm_timeout_s / 3，由编排器装配期断言）--
    l15_timeout_s: float = 0.9

    # -- L1 检测器 --
    detector: Literal["stub", "yolo"] = "stub"
    yolo_weights: str = "yolov8n.pt"

    # -- 可观测性 --
    sample_mode: SampleModeStr = "steady"
    trace_file: str | None = None
    log_level: LogLevel = "INFO"

    # -- 安全 / 网关入口 --
    api_key: str | None = None  # 设置后要求请求头 X-API-Key 校验
    cors_origins: list[str] = []

    # -- 登录机制（v0）：启用后 /v1/* 由 X-API-Key 切换为 Bearer 令牌鉴权 --
    auth_enabled: bool = False
    auth_db: str = "svsg_auth.db"
    auth_token_ttl_s: int = 604800  # 令牌有效期：7 天
    auth_pbkdf2_iters: int = 200_000

    @model_validator(mode="after")
    def _checks(self) -> Settings:
        self.llm_base_url = (
            self.llm_base_url.strip() if isinstance(self.llm_base_url, str) else None
        ) or None
        self.llm_api_key = self.llm_api_key.strip()
        self.llm_model = self.llm_model.strip()
        self.api_key = (
            self.api_key.strip() if isinstance(self.api_key, str) else None
        ) or None
        self.trace_file = (
            self.trace_file.strip() if isinstance(self.trace_file, str) else None
        ) or None

        if self.llm_provider == "openai" and not self.llm_base_url:
            raise ValueError(
                "SVSG_LLM_PROVIDER=openai 时必须提供 SVSG_LLM_BASE_URL"
            )
        if self.llm_provider == "openai" and not self.llm_api_key:
            raise ValueError(
                "SVSG_LLM_PROVIDER=openai 时必须提供非空的 SVSG_LLM_API_KEY"
            )
        if self.llm_provider == "openai" and not self.llm_model:
            raise ValueError(
                "SVSG_LLM_PROVIDER=openai 时必须提供非空的 SVSG_LLM_MODEL"
            )
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    """返回进程级缓存配置实例；用 cache_clear 复位（测试/热更新场景）。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def cache_clear() -> None:
    """清除配置缓存（供测试或程序化重载）。"""
    global _settings
    _settings = None
