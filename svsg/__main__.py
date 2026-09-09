"""命令行入口：``python -m svsg`` 启动 API 服务。

监听地址/端口等由 SVSG_* 环境变量（或 .env）驱动，默认 0.0.0.0:3001。
生产环境建议经 Gunicorn（Linux，多 worker）或直接复用 uvicorn 的
``--workers``（首次适配阶段单 worker 足够）。
"""

from __future__ import annotations

import uvicorn

from svsg.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "svsg.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
