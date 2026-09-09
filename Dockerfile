FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 仅安装运行时依赖（api extra）；用到真实检测器再追加 .[ml]
COPY pyproject.toml ./
COPY svsg ./svsg
RUN pip install ".[api]"

EXPOSE 3002

CMD ["python", "-m", "svsg"]