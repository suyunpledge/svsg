# SVSG — 结构化视觉语义网关 V5.2.1

图片 + 问题 → L1 检测编译 → L1.5 声明验证 → L3 LLM 编排 → 带证据锚定的结构化答案。

## 快速开始

```bash
pip install -e ".[api,llm]"        # 基础服务
pip install -e ".[ml]"             # 需要 YOLO 真实检测时
cp .env.example .env               # 按需修改（默认 127.0.0.1:3002）
python -m svsg
```

健康检查：`GET /healthz`（无鉴权）

## 鉴权

- **X-API-Key**：`.env` 中 `SVSG_API_KEY` 非空时启用，请求头 `X-API-Key: <key>`。
- **Bearer 令牌**：`SVSG_AUTH_ENABLED=1` 后可用 `/auth/register|login|logout`，`/v1/*` 改为 Bearer 鉴权（替代 X-API-Key）。
- AI Platform 等调用方在设置界面填入同一 key 即可。

## 主要端点

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | /v1/analyze | 是 | 传入 L1 已产出的 IR 载荷 |
| POST | /v1/analyze-image | 是 | 上传图片+问题，完整闭环 |
| GET | /healthz | 否 | 健康检查 |

## 架构分层

- **contracts**：跨层契约（IR Schema、错误码、枚举）唯一来源
- **l1_compiler**：检测器适配（stub/demo/YOLO）→ 几何关系 → 不确定性 → IR
- **l15_verifier**：视觉验证服务（超时隔离 + 静默降级）
- **l2_runtime**：FSM（S0~S6）、意图路由、冲突解决、证据锚定
- **l3_orchestrator**：LLM 编排循环、双通道答案、锚定否决重生成

## 测试与静态检查

```bash
python -m pytest -q                # 108 个测试
python -m ruff check .             # 全绿
python -m mypy --python-version 3.12   # 0 error
```

## 运维

- 看门狗：计划任务 `Svsg-Watchdog`（登录 + 每 5 分钟）确保 3002 在线
- 日志：`_svsg.out.log` / `_svsg.err.log` / `watchdog-3002.log`
- 版本控制：git（`archive/pre-git-backups/` 保留 git 化前手工备份）

