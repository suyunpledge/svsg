# SVSG — 结构化视觉语义网关 V5.2.1

## 两个核心能力

### 1. 给 LLM 配上准确的 VLM 眼镜

传统的视觉问答是把图片直接扔给 VLM（如 GPT-4V、Claude Vision），让它"看图说话"。问题是 VLM 的视觉感知不稳定——同一张图问两次可能给出矛盾的答案，计数经常出错，空间关系判断模糊。

SVSG 走了一条完全不同的路：**不让 VLM 直接看图，而是先用精确的检测器把图片翻译成结构化的 IR（中间表示），再把 IR 喂给 LLM**。

```
传统方式：  图片 → VLM（黑盒）→ 答案（不可靠）
SVSG 方式：  图片 → L1 检测编译 → IR（结构化证据）→ LLM（白盒推理）→ 答案（可审计）
```

这意味着：
- **LLM 不需要"看"图**，它只需要读懂结构化的检测结果（类别、位置、数量、关系）
- **检测精度由 YOLO 等专业模型保证**，不依赖 VLM 的模糊视觉能力
- **每条结论都能追溯到具体的检测证据**，不是"我觉得图里有三个苹果"

### 2. 带证据锚定的视觉审核

在 V1 能力的基础上，SVSG 还提供完整的审核链路：L1 检测 → L1.5 声明验证 → L3 编排 → 证据锚定校验。LLM 的每条输出都会被锚定到 IR 中的检测证据，编造或无依据的断言会被自动否决。

## 快速开始

```bash
pip install -e ".[api,llm]"        # 基础服务
pip install -e ".[ml]"             # 需要 YOLO 真实检测时
cp .env.example .env               # 按需修改（默认 127.0.0.1:3002）
python -m svsg
```

健康检查：`GET /healthz`（无鉴权）

## 鉴权

- **X-API-Key**：`.env` 中 `SVSG_API_KEY` 非空时启用，请求头 `X-API-Key: ***`
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

## 与传统 VLM 方案的对比

| | 传统 VLM 直答 | SVSG 结构化方案 |
|---|---|---|
| 视觉感知 | VLM 黑盒，不可控 | YOLO 等专业检测器，精确 |
| 计数准确率 | 经常出错 | 像素级检测，精确计数 |
| 空间关系 | 模糊描述 | 结构化 IR（left_of/right_of/contains 等） |
| 可审计性 | 无法追溯 | 每条结论锚定到检测证据 |
| 幻觉风险 | 高（VLM 可能编造） | 低（锚定校验自动否决无依据断言） |
| 成本 | 每次都调 VLM（贵） | 检测一次，LLM 只读文本（便宜） |

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
