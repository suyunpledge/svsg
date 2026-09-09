"""S0 校验管道：IR 载荷校验（S2 硬拒绝）与依赖软检查（S3）。

职责划分与 contracts 层一致：
- validate_ir_payload：S2 裁判入口（IR 可能源自本机 L1，也可能跨进程序列化
  传输，因此入口处统一重跑 Schema 硬校验）；
- check_dependencies：S3 软检查（悬空 target_id → 软重试，最多 1 次）。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from svsg.contracts import IR, check_relation_references, from_validation_error


def validate_ir_payload(payload: dict[str, Any] | IR) -> IR:
    """S0 → S1/S2 的校验入口。

    接受 dict（跨进程序列化载荷）或已构造的 IR（本机直连，round-trip 复核）。
    校验失败抛 SVSGError（E1xxx → S2），由编排器映射为 VALIDATION_FAILED 事件。
    """
    try:
        if isinstance(payload, IR):
            return IR.model_validate(payload.model_dump(by_alias=True))
        return IR.model_validate(payload)
    except ValidationError as exc:
        raise from_validation_error(exc) from exc


def check_dependencies(ir: IR) -> list[tuple[int, str, int]]:
    """S0 → S3 的软检查入口：返回悬空引用列表 [(source_id, rel_type, target_id)]。

    非空时编排器应触发 DEPENDENCY_MISSING 事件（软重试一次；L1 重建 IR）。
    """
    return [
        (src, rel.value, tgt)
        for src, rel, tgt in check_relation_references(ir)
    ]
