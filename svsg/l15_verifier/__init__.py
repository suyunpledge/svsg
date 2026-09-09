"""L1.5 视觉验证服务：按需局部验证，输出结构化验证报告。

- service：VerifierBackend 协议 + L15Verifier（asyncio.wait_for 强制超时，
  绝不抛异常、绝不重试，超时/故障统一返回非成功状态的报告）
- fallback：静默降级策略（超时预算断言 + 降级实例集合推导）
"""

from .fallback import assert_timeout_budget, silent_degrade
from .service import (
    L15Verifier,
    StubVerifierBackend,
    VerifierBackend,
    VerifierConfig,
)

__all__ = [
    "assert_timeout_budget",
    "silent_degrade",
    "L15Verifier",
    "StubVerifierBackend",
    "VerifierBackend",
    "VerifierConfig",
]
