"""L1.5 视觉验证服务测试：超时注入、故障路径、幂等性、预算断言。"""

from __future__ import annotations

import asyncio
import sys

from svsg.contracts import VerificationStatus
from svsg.l15_verifier import (
    L15Verifier,
    StubVerifierBackend,
    VerifierConfig,
    assert_timeout_budget,
    silent_degrade,
)


def _run(coro):
    return asyncio.run(coro)


def test_success_batch():
    backend = StubVerifierBackend(batches=[{1: {"class": "screw", "confidence": 0.9}}])
    verifier = L15Verifier(backend, VerifierConfig(timeout_s=1.0))
    report = _run(verifier.verify_instances("img_1", [1]))
    assert report.status == VerificationStatus.SUCCESS
    assert len(report.results) == 1
    assert report.results[0].class_label == "screw"
    assert report.elapsed_ms is not None


def test_timeout_returns_silent_report():
    backend = StubVerifierBackend(batches=[{1: {"class": "screw"}}], delay_s=0.3)
    verifier = L15Verifier(backend, VerifierConfig(timeout_s=0.05))
    report = _run(verifier.verify_instances("img_1", [1, 2]))
    assert report.status == VerificationStatus.TIMEOUT
    assert report.results == []  # 保守：部分输出不采信
    assert silent_degrade([1, 2], report) == {1, 2}


def test_backend_error_same_path_as_timeout():
    backend = StubVerifierBackend(fail_with=RuntimeError("gpu on fire"))
    verifier = L15Verifier(backend, VerifierConfig(timeout_s=0.5))
    report = _run(verifier.verify_instances("img_1", [1]))
    assert report.status == VerificationStatus.ERROR
    assert report.results == []
    assert silent_degrade([1], report) == {1}


def test_empty_request_short_circuits():
    verifier = L15Verifier(StubVerifierBackend())
    report = _run(verifier.verify_instances("img_1", []))
    assert report.status == VerificationStatus.SUCCESS
    assert report.results == []


def test_idempotent_calls():
    batch = {1: {"class": "screw", "confidence": 0.9}}
    backend = StubVerifierBackend(batches=[batch, dict(batch)])
    verifier = L15Verifier(backend, VerifierConfig(timeout_s=1.0))
    r1 = _run(verifier.verify_instances("img_1", [1]))
    r2 = _run(verifier.verify_instances("img_1", [1]))
    assert r1.status == r2.status == VerificationStatus.SUCCESS
    assert [r.class_label for r in r1.results] == [r.class_label for r in r2.results]
    assert backend.calls == [[1], [1]]


def test_timeout_budget_assertion():
    assert_timeout_budget(0.9, 3.0)  # 0.9 < 1.0 ✓
    try:
        assert_timeout_budget(1.0, 3.0)  # 不严格小于 1/3 → 违反
    except ValueError as exc:
        assert "1/3" in str(exc)
    else:
        raise AssertionError("预算违规应抛 ValueError")


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
