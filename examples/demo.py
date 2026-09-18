"""SVSG Demo：对比无锚定 LLM 幻觉 vs SVSG 证据锚定拦截效果。

Usage:
    python -m examples.demo

No real image / GPU / API key required — all stub/demo components.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from svsg.contracts import Claim, ErrorCode, L3Answer
from svsg.l1_compiler.compiler import ImageMeta, L1Compiler, L1CompilerConfig
from svsg.l1_compiler.detector_adapter import RawDetection, StubDetector
from svsg.l2_runtime.evidence_anchor import run_anchor
from svsg.contracts.enums import SceneType


# ── Demo data ──────────────────────────────────────────────────

# IR will have 2 instances: screw + washer
# The "hallucinated" LLM answer claims count=3 (fabricated 1 extra)

STUB_DETECTIONS = [
    RawDetection(
        label="screw",
        bbox=(50, 50, 120, 120),
        detection_score=0.93,
        local_edge_sharpness=0.80,
    ),
    RawDetection(
        label="washer",
        bbox=(200, 150, 280, 220),
        detection_score=0.88,
        local_edge_sharpness=0.76,
    ),
]

HALLUCINATED_ANSWER = L3Answer(
    claims=[
        {"field": "count", "value": 3, "instance_id": None},  # hallucinated
    ],
    final_answer="图中可以看到 3 个紧固件（2 颗螺丝和 1 个垫圈），分布在左上和右侧区域。",
)

CORRECTED_ANSWER = L3Answer(
    claims=[
        {"field": "count", "value": 2, "instance_id": None},
    ],
    final_answer="图中可以看到 2 个紧固件：1 颗螺丝（左上）和 1 个垫圈（右侧），没有遮挡。",
)


def build_ir():
    """Build IR from StubDetector — no real image needed."""
    stub = StubDetector(detections=STUB_DETECTIONS)
    compiler = L1Compiler(
        detector=stub,
        config=L1CompilerConfig(scene_type=SceneType.ORTHOGRAPHIC),
    )
    meta = ImageMeta(image_id="demo-img-001", width=640, height=480)
    ir = compiler.compile(image=b"", meta=meta)  # StubDetector ignores image
    return ir


async def main():
    ir = build_ir()

    print("=" * 60)
    print("SVSG Demo — Evidence Anchoring vs. LLM Hallucination")
    print("=" * 60)

    # Step 1: IR summary
    print("\n[IR Summary]")
    print(f"  instances: {len(ir.detections)}")
    for d in ir.detections:
        print(f"  instance {d.instance_id}: class={d.object_class}, "
              f"bbox={d.bbox_px}, score={d.detection_score:.2f}")

    # Step 2: No anchoring — LLM hallucinates
    print("\n--- Without Anchoring (raw LLM) ---")
    print(f"  Q: How many fasteners in the image?")
    print(f"  A: {HALLUCINATED_ANSWER.final_answer}")
    print(f"     [no evidence | LLM fabricated count={HALLUCINATED_ANSWER.claims[0].value}]")

    # Step 3: SVSG anchoring check
    print("\n--- SVSG Anchor Check ---")
    verdict = run_anchor(
        ir=ir,
        answer=HALLUCINATED_ANSWER,
        executed_verifications=set(),
        report=None,
    )

    if verdict.passed:
        print("  PASS (unexpected: hallucination not caught!)")
    else:
        print(f"  BLOCKED! {len(verdict.violations)} violation(s):")
        for v in verdict.violations:
            print(f"    [{v.code.value}] {v.reason}")

    # Step 4: Retry with corrected answer
    print("\n--- Retry With Corrected Answer ---")
    print("  LLM receives corrective feedback: count should be 2, regenerating...")

    retry_verdict = run_anchor(
        ir=ir,
        answer=CORRECTED_ANSWER,
        executed_verifications=set(),
        report=None,
    )

    if retry_verdict.passed:
        print(f"  PASS after correction!")
        print(f"  A: {CORRECTED_ANSWER.final_answer}")
    else:
        print(f"  STILL FAILED: {len(retry_verdict.violations)} violation(s)")

    # Step 5: final delivery
    print("\n--- Final Delivery ---")
    print("  Delivered: count=2 (consistent with IR, evidence-anchored)")
    print("  Source: IR instance 1=screw, 2=washer")


if __name__ == "__main__":
    asyncio.run(main())
