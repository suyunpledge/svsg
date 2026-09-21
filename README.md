# SVSG — Structured Visual Semantic Gateway

**Lets an LLM understand images without relying on VLM's fuzzy vision.**

## The Problem

The conventional approach is to hand the image to GPT-4V / Claude Vision and let it "look and describe." But VLM visual perception is unstable:

- Ask about the same image twice and the answers can contradict each other
- Counting is often wrong ("the image has 3 apples" when there are actually 5)
- Spatial relationships are described vaguely ("the cup on the left" — which one is that exactly)

## SVSG's Approach

**First translate the image into a structured IR (intermediate representation) using a detector, then feed that IR to the LLM for reasoning.**

```
Conventional:  Image → VLM (black box) → Answer (unreliable)
SVSG:          Image → YOLO detection → IR (classes/positions/counts/relations) → LLM (white-box reasoning) → Answer (auditable)
```

The LLM doesn't need to "see" the image — it only needs to understand the structured detection results. Every conclusion can be traced back to concrete detection evidence.

## Quick Start

```bash
pip install -e ".[api,llm]"    # base service
python -m svsg                 # start (default 127.0.0.1:3002)
```

```bash
# Upload an image + ask a question
curl -X POST http://127.0.0.1:3002/v1/analyze-image \
  -F "image=@photo.jpg" \
  -F "question=How many red objects are in the image?"
```

## Two Core Capabilities

### 1. Give the LLM a pair of accurate VLM "glasses"

A precise detector (YOLO) handles "seeing," and the LLM handles only "reasoning." A clear division of labor, each doing what it's best at.

### 2. Evidence-anchored review

Every LLM output is anchored back to the detection evidence in the IR. Fabricated or unsupported claims are automatically rejected.

## Architecture

```
L1 detection compilation → L1.5 declaration verification → L2 runtime (FSM) → L3 LLM orchestration → evidence anchoring
```

| Layer | Responsibility |
|---|---|
| L1 | Detector adapter (YOLO/stub) → geometric relations → IR |
| L1.5 | Visual verification (timeout isolation + silent degradation) |
| L2 | FSM state machine, intent routing, conflict resolution, evidence anchoring |
| L3 | LLM orchestration loop, dual-channel answers, anchor-triggered veto and regeneration |

## Comparison with Conventional VLM

| | Conventional VLM direct answer | SVSG |
|---|---|---|
| Counting | Often wrong | Pixel-level detection, precise |
| Spatial relations | Vague descriptions | Structured IR (left_of/right_of/contains) |
| Auditability | Can't be traced | Every conclusion anchored to detection evidence |
| Hallucination | High | Low (anchor verification auto-vetoes unsupported claims) |
| Cost | Every call hits the VLM (expensive) | Detect once, LLM only reads text (cheap) |

## Authentication

```bash
# Method 1: API Key
curl -H "X-API-Key: ***" ...

# Method 2: Bearer token (requires prior registration)
curl -H "Authorization: Bearer ***" ...
```

## Testing

```bash
python -m pytest -q    # 108 tests
```

## License

MIT
