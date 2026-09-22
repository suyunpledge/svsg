"""Regression cases for evidence boundaries, backend failures and auth races."""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from test_evidence_anchor import _answer, _ir, _report
from test_orchestrator import _orch, _payload

from svsg import auth
from svsg.api import create_app
from svsg.config import Settings
from svsg.contracts import MetricType, SceneType, SVSGError, VerificationResult
from svsg.l1_compiler import ImageMeta, L1Compiler, RawDetection, StubDetector
from svsg.l1_compiler.geometry_engine import build_nearest_relations
from svsg.l2_runtime import anchor_claims
from svsg.l15_verifier import L15Verifier


@pytest.mark.parametrize("claim", [
    {"field": "class", "value": "invented"},
    {"field": "exists", "value": True},
    {"instance_id": 1, "field": "exists", "value": "true"},
    {"instance_id": 1, "field": "exists", "value": 1},
    {"field": "count:washer", "value": True},
    {"field": "count", "value": 3.0},
    {"field": "count:", "value": 0},
    {"instance_id": 1, "field": "relation:left_of", "value": "2"},
    {"instance_id": 1, "field": "relation:contains", "value": True},
    {"instance_id": 1, "field": "relation:contains", "value": 1},
    {"instance_id": 1, "field": "relation:invented", "value": 2},
    {"instance_id": 1, "field": "unrecognized", "value": "anything"},
])
def test_malformed_claims_fail_closed(claim):
    assert anchor_claims(_ir(), _answer([claim]))


def test_nearest_claim_checks_all_instances_and_scene_gate():
    ir = _ir()
    nearest = _answer([{"instance_id": 2, "field": "relation:nearest_to", "value": 3}])
    assert anchor_claims(ir, nearest) == []
    wrong = _answer([{"instance_id": 2, "field": "relation:nearest_to", "value": 1}])
    assert anchor_claims(ir, wrong)
    ir.scene_type = SceneType.UNKNOWN
    assert anchor_claims(ir, nearest)


def test_distance_claim_without_numeric_evidence_is_rejected_not_crashed():
    answer = _answer([{"instance_id": 1, "field": "relation:distance_to", "value": 2}])
    assert anchor_claims(_ir(), answer)


def test_other_image_report_is_not_accepted_as_evidence():
    report = _report([{"instance_id": 1, "attributes": {"color": "silver"}}])
    report.image_id = "another_image"
    answer = _answer([{"instance_id": 1, "field": "attribute:color", "value": "silver"}])
    assert anchor_claims(_ir(), answer, report)


@pytest.mark.parametrize("instance_id", [True, "1", 1.0])
def test_claim_instance_ids_are_not_coerced(instance_id):
    with pytest.raises(ValidationError):
        _answer([{"instance_id": instance_id, "field": "class", "value": "screw"}])


@pytest.mark.parametrize("claim", [
    {"field": "count", "value": 3},
    {"field": "count:screw", "value": 2},
    {"instance_id": 2, "field": "class", "value": "screw"},
    {"instance_id": 1, "field": "relation:left_of", "value": 2},
])
def test_negative_existence_cannot_be_bypassed_by_other_fields(claim):
    report = _report([{"instance_id": 2, "exists": False}])
    assert anchor_claims(_ir(), _answer([claim]), report)


def test_class_count_cannot_bypass_ambiguous_verification():
    report = _report([{"instance_id": 2, "class": "ambiguous"}])
    assert anchor_claims(_ir(), _answer([{"field": "count:screw", "value": 2}]), report)
    # Ambiguous class does not contradict the total number of detected objects.
    assert anchor_claims(_ir(), _answer([{"field": "count", "value": 3}]), report) == []


def test_malformed_claim_uses_regeneration_path():
    from svsg.l15_verifier import StubVerifierBackend

    bad = _answer([{"field": "unknown", "value": "invented"}])
    good = _answer([{"field": "count", "value": 3}])
    orch, llm = _orch(StubVerifierBackend(), [bad, good])
    result = asyncio.run(orch.run("q", _payload(verify2=False)))
    assert result.delivered
    assert "corrective_feedback" in llm.calls[1]["context"]


def test_valid_password_hash_roundtrip():
    stored = auth.hash_password("密码-pass", 1000)
    assert auth.verify_password("密码-pass", stored)
    assert not auth.verify_password("incorrect", stored)


class ReturnedBackend:
    def __init__(self, result):
        self.result = result

    async def verify_batch(self, image_id, instance_ids):
        return self.result


@pytest.mark.parametrize("output", [
    None,
    [{"instance_id": "bad"}],
    [VerificationResult(instance_id=99)],
    [VerificationResult(instance_id=1), VerificationResult(instance_id=1)],
])
def test_invalid_verifier_output_degrades_without_escaping(output):
    report = asyncio.run(L15Verifier(ReturnedBackend(output)).verify_instances("image", [1]))
    assert report.status.value == "error"
    assert report.results == []


@pytest.mark.parametrize("bbox", [(-1, 0, 10, 10), (0, 0, 0, 10)])
def test_invalid_detector_bbox_maps_to_s2(bbox):
    compiler = L1Compiler(StubDetector([RawDetection("part", bbox, 0.9)]))
    with pytest.raises(SVSGError) as caught:
        compiler.compile(b"image", ImageMeta("image", 100, 100))
    assert caught.value.code.value == "E1002"


def test_invalid_detector_output_api_returns_business_rejection():
    compiler = L1Compiler(StubDetector([RawDetection("part", (0, 0, 0, 10), 0.9)]))
    client = TestClient(create_app(compiler=compiler, settings=Settings(_env_file=None)))
    image = io.BytesIO()
    Image.new("RGB", (20, 20)).save(image, "PNG")
    response = client.post("/v1/analyze-image", data={"query": "q"},
                           files={"file": ("image.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["error"]["code"] == "E1002"


def test_intrinsics_do_not_convert_pixel_distances_to_physical_units():
    relations = build_nearest_relations([1, 2], [(0, 0, 10, 10), (20, 0, 30, 10)],
                                       scene_type=SceneType.PERSPECTIVE, has_intrinsics=True)
    assert relations[1][0].metric_type == MetricType.IMAGE_PLANE_PX


@pytest.mark.parametrize("scene", ["orthographic", "perspective"])
def test_llm_is_not_told_pixel_only_ir_has_physical_calibration(scene):
    from svsg.l15_verifier import StubVerifierBackend

    orch, llm = _orch(StubVerifierBackend(), [_answer([{"field": "count", "value": 3}])])
    payload = _payload(verify2=False)
    payload["scene_type"] = scene
    payload["camera_intrinsics"] = {"fx": 800, "fy": 800, "cx": 960, "cy": 540}
    assert asyncio.run(orch.run("distance?", payload)).delivered
    assert "严禁将像素距离表述为物理距离" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize("stored", [
    "pbkdf2_sha256$bad$00$00", "pbkdf2_sha256$1000$not-hex$00",
    "pbkdf2_sha256$0$00$00", "pbkdf2_sha256$1000$00$非ASCII",
])
def test_malformed_password_hash_fails_closed(stored):
    assert auth.verify_password("password", stored) is False


def test_concurrent_first_registrations_create_only_one_admin(tmp_path, monkeypatch):
    store = auth.AuthStore(str(tmp_path / "auth.db"))
    barrier = Barrier(2)
    original = auth.hash_password

    def synchronized_hash(password, iters):
        barrier.wait(timeout=5)
        return original(password, iters)

    monkeypatch.setattr(auth, "hash_password", synchronized_hash)
    with ThreadPoolExecutor(max_workers=2) as pool:
        users = list(pool.map(
            lambda name: store.create_user(name, "secret123", iters=1000),
            ["alice", "bobby"],
        ))
    assert sum(user["is_admin"] for user in users) == 1
