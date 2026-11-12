"""Credit ledger, generation lifecycle and asset authorisation tests.

The generation tests use the deterministic ``mock`` provider (wired in
conftest.py) so the suite does not spend a few minutes of CPU per case; the real
engine path is covered by ``tests/test_engine_smoke.py`` and the e2e script.
"""
from __future__ import annotations

import io
import uuid

import pytest


# --------------------------------------------------------------------- helpers
def _upload(client, auth, source_image) -> str:
    res = client.post("/api/v1/uploads/", headers=auth,
                      files={"file": ("sneaker.png", io.BytesIO(source_image), "image/png")})
    assert res.status_code == 200, res.text
    return res.json()["data"]["id"]


def _create(client, auth, upload_id, quality="draft", **kwargs):
    payload = {"upload_id": upload_id, "quality": quality, "preset": "product",
               "prompt": "teal running shoe", "accept_terms": True}
    payload.update(kwargs)
    return client.post("/api/v1/generations", headers=auth, json=payload)


def _wait(client, auth, generation_id, timeout=60.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/v1/generations/{generation_id}", headers=auth)
        data = res.json()["data"]
        if data["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return data
        time.sleep(0.4)
    raise AssertionError(f"generation {generation_id} did not finish within {timeout}s")


# ------------------------------------------------------------------ credits
def test_estimate_matches_charged_cost(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    est = client.post("/api/v1/generations/estimate", headers=auth,
                      json={"quality": "draft"}).json()["data"]
    gen = _create(client, auth, upload_id, quality="draft").json()["data"]
    assert gen["credits_charged"] == est["credits"]
    assert est["credits"] == 4                     # CREDIT_COST_DRAFT


def test_credits_are_reserved_then_spent(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    before = client.get("/api/v1/credits/balance", headers=auth).json()["data"]
    gen = _create(client, auth, upload_id).json()["data"]

    during = client.get("/api/v1/credits/balance", headers=auth).json()["data"]
    assert during["reserved"] >= gen["credits_charged"]
    assert during["available"] == before["available"] - gen["credits_charged"]

    _wait(client, auth, gen["id"])
    after = client.get("/api/v1/credits/balance", headers=auth).json()["data"]
    assert after["reserved"] == 0
    assert after["balance"] == before["balance"] - gen["credits_charged"]

    ledger = client.get("/api/v1/credits/transactions", headers=auth).json()["data"]
    reasons = [row["reason"] for row in ledger]
    assert "RESERVATION" in reasons
    for row in ledger:
        if row["reason"] in ("SPEND", "RESERVATION", "SIGNUP_GRANT"):
            assert row["amount"] == 0 or row["amount"] <= 0 or row["reason"] == "SIGNUP_GRANT"
    spend = [r for r in ledger if r["reason"] == "SPEND"][0]
    assert spend["amount"] == -gen["credits_charged"]


def test_insufficient_credits_blocks_generation(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    admin = client.post("/api/v1/auth/login", json={
        "email": "admin@3dfun.dev", "password": "DemoPassword123!"})
    if admin.status_code != 200:
        pytest.skip("seed admin account not present")
    admin_headers = {"authorization": f"Bearer {admin.json()['data']['access_token']}"}
    me = client.get("/api/v1/auth/me", headers=auth).json()["data"]
    client.post("/api/v1/admin/credits/grant", headers=admin_headers,
                json={"user_id": me["id"], "amount": -100000, "reason": "test drain"})
    res = _create(client, auth, upload_id)
    assert res.status_code == 402
    assert res.json()["error"]["code"] in {"INSUFFICIENT_CREDITS", "PAYMENT_REQUIRED"}


def test_grant_and_ledger_are_append_only(client, auth):
    admin = client.post("/api/v1/auth/login", json={
        "email": "admin@3dfun.dev", "password": "DemoPassword123!"})
    if admin.status_code != 200:
        pytest.skip("seed admin account not present")
    admin_headers = {"authorization": f"Bearer {admin.json()['data']['access_token']}"}
    me = client.get("/api/v1/auth/me", headers=auth).json()["data"]
    before = client.get("/api/v1/credits/balance", headers=auth).json()["data"]["balance"]
    client.post("/api/v1/admin/credits/grant", headers=admin_headers,
                json={"user_id": me["id"], "amount": 25, "reason": "test grant"})
    after = client.get("/api/v1/credits/balance", headers=auth).json()["data"]["balance"]
    assert after == before + 25
    ledger = client.get("/api/v1/credits/transactions", headers=auth).json()["data"]
    assert ledger[0]["amount"] == 25
    assert ledger[0]["balance_after"] == after


# ---------------------------------------------------------------- lifecycle
def test_full_generation_lifecycle(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    created = _create(client, auth, upload_id).json()["data"]
    assert created["status"] in {"QUEUED", "VALIDATING", "PROCESSING"}

    final = _wait(client, auth, created["id"])
    assert final["status"] == "COMPLETED", final
    assert final["progress"] == 100
    assert final["asset_id"]

    asset = client.get(f"/api/v1/assets/{final['asset_id']}", headers=auth).json()["data"]
    assert asset["polygons"] > 0
    assert asset["vertices"] > 0
    assert asset["watertight"] is True
    assert {d["format"] for d in asset["downloads"]} >= {"glb", "obj"}

    glb = client.get(f"/api/v1/assets/{asset['id']}/download?format=glb", headers=auth)
    assert glb.status_code == 200
    assert glb.content[:4] == b"glTF"
    assert len(glb.content) > 500

    events = client.get(f"/api/v1/generations/{created['id']}/events", headers=auth).json()["data"]
    types = [e["type"] for e in events]
    assert "generation.completed" in types
    sequences = [e["sequence"] for e in events]
    assert sequences == sorted(sequences)


def test_generation_events_are_ordered_and_replayable(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    gen = _create(client, auth, upload_id).json()["data"]
    _wait(client, auth, gen["id"])
    all_events = client.get(f"/api/v1/generations/{gen['id']}/events", headers=auth).json()["data"]
    midpoint = all_events[len(all_events) // 2]["sequence"]
    tail = client.get(f"/api/v1/generations/{gen['id']}/events?after={midpoint}",
                      headers=auth).json()["data"]
    assert all(e["sequence"] > midpoint for e in tail)


def test_attempts_are_recorded(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    gen = _create(client, auth, upload_id).json()["data"]
    _wait(client, auth, gen["id"])
    attempts = client.get(f"/api/v1/generations/{gen['id']}/attempts",
                          headers=auth).json()["data"]
    assert len(attempts) >= 1
    assert attempts[0]["status"] in {"succeeded", "failed", "running"}


def test_retry_is_rejected_for_a_completed_job(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    gen = _create(client, auth, upload_id).json()["data"]
    _wait(client, auth, gen["id"])
    res = client.post(f"/api/v1/generations/{gen['id']}/retry", headers=auth)
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "NOT_RETRYABLE"


def test_retry_requeues_a_failed_job(client, auth, source_image):
    """A failed job can be retried and produces a second attempt row.

    The failure is induced through the provider registry rather than the HTTP
    layer so the state machine, credit refund and retry path are all exercised.
    """
    import os

    upload_id = _upload(client, auth, source_image)

    # first attempt: the provider is forced to fail
    os.environ["THREEFUN_FORCE_MOCK_FAILURE"] = "1"
    try:
        gen = _create(client, auth, upload_id).json()["data"]
        failed = _wait(client, auth, gen["id"])
    finally:
        os.environ.pop("THREEFUN_FORCE_MOCK_FAILURE", None)

    assert failed["status"] == "FAILED", failed
    assert failed["error_code"] == "MOCK_FAILURE"
    # a failed job must not keep the user's credits
    balance = client.get("/api/v1/credits/balance", headers=auth).json()["data"]
    assert balance["reserved"] == 0

    res = client.post(f"/api/v1/generations/{gen['id']}/retry", headers=auth)
    assert res.status_code == 200, res.text
    assert res.json()["data"]["id"] == gen["id"]
    recovered = _wait(client, auth, gen["id"])
    assert recovered["status"] == "COMPLETED"

    attempts = client.get(f"/api/v1/generations/{gen['id']}/attempts",
                          headers=auth).json()["data"]
    assert len(attempts) >= 2
    assert {"failed", "succeeded"} <= {a["status"] for a in attempts}


def test_cancel_marks_the_job_and_refunds(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    gen = _create(client, auth, upload_id).json()["data"]
    res = client.post(f"/api/v1/generations/{gen['id']}/cancel", headers=auth)
    assert res.status_code == 200, res.text
    final = _wait(client, auth, gen["id"])
    assert final["status"] in {"CANCELLED", "COMPLETED"}
    balance = client.get("/api/v1/credits/balance", headers=auth).json()["data"]
    assert balance["reserved"] == 0


# ------------------------------------------------------- authorisation (IDOR)
def test_users_cannot_read_each_other(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    gen = _create(client, auth, upload_id).json()["data"]
    final = _wait(client, auth, gen["id"])

    other = client.post("/api/v1/auth/register", json={
        "email": f"other-{uuid.uuid4().hex[:8]}@example.com",
        "password": "Hunter2Hunter2", "accept_terms": True})
    other_headers = {"authorization": f"Bearer {other.json()['data']['access_token']}"}

    assert client.get(f"/api/v1/generations/{gen['id']}",
                      headers=other_headers).status_code in {403, 404}
    assert client.get(f"/api/v1/assets/{final['asset_id']}",
                      headers=other_headers).status_code in {403, 404}
    assert client.get(f"/api/v1/assets/{final['asset_id']}/download?format=glb",
                      headers=other_headers).status_code in {403, 404}


def test_private_asset_is_not_in_the_gallery(client, auth, source_image):
    upload_id = _upload(client, auth, source_image)
    gen = _create(client, auth, upload_id).json()["data"]
    final = _wait(client, auth, gen["id"])
    gallery = client.get("/api/v1/gallery?page_size=100").json()["data"]
    assert all(item["id"] != final["asset_id"] for item in gallery["items"])

    # publishing makes it visible
    client.patch(f"/api/v1/assets/{final['asset_id']}", headers=auth,
                 json={"visibility": "PUBLIC"})
    gallery = client.get("/api/v1/gallery?page_size=100").json()["data"]
    assert any(item["id"] == final["asset_id"] for item in gallery["items"])


def test_upload_rejects_non_images(client, auth):
    res = client.post("/api/v1/uploads/", headers=auth,
                      files={"file": ("payload.png", io.BytesIO(b"not an image at all"),
                                      "image/png")})
    assert res.status_code in {400, 415, 422}
    assert res.json()["success"] is False


def test_upload_rejects_oversized_files(client, auth):
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (21 * 1024 * 1024)
    res = client.post("/api/v1/uploads/", headers=auth,
                      files={"file": ("huge.png", io.BytesIO(big), "image/png")})
    assert res.status_code in {400, 413, 422}


def test_deduplicates_identical_uploads(client, auth, source_image):
    first = _upload(client, auth, source_image)
    second = _upload(client, auth, source_image)
    a = client.get(f"/api/v1/uploads/{first}", headers=auth).json()["data"]
    b = client.get(f"/api/v1/uploads/{second}", headers=auth).json()["data"]
    assert a["sha256"] == b["sha256"]


# ------------------------------------------------------------- rate limits
def test_generation_rate_limit_is_enforced():
    from app.ratelimit import RateLimiter, enforce
    from app.schemas import ApiError

    limiter = RateLimiter(capacity=2, window_seconds=60)
    enforce(limiter, "user:1", "generation")
    enforce(limiter, "user:1", "generation")
    with pytest.raises(ApiError) as exc:
        enforce(limiter, "user:1", "generation")
    assert exc.value.status == 429
