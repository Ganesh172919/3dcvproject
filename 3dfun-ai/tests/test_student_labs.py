"""Tests for student curriculum labs, billing wallet, and coupons."""
from __future__ import annotations

import uuid


def _register(client):
    email = f"student-{uuid.uuid4().hex[:10]}@example.com"
    res = client.post("/api/v1/auth/register", json={
        "email": email, "password": "Hunter2Hunter2", "display_name": "Student Tester",
        "accept_terms": True,
    })
    token = res.json()["data"]["access_token"]
    return {"authorization": f"Bearer {token}"}, email


def test_student_labs_catalog(client):
    res = client.get("/api/v1/student-labs")
    assert res.status_code == 200
    data = res.json()["data"]
    assert "labs" in data
    assert len(data["labs"]) == 4
    lab_ids = {lab["id"] for lab in data["labs"]}
    assert "lab1_depth" in lab_ids
    assert "lab2_occupancy" in lab_ids
    assert "lab3_3dgs" in lab_ids
    assert "lab4_evaluator" in lab_ids


def test_billing_wallet_and_coupon(client):
    auth, _ = _register(client)
    res = client.get("/api/v1/billing/wallet", headers=auth)
    assert res.status_code == 200
    wallet = res.json()["data"]
    assert wallet["balance"] >= 50
    assert wallet["tier"] == "educational"

    # Redeem coupon
    c_res = client.post("/api/v1/billing/redeem-coupon", headers=auth, json={"code": "STUDENT3D"})
    assert c_res.status_code == 200
    c_data = c_res.json()["data"]
    assert c_data["credits_added"] == 20
    assert c_data["new_balance"] == wallet["balance"] + 20

    # Duplicate coupon is rejected
    dup_res = client.post("/api/v1/billing/redeem-coupon", headers=auth, json={"code": "STUDENT3D"})
    assert dup_res.status_code == 409


def test_student_lab_execution(client):
    auth, _ = _register(client)
    # Launch lab4_evaluator (fast synthetic evaluation)
    res = client.post("/api/v1/student-labs/lab4_evaluator/run", headers=auth, json={"config": {}})
    assert res.status_code == 200
    body = res.json()["data"]
    run_id = body["run_id"]
    assert run_id.startswith("lbr_")

    # List runs
    runs_res = client.get("/api/v1/student-labs/runs", headers=auth)
    assert runs_res.status_code == 200
    runs = runs_res.json()["data"]
    assert any(r["id"] == run_id for r in runs)

    # Get single run
    run_res = client.get(f"/api/v1/student-labs/runs/{run_id}", headers=auth)
    assert run_res.status_code == 200
    assert run_res.json()["data"]["id"] == run_id
