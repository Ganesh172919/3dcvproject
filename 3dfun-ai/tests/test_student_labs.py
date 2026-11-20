"""Tests for student curriculum labs, billing wallet, and coupons."""
from __future__ import annotations

import time
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
    assert len(data["labs"]) == 6
    lab_ids = {lab["id"] for lab in data["labs"]}
    assert "lab1_depth" in lab_ids
    assert "lab2_occupancy" in lab_ids
    assert "lab3_3dgs" in lab_ids
    assert "lab4_evaluator" in lab_ids
    assert "lab5_triplane" in lab_ids
    assert "lab6_pbr" in lab_ids


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
    before = client.get("/api/v1/billing/wallet", headers=auth).json()["data"]["available"]
    res = client.post("/api/v1/student-labs/lab4_evaluator/run", headers=auth,
                      json={"config": {"num_points": 128, "seed": 17}})
    assert res.status_code == 200
    body = res.json()["data"]
    run_id = body["run_id"]
    assert run_id.startswith("lbr_")
    assert body["balance"] == before - body["credits_charged"]

    # List runs
    runs_res = client.get("/api/v1/student-labs/runs", headers=auth)
    assert runs_res.status_code == 200
    runs = runs_res.json()["data"]
    assert any(r["id"] == run_id for r in runs)

    for _ in range(100):
        run_res = client.get(f"/api/v1/student-labs/runs/{run_id}", headers=auth)
        run = run_res.json()["data"]
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert run["id"] == run_id and run["status"] == "completed", run
    assert run["config"]["num_points"] == 128
    assert run["artifacts"] == {"results_json": "lab4_results.json"}
    artifact_url = f"/api/v1/student-labs/runs/{run_id}/artifacts/lab4_results.json"
    report = client.get(artifact_url, headers=auth)
    assert report.status_code == 200
    assert report.json()["point_counts"] == {"prediction": 128, "reference": 128}
    other, _ = _register(client)
    assert client.get(artifact_url, headers=other).status_code == 404


def test_student_lab6_pbr_execution(client):
    auth, _ = _register(client)
    before = client.get("/api/v1/billing/wallet", headers=auth).json()["data"]["available"]
    res = client.post(
        "/api/v1/student-labs/lab6_pbr/run",
        headers=auth,
        json={"config": {"filter_radius": 3, "roughness_base": 0.4, "metallic_factor": 0.1}},
    )
    assert res.status_code == 200
    body = res.json()["data"]
    run_id = body["run_id"]
    assert body["credits_charged"] == 5
    assert body["balance"] == before - 5

    for _ in range(100):
        run_res = client.get(f"/api/v1/student-labs/runs/{run_id}", headers=auth)
        run = run_res.json()["data"]
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert run["status"] == "completed", run
    assert "mean_albedo_luminance" in run["metrics"]
    assert "albedo" in run["artifacts"]


def test_student_lab5_triplane_execution(client):
    auth, _ = _register(client)
    before = client.get("/api/v1/billing/wallet", headers=auth).json()["data"]["available"]
    res = client.post(
        "/api/v1/student-labs/lab5_triplane/run",
        headers=auth,
        json={"config": {"primitive": "cube", "epochs": 1, "lr": 0.01, "plane_res": 16}},
    )
    assert res.status_code == 200
    body = res.json()["data"]
    run_id = body["run_id"]
    assert body["credits_charged"] == 10
    assert body["balance"] == before - 10

    for _ in range(400):
        run_res = client.get(f"/api/v1/student-labs/runs/{run_id}", headers=auth)
        run = run_res.json()["data"]
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert run["status"] == "completed", run
    assert "accuracy" in run["metrics"]
    assert "mesh" in run["artifacts"]


def test_invalid_lab_config_keeps_balance(client):
    auth, _ = _register(client)
    wallet = "/api/v1/billing/wallet"
    before = client.get(wallet, headers=auth).json()["data"]["available"]
    response = client.post("/api/v1/student-labs/lab3_3dgs/run", headers=auth,
                           json={"config": {"num_points": 3001}})
    assert response.status_code == 422
    assert client.get(wallet, headers=auth).json()["data"]["available"] == before
