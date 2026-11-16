"""Tests for Agentic AI Chat tools and dual-persona assistant."""
from __future__ import annotations

import uuid


def _register(client):
    email = f"chat-{uuid.uuid4().hex[:10]}@example.com"
    res = client.post("/api/v1/auth/register", json={
        "email": email, "password": "Hunter2Hunter2", "display_name": "Chat Student",
        "accept_terms": True,
    })
    token = res.json()["data"]["access_token"]
    return {"authorization": f"Bearer {token}"}, email


def test_chat_status(client):
    auth, _ = _register(client)
    res = client.get("/api/v1/chat/status", headers=auth)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["configured"] is True
    assert len(data["personas"]) == 2
    persona_ids = {p["id"] for p in data["personas"]}
    assert "academic" in persona_ids
    assert "creative" in persona_ids
    assert len(data["available_tools"]) == 5


def test_chat_academic_persona_latex(client):
    auth, _ = _register(client)
    payload = {
        "messages": [
            {"role": "user", "text": "Explain marching cubes and signed distance field calculation."}
        ],
        "persona": "academic",
    }
    res = client.post("/api/v1/chat/message", headers=auth, json=payload)
    assert res.status_code == 200
    reply = res.json()["data"]["text"]
    assert "$" in reply or "marching" in reply.lower() or "sdf" in reply.lower()


def test_chat_creative_persona(client):
    auth, _ = _register(client)
    payload = {
        "messages": [
            {"role": "user", "text": "How do I optimize lighting for 3D printing and game engines?"}
        ],
        "persona": "creative",
    }
    res = client.post("/api/v1/chat/message", headers=auth, json=payload)
    assert res.status_code == 200
    reply = res.json()["data"]["text"]
    assert len(reply) > 50


def test_chat_tool_inspect_tensor(client):
    auth, _ = _register(client)
    payload = {
        "messages": [
            {"role": "user", "text": "Inspect depth tensor statistics and dimensions"}
        ],
        "persona": "academic",
    }
    res = client.post("/api/v1/chat/message", headers=auth, json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["tool_call"] is not None
    assert data["tool_call"]["action"] == "inspect_stage_tensor"
    assert "telemetry" in data["tool_call"]
    assert "512" in str(data["tool_call"]["telemetry"]["shape"])


def test_chat_nerf_vs_3dgs_comparison(client):
    auth, _ = _register(client)
    payload = {
        "messages": [
            {"role": "user", "text": "Compare NeRF vs 3D Gaussian Splatting (3DGS) inference speed"}
        ],
        "persona": "academic",
    }
    res = client.post("/api/v1/chat/message", headers=auth, json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    assert "NeRF" in data["text"] and "Gaussian" in data["text"]

