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


def test_chat_openrouter_free_models_catalog(client):
    auth, _ = _register(client)
    res = client.get("/api/v1/chat/status", headers=auth)
    assert res.status_code == 200
    data = res.json()["data"]
    assert "models" in data
    model_ids = {m["id"] for m in data["models"]}

    expected_free_models = [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "baidu/ernie-5.5-flash-preview:free",
        "thinkingmachines/inkling-2-small:free",
        "nvidia/nemotron-3.5-lightning:free",
        "dots-studio/dots-3-note-preview:free",
        "particle-ai/particle-1:free",
        "cohere/north-mini-code:free",
        "inclusionai/ling-3.2-flash:free",
        "recursive/flint-v1.0:free",
        "perplexity/pplx-decider-v1.3:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "liquid/lfm-3-1.7b:free",
    ]
    for m in expected_free_models:
        assert m in model_ids, f"Expected free model {m} not found in catalog"


def test_chat_model_selection_and_cohere_code(client):
    auth, _ = _register(client)
    # 1. Test Nemotron 3 Ultra selection
    payload_nemotron = {
        "messages": [
            {"role": "user", "text": "Explain monocular depth estimation SILog loss"}
        ],
        "persona": "academic",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    }
    res = client.post("/api/v1/chat/message", headers=auth, json=payload_nemotron)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert "SILog" in data["text"]

    # 2. Test Cohere North Mini Code model returning PyTorch snippet
    payload_cohere = {
        "messages": [
            {"role": "user", "text": "Explain depth estimation loss with code"}
        ],
        "persona": "academic",
        "model": "cohere/north-mini-code:free",
    }
    res2 = client.post("/api/v1/chat/message", headers=auth, json=payload_cohere)
    assert res2.status_code == 200
    data2 = res2.json()["data"]
    assert data2["model"] == "cohere/north-mini-code:free"
    assert len(data2["text"]) > 50
    assert "loss" in data2["text"].lower() or "depth" in data2["text"].lower()


def test_chat_creative_3d_printing(client):
    auth, _ = _register(client)
    payload = {
        "messages": [
            {"role": "user", "text": "What are the 3D printing and slicing requirements for STL export?"}
        ],
        "persona": "creative",
        "model": "nvidia/nemotron-3.5-super-vl:free",
    }
    res = client.post("/api/v1/chat/message", headers=auth, json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    text_lower = data["text"].lower()
    assert "3d printing" in text_lower or "slicing" in text_lower or "stl" in text_lower
    assert "manifold" in text_lower or "overhang" in text_lower

