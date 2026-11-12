"""Shared pytest fixtures.

Every test run gets an isolated temporary database and storage root, so the
suite never touches ``data/3dfun.db`` or the developer's stored assets.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _isolate() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="3dfun-tests-"))
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp / 'test.db'}"
    os.environ["STORAGE_ROOT"] = str(tmp / "storage")
    os.environ["ENVIRONMENT"] = "test"
    os.environ["EMAIL_TRANSPORT"] = "console"
    os.environ["GENERATION_PROVIDER"] = "mock"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"
    os.environ["RATE_LIMIT_WINDOW"] = "1"
    os.environ["AUTH_RATE_LIMIT_PER_MINUTE"] = "10000"
    os.environ["CREDITS_SIGNUP_GRANT"] = "60"
    return tmp


TMP_ROOT = _isolate()


@pytest.fixture(scope="session")
def app():
    from app.main import create_app

    application = create_app()
    from app.db import SessionLocal, init_db

    init_db()

    # Providers, feature flags and a believable admin account, exactly as a real
    # deployment gets them from `python -m app.seed`.
    from app import seed

    with SessionLocal() as db:
        seed.seed_providers(db)
        seed.seed_flags(db)
        seed._ensure_user(db, seed.ADMIN_EMAIL, "admin", 1000)

    # The suite drives many short bursts from one client IP; relax the buckets so
    # rate limiting is tested deliberately (tests/test_ratelimit.py), not by accident.
    from app import ratelimit

    ratelimit.auth_limiter.capacity = ratelimit.auth_limiter.refill_rate = 100000.0
    ratelimit.upload_limiter.capacity = ratelimit.upload_limiter.refill_rate = 100000.0
    ratelimit.generation_limiter.capacity = ratelimit.generation_limiter.refill_rate = 100000.0
    ratelimit.limiter.capacity = ratelimit.limiter.refill_rate = 100000.0
    return application


@pytest.fixture(scope="session")
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def source_image() -> bytes:
    path = ROOT / "engine" / "testdata" / "sneaker.png"
    if path.exists():
        return path.read_bytes()
    # minimal valid PNG (1x1) so upload tests still run without the fixture
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAF"
        "AAH9Cf8AAAAASUVORK5CYII="
    )


@pytest.fixture
def user_token(client) -> str:
    import uuid

    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    res = client.post("/api/v1/auth/register", json={
        "email": email, "password": "Hunter2Hunter2", "display_name": "Test User",
        "accept_terms": True,
    })
    assert res.status_code == 200, res.text
    return res.json()["data"]["access_token"]


@pytest.fixture
def auth(user_token):
    return {"authorization": f"Bearer {user_token}"}
