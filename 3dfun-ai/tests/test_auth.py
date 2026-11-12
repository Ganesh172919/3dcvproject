"""Tests for authentication, sessions, RBAC and account security."""
from __future__ import annotations

import uuid

from app.security import hash_password, verify_password


def _register(client, password="Hunter2Hunter2", email=None):
    email = email or f"auth-{uuid.uuid4().hex[:10]}@example.com"
    return client.post("/api/v1/auth/register", json={
        "email": email, "password": password, "display_name": "Auth Test",
        "accept_terms": True,
    }), email


# --------------------------------------------------------------------------- unit
def test_password_hashing_roundtrip():
    digest = hash_password("CorrectHorseBattery")
    assert digest.startswith("$argon2")
    assert verify_password("CorrectHorseBattery", digest)
    assert not verify_password("wrong-password", digest)


def test_password_hash_is_salted():
    assert hash_password("same-password") != hash_password("same-password")


def test_token_encode_decode_roundtrip():
    from app.security import create_access_token, decode_access_token

    token = create_access_token("usr_test", "user", "ses_test")
    claims = decode_access_token(token)
    assert claims["sub"] == "usr_test"
    assert claims["sid"] == "ses_test"


def test_totp_verification_window():
    from app.security import generate_totp_secret, totp_now, verify_totp

    secret = generate_totp_secret()
    code = totp_now(secret)
    assert len(code) == 6
    assert verify_totp(secret, code)
    assert not verify_totp(secret, "000000" if code != "000000" else "111111")


# -------------------------------------------------------------------- integration
def test_register_returns_token_and_credit_grant(client):
    res, email = _register(client)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    data = body["data"]
    assert data["user"]["email"] == email
    assert data["user"]["role"] == "user"
    assert data["access_token"]
    assert body["meta"]["requestId"].startswith("req_")


def test_registration_grants_signup_credits(client, auth):
    balance = client.get("/api/v1/credits/balance", headers=auth).json()["data"]
    assert balance["balance"] >= 60
    ledger = client.get("/api/v1/credits/transactions", headers=auth).json()["data"]
    reasons = {row["reason"] for row in ledger}
    assert "SIGNUP_GRANT" in reasons


def test_duplicate_email_is_rejected(client):
    first, email = _register(client)
    assert first.status_code == 200
    second, _ = _register(client, email=email)
    assert second.status_code == 409 or second.status_code == 400
    assert second.json()["error"]["code"] in {"EMAIL_TAKEN", "CONFLICT", "VALIDATION_ERROR"}


def test_weak_password_is_rejected(client):
    res, _ = _register(client, password="short")
    assert res.status_code == 422 or res.status_code == 400


def test_login_logout_login(client):
    _, email = _register(client)
    login = client.post("/api/v1/auth/login",
                        json={"email": email, "password": "Hunter2Hunter2"})
    assert login.status_code == 200
    token = login.json()["data"]["access_token"]
    headers = {"authorization": f"Bearer {token}"}
    assert client.get("/api/v1/auth/me", headers=headers).json()["data"]["email"] == email
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 200
    # the session is revoked server-side, so the token stops resolving to a user
    assert client.get("/api/v1/auth/me", headers=headers).json()["data"] is None


def test_login_with_bad_password_does_not_leak(client):
    _, email = _register(client)
    res = client.post("/api/v1/auth/login", json={"email": email, "password": "nope-nope-nope"})
    assert res.status_code == 401
    assert "password" in res.json()["error"]["message"].lower()


def test_unauthenticated_requests_are_401(client):
    # /auth/me is deliberately readable without a token (it returns null user so
    # the SPA can boot anonymously); everything else must be closed.
    client.cookies.clear()          # earlier tests share this client's cookie jar
    assert client.get("/api/v1/auth/me").status_code == 200
    assert client.get("/api/v1/auth/me").json()["data"] is None
    for path in ("/api/v1/generations", "/api/v1/assets", "/api/v1/credits/balance",
                 "/api/v1/uploads/", "/api/v1/admin/overview"):
        assert client.get(path).status_code == 401, path


def test_sessions_listing_and_revocation(client):
    _, email = _register(client)
    tokens = []
    for _ in range(2):
        res = client.post("/api/v1/auth/login",
                          json={"email": email, "password": "Hunter2Hunter2"})
        tokens.append(res.json()["data"]["access_token"])
    headers = {"authorization": f"Bearer {tokens[0]}"}
    sessions = client.get("/api/v1/auth/sessions", headers=headers).json()["data"]
    assert len(sessions) >= 3          # register + 2 logins
    victim = sessions[-1]
    assert client.delete(f"/api/v1/auth/sessions/{victim['id']}",
                         headers=headers).status_code == 200
    remaining = client.get("/api/v1/auth/sessions", headers=headers).json()["data"]
    assert all(s["id"] != victim["id"] for s in remaining)


def test_role_escalation_is_impossible_via_profile(client, auth):
    res = client.patch("/api/v1/auth/me", headers=auth, json={"display_name": "Renamed"})
    assert res.status_code == 200
    assert res.json()["data"]["role"] == "user"
    # an unknown field asking for admin must not be honoured
    res2 = client.patch("/api/v1/auth/me", headers=auth, json={"role": "admin"})
    assert client.get("/api/v1/auth/me", headers=auth).json()["data"]["role"] == "user"
    assert res2.status_code in (200, 422)


def test_non_admin_cannot_reach_admin_api(client, auth):
    for path in ("/api/v1/admin/overview", "/api/v1/admin/users", "/api/v1/admin/audit-logs"):
        res = client.get(path, headers=auth)
        assert res.status_code == 403, path
        assert res.json()["error"]["code"] in {"FORBIDDEN", "ADMIN_REQUIRED"}


def test_csrf_not_required_for_bearer_tokens(client, auth):
    """Bearer clients cannot be forged cross-site, so they skip the double-submit."""
    res = client.post("/api/v1/generations/estimate", headers=auth, json={"quality": "draft"})
    assert res.status_code == 200, res.text


def test_forgot_password_is_enumeration_safe(client):
    _, email = _register(client)
    known = client.post("/api/v1/auth/forgot-password", json={"email": email})
    unknown = client.post("/api/v1/auth/forgot-password",
                          json={"email": "nobody@example.com"})
    assert known.status_code == unknown.status_code == 200
    assert known.json()["data"]["message"] == unknown.json()["data"]["message"]


def test_rate_limiter_allows_then_blocks():
    from app.ratelimit import RateLimiter

    limiter = RateLimiter(capacity=3, window_seconds=60)
    results = [limiter.check("user:1")[0] for _ in range(4)]
    assert results[:3] == [True, True, True]
    assert results[3] is False
    assert limiter.retry_after("user:1") >= 1
