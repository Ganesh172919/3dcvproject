"""Security primitives: password hashing, JWT, TOTP, HMAC helpers."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
import secrets
import struct
import time
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from .config import settings
from .db import utcnow

_ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16)


# ---------------------------------------------------------------------------
# passwords
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, stored: str) -> bool:
    try:
        return _ph.verify(stored, password)
    except (VerifyMismatchError, InvalidHashError, Exception):
        return False


def needs_rehash(stored: str) -> bool:
    try:
        return _ph.check_needs_rehash(stored)
    except Exception:
        return False


def password_strength(password: str) -> list[str]:
    problems = []
    if len(password) < 10:
        problems.append("Use at least 10 characters.")
    if not any(c.islower() for c in password):
        problems.append("Add a lowercase letter.")
    if not any(c.isupper() for c in password):
        problems.append("Add an uppercase letter.")
    if not any(c.isdigit() for c in password):
        problems.append("Add a digit.")
    if password.lower() in {"password12", "qwertyuiop", "letmein123", "3dfun12345"}:
        problems.append("This password is too common.")
    return problems


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
def create_access_token(user_id: str, role: str, session_id: str, ttl: int | None = None) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "sid": session_id,
        "type": "access",
        "iat": now,
        "nbf": now,
        "exp": now + (ttl or settings.access_token_ttl),
        "iss": "3dfun-ai",
    }
    return jwt.encode(payload, settings.jwt_access_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_access_secret, algorithms=["HS256"],
                      options={"require": ["exp", "sub", "sid"]})


def create_refresh_token() -> tuple[str, str]:
    """Returns (plaintext, sha256 hash). Only the hash is stored."""
    raw = secrets.token_urlsafe(48)
    return raw, sha256_hex(raw)


def sha256_hex(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def random_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def sign_payload(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def verify_signature(payload: str, signature: str, secret: str) -> bool:
    return hmac.compare_digest(sign_payload(payload, secret), signature)


# ---------------------------------------------------------------------------
# TOTP (RFC 6238, SHA-1, 30 s, 6 digits)
# ---------------------------------------------------------------------------
def generate_totp_secret() -> str:
    return base64.b32encode(os.urandom(20)).decode().rstrip("=")


def _totp_at(secret: str, counter: int, digits: int = 6) -> str:
    padded = secret + "=" * ((8 - len(secret) % 8) % 8)
    key = base64.b32decode(padded.upper())
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10 ** digits)).zfill(digits)


def totp_now(secret: str, step: int = 30, digits: int = 6) -> str:
    return _totp_at(secret, int(time.time() // step), digits)


def verify_totp(secret: str, code: str, window: int = 1, step: int = 30) -> bool:
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit():
        return False
    counter = int(time.time() // step)
    for offset in range(-window, window + 1):
        if constant_time_equals(_totp_at(secret, counter + offset), code):
            return True
    return False


def totp_provisioning_uri(secret: str, email: str, issuer: str = "3DFUN AI") -> str:
    from urllib.parse import quote
    label = quote(f"{issuer}:{email}")
    return (f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer)}"
            f"&algorithm=SHA1&digits=6&period=30")


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def utc_from_timestamp(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)


def is_expired(when: dt.datetime | None) -> bool:
    if when is None:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when < utcnow()
