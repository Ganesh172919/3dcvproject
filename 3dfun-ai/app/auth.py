"""FastAPI dependencies for authentication, CSRF and RBAC."""
from __future__ import annotations

import datetime as dt
from typing import Annotated

import jwt
from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import Generation, User, UserSession, get_db, utcnow
from .schemas import ApiError
from .security import decode_access_token, is_expired

ACCESS_COOKIE = "3dfun_access"
REFRESH_COOKIE = "3dfun_refresh"
CSRF_COOKIE = "3dfun_csrf"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


def _token_from_request(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get(ACCESS_COOKIE)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = _token_from_request(request)
    if not token:
        raise ApiError("UNAUTHENTICATED", "Authentication required.", status=401)
    try:
        claims = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise ApiError("TOKEN_EXPIRED", "Session expired. Refresh and retry.", status=401,
                       retryable=True)
    except jwt.PyJWTError:
        raise ApiError("INVALID_TOKEN", "The access token is invalid.", status=401)

    session = db.get(UserSession, claims.get("sid", ""))
    if session is None or session.revoked_at is not None or is_expired(session.expires_at):
        raise ApiError("SESSION_REVOKED", "This session is no longer valid.", status=401)
    user = db.get(User, claims["sub"])
    if user is None or user.deleted_at is not None or not user.is_active:
        raise ApiError("ACCOUNT_DISABLED", "This account is not available.", status=403)

    # CSRF: cookie-authenticated state-changing requests must echo the token.
    # Requests that authenticate with an explicit bearer header cannot be forged
    # cross-site (the browser never attaches that header automatically), so the
    # double-submit check only applies to ambient cookie credentials.
    bearer = request.headers.get("authorization", "").lower().startswith("bearer ")
    if request.method not in SAFE_METHODS and not bearer and request.cookies.get(ACCESS_COOKIE):
        supplied = request.headers.get("x-csrf-token") or request.headers.get("x-xsrf-token")
        if not supplied or supplied != session.csrf_token:
            raise ApiError("CSRF_FAILED", "Missing or invalid CSRF token.", status=403)

    if request.cookies.get(ACCESS_COOKIE):
        session.last_seen_at = utcnow()
        db.commit()
    return user


def get_optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    try:
        return get_current_user(request, db)
    except ApiError:
        return None


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise ApiError("FORBIDDEN", "Administrator access required.", status=403)
    return user


def require_verified(user: User = Depends(get_current_user)) -> User:
    if not user.email_verified:
        raise ApiError("EMAIL_NOT_VERIFIED", "Verify your email address first.", status=403)
    return user


def owned_generation(db: Session, generation_id: str, user: User,
                     allow_admin: bool = True) -> Generation:
    generation = db.get(Generation, generation_id)
    if generation is None:
        raise ApiError("NOT_FOUND", "Generation not found.", status=404)
    if generation.user_id != user.id and not (allow_admin and user.role == "admin"):
        # IDOR protection: never reveal existence to a non-owner
        raise ApiError("NOT_FOUND", "Generation not found.", status=404)
    return generation


CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
DB = Annotated[Session, Depends(get_db)]


def session_for_request(db: Session, request: Request) -> UserSession | None:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        return None
    try:
        claims = decode_access_token(token)
    except jwt.PyJWTError:
        return None
    return db.get(UserSession, claims.get("sid", ""))


def purge_expired_sessions(db: Session) -> int:
    now = utcnow()
    rows = db.execute(select(UserSession).where(UserSession.expires_at < now)).scalars().all()
    for row in rows:
        row.revoked_at = row.revoked_at or now
    db.commit()
    return len(rows)
