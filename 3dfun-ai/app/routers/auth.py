"""Authentication, session management and account security endpoints."""
from __future__ import annotations

import datetime as dt
import secrets

from fastapi import APIRouter, Request, Response
from sqlalchemy import select

from .. import credits, notifications
from ..auth import (ACCESS_COOKIE, CSRF_COOKIE, REFRESH_COOKIE, client_ip,
                    get_current_user, get_optional_user, purge_expired_sessions)
from ..config import settings
from ..db import (AuditLog, User, UserSession, VerificationToken, get_db, utcnow)
from ..ratelimit import auth_limiter, enforce, login_locked
from ..schemas import (ApiError, LoginRequest, RegisterRequest, UpdateProfileRequest,
                       UserPublic, ok)
from ..security import (create_access_token, create_refresh_token, hash_password,
                        is_expired, password_strength, random_token, sha256_hex,
                        totp_now, totp_provisioning_uri, verify_password, verify_totp,
                        generate_totp_secret)
from ..serializers import user_public
from fastapi import Depends
from sqlalchemy.orm import Session

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_FAILED_LOGINS = settings.login_max_attempts
LOCK_MINUTES = 15


def _set_auth_cookies(response: Response, access: str, refresh: str, csrf: str) -> None:
    common = dict(httponly=True, secure=settings.cookie_secure, samesite="lax",
                  path="/")
    response.set_cookie(ACCESS_COOKIE, access, max_age=settings.access_token_ttl, **common)
    response.set_cookie(REFRESH_COOKIE, refresh, max_age=settings.refresh_token_ttl, **common)
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, secure=settings.cookie_secure,
                        samesite="lax", max_age=settings.refresh_token_ttl, path="/")


def _clear_auth_cookies(response: Response) -> None:
    for name in (ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE):
        response.delete_cookie(name, path="/")


def _start_session(db: Session, user: User, request: Request) -> tuple[str, str, str]:
    raw_refresh, refresh_hash = create_refresh_token()
    session = UserSession(
        user_id=user.id,
        refresh_token_hash=refresh_hash,
        csrf_token=random_token(24),
        user_agent=request.headers.get("user-agent", "")[:255],
        ip=client_ip(request),
        expires_at=utcnow() + dt.timedelta(seconds=settings.refresh_token_ttl),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    access = create_access_token(user.id, user.role, session.id)
    return access, raw_refresh, session.csrf_token


@router.post("/register")
def register(payload: RegisterRequest, request: Request, response: Response,
             db: Session = Depends(get_db)):
    enforce(auth_limiter, client_ip(request), "registration")
    if not settings.allow_registration:
        raise ApiError("REGISTRATION_CLOSED", "Registration is disabled.", status=403)
    problems = password_strength(payload.password)
    if problems:
        raise ApiError("WEAK_PASSWORD", "Choose a stronger password.", status=422,
                       details=[{"issue": p} for p in problems])
    email = payload.email.lower()
    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise ApiError("EMAIL_TAKEN", "An account with that email already exists.", status=409)

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name or email.split("@")[0][:40],
        email_verified=False,
        role="user",
        terms_accepted_at=utcnow() if payload.accept_terms else None,
        privacy_accepted_at=utcnow() if payload.accept_terms else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    credits.grant(db, user.id, settings.signup_credits, idempotency_key=f"signup:{user.id}")
    token_raw = random_token(24)
    db.add(VerificationToken(user_id=user.id, kind="email_verify",
                             token_hash=sha256_hex(token_raw),
                             expires_at=utcnow() + dt.timedelta(hours=24)))
    db.add(AuditLog(actor_id=user.id, action="auth.register", target_type="user",
                    target_id=user.id, ip=client_ip(request)))
    db.commit()

    notifications.send_verify_email(user, token_raw)
    notifications.send_welcome(user, settings.signup_credits)
    access, refresh, csrf = _start_session(db, user, request)
    _set_auth_cookies(response, access, refresh, csrf)
    wallet = credits.get_wallet(db, user.id)
    return ok({
        "user": user_public(user, wallet.balance),
        "access_token": access,
        "expires_in": settings.access_token_ttl,
        "csrf_token": csrf,
        # In local development the verification link is returned so the flow is
        # testable without a mail server; never enabled in production.
        "verification_token": None if settings.is_production else token_raw,
    })


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response,
          db: Session = Depends(get_db)):
    ip = client_ip(request)
    enforce(auth_limiter, ip, "login")
    user = db.execute(select(User).where(User.email == payload.email.lower())).scalar_one_or_none()
    if user is None:
        raise ApiError("INVALID_CREDENTIALS", "Email or password is incorrect.", status=401)
    if login_locked(user):
        raise ApiError("ACCOUNT_LOCKED",
                       "Too many failed attempts. Try again later or reset your password.",
                       status=423, retryable=True)
    if not verify_password(payload.password, user.password_hash):
        user.failed_logins += 1
        if user.failed_logins >= MAX_FAILED_LOGINS:
            user.locked_until = utcnow() + dt.timedelta(minutes=LOCK_MINUTES)
            user.failed_logins = 0
            db.add(AuditLog(actor_id=user.id, action="auth.lockout", ip=ip))
        db.commit()
        raise ApiError("INVALID_CREDENTIALS", "Email or password is incorrect.", status=401)
    if user.deleted_at is not None or not user.is_active:
        raise ApiError("ACCOUNT_DISABLED", "This account is not available.", status=403)
    if user.mfa_enabled:
        if not payload.mfa_code:
            raise ApiError("MFA_REQUIRED", "Enter your authenticator code.", status=401,
                           details=[{"field": "mfa_code"}])
        if not verify_totp(user.mfa_secret or "", payload.mfa_code):
            raise ApiError("MFA_INVALID", "That authenticator code is not valid.", status=401)

    user.failed_logins = 0
    user.locked_until = None
    user.last_login_at = utcnow()
    db.add(AuditLog(actor_id=user.id, action="auth.login", ip=ip))
    db.commit()
    access, refresh, csrf = _start_session(db, user, request)
    _set_auth_cookies(response, access, refresh, csrf)
    wallet = credits.get_wallet(db, user.id)
    return ok({"user": user_public(user, wallet.balance), "access_token": access,
               "expires_in": settings.access_token_ttl, "csrf_token": csrf})


@router.post("/refresh")
def refresh_session(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get(REFRESH_COOKIE)
    if not raw:
        raise ApiError("NO_REFRESH_TOKEN", "No refresh token present.", status=401)
    token_hash = sha256_hex(raw)
    session = db.execute(select(UserSession).where(
        UserSession.refresh_token_hash == token_hash)).scalar_one_or_none()
    if session is None or session.revoked_at is not None or is_expired(session.expires_at):
        _clear_auth_cookies(response)
        raise ApiError("SESSION_REVOKED", "Please sign in again.", status=401)
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise ApiError("ACCOUNT_DISABLED", "This account is not available.", status=403)
    # rotating refresh tokens: the old one is invalidated immediately
    new_raw, new_hash = create_refresh_token()
    session.refresh_token_hash = new_hash
    session.csrf_token = random_token(24)
    session.last_seen_at = utcnow()
    session.expires_at = utcnow() + dt.timedelta(seconds=settings.refresh_token_ttl)
    db.commit()
    access = create_access_token(user.id, user.role, session.id)
    _set_auth_cookies(response, access, new_raw, session.csrf_token)
    return ok({"access_token": access, "expires_in": settings.access_token_ttl,
               "csrf_token": session.csrf_token})


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get(REFRESH_COOKIE)
    if raw:
        session = db.execute(select(UserSession).where(
            UserSession.refresh_token_hash == sha256_hex(raw))).scalar_one_or_none()
        if session:
            session.revoked_at = utcnow()
            db.commit()
    _clear_auth_cookies(response)
    return ok({"loggedOut": True})


@router.get("/me")
def me(user: User | None = Depends(get_optional_user), db: Session = Depends(get_db)):
    """Session probe.

    Returns the signed-in user, or ``null`` when no valid session is presented so
    the SPA can boot anonymously without treating it as an error.
    """
    if user is None:
        return ok(None)
    wallet = credits.get_wallet(db, user.id)
    return ok(user_public(user, wallet.balance))


@router.patch("/me")
def update_me(payload: UpdateProfileRequest, request: Request,
              user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if payload.display_name is not None:
        user.display_name = payload.display_name.strip()[:80]
    if payload.password is not None:
        problems = password_strength(payload.password)
        if problems:
            raise ApiError("WEAK_PASSWORD", "Choose a stronger password.", status=422,
                           details=[{"issue": p} for p in problems])
        user.password_hash = hash_password(payload.password)
        notifications.send_security_alert(user, "password changed")
    db.add(AuditLog(actor_id=user.id, action="auth.profile_update", ip=client_ip(request)))
    db.commit()
    wallet = credits.get_wallet(db, user.id)
    return ok(user_public(user, wallet.balance))


@router.post("/forgot-password")
def forgot_password(payload: dict, request: Request, db: Session = Depends(get_db)):
    enforce(auth_limiter, client_ip(request), "password reset")
    email = str(payload.get("email", "")).lower()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    token = None
    if user is not None:
        raw = random_token(24)
        db.add(VerificationToken(user_id=user.id, kind="password_reset",
                                 token_hash=sha256_hex(raw),
                                 expires_at=utcnow() + dt.timedelta(minutes=60)))
        db.commit()
        notifications.send_password_reset(user, raw)
        token = None if settings.is_production else raw
    # always the same response: no account enumeration
    return ok({"message": "If that address exists, a reset link has been sent.",
               "reset_token": token})


@router.post("/reset-password")
def reset_password(payload: dict, request: Request, db: Session = Depends(get_db)):
    enforce(auth_limiter, client_ip(request), "password reset")
    token = str(payload.get("token", ""))
    password = str(payload.get("password", ""))
    problems = password_strength(password)
    if problems:
        raise ApiError("WEAK_PASSWORD", "Choose a stronger password.", status=422,
                       details=[{"issue": p} for p in problems])
    record = db.execute(select(VerificationToken).where(
        VerificationToken.token_hash == sha256_hex(token),
        VerificationToken.kind == "password_reset")).scalar_one_or_none()
    if record is None or record.used_at is not None or is_expired(record.expires_at):
        raise ApiError("INVALID_TOKEN", "This reset link is invalid or has expired.", status=400)
    user = db.get(User, record.user_id)
    if user is None:
        raise ApiError("INVALID_TOKEN", "This reset link is invalid.", status=400)
    user.password_hash = hash_password(password)
    user.locked_until = None
    user.failed_logins = 0
    record.used_at = utcnow()
    for session in db.execute(select(UserSession).where(UserSession.user_id == user.id)).scalars():
        session.revoked_at = utcnow()
    db.add(AuditLog(actor_id=user.id, action="auth.password_reset", ip=client_ip(request)))
    db.commit()
    notifications.send_security_alert(user, "password reset")
    return ok({"message": "Password updated. All sessions were signed out."})


@router.post("/verify-email")
def verify_email(payload: dict, db: Session = Depends(get_db)):
    token = str(payload.get("token", ""))
    record = db.execute(select(VerificationToken).where(
        VerificationToken.token_hash == sha256_hex(token),
        VerificationToken.kind == "email_verify")).scalar_one_or_none()
    if record is None or record.used_at is not None or is_expired(record.expires_at):
        raise ApiError("INVALID_TOKEN", "This verification link is invalid or expired.", status=400)
    user = db.get(User, record.user_id)
    if user is None:
        raise ApiError("INVALID_TOKEN", "This verification link is invalid.", status=400)
    user.email_verified = True
    record.used_at = utcnow()
    db.commit()
    return ok({"verified": True})


# ---------------------------------------------------------------------------
# MFA
# ---------------------------------------------------------------------------
@router.post("/mfa/setup")
def mfa_setup(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    secret = generate_totp_secret()
    user.mfa_secret = secret
    db.commit()
    return ok({"secret": secret, "otpauth_uri": totp_provisioning_uri(secret, user.email),
               "current_code": totp_now(secret)})


@router.post("/mfa/enable")
def mfa_enable(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    code = str(payload.get("code", ""))
    if not user.mfa_secret:
        raise ApiError("MFA_NOT_CONFIGURED", "Start MFA setup first.", status=400)
    if not verify_totp(user.mfa_secret, code):
        raise ApiError("MFA_INVALID", "That authenticator code is not valid.", status=400)
    user.mfa_enabled = True
    db.commit()
    notifications.send_security_alert(user, "two-factor authentication enabled")
    return ok({"mfa_enabled": True})


@router.post("/mfa/disable")
def mfa_disable(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_totp(user.mfa_secret or "", str(payload.get("code", ""))):
        raise ApiError("MFA_INVALID", "That authenticator code is not valid.", status=400)
    user.mfa_enabled = False
    user.mfa_secret = None
    db.commit()
    notifications.send_security_alert(user, "two-factor authentication disabled")
    return ok({"mfa_enabled": False})


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
@router.get("/sessions")
def list_sessions(request: Request, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    purge_expired_sessions(db)
    current = request.cookies.get(REFRESH_COOKIE)
    current_hash = sha256_hex(current) if current else ""
    rows = db.execute(select(UserSession).where(UserSession.user_id == user.id,
                                                UserSession.revoked_at.is_(None))
                      .order_by(UserSession.last_seen_at.desc())).scalars().all()
    return ok([{
        "id": s.id,
        "current": s.refresh_token_hash == current_hash,
        "user_agent": s.user_agent,
        "ip": s.ip,
        "created_at": s.created_at,
        "last_seen_at": s.last_seen_at,
        "expires_at": s.expires_at,
    } for s in rows])


@router.delete("/sessions/{session_id}")
def revoke_session(session_id: str, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    session = db.get(UserSession, session_id)
    if session is None or session.user_id != user.id:
        raise ApiError("NOT_FOUND", "Session not found.", status=404)
    session.revoked_at = utcnow()
    db.commit()
    return ok({"revoked": session_id})


@router.delete("/me")
def delete_account(request: Request, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Soft-delete with immediate session revocation (retention policy applies)."""
    user.deleted_at = utcnow()
    user.is_active = False
    for session in db.execute(select(UserSession).where(UserSession.user_id == user.id)).scalars():
        session.revoked_at = utcnow()
    db.add(AuditLog(actor_id=user.id, action="account.delete", target_type="user",
                    target_id=user.id, ip=client_ip(request)))
    db.commit()
    notifications.send_email(user.email, "Your 3DFUN AI account was deleted",
                             "Account deleted",
                             "Your account has been deactivated. Assets are retained for 30 days "
                             "for recovery, then purged.")
    response = Response(status_code=200)
    _clear_auth_cookies(response)
    return ok({"deleted": True})
