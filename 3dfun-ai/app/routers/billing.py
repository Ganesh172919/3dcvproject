"""Credits ledger endpoints and billing (Stripe-compatible, optional)."""
from __future__ import annotations

import hashlib
import hmac
import json
import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import credits, notifications
from ..auth import get_current_user
from ..config import settings
from ..db import (AuditLog, CreditReason, Invoice, User, WebhookEvent, get_db)
from ..schemas import ApiError, CheckoutRequest, ok
from ..serializers import transaction_public

router = APIRouter(tags=["credits"])


@router.get("/credits/balance")
def balance(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    wallet = credits.get_wallet(db, user.id)
    return ok({
        "balance": wallet.balance,
        "reserved": wallet.reserved,
        "available": credits.available(wallet.balance, wallet.reserved),
        "lifetime_granted": wallet.lifetime_granted,
        "lifetime_spent": wallet.lifetime_spent,
        "costs": settings.credit_costs,
    })


@router.get("/credits/transactions")
def transactions(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                 limit: int = 50, offset: int = 0):
    rows = credits.transactions(db, user.id, limit=min(limit, 200), offset=offset)
    return ok([transaction_public(t) for t in rows])


@router.get("/credits/packs")
def packs():
    return ok([{"id": key, **value} for key, value in credits.PACKS.items()])


@router.post("/billing/checkout")
def checkout(payload: CheckoutRequest, user: User = Depends(get_current_user),
             db: Session = Depends(get_db)):
    pack = credits.PACKS.get(payload.pack)
    if pack is None:
        raise ApiError("INVALID_PACK", "Unknown credit pack.", status=422)
    if settings.stripe_secret_key:
        # Stripe Checkout Session via the REST API (no SDK dependency).
        import urllib.parse
        import urllib.request
        body = urllib.parse.urlencode({
            "mode": "payment",
            "success_url": f"{settings.app_url}/dashboard?checkout=success",
            "cancel_url": f"{settings.app_url}/pricing?checkout=cancelled",
            "customer_email": user.email,
            "metadata[user_id]": user.id,
            "metadata[credits]": str(pack["credits"]),
            "metadata[pack]": payload.pack,
            "line_items[0][quantity]": "1",
            "line_items[0][price_data][currency]": "usd",
            "line_items[0][price_data][unit_amount]": str(pack["amount_cents"]),
            "line_items[0][price_data][product_data][name]": f"3DFUN AI · {payload.pack} pack",
        }).encode()
        req = urllib.request.Request("https://api.stripe.com/v1/checkout/sessions", data=body,
                                     headers={"Authorization": f"Bearer {settings.stripe_secret_key}",
                                              "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                session = json.loads(resp.read().decode())
            return ok({"checkout_url": session["url"], "pack": payload.pack,
                       "credits": pack["credits"], "amount_cents": pack["amount_cents"],
                       "mode": "stripe"})
        except Exception as exc:
            raise ApiError("STRIPE_ERROR", f"Checkout could not be created: {exc}", status=502)
    # Development mode: grant immediately so billing never blocks local work.
    credits.grant(db, user.id, pack["credits"], reason=CreditReason.PURCHASE,
                  idempotency_key=f"devcheckout:{user.id}:{payload.pack}:{int(time.time() // 30)}",
                  meta={"pack": payload.pack, "mode": "development"})
    db.add(Invoice(user_id=user.id, amount_cents=pack["amount_cents"], credits=pack["credits"],
                   status="paid", external_id=f"dev_{payload.pack}_{int(time.time())}"))
    db.commit()
    notifications.send_email(user.email, "Credits added to your account",
                             "Purchase complete",
                             f"{pack['credits']} credits were added to your 3DFUN AI wallet.")
    wallet = credits.get_wallet(db, user.id)
    return ok({"checkout_url": f"{settings.app_url}/dashboard?checkout=dev-success",
               "pack": payload.pack, "credits": pack["credits"],
               "amount_cents": pack["amount_cents"], "mode": "development",
               "balance": wallet.balance})


@router.post("/billing/portal")
def portal(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    invoices = db.execute(select(Invoice).where(Invoice.user_id == user.id)
                          .order_by(Invoice.created_at.desc()).limit(20)).scalars().all()
    return ok({"mode": "stripe" if settings.stripe_secret_key else "development",
               "invoices": [{"id": i.id, "amount_cents": i.amount_cents, "credits": i.credits,
                             "status": i.status, "created_at": i.created_at} for i in invoices]})


def _verify_stripe_signature(payload: bytes, header: str, secret: str, tolerance: int = 300) -> bool:
    """HMAC-SHA256 over `t.body` per Stripe's scheme; constant-time compare."""
    parts = dict(item.split("=", 1) for item in header.split(",") if "=" in item)
    timestamp = parts.get("t", "")
    signature = parts.get("v1", "")
    if not timestamp or not signature:
        return False
    if abs(time.time() - int(timestamp)) > tolerance:
        return False
    expected = hmac.new(secret.encode(), f"{timestamp}.{payload.decode()}".encode(),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/billing/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    header = request.headers.get("stripe-signature", "")
    if settings.stripe_webhook_secret and not _verify_stripe_signature(
            payload, header, settings.stripe_webhook_secret):
        raise ApiError("INVALID_SIGNATURE", "Webhook signature verification failed.", status=400)
    try:
        event = json.loads(payload.decode() or "{}")
    except json.JSONDecodeError:
        raise ApiError("INVALID_PAYLOAD", "Malformed webhook body.", status=400)
    external_id = str(event.get("id", ""))
    if not external_id:
        raise ApiError("INVALID_PAYLOAD", "Missing event id.", status=400)
    existing = db.execute(select(WebhookEvent).where(
        WebhookEvent.external_id == external_id)).scalar_one_or_none()
    if existing is not None and existing.processed:
        return ok({"received": True, "idempotent": True})   # replay-safe

    record = existing or WebhookEvent(external_id=external_id, provider="stripe",
                                      event_type=str(event.get("type", "")),
                                      payload=event)
    if existing is None:
        db.add(record)
    try:
        if event.get("type") == "checkout.session.completed":
            session = event["data"]["object"]
            metadata = session.get("metadata", {}) or {}
            user_id = metadata.get("user_id")
            pack = metadata.get("pack", "starter")
            amount = int(metadata.get("credits") or credits.PACKS.get(pack, {}).get("credits", 0))
            if user_id and amount:
                credits.grant(db, user_id, amount, reason=CreditReason.PURCHASE,
                              idempotency_key=f"stripe:{external_id}",
                              meta={"pack": pack})
                db.add(Invoice(user_id=user_id, amount_cents=int(session.get("amount_total", 0)),
                               credits=amount, status="paid", external_id=session.get("id")))
        record.processed = True
        db.commit()
    except Exception as exc:                       # pragma: no cover - defensive
        record.error = str(exc)
        db.commit()
        raise ApiError("WEBHOOK_PROCESSING_FAILED", str(exc), status=500, retryable=True)
    return ok({"received": True, "idempotent": False})


@router.post("/admin/credits/grant")
def admin_grant(payload: dict, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    if user.role != "admin":
        raise ApiError("FORBIDDEN", "Admin access required.", status=403)
    target = db.get(User, str(payload.get("user_id", "")))
    if target is None:
        raise ApiError("NOT_FOUND", "User not found.", status=404)
    amount = int(payload.get("amount", 0))
    wallet = credits.grant(db, target.id, amount, reason=CreditReason.ADMIN_ADJUSTMENT,
                           meta={"by": user.id, "reason": payload.get("reason", "")})
    db.add(AuditLog(actor_id=user.id, action="credits.admin_grant", target_type="user",
                    target_id=target.id, metadata_json={"amount": amount}))
    db.commit()
    return ok({"user_id": target.id, "balance": wallet.balance})
