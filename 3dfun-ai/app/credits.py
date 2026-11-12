"""Immutable credit ledger with reservation, spend, refund and grants.

Invariants
----------
1. ``credit_transactions`` is append-only; corrections are new rows.
2. ``wallet.balance`` always equals the sum of ledger amounts for that wallet.
3. A generation reserves credits (`reserved` grows, `balance` unchanged) before
   work starts, then either spends (reserved -> balance debit) or releases.
4. Every mutation is idempotent when an ``idempotency_key`` is supplied.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (CreditReason, CreditTransaction, CreditWallet, Generation,
                 User, utcnow)
from .schemas import ApiError

PACKS: dict[str, dict[str, int]] = {
    "starter": {"credits": 200, "amount_cents": 900},
    "studio": {"credits": 800, "amount_cents": 2900},
    "scale": {"credits": 3000, "amount_cents": 8900},
}


def get_wallet(db: Session, user_id: str) -> CreditWallet:
    wallet = db.execute(select(CreditWallet).where(CreditWallet.user_id == user_id)).scalar_one_or_none()
    if wallet is None:
        wallet = CreditWallet(user_id=user_id, balance=0)
        db.add(wallet)
        db.flush()
    return wallet


def _locked_wallet(db: Session, user_id: str) -> CreditWallet:
    """Row-locked wallet read (no-op on SQLite, real on PostgreSQL)."""
    stmt = select(CreditWallet).where(CreditWallet.user_id == user_id)
    if db.bind and db.bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()
    wallet = db.execute(stmt).scalar_one_or_none()
    if wallet is None:
        wallet = CreditWallet(user_id=user_id, balance=0)
        db.add(wallet)
        db.flush()
    return wallet


def _ledger(db: Session, wallet: CreditWallet, user_id: str, reason: CreditReason,
            amount: int, generation_id: str | None = None,
            idempotency_key: str | None = None, meta: dict[str, Any] | None = None) -> CreditTransaction:
    if idempotency_key:
        existing = db.execute(
            select(CreditTransaction).where(CreditTransaction.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
        if existing:
            return existing
    entry = CreditTransaction(
        wallet_id=wallet.id, user_id=user_id, generation_id=generation_id,
        reason=reason, amount=amount, balance_after=wallet.balance,
        idempotency_key=idempotency_key, metadata_json=meta or {})
    db.add(entry)
    wallet.version += 1
    wallet.updated_at = utcnow()
    return entry


def grant(db: Session, user_id: str, amount: int, reason: CreditReason = CreditReason.SIGNUP_GRANT,
          idempotency_key: str | None = None, meta: dict[str, Any] | None = None) -> CreditWallet:
    wallet = _locked_wallet(db, user_id)
    wallet.balance += amount
    wallet.lifetime_granted += max(amount, 0)
    _ledger(db, wallet, user_id, reason, amount, idempotency_key=idempotency_key, meta=meta)
    db.commit()
    db.refresh(wallet)
    return wallet


def available(balance: int, reserved: int) -> int:
    return max(balance - reserved, 0)


def reserve(db: Session, user_id: str, generation: Generation, amount: int) -> CreditWallet:
    wallet = _locked_wallet(db, user_id)
    if available(wallet.balance, wallet.reserved) < amount:
        raise ApiError("INSUFFICIENT_CREDITS",
                       "Not enough credits for this generation.",
                       status=402,
                       details=[{"required": amount,
                                 "available": available(wallet.balance, wallet.reserved)}])
    wallet.reserved += amount
    generation.credits_reserved = amount
    # The price is known the moment the job is queued, so it is exposed in the
    # API straight away; `spend()` turns this into an actual wallet debit when
    # the generation succeeds and `refund_failed()` zeroes it on failure.
    generation.credits_charged = amount
    _ledger(db, wallet, user_id, CreditReason.RESERVATION, 0, generation.id,
            idempotency_key=f"reserve:{generation.id}", meta={"reserved": amount})
    db.commit()
    db.refresh(wallet)
    return wallet


def release(db: Session, user_id: str, generation: Generation) -> CreditWallet:
    wallet = _locked_wallet(db, user_id)
    amount = generation.credits_reserved or 0
    wallet.reserved = max(0, wallet.reserved - amount)
    if amount:
        _ledger(db, wallet, user_id, CreditReason.REFUND, 0, generation.id,
                idempotency_key=f"release:{generation.id}", meta={"released": amount})
    generation.credits_reserved = 0
    generation.credits_charged = 0
    db.commit()
    db.refresh(wallet)
    return wallet


def spend(db: Session, user_id: str, generation: Generation) -> CreditWallet:
    """Commit a reservation: reserved -> balance debit (idempotent per generation)."""
    wallet = _locked_wallet(db, user_id)
    amount = generation.credits_reserved or 0
    key = f"spend:{generation.id}"
    existing = db.execute(
        select(CreditTransaction).where(CreditTransaction.idempotency_key == key)
    ).scalar_one_or_none()
    if existing is not None:
        return wallet
    wallet.reserved = max(0, wallet.reserved - amount)
    wallet.balance = max(0, wallet.balance - amount)
    wallet.lifetime_spent += amount
    generation.credits_reserved = 0
    generation.credits_charged = amount
    _ledger(db, wallet, user_id, CreditReason.SPEND, -amount, generation.id,
            idempotency_key=key, meta={"quality": generation.quality})
    db.commit()
    db.refresh(wallet)
    return wallet


def refund_failed(db: Session, user_id: str, generation: Generation) -> CreditWallet:
    """Give back a charge for a failed generation (only if it was charged)."""
    wallet = _locked_wallet(db, user_id)
    key = f"refund:{generation.id}"
    existing = db.execute(
        select(CreditTransaction).where(CreditTransaction.idempotency_key == key)
    ).scalar_one_or_none()
    if existing is not None or not generation.credits_charged:
        return wallet
    amount = generation.credits_charged
    wallet.balance += amount
    _ledger(db, wallet, user_id, CreditReason.REFUND, amount, generation.id,
            idempotency_key=key, meta={"refunded": amount})
    generation.credits_charged = 0
    db.commit()
    db.refresh(wallet)
    return wallet


def transactions(db: Session, user_id: str, limit: int = 50, offset: int = 0) -> list[CreditTransaction]:
    return list(db.execute(
        select(CreditTransaction).where(CreditTransaction.user_id == user_id)
        .order_by(CreditTransaction.created_at.desc()).limit(limit).offset(offset)
    ).scalars())


def consistency_error(db: Session, user_id: str) -> int | None:
    """Return non-zero when the wallet disagrees with the ledger (for health checks)."""
    wallet = get_wallet(db, user_id)
    total = db.execute(
        select(CreditTransaction).where(CreditTransaction.reason != CreditReason.RESERVATION)
    ).scalars()
    ledger_sum = sum(t.amount for t in total)
    if wallet.balance != ledger_sum:
        return wallet.balance - ledger_sum
    return None
