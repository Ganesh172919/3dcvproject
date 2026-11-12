"""Generation endpoints: estimate, create, list, detail, cancel, retry, events."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .. import credits, events, metrics
from ..auth import get_current_user, owned_generation
from ..config import settings
from ..db import (AuditLog, Generation, GenerationEvent, GenerationStatus, Provider,
                  Upload, User, get_db, utcnow)
from .. import jobs, providers
from ..jobs import runner
from ..providers import get_provider
from ..ratelimit import enforce, generation_limiter
from ..schemas import (ApiError, CreateGenerationRequest, EstimateRequest, ok)
from ..serializers import generation_public
from ..state_machine import assert_transition, is_retryable

router = APIRouter(prefix="/generations", tags=["generations"])


def _queue_position(db: Session) -> int:
    return db.execute(select(func.count()).select_from(Generation).where(
        Generation.status.in_([GenerationStatus.QUEUED]))).scalar_one()


def _pick_provider(db: Session, privacy_mode: bool, engine: str | None = None) -> str:
    """Weighted provider routing with health fallback (recorded in metadata)."""
    if engine:
        return engine
    if settings.generation_provider and settings.generation_provider not in {"local", ""} \
            and settings.environment in {"test", "development"}:
        # A single explicit provider wins in dev/test (e.g. GENERATION_PROVIDER=mock).
        return settings.generation_provider
    rows = db.execute(select(Provider).where(Provider.enabled.is_(True))
                      .order_by(Provider.weight.desc())).scalars().all()
    if not rows:
        return settings.generation_provider
    if privacy_mode:
        no_retention = [p for p in rows if "no-retention" in (p.notes or "")]
        if no_retention:
            return no_retention[0].slug
    for provider in rows:
        if provider.healthy:
            return provider.slug
    return rows[0].slug


@router.post("/estimate")
def estimate(payload: EstimateRequest, user: User = Depends(get_current_user),
             db: Session = Depends(get_db)):
    provider_slug = _pick_provider(db, privacy_mode=False, engine=payload.options.engine)
    provider = providers.get_provider(provider_slug)
    cost = provider.estimate_cost(payload.quality, payload.options.model_dump())
    wallet = credits.get_wallet(db, user.id)
    return ok({
        "credits": cost.credits,
        "balance": wallet.balance,
        "available": credits.available(wallet.balance, wallet.reserved),
        "estimated_seconds": cost.estimated_seconds,
        "provider": cost.provider,
        "model": getattr(provider, "model_name", settings.generation_model),
        "queue_position": _queue_position(db),
    })


@router.post("")
def create_generation(payload: CreateGenerationRequest, request: Request,
                      user: User = Depends(get_current_user),
                      db: Session = Depends(get_db),
                      idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    enforce(generation_limiter, user.id, "generation")
    if not payload.accept_terms:
        raise ApiError("TERMS_REQUIRED", "You must accept the generation terms.", status=422)

    key = payload.idempotency_key or idempotency_key
    if key:
        existing = db.execute(select(Generation).where(
            Generation.user_id == user.id,
            Generation.idempotency_key == key)).scalar_one_or_none()
        if existing is not None:
            return ok(generation_public(existing))

    upload = db.get(Upload, payload.upload_id)
    if upload is None or upload.user_id != user.id:
        raise ApiError("UPLOAD_NOT_FOUND", "Upload the reference image first.", status=404)

    provider_slug = _pick_provider(db, payload.privacy_mode, payload.options.engine)
    # resolve through the module attribute so provider routing stays patchable
    provider = providers.get_provider(provider_slug)
    cost = provider.estimate_cost(payload.quality, payload.options.model_dump())

    gen = Generation(
        user_id=user.id, upload_id=upload.id, provider_slug=provider_slug,
        model=getattr(provider, "model_name", settings.generation_model),
        quality=payload.quality, preset=payload.preset,
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        options=payload.options.model_dump(), privacy_mode=payload.privacy_mode,
        status=GenerationStatus.QUEUED, stage="QUEUED", progress=0,
        idempotency_key=key, queue_position=_queue_position(db) + 1,
        config={"provider_reason": "weighted-routing", "estimated_credits": cost.credits},
    )
    db.add(gen)
    db.flush()
    credits.reserve(db, user.id, gen, cost.credits)     # raises 402 when short
    db.add(AuditLog(actor_id=user.id, action="generation.create", target_type="generation",
                    target_id=gen.id, ip=request.client.host if request.client else ""))
    db.commit()
    events.record_event(db, gen.id, "generation.queued", "QUEUED", 0,
                        {"creditsReserved": cost.credits, "provider": provider_slug})
    metrics.inc("generation_created_total", quality=payload.quality, provider=provider_slug)
    gen.queue_position = 0
    db.commit()
    runner.enqueue(gen.id)
    return ok(generation_public(gen))


@router.get("")
def list_generations(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                     page: int = 1, page_size: int = 20, status: str | None = None,
                     q: str | None = None):
    page = max(1, page)
    page_size = min(max(page_size, 1), 100)
    stmt = select(Generation).where(Generation.user_id == user.id)
    count_stmt = select(func.count()).select_from(Generation).where(Generation.user_id == user.id)
    if status:
        try:
            wanted = GenerationStatus(status.upper())
        except ValueError:
            raise ApiError("INVALID_FILTER", f"Unknown status '{status}'.", status=422)
        stmt = stmt.where(Generation.status == wanted)
        count_stmt = count_stmt.where(Generation.status == wanted)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(func.lower(Generation.prompt).like(like))
        count_stmt = count_stmt.where(func.lower(Generation.prompt).like(like))
    total = db.execute(count_stmt).scalar_one()
    rows = db.execute(stmt.options(selectinload(Generation.asset)).order_by(Generation.created_at.desc())
                      .offset((page - 1) * page_size).limit(page_size)).scalars().all()
    source_ids = {g.upload_id for g in rows if g.upload_id}
    uploads = {u.id: u for u in db.scalars(select(Upload).where(
        Upload.id.in_(source_ids), Upload.user_id == user.id,
    ))} if source_ids else {}
    return ok({"items": [generation_public(g, reference=uploads.get(g.upload_id)) for g in rows], "total": total,
               "page": page, "page_size": page_size})


@router.get("/{generation_id}")
def get_generation(generation_id: str, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    return ok(generation_public(gen))


@router.post("/{generation_id}/cancel")
def cancel_generation(generation_id: str, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    if gen.status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED,
                      GenerationStatus.OPTIMIZING, GenerationStatus.RENDERING_PREVIEW):
        raise ApiError("NOT_CANCELLABLE", "This generation already finished.", status=409)
    gen.cancel_requested = True
    if gen.status in (GenerationStatus.QUEUED, GenerationStatus.VALIDATING):
        credits.release(db, gen.user_id, gen)
        gen.status = GenerationStatus.CANCELLED
        gen.stage = "CANCELLED"
        gen.completed_at = utcnow()
        db.commit()
        events.record_event(db, gen.id, "generation.cancelled", "CANCELLED", gen.progress)
    else:
        gen.status = GenerationStatus.CANCEL_REQUESTED
        db.commit()
        events.record_event(db, gen.id, "generation.cancel_requested", "CANCEL_REQUESTED",
                            gen.progress)
    return ok(generation_public(gen))


@router.post("/{generation_id}/retry")
def retry_generation(generation_id: str, request: Request,
                     user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    if gen.status not in (GenerationStatus.FAILED, GenerationStatus.CANCELLED):
        raise ApiError("NOT_RETRYABLE", "Only failed or cancelled jobs can be retried.", status=409)
    cost = providers.get_provider(gen.provider_slug).estimate_cost(gen.quality, gen.options or {})
    assert_transition(gen.status, GenerationStatus.QUEUED)
    gen.status = GenerationStatus.QUEUED
    gen.stage = "QUEUED"
    gen.progress = 0
    gen.error_code = None
    gen.error_message = None
    gen.cancel_requested = False
    gen.completed_at = None
    credits.reserve(db, gen.user_id, gen, cost.credits)
    events.record_event(db, gen.id, "generation.queued", "QUEUED", 0, {"retry": True})
    db.add(AuditLog(actor_id=user.id, action="generation.retry", target_type="generation",
                    target_id=gen.id))
    db.commit()
    runner.enqueue(gen.id)
    return ok(generation_public(gen))


@router.get("/{generation_id}/events")
def list_events(generation_id: str, after: int = 0, limit: int = 200,
                user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    rows = db.execute(select(GenerationEvent).where(
        GenerationEvent.generation_id == gen.id, GenerationEvent.sequence > after)
        .order_by(GenerationEvent.sequence.asc()).limit(min(limit, 500))).scalars().all()
    from ..events import serialize
    return ok([serialize(r) for r in rows])


@router.get("/{generation_id}/attempts")
def list_attempts(generation_id: str, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    return ok([{
        "id": a.id, "attempt_no": a.attempt_no, "status": a.status,
        "provider": a.provider_slug, "error_code": a.error_code,
        "error_message": a.error_message, "metrics": a.metrics,
        "started_at": a.started_at, "finished_at": a.finished_at,
        "duration_ms": a.duration_ms,
    } for a in gen.attempts])
