"""Administration endpoints (RBAC-protected, every mutation audited)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import credits, metrics, notifications
from ..auth import require_admin
from ..config import settings
from ..db import (Asset, AuditLog, CreditReason, FeatureFlag, Generation, GenerationStatus,
                  Provider, Report, Upload, User, get_db, utcnow)
from ..jobs import backend as queue_backend
from ..jobs import queue_stats, runner
from ..providers import provider_health
from ..schemas import ApiError, AdminUserPatch, FeatureFlagPatch, ok
from ..serializers import generation_public
from ..storage import storage

router = APIRouter(prefix="/admin", tags=["admin"])


def _audit(db: Session, actor: User, action: str, target_type: str, target_id: str,
           request: Request | None = None, meta: dict | None = None) -> None:
    db.add(AuditLog(actor_id=actor.id, action=action, target_type=target_type,
                    target_id=target_id, metadata_json=meta or {},
                    ip=(request.client.host if request and request.client else "")))
    db.commit()


@router.get("/overview")
def overview(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    from .. import jobs as jobs_mod
    stats = queue_stats(db)
    users = db.execute(select(func.count()).select_from(User)).scalar_one()
    gens = db.execute(select(func.count()).select_from(Generation)).scalar_one()
    assets = db.execute(select(func.count()).select_from(Asset)).scalar_one()
    completed = db.execute(select(func.count()).select_from(Generation).where(
        Generation.status == GenerationStatus.COMPLETED)).scalar_one()
    failed = db.execute(select(func.count()).select_from(Generation).where(
        Generation.status == GenerationStatus.FAILED)).scalar_one()
    storage_bytes = 0
    try:
        storage_bytes = storage.size_of_prefix("users/")
    except Exception:                                  # pragma: no cover
        storage_bytes = 0
    return ok({
        "users": users, "generations": gens, "assets": assets,
        "completed": completed, "failed": failed,
        "success_rate": round(completed / gens * 100, 1) if gens else 0.0,
        "queue": {"queued": stats.queued, "running": stats.running,
                  "completed_today": stats.completed_today, "failed_today": stats.failed_today},
        "runner": {"inflight": runner.inflight(), "processed": runner.processed,
                   "failed": runner.failed, "backend": queue_backend()},
        "storage_bytes": storage_bytes,
        "metrics": metrics.snapshot(),
    })


@router.get("/users")
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db),
               page: int = 1, page_size: int = 25, q: str | None = None):
    page = max(1, page)
    page_size = min(max(page_size, 1), 100)
    base = select(User)
    count = select(func.count()).select_from(User)
    if q:
        like = f"%{q.lower()}%"
        base = base.where(func.lower(User.email).like(like))
        count = count.where(func.lower(User.email).like(like))
    total = db.execute(count).scalar_one()
    rows = db.execute(base.order_by(User.created_at.desc())
                      .offset((page - 1) * page_size).limit(page_size)).scalars().all()
    items = []
    for u in rows:
        wallet = credits.get_wallet(db, u.id)
        gens = db.execute(select(func.count()).select_from(Generation).where(
            Generation.user_id == u.id)).scalar_one()
        items.append({
            "id": u.id, "email": u.email, "display_name": u.display_name, "role": u.role,
            "is_active": u.is_active, "email_verified": u.email_verified,
            "mfa_enabled": u.mfa_enabled, "credits": wallet.balance,
            "generations": gens, "created_at": u.created_at,
            "last_login_at": u.last_login_at,
            "deleted": u.deleted_at is not None,
        })
    return ok({"items": items, "total": total, "page": page, "page_size": page_size})


@router.patch("/users/{user_id}")
def patch_user(user_id: str, payload: AdminUserPatch, request: Request,
               admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = db.get(User, user_id)
    if target is None:
        raise ApiError("NOT_FOUND", "User not found.", status=404)
    changes: dict = {}
    if payload.role and payload.role != target.role:
        if target.id == admin.id and payload.role != "admin":
            raise ApiError("SELF_DEMOTION", "You cannot remove your own admin role.", status=409)
        changes["role"] = f"{target.role} -> {payload.role}"
        target.role = payload.role
    if payload.is_active is not None and payload.is_active != target.is_active:
        changes["is_active"] = payload.is_active
        target.is_active = payload.is_active
        if not payload.is_active:
            from ..db import UserSession
            for s in db.execute(select(UserSession).where(UserSession.user_id == target.id)).scalars():
                s.revoked_at = utcnow()
    db.commit()
    if payload.credits_delta:
        credits.grant(db, target.id, payload.credits_delta,
                      reason=CreditReason.ADMIN_ADJUSTMENT,
                      meta={"by": admin.id, "reason": payload.reason})
        changes["credits_delta"] = payload.credits_delta
    _audit(db, admin, "admin.user_patch", "user", target.id, request,
           {"changes": changes, "reason": payload.reason})
    wallet = credits.get_wallet(db, target.id)
    return ok({"id": target.id, "role": target.role, "is_active": target.is_active,
               "credits": wallet.balance, "changes": changes})


@router.get("/jobs")
def list_jobs(admin: User = Depends(require_admin), db: Session = Depends(get_db),
              page: int = 1, page_size: int = 25, status: str | None = None):
    page = max(1, page)
    page_size = min(max(page_size, 1), 100)
    base = select(Generation)
    count = select(func.count()).select_from(Generation)
    if status:
        try:
            wanted = GenerationStatus(status.upper())
        except ValueError:
            raise ApiError("INVALID_FILTER", "Unknown status.", status=422)
        base = base.where(Generation.status == wanted)
        count = count.where(Generation.status == wanted)
    total = db.execute(count).scalar_one()
    rows = db.execute(base.order_by(Generation.created_at.desc())
                      .offset((page - 1) * page_size).limit(page_size)).scalars().all()
    items = []
    for g in rows:
        owner = db.get(User, g.user_id)
        items.append({
            "id": g.id, "user_email": owner.email if owner else "deleted user",
            "status": g.status.value, "stage": g.stage, "progress": g.progress,
            "quality": g.quality, "provider": g.provider_slug,
            "created_at": g.created_at, "duration_ms": g.duration_ms,
            "error_code": g.error_code, "credits": g.credits_charged,
        })
    return ok({"items": items, "total": total, "page": page, "page_size": page_size})


@router.get("/providers")
def list_providers(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.execute(select(Provider).order_by(Provider.weight.desc())).scalars().all()
    out = []
    for p in rows:
        health = provider_health(p.slug)
        gens = db.execute(select(func.count()).select_from(Generation).where(
            Generation.provider_slug == p.slug)).scalar_one()
        funded = db.execute(select(func.count()).select_from(Generation).where(
            Generation.provider_slug == p.slug,
            Generation.status == GenerationStatus.COMPLETED)).scalar_one()
        out.append({
            "id": p.id, "slug": p.slug, "display_name": p.display_name, "kind": p.kind,
            "enabled": p.enabled, "healthy": p.healthy and health.get("healthy", False),
            "weight": p.weight, "concurrency": p.concurrency,
            "generations": gens,
            "success_rate": round(funded / gens * 100, 1) if gens else None,
            "probe": health, "notes": p.notes,
        })
    return ok(out)


@router.get("/audit-logs")
def audit_logs(admin: User = Depends(require_admin), db: Session = Depends(get_db),
               page: int = 1, page_size: int = 50, action: str | None = None):
    page = max(1, page)
    page_size = min(max(page_size, 1), 200)
    base = select(AuditLog)
    count = select(func.count()).select_from(AuditLog)
    if action:
        base = base.where(AuditLog.action.like(f"%{action}%"))
        count = count.where(AuditLog.action.like(f"%{action}%"))
    total = db.execute(count).scalar_one()
    rows = db.execute(base.order_by(AuditLog.created_at.desc())
                      .offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return ok({"items": [{
        "id": a.id, "actor_id": a.actor_id, "action": a.action,
        "target_type": a.target_type, "target_id": a.target_id, "ip": a.ip,
        "metadata": a.metadata_json, "created_at": a.created_at,
    } for a in rows], "total": total, "page": page, "page_size": page_size})


@router.get("/feature-flags")
def feature_flags(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.execute(select(FeatureFlag).order_by(FeatureFlag.key)).scalars().all()
    return ok([{"id": f.id, "key": f.key, "description": f.description, "enabled": f.enabled,
                "rollout_pct": f.rollout_pct, "allowlist": f.allowlist or [],
                "updated_at": f.updated_at} for f in rows])


@router.patch("/feature-flags/{flag_id}")
def patch_flag(flag_id: str, payload: FeatureFlagPatch, request: Request,
               admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    flag = db.get(FeatureFlag, flag_id)
    if flag is None:
        raise ApiError("NOT_FOUND", "Feature flag not found.", status=404)
    if payload.enabled is not None:
        flag.enabled = payload.enabled
    if payload.rollout_pct is not None:
        flag.rollout_pct = payload.rollout_pct
    if payload.allowlist is not None:
        flag.allowlist = payload.allowlist[:200]
    db.commit()
    _audit(db, admin, "admin.flag_patch", "feature_flag", flag.id, request,
           {"enabled": flag.enabled, "rollout_pct": flag.rollout_pct})
    return ok({"id": flag.id, "key": flag.key, "enabled": flag.enabled,
               "rollout_pct": flag.rollout_pct, "allowlist": flag.allowlist})


@router.get("/moderation")
def moderation_queue(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    reports = db.execute(select(Report).where(Report.state == "OPEN")
                         .order_by(Report.created_at.desc()).limit(100)).scalars().all()
    pending = db.execute(select(Asset).where(Asset.moderation_state == "REVIEW")
                         .limit(100)).scalars().all()
    return ok({
        "reports": [{"id": r.id, "asset_id": r.asset_id, "reason": r.reason,
                     "details": r.details, "created_at": r.created_at} for r in reports],
        "pending_assets": [{"id": a.id, "name": a.name, "owner_id": a.user_id,
                            "state": a.moderation_state} for a in pending],
    })


@router.post("/moderation/{asset_id}")
def moderate(asset_id: str, payload: dict, request: Request,
             admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    decision = str(payload.get("decision", "APPROVE")).upper()
    if decision not in ("APPROVE", "REJECT", "HIDE"):
        raise ApiError("INVALID_DECISION", "decision must be APPROVE, REJECT or HIDE.", status=422)
    asset.moderation_state = {"APPROVE": "APPROVED", "REJECT": "REJECTED",
                              "HIDE": "HIDDEN"}[decision]
    if decision == "REJECT":
        asset.visibility = asset.visibility.__class__.PRIVATE
    for report in db.execute(select(Report).where(Report.asset_id == asset.id)).scalars():
        report.state = "RESOLVED"
    db.commit()
    _audit(db, admin, f"admin.moderation_{decision.lower()}", "asset", asset.id, request,
           {"note": payload.get("note", "")[:500]})
    return ok({"asset_id": asset.id, "moderation_state": asset.moderation_state})


@router.post("/jobs/{generation_id}/cancel")
def admin_cancel_job(generation_id: str, request: Request,
                     admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    gen = db.get(Generation, generation_id)
    if gen is None:
        raise ApiError("NOT_FOUND", "Generation not found.", status=404)
    gen.cancel_requested = True
    db.commit()
    _audit(db, admin, "admin.job_cancel", "generation", gen.id, request)
    return ok(generation_public(gen))


@router.get("/storage")
def storage_report(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.execute(select(func.count()).select_from(User)).scalar_one()
    uploads = db.execute(select(func.count()).select_from(Upload)).scalar_one()
    upload_bytes = db.execute(select(func.coalesce(func.sum(Upload.size_bytes), 0))).scalar_one()
    asset_bytes = db.execute(select(func.coalesce(func.sum(Asset.size_bytes), 0))).scalar_one()
    return ok({"driver": settings.storage_driver, "users": users, "uploads": uploads,
               "upload_bytes": int(upload_bytes), "asset_bytes": int(asset_bytes),
               "total_bytes": int(upload_bytes) + int(asset_bytes),
               "max_upload_mb": settings.max_upload_mb})


@router.get("/analytics")
def analytics(admin: User = Depends(require_admin), db: Session = Depends(get_db),
              days: int = 14):
    days = min(max(days, 1), 90)
    since = utcnow() - dt.timedelta(days=days)
    rows = db.execute(select(Generation).where(Generation.created_at >= since)).scalars().all()
    by_day: dict[str, dict[str, int]] = {}
    for g in rows:
        day = g.created_at.strftime("%Y-%m-%d")
        bucket = by_day.setdefault(day, {"total": 0, "completed": 0, "failed": 0,
                                         "credits": 0, "duration_ms": 0})
        bucket["total"] += 1
        bucket["credits"] += g.credits_charged or 0
        if g.status == GenerationStatus.COMPLETED:
            bucket["completed"] += 1
            bucket["duration_ms"] += g.duration_ms or 0
        elif g.status == GenerationStatus.FAILED:
            bucket["failed"] += 1
    series = [{"date": d, **v} for d, v in sorted(by_day.items())]
    return ok({"days": days, "series": series,
               "totals": {"generations": len(rows),
                          "credits": sum(g.credits_charged or 0 for g in rows)}})
