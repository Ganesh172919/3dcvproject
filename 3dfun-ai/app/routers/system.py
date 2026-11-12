"""Health, readiness, metrics and OpenAPI endpoints."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .. import events, metrics as metrics_mod
from ..config import settings
from ..db import User, get_db, utcnow
from ..jobs import backend as queue_backend
from ..jobs import runner
from ..recon_bridge import engine_available
from ..schemas import ok

router = APIRouter(tags=["system"])
START = utcnow()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    checks: dict = {}
    status = "ok"
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok", "driver": db.bind.dialect.name}
    except Exception as exc:
        checks["database"] = {"status": "fail", "error": str(exc)}
        status = "fail"

    engine = engine_available()
    checks["engine"] = {"status": "ok" if engine.get("available") else "degraded", **engine}
    if not engine.get("available") and status == "ok":
        status = "degraded"

    checks["queue"] = {"status": "ok", "inflight": runner.inflight(),
                       "backend": queue_backend()["kind"],
                       "event_bus": events.backend_name()}
    checks["storage"] = {"status": "ok", "driver": settings.storage_driver}
    return ok({
        "status": status,
        "version": settings.version,
        "time": utcnow().isoformat(),
        "uptime_seconds": round((utcnow() - START).total_seconds(), 1),
        "checks": checks,
    })


@router.get("/health/ready")
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        return Response(status_code=503, content='{"ready": false}')
    return ok({"ready": True, "version": settings.version})


@router.get("/metrics")
def metrics_endpoint():
    return Response(content=metrics_mod.prometheus_text(), media_type="text/plain; version=0.0.4")


@router.get("/metrics/json")
def metrics_json():
    return ok(metrics_mod.snapshot())


@router.get("/meta")
def meta(db: Session = Depends(get_db)):
    from recon.depth import midas_available
    from recon.runtime import runtime_status
    from recon.triposr import available as triposr_available

    return ok({
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "api_version": "v1",
        "inference": runtime_status(),
        "features": {
            "registration": settings.allow_registration,
            "billing": "stripe" if settings.stripe_secret_key else "development",
            "email": settings.email_provider,
            "storage": settings.storage_driver,
            "provider": settings.generation_provider,
            "triposr": triposr_available(),
            "midas": midas_available(),
            "demo_account": settings.environment == "development" and db.scalar(
                select(User.id).where(User.email == "demo@3dfun.dev", User.is_active.is_(True))
            ) is not None,
        },
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "rate_limit_per_minute": settings.rate_limit_max,
            "job_timeout_seconds": settings.job_timeout_seconds,
        },
        "credits": settings.credit_costs,
    })


@router.get("/openapi.json", include_in_schema=False)
def openapi_json():
    from ..main import app
    return app.openapi()


# --------------------------------------------------------------------------
# benchmark results (read-only view over results/, produced by make benchmark)
# --------------------------------------------------------------------------
@router.get("/results/summary")
def results_summary():
    """Serve the benchmark summary the paper and the deck are generated from."""
    import csv
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "results"
    payload: dict = {"available": False, "root": str(root)}
    table = root / "summary" / "main_table.csv"
    if table.exists():
        with open(table, newline="", encoding="utf-8") as fh:
            payload["methods"] = list(csv.DictReader(fh))
        payload["available"] = True
    ablation = root / "summary" / "ablation_table.csv"
    if ablation.exists():
        with open(ablation, newline="", encoding="utf-8") as fh:
            payload["ablation"] = list(csv.DictReader(fh))
    timings = root / "summary" / "stage_timings.csv"
    if timings.exists():
        with open(timings, newline="", encoding="utf-8") as fh:
            payload["stage_timings"] = list(csv.DictReader(fh))
    failures = root / "summary" / "failures.csv"
    if failures.exists():
        with open(failures, newline="", encoding="utf-8") as fh:
            payload["failures"] = list(csv.DictReader(fh))
    env_file = root / "environment.json"
    if env_file.exists():
        payload["environment"] = json.loads(env_file.read_text(encoding="utf-8"))
    dataset = root.parent / "benchmarks" / "data" / "dataset.json"
    if dataset.exists():
        data = json.loads(dataset.read_text(encoding="utf-8"))
        payload["dataset"] = {"version": data["version"], "scenes": data["scenes"],
                              "checksum": data["checksum"], "seed": data["seed"]}
    return ok(payload)
