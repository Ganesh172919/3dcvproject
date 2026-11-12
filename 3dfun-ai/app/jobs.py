"""Generation job runner.

A bounded thread pool plays the role of a BullMQ/Celery queue in this
single-node deployment. The interface (enqueue -> worker -> provider -> asset
pipeline) is identical to a distributed deployment; `jobs.backend` reports the
active implementation so the docs and the admin UI never overstate it.

Job lifecycle
-------------
reserve credits (API) -> QUEUED -> VALIDATING -> ... -> COMPLETED
                                   \\-> FAILED (credits refunded, retryable flag)
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from . import credits, events, metrics, notifications
from .config import settings
from .db import (Asset, Generation, GenerationAttempt, GenerationStatus, Provider,
                 SessionLocal, Upload, UsageRecord, new_id, utcnow)
from . import providers as provider_registry
from .providers import GenerationInput
from .state_machine import TERMINAL, assert_transition, is_retryable, status_for_stage
from .storage import asset_prefix, content_type_for, storage

log = logging.getLogger(__name__)

QUEUES = ("generation-orchestration", "provider-dispatch", "asset-processing",
          "preview-rendering", "notification-delivery", "cleanup")

ASSET_EXTENSION_TYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".obj": "text/plain",
    ".mtl": "text/plain",
    ".ply": "application/octet-stream",
    ".stl": "model/stl",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
    ".gif": "image/gif",
    ".json": "application/json",
}
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024


@dataclass
class QueueStats:
    queued: int
    running: int
    completed_today: int
    failed_today: int


class JobRunner:
    def __init__(self, concurrency: int | None = None):
        self._pool = ThreadPoolExecutor(max_workers=concurrency or settings.job_concurrency,
                                        thread_name_prefix="recon")
        self._inflight: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.processed = 0
        self.failed = 0

    # -- public API ------------------------------------------------------
    def enqueue(self, generation_id: str) -> None:
        with self._lock:
            if generation_id in self._inflight:
                log.info("generation %s already in flight", generation_id)
                return
            fut = self._pool.submit(self._run_safe, generation_id)
            self._inflight[generation_id] = fut
        metrics.inc("generation_enqueued_total")

    def inflight(self) -> int:
        with self._lock:
            return len(self._inflight)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=False)

    # -- internals -------------------------------------------------------
    def _run_safe(self, generation_id: str) -> None:
        try:
            self._run(generation_id)
            self.processed += 1
        except Exception:
            self.failed += 1
            log.exception("unhandled worker error for %s", generation_id)
            with SessionLocal() as db:
                generation = db.get(Generation, generation_id)
                if generation is not None and generation.status not in TERMINAL:
                    self._fail(db, generation, "ENGINE_ERROR", "The worker could not finish this run.", True)
        finally:
            with self._lock:
                self._inflight.pop(generation_id, None)

    def _update(self, db, gen: Generation, status: GenerationStatus | None = None,
                stage: str | None = None, progress: int | None = None) -> None:
        if status is not None and status != gen.status:
            try:
                assert_transition(gen.status, status)
                gen.status = status
            except Exception:
                log.warning("skipping illegal transition %s -> %s", gen.status, status)
        if stage:
            gen.stage = stage
        if progress is not None:
            gen.progress = max(gen.progress, min(99, progress))
        db.commit()

    def _run(self, generation_id: str) -> None:
        db = SessionLocal()
        started = time.perf_counter()
        workdir = None
        try:
            gen = db.get(Generation, generation_id)
            if gen is None:
                log.error("generation %s vanished", generation_id)
                return
            if gen.status == GenerationStatus.CANCELLED:
                return
            if gen.status == GenerationStatus.CANCEL_REQUESTED:
                self._finish_cancelled(db, gen)
                return

            upload = db.get(Upload, gen.upload_id) if gen.upload_id else None
            if upload is None:
                self._fail(db, gen, "UPLOAD_MISSING", "The source upload is missing.", False)
                return
            source = storage.read_bytes(upload.storage_key)

            self._update(db, gen, GenerationStatus.VALIDATING, "VALIDATING", 2)
            gen.started_at = utcnow()
            db.commit()
            events.record_event(db, gen.id, "generation.started", "VALIDATING", 2)

            attempt_no = (db.execute(
                select(GenerationAttempt).where(GenerationAttempt.generation_id == gen.id)
            ).scalars().all().__len__()) + 1
            attempt = GenerationAttempt(generation_id=gen.id, attempt_no=attempt_no,
                                       provider_slug=gen.provider_slug)
            db.add(attempt)
            db.commit()

            workdir = Path(tempfile.mkdtemp(prefix=f"3dfun-{gen.id}-"))
            provider = provider_registry.get_provider(gen.provider_slug)

            def on_progress(stage: str, pct: int, message: str) -> None:
                if stage == "COMPLETED":
                    return  # Completion is published once the asset and charges are saved.
                db2 = SessionLocal()
                try:
                    g = db2.get(Generation, gen.id)
                    if g is None or g.cancel_requested:
                        return
                    self._update(db2, g, status_for_stage(stage), stage, pct)
                    g.progress = min(99, max(g.progress, pct))
                    db2.commit()
                    events.record_event(db2, g.id, "generation.progress_updated", stage, pct,
                                        {"message": message})
                finally:
                    db2.close()

            payload = GenerationInput(
                generation_id=gen.id, quality=gen.quality, options=gen.options or {},
                source_image=source, workdir=workdir, progress=on_progress,
                cancelled=lambda: _is_cancelled(gen.id))

            metrics.inc("generation_started_total", provider=provider.slug)
            t0 = time.perf_counter()
            if hasattr(provider, "run_blocking"):
                status = provider.run_blocking(payload)     # local / mock
            else:
                status = self._run_remote(provider, payload)  # http
            provider_ms = (time.perf_counter() - t0) * 1000

            db.refresh(gen)
            if status.state == "cancelled" or gen.cancel_requested:
                attempt.status = "cancelled"
                attempt.finished_at = utcnow()
                attempt.duration_ms = int(provider_ms)
                self._finish_cancelled(db, gen)
                return

            if status.state == "failed":
                attempt.status = "failed"
                attempt.error_code = status.error_code
                attempt.error_message = status.error_message
                attempt.finished_at = utcnow()
                attempt.duration_ms = int(provider_ms)
                db.commit()
                self._fail(db, gen, status.error_code or "PROVIDER_FAILED",
                           status.error_message or "Generation failed.",
                           status.retryable)
                return

            # ---- asset ingestion -----------------------------------------
            self._update(db, gen, progress=96)
            asset = self._ingest_assets(db, gen, Path(status.artifact_dir or workdir),
                                        status.manifest or {})

            attempt.status = "succeeded"
            attempt.metrics = {"provider_ms": round(provider_ms, 1)}
            attempt.finished_at = utcnow()
            attempt.duration_ms = int(provider_ms)
            gen.provider_job_id = gen.provider_job_id or f"{provider.slug}:{gen.id}"
            db.commit()

            credits.spend(db, gen.user_id, gen)           # commit the reservation
            gen.status = GenerationStatus.COMPLETED
            gen.stage = "COMPLETED"
            gen.progress = 100
            gen.completed_at = utcnow()
            gen.duration_ms = int((time.perf_counter() - started) * 1000)
            db.add(UsageRecord(user_id=gen.user_id, generation_id=gen.id,
                               provider_slug=gen.provider_slug, stage="total",
                               duration_ms=gen.duration_ms, credits=gen.credits_charged,
                               storage_bytes=asset.size_bytes))
            db.commit()
            events.record_event(db, gen.id, "generation.completed", "COMPLETED", 100,
                                {"assetId": asset.id, "durationMs": gen.duration_ms})
            notifications.notify_generation_complete(db, gen, asset)
            metrics.observe("generation_duration_ms", gen.duration_ms, quality=gen.quality)
        finally:
            db.close()
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)

    def _run_remote(self, provider, payload: GenerationInput):
        """Poll an HTTP provider until it finishes or times out."""
        job = provider.create_generation(payload)
        deadline = time.time() + settings.job_timeout_seconds
        while time.time() < deadline:
            status = provider.get_generation(job.provider_job_id)
            payload.progress(status.stage or "PROCESSING", status.progress, "provider update")
            if status.state in ("succeeded", "failed", "cancelled"):
                return status
            time.sleep(3)
        return type(status)(state="failed", error_code="TIMEOUT",
                            error_message="Provider did not finish in time.", retryable=True)

    def _ingest_assets(self, db, gen: Generation, outdir: Path, manifest: dict[str, Any]) -> Asset:
        keep_suffixes = {".glb", ".gltf", ".bin", ".obj", ".mtl", ".ply", ".stl",
                         ".png", ".jpg", ".jpeg", ".mp4", ".gif", ".json"}
        files: dict[str, str] = {}
        total_bytes = 0
        asset_id = new_id("ast")
        prefix = asset_prefix(gen.user_id, gen.id, asset_id)
        try:
            for path in sorted(outdir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in keep_suffixes:
                    continue
                size = path.stat().st_size
                if size > MAX_ARTIFACT_BYTES:
                    log.warning("skipping oversized artifact %s (%d bytes)", path, size)
                    continue
                rel = path.relative_to(outdir).as_posix()
                key = f"{prefix}/{rel}"
                ctype = ASSET_EXTENSION_TYPES.get(path.suffix.lower(), content_type_for(path.name))
                storage.put_file(key, path, ctype)
                files[rel] = key
                total_bytes += size
            if not any(name in files for name in ("model.glb", "model.gltf", "model.obj", "model.ply", "model.stl")):
                raise RuntimeError("Reconstruction did not produce a mesh file.")
        except Exception:
            storage.delete_prefix(prefix)
            raise

        mesh = manifest.get("mesh", {}) or {}
        asset = Asset(
            id=asset_id,
            user_id=gen.user_id,
            generation_id=gen.id,
            name=f"{gen.preset.replace('_', ' ').title()} · {gen.id[-6:]}",
            description=gen.prompt[:2000],
            tags=[gen.preset, gen.quality],
            storage_prefix=prefix,
            files={rel: rel for rel in files},
            manifest=manifest,
            metrics=mesh,
            preview_key=files.get("preview.png"),
            cover_key=files.get("preview.png"),
            turntable_key=files.get("turntable.mp4") or files.get("turntable.gif"),
            polygons=int(mesh.get("faces") or 0),
            vertices=int(mesh.get("vertices") or 0),
            watertight=bool(mesh.get("watertight_trimesh") or mesh.get("watertight") or False),
            size_bytes=total_bytes,
        )
        db.add(asset)
        asset._file_keys = files        # type: ignore[attr-defined]
        return asset

    def _finish_cancelled(self, db, gen: Generation) -> None:
        credits.release(db, gen.user_id, gen)
        gen.status = GenerationStatus.CANCELLED
        gen.stage = "CANCELLED"
        gen.completed_at = utcnow()
        db.commit()
        events.record_event(db, gen.id, "generation.cancelled", "CANCELLED", gen.progress)

    def _fail(self, db, gen: Generation, code: str, message: str, retryable: bool) -> None:
        credits.refund_failed(db, gen.user_id, gen)
        credits.release(db, gen.user_id, gen)
        gen.status = GenerationStatus.FAILED
        gen.stage = "FAILED"
        gen.error_code = code
        gen.error_message = message
        gen.support_code = gen.id[-8:].upper()
        gen.completed_at = utcnow()
        db.commit()
        events.record_event(db, gen.id, "generation.failed", "FAILED", gen.progress,
                            {"code": code, "message": message,
                             "retryable": retryable or is_retryable(code),
                             "supportCode": gen.support_code})
        notifications.notify_generation_failed(db, gen)
        metrics.inc("generation_failed_total", code=code)


def _is_cancelled(generation_id: str) -> bool:
    db = SessionLocal()
    try:
        gen = db.get(Generation, generation_id)
        return bool(gen and gen.cancel_requested)
    finally:
        db.close()


runner = JobRunner()


def queue_stats(db) -> QueueStats:
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = db.execute(select(Generation)).scalars().all()
    queued = sum(1 for g in rows if g.status == GenerationStatus.QUEUED)
    running = sum(1 for g in rows if g.status in (
        GenerationStatus.VALIDATING, GenerationStatus.PROCESSING,
        GenerationStatus.GENERATING_GEOMETRY, GenerationStatus.GENERATING_TEXTURE,
        GenerationStatus.OPTIMIZING, GenerationStatus.RENDERING_PREVIEW))
    completed_today = sum(1 for g in rows if g.status == GenerationStatus.COMPLETED
                          and _aware(g.completed_at) >= today)
    failed_today = sum(1 for g in rows if g.status == GenerationStatus.FAILED
                       and _aware(g.completed_at) >= today)
    return QueueStats(queued, running, completed_today, failed_today)


def _aware(value):
    import datetime as dt
    if value is None:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value


def backend() -> dict[str, Any]:
    return {
        "kind": "thread-pool",
        "workers": settings.job_concurrency,
        "queues": list(QUEUES),
        "redis_configured": bool(settings.redis_url),
        "note": ("single-node runner; set REDIS_URL and run workers/queue_worker.py "
                 "for horizontal scaling"),
    }
