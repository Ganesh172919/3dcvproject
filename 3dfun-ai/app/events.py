"""Event bus for real-time generation progress (SSE) with replay support.

In-process pub/sub by default (single-node deployments and the dev server). The
same interface is used by a Redis pub/sub implementation when REDIS_URL is set,
so horizontal scaling only requires swapping the backend.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import GenerationEvent, utcnow


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, generation_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subs.get(generation_id, []))
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:               # pragma: no cover - slow consumer
                pass

    def subscribe(self, generation_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs[generation_id].append(q)
        return q

    def unsubscribe(self, generation_id: str, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs.get(generation_id, []):
                self._subs[generation_id].remove(q)
            if not self._subs.get(generation_id):
                self._subs.pop(generation_id, None)

    def subscriber_count(self, generation_id: str) -> int:
        with self._lock:
            return len(self._subs.get(generation_id, []))


bus = EventBus()


def record_event(db: Session, generation_id: str, event_type: str, stage: str,
                 progress: int, payload: dict[str, Any] | None = None) -> GenerationEvent:
    """Persist an event (for replay) and publish it to live subscribers."""
    last = db.execute(
        select(GenerationEvent).where(GenerationEvent.generation_id == generation_id)
        .order_by(GenerationEvent.sequence.desc()).limit(1)
    ).scalar_one_or_none()
    seq = (last.sequence + 1) if last else 1
    evt = GenerationEvent(generation_id=generation_id, sequence=seq, type=event_type,
                          stage=stage, progress=progress, payload=payload or {},
                          occurred_at=utcnow())
    db.add(evt)
    db.commit()
    db.refresh(evt)
    bus.publish(generation_id, serialize(evt))
    return evt


def serialize(evt: GenerationEvent) -> dict[str, Any]:
    return {
        "id": evt.id,
        "type": evt.type,
        "generationId": evt.generation_id,
        "sequence": evt.sequence,
        "stage": evt.stage,
        "progress": evt.progress,
        "occurredAt": evt.occurred_at.isoformat(),
        "data": evt.payload or {},
    }


def replay(db: Session, generation_id: str, after_sequence: int = 0,
           limit: int = 500) -> list[dict[str, Any]]:
    rows = db.execute(
        select(GenerationEvent)
        .where(GenerationEvent.generation_id == generation_id,
               GenerationEvent.sequence > after_sequence)
        .order_by(GenerationEvent.sequence.asc()).limit(limit)
    ).scalars().all()
    return [serialize(r) for r in rows]


def sse_stream(db: Session, generation_id: str, after_sequence: int = 0,
               heartbeat_seconds: float = 15.0, max_seconds: float = 900.0) -> Iterator[str]:
    """Yield SSE frames: history replay, then live events with heartbeats."""
    started = time.time()
    last_seq = after_sequence

    def frame(event: dict[str, Any]) -> str:
        return (f"id: {event['sequence']}\n"
                f"event: {event['type']}\n"
                f"data: {json.dumps(event)}\n\n")

    for event in replay(db, generation_id, after_sequence):
        last_seq = max(last_seq, event["sequence"])
        yield frame(event)

    q = bus.subscribe(generation_id)
    last_emit = time.time()
    try:
        while time.time() - started < max_seconds:
            try:
                event = q.get(timeout=1.0)
                if event["sequence"] > last_seq:
                    last_seq = event["sequence"]
                    yield frame(event)
                    last_emit = time.time()
                    if event["type"] in ("generation.completed", "generation.failed",
                                         "generation.cancelled"):
                        return
            except queue.Empty:
                if time.time() - last_emit > heartbeat_seconds:
                    last_emit = time.time()
                    yield ": heartbeat\n\n"
                # safety net: pick up events written by another process
                for event in replay(db, generation_id, last_seq, limit=50):
                    last_seq = event["sequence"]
                    yield frame(event)
    finally:
        bus.unsubscribe(generation_id, q)


def backend_name() -> str:
    return "redis" if settings.redis_url else "in-process"
