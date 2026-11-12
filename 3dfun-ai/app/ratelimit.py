"""Rate limiting (token bucket, in-process; Redis-ready interface)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .config import settings
from .schemas import ApiError


@dataclass
class Bucket:
    tokens: float
    updated: float
    capacity: float


class RateLimiter:
    def __init__(self, capacity: int, window_seconds: int):
        self.capacity = float(capacity)
        self.refill_rate = float(capacity) / max(window_seconds, 1)
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = Bucket(tokens=self.capacity, updated=now, capacity=self.capacity)
                self._buckets[key] = bucket
            elapsed = now - bucket.updated
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
            bucket.updated = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, bucket.tokens
            return False, bucket.tokens

    def retry_after(self, key: str) -> int:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return 1
            deficit = max(0.0, 1.0 - bucket.tokens)
            return max(1, int(deficit / max(self.refill_rate, 1e-6)))

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


limiter = RateLimiter(settings.rate_limit_max, settings.rate_limit_window)
auth_limiter = RateLimiter(10, 60)          # 10 auth attempts / minute / IP
upload_limiter = RateLimiter(30, 60)
generation_limiter = RateLimiter(12, 60)


def enforce(l: RateLimiter, key: str, scope: str, cost: float = 1.0) -> None:
    allowed, remaining = l.check(key, cost)
    if not allowed:
        raise ApiError("RATE_LIMITED",
                       f"Too many {scope} requests. Try again shortly.",
                       status=429, retryable=True,
                       details=[{"retryAfterSeconds": l.retry_after(key)}])


def login_locked(db_user) -> bool:
    from .db import utcnow
    if db_user.locked_until is None:
        return False
    locked_until = db_user.locked_until
    if locked_until.tzinfo is None:
        import datetime as dt
        locked_until = locked_until.replace(tzinfo=dt.timezone.utc)
    return locked_until > utcnow()
