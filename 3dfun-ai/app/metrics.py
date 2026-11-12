"""In-process metrics registry (Prometheus text format on /metrics).

Deliberately dependency-free so the API runs in constrained environments; the
same series names are compatible with a Prometheus scrape.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable

_LOCK = threading.Lock()
_COUNTERS: dict[str, float] = defaultdict(float)
_GAUGES: dict[str, float] = {}
_HISTOGRAMS: dict[str, list[float]] = defaultdict(list)
_TIMESERIES: dict[str, deque] = defaultdict(lambda: deque(maxlen=600))
_START = time.time()

_LATENCY_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    key = _key(name, labels)
    with _LOCK:
        _COUNTERS[key] += value


def observe(name: str, value: float, **labels: str) -> None:
    key = _key(name, labels)
    with _LOCK:
        bucket = _HISTOGRAMS[key]
        bucket.append(value)
        if len(bucket) > 5000:
            del bucket[:1000]


def gauge(name: str, value: float, **labels: str) -> None:
    key = _key(name, labels)
    with _LOCK:
        _GAUGES[key] = value


def record_timing(name: str, seconds: float, **labels: str) -> None:
    observe(name, seconds * 1000.0, **labels)
    with _LOCK:
        _TIMESERIES[name].append((time.time(), seconds * 1000.0))


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    body = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{body}}}"


def snapshot() -> dict:
    with _LOCK:
        counters = dict(_COUNTERS)
        gauges = dict(_GAUGES)
        hist = {}
        for key, values in _HISTOGRAMS.items():
            if not values:
                continue
            sv = sorted(values)
            n = len(sv)
            hist[key] = {
                "count": n,
                "p50": round(sv[int(n * 0.5)], 2),
                "p90": round(sv[min(int(n * 0.9), n - 1)], 2),
                "p95": round(sv[min(int(n * 0.95), n - 1)], 2),
                "p99": round(sv[min(int(n * 0.99), n - 1)], 2),
                "max": round(sv[-1], 2),
                "mean": round(sum(sv) / n, 2),
            }
    return {"counters": counters, "gauges": gauges, "histograms": hist,
            "uptime_seconds": round(time.time() - _START, 1)}


def prometheus_text() -> str:
    snap = snapshot()
    lines: list[str] = [f"# 3dfun_ai metrics, uptime {snap['uptime_seconds']}s"]
    for key, val in sorted(snap["counters"].items()):
        lines.append(f"# TYPE {key.split('{')[0]} counter")
        lines.append(f"{key} {val:g}")
    for key, val in sorted(snap["gauges"].items()):
        lines.append(f"# TYPE {key.split('{')[0]} gauge")
        lines.append(f"{key} {val:g}")
    for key, stats in sorted(snap["histograms"].items()):
        base = key.split("{")[0]
        labels = key[len(base):]
        for b in _LATENCY_BUCKETS:
            inner = labels[1:-1] + ("," if labels else "") + f'le="{b}"'
            lines.append(f"{base}_bucket{{{inner}}} {sum(1 for _ in [stats]) if stats['max'] <= b else 0}")
        lines.append(f"# TYPE {base} summary")
        for q in ("p50", "p90", "p95", "p99"):
            inner = labels[1:-1] + ("," if labels else "") + f'quantile="0.{q[1:]}"'
            lines.append(f"{base}{{{inner}}} {stats[q]}")
        lines.append(f"{base}_count{labels} {stats['count']}")
    return "\n".join(lines) + "\n"


class Timer:
    def __init__(self, name: str, **labels: str):
        self.name = name
        self.labels = labels
        self.t0 = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        record_timing(self.name, time.perf_counter() - self.t0, **self.labels)


def timed(name: str, **labels: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with Timer(name, **labels):
                return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco
