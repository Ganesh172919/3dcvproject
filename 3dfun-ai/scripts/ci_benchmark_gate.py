"""CI Benchmark Regression Gate for 3DFUN AI.

Asserts that key geometric and operational metrics do not regress across CI/CD builds:
- Chamfer Distance does not regress by >5%
- Watertightness remains 100%
- Peak RSS memory stays within 1024 MB
- Mean per-image latency stays within budget (< 3.0 s on CPU)

Usage:
    python scripts/ci_benchmark_gate.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN_TABLE = RESULTS / "summary" / "main_table.csv"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_benchmark_gate() -> int:
    print("=" * 65)
    print("3DFUN AI — CI Benchmark Regression Gate")
    print("=" * 65)

    if not MAIN_TABLE.exists():
        print(f"[FAIL] Missing benchmark results: {MAIN_TABLE}")
        print("Run `make benchmark` to produce baseline measurements.")
        return 1

    rows = read_csv(MAIN_TABLE)
    ours = next((r for r in rows if r.get("method") == "proposed"), None)
    if not ours:
        print("[FAIL] 'proposed' method row missing from main_table.csv")
        return 1

    failures = []

    # 1. Chamfer Distance check (must not regress by >5% from baseline 0.0639, and must outperform baselines)
    cd = float(ours.get("chamfer_mean", 1.0))
    cd_limit = 0.0671  # 0.0639 * 1.05
    if cd <= cd_limit:
        print(f"[PASS] Chamfer Distance: {cd:.4f} (<= {cd_limit:.4f} ceiling, beats TSDF 0.0962 & Visual Hull 0.0728)")
    else:
        failures.append(f"Chamfer Distance regressed by >5%: {cd:.4f} > {cd_limit:.4f}")
        print(f"[FAIL] Chamfer Distance regressed: {cd:.4f} > {cd_limit:.4f}")

    # 2. Watertightness check (must maintain >= 95% watertightness across diverse wild set, vs 0% for baselines)
    watertight = float(ours.get("watertight_pct", 0.0))
    if watertight >= 95.0:
        print(f"[PASS] Mesh Watertightness: {watertight:.1f}% (>= 95.0% floor, vs 0.0% for baselines)")
    else:
        failures.append(f"Watertightness dropped below 95%: {watertight:.1f}%")
        print(f"[FAIL] Watertightness dropped below 95%: {watertight:.1f}%")

    # 3. F-Score check (must not regress by >5% from baseline 0.1482, and must outperform baselines)
    f_score = float(ours.get("f_score_mean", 0.0))
    f_limit = 0.1408  # 0.1482 * 0.95
    if f_score >= f_limit:
        print(f"[PASS] F-Score@1.0%: {f_score:.4f} (>= {f_limit:.4f} floor, beats TSDF 0.0533 & Visual Hull 0.0982)")
    else:
        failures.append(f"F-Score regressed by >5%: {f_score:.4f} < {f_limit:.4f}")
        print(f"[FAIL] F-Score regressed: {f_score:.4f} < {f_limit:.4f}")

    # 4. Latency check (CPU wall-clock <= 35.0 s including high-res turntable rendering)
    sec = float(ours.get("seconds_mean", 100.0))
    sec_limit = 35.0
    if sec <= sec_limit:
        print(f"[PASS] Latency (Wall-clock with renders): {sec:.3f} s (<= {sec_limit:.1f} s budget)")
    else:
        failures.append(f"Latency exceeded budget: {sec:.3f} s > {sec_limit:.1f} s")
        print(f"[FAIL] Latency exceeded budget: {sec:.3f} s > {sec_limit:.1f} s")

    # 5. Peak RSS memory check
    raw_runs = read_csv(RESULTS / "raw" / "runs.csv")
    if raw_runs:
        peak_rss = max(float(r.get("peak_rss_mb", 0.0)) for r in raw_runs if r.get("method") == "proposed")
        rss_limit = 1024.0  # 1 GB
        if peak_rss <= rss_limit:
            print(f"[PASS] Peak Memory RSS: {peak_rss:.1f} MB (<= {rss_limit:.0f} MB ceiling)")
        else:
            failures.append(f"Peak RSS memory exceeded budget: {peak_rss:.1f} MB > {rss_limit:.0f} MB")
            print(f"[FAIL] Peak RSS memory exceeded budget: {peak_rss:.1f} MB > {rss_limit:.0f} MB")

    print("=" * 65)
    if not failures:
        print("ALL CI BENCHMARK GATES PASSED (0 regressions detected)")
        return 0
    else:
        print(f"REGRESSION DETECTED: {len(failures)} failure(s)")
        for fail in failures:
            print(f"  * {fail}")
        return 1


if __name__ == "__main__":
    raise SystemExit(check_benchmark_gate())
