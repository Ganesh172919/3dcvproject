#!/usr/bin/env python3
"""Verify that every dependency the project imports is actually installed.

    python3 scripts/verify_install.py          # report
    python3 scripts/verify_install.py --json   # machine readable

Exit code 0 when everything required imports, 1 otherwise. Optional
dependencies are reported but never fail the check.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

REQUIRED = [
    ("fastapi", "web framework"),
    ("uvicorn", "ASGI server"),
    ("pydantic", "request validation"),
    ("multipart", "multipart uploads"),
    ("sqlalchemy", "ORM"),
    ("argon2", "password hashing"),
    ("jwt", "JWT tokens"),
    ("numpy", "numerics"),
    ("scipy", "numerics"),
    ("cv2", "image IO"),
    ("onnxruntime", "depth model runtime"),
    ("trimesh", "mesh IO / validation"),
    ("skimage", "marching cubes"),
    ("PIL", "image encoding"),
]

OPTIONAL = [
    ("pptx", "PowerPoint generation"),
    ("reportlab", "PDF generation"),
    ("matplotlib", "benchmark charts"),
    ("httpx", "integration tests"),
    ("pytest", "test runner"),
    ("playwright", "browser smoke test"),
    ("fast_simplification", "optional decimation"),
    ("open3d", "optional mesh ops"),
]

MODEL_FILES = [
    ("engine/models/depth_anything_v2_small_quantized.onnx", 27_000_000),
    ("engine/models/u2netp.onnx", 4_500_000),
]


def check(modules):
    rows = []
    for name, purpose in modules:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            rows.append({"module": name, "ok": True, "version": str(version), "purpose": purpose})
        except Exception as exc:                     # noqa: BLE001
            rows.append({"module": name, "ok": False, "error": str(exc), "purpose": purpose})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    required = check(REQUIRED)
    optional = check(OPTIONAL)

    missing_models = []
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for rel, minimum in MODEL_FILES:
        path = root / rel
        size = path.stat().st_size if path.exists() else 0
        if size < minimum:
            missing_models.append({"path": rel, "size": size, "expected_at_least": minimum})

    if args.json:
        print(json.dumps({"required": required, "optional": optional,
                          "models": missing_models,
                          "ok": all(r["ok"] for r in required) and not missing_models},
                         indent=2))
    else:
        print("3DFUN AI dependency check\n")
        for row in required:
            mark = "ok  " if row["ok"] else "MISS"
            print(f"  [{mark}] {row['module']:<20} {row.get('version', '') or row.get('error', '')}")
        print("\n  optional:")
        for row in optional:
            mark = "ok  " if row["ok"] else "--  "
            print(f"  [{mark}] {row['module']:<20} {row.get('version', '') or row.get('error', '')}")
        print("\n  model files:")
        if not missing_models:
            print("  [ok  ] depth + matte ONNX weights present")
        for row in missing_models:
            print(f"  [MISS] {row['path']} ({row['size']} bytes)")

    return 0 if all(r["ok"] for r in required) and not missing_models else 1


if __name__ == "__main__":
    sys.exit(main())
