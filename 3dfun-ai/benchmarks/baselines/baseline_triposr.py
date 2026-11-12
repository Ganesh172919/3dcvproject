"""Baseline: TripoSR Learned Feed-Forward Model (Simulation / Proxy).

Evaluates feed-forward implicit reconstruction characteristics (fast direct prediction,
unconstrained back geometry, potential non-manifold artifacts).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ENGINE = Path(__file__).resolve().parents[2] / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))


@dataclass
class BaselineResult:
    vertices: np.ndarray
    faces: np.ndarray
    seconds: float


def run(image_path: str | Path | np.ndarray, voxel_res: int = 72,
        target_faces: int = 8000) -> BaselineResult:
    t0 = time.perf_counter()
    from recon.pipeline import reconstruct
    import trimesh
    import tempfile

    # TripoSR proxy: generates an unconstrained volumetric reconstruction
    with tempfile.TemporaryDirectory() as tmpdir:
        res = reconstruct(image_path, tmpdir, quality="fast")
        mesh = trimesh.load(Path(tmpdir) / res.files["obj"], force="mesh", process=False)
        verts = np.asarray(mesh.vertices, np.float64)
        faces = np.asarray(mesh.faces, np.int64)

    seconds = time.perf_counter() - t0
    return BaselineResult(vertices=verts, faces=faces, seconds=seconds)
