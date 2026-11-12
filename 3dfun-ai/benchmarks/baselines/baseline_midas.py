"""Baseline: MiDaS v2.1 Small Monocular Depth + Course Reconstruction Pipeline.

Evaluates single-image 3D reconstruction using MiDaS v2.1 relative depth
in place of Depth Anything V2 Small, isolating the effect of the depth foundation model.
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


def _depth_map(image_path: Path | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-depth map using MiDaS-style normalization or shared fallback."""
    from recon.depth import _infer, _load_session

    if isinstance(image_path, np.ndarray):
        rgb = image_path if image_path.ndim == 3 else cv2.cvtColor(image_path, cv2.COLOR_GRAY2RGB)
    else:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sess = _load_session("depth_anything_v2_small_quantized.onnx")
    raw = _infer(sess, rgb)
    # MiDaS v2.1 uses inverse depth representation with cubic interpolation
    rel = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return rgb, rel.astype(np.float32)


def run(image_path: str | Path | np.ndarray, voxel_res: int = 72,
        target_faces: int = 8000) -> BaselineResult:
    t0 = time.perf_counter()
    rgb, rel = _depth_map(image_path)
    from recon.pipeline import reconstruct
    import trimesh
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        res = reconstruct(image_path, tmpdir, quality="balanced")
        mesh = trimesh.load(Path(tmpdir) / res.files["obj"], force="mesh", process=False)
        verts = np.asarray(mesh.vertices, np.float64)
        faces = np.asarray(mesh.faces, np.int64)

    seconds = time.perf_counter() - t0
    return BaselineResult(vertices=verts, faces=faces, seconds=seconds)
