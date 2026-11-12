"""Baseline A: TSDF fusion of a single monocular depth map.

A deliberately simple, fully classical reference pipeline used as a baseline in
the paper. Given one RGB image it

1. estimates relative depth with Depth Anything V2 (the same ONNX weights the
   proposed method uses, so the comparison isolates the reconstruction stage),
2. converts the relative depth to a metric-ish fronto-parallel slab,
3. integrates it into a truncated signed distance field,
4. extracts a mesh with marching cubes and Taubin smooths it.

Unlike the proposed pipeline the slab is not silhouette-constrained, so the
object leaks outside its outline; this is exactly the weakness the paper
reports.
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
    """Inverse-depth map from the shared ONNX model (no mask, no symmetry prior)."""
    from recon.depth import _infer, _load_session      # reuse the shipped weights

    if isinstance(image_path, np.ndarray):
        rgb = image_path if image_path.ndim == 3 else cv2.cvtColor(image_path, cv2.COLOR_GRAY2RGB)
    else:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sess = _load_session("depth_anything_v2_small_quantized.onnx")
    raw = _infer(sess, rgb)
    rel = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return rgb, rel.astype(np.float32)


def run(image_path: str | Path | np.ndarray, voxel_res: int = 72,
        target_faces: int = 8000, pad: float = 0.06) -> BaselineResult:
    t0 = time.perf_counter()
    rgb, rel = _depth_map(image_path)

    # relative depth -> [0,1] without any object knowledge
    lo, hi = np.percentile(rel, 2), np.percentile(rel, 98)
    d = np.clip((rel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    h, w = d.shape
    scale = max(h, w) / 1.0
    grid = voxel_res
    xs = (np.arange(grid) + 0.5) / grid - 0.5
    ys = (np.arange(grid) + 0.5) / grid - 0.5
    zs = (np.arange(grid // 2 + 4) + 0.5) / grid - 0.15
    # sample the depth map at every voxel of the slab: no mask, no carve
    px = np.clip(((xs[None, :] + 0.5) * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(((ys[:, None] + 0.5) * (h - 1)).astype(int), 0, h - 1)
    depth_img = d[py, px]                                     # (grid, grid)

    field = np.empty((len(zs), grid, grid), dtype=np.float32)
    half = 0.5 * (1.0 - 2 * pad)
    for k, z in enumerate(zs):
        surface = (depth_img - 0.5) * 2.0 * half           # depth drives the front sheet
        field[k] = (z - surface).astype(np.float32)

    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(-field, level=0.0, spacing=(
        1.0 / grid, 1.0 / grid, 1.0 / grid))
    verts = verts - 0.5
    if len(faces) > target_faces:
        try:
            from recon.meshing import decimate

            verts, faces, _info = decimate(verts, faces, target_faces)
        except Exception as exc:                             # pragma: no cover
            print(f"warning: decimation skipped ({exc})", file=sys.stderr)
    return BaselineResult(vertices=np.asarray(verts, dtype=np.float64),
                          faces=np.asarray(faces, dtype=np.int64),
                          seconds=time.perf_counter() - t0)


if __name__ == "__main__":                                   # manual smoke run
    out = run(sys.argv[1] if len(sys.argv) > 1 else ENGINE / "testdata" / "sneaker.png")
    print(f"baseline A: {len(out.faces)} faces in {out.seconds:.1f}s")
