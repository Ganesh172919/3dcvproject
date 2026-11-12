"""Baseline B: visual hull (shape-from-silhouette) with a depth-derived back plane.

The classical "space carving" reference for single-view reconstruction:

1. the object silhouette is matted (the same U²-Net weights the proposed method
   uses, so the comparison isolates the reconstruction stage),
2. the silhouette is back-projected from the reference camera, producing an
   extruded prism (the visual hull of one view),
3. the prism is intersected with a half-space cut derived from the depth map so
   it is not a full extrusion, and
4. the isosurface is extracted with marching cubes.

The hull cannot express concavities and produces flat, extruded sides — the
failure mode reported in the paper.
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


def _matte_and_depth(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from recon.depth import _infer, _load_session
    from recon.preprocessing import _clean_alpha, _grabcut_alpha, _u2net_alpha

    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    alpha = _u2net_alpha(rgb)
    if alpha is None:
        alpha = _grabcut_alpha(rgb)
    alpha = _clean_alpha(alpha, 1)
    sess = _load_session("depth_anything_v2_small_quantized.onnx")
    raw = _infer(sess, rgb)
    depth = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return (alpha > 0.5).astype(np.float32), depth.astype(np.float32)


def run(image_path: str | Path | np.ndarray, voxel_res: int = 72,
        target_faces: int = 8000) -> BaselineResult:
    t0 = time.perf_counter()
    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    mask, depth = _matte_and_depth(image)

    # normalise the silhouette to a square canvas (no centring assumption beyond bbox)
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size < 32:
        raise ValueError("silhouette too small")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.05 * max(x1 - x0, y1 - y0))
    crop_m = mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    crop_d = depth[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]

    side = max(crop_m.shape)
    canvas_m = np.zeros((side, side), np.float32)
    canvas_d = np.zeros((side, side), np.float32)
    oy, ox = (side - crop_m.shape[0]) // 2, (side - crop_m.shape[1]) // 2
    canvas_m[oy:oy + crop_m.shape[0], ox:ox + crop_m.shape[1]] = crop_m
    canvas_d[oy:oy + crop_d.shape[0], ox:ox + crop_d.shape[1]] = crop_d

    lo, hi = np.percentile(canvas_d[canvas_m > 0.5], 2), np.percentile(canvas_d[canvas_m > 0.5], 98)
    d = np.clip((canvas_d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    g = voxel_res
    z_steps = g // 2 + 4
    zs = (np.arange(z_steps) + 0.5) / g - 0.2
    # silhouette mask upsampled to the voxel grid, no depth shaping of the sides
    m_vox = cv2.resize(canvas_m, (g, g), interpolation=cv2.INTER_NEAREST)
    d_vox = cv2.resize(d, (g, g), interpolation=cv2.INTER_LINEAR)
    field = np.empty((z_steps, g, g), np.float32)
    for k, z in enumerate(zs):
        # inside the prism: z between the depth-driven front sheet and a flat back plane
        front = (0.5 - d_vox) * 0.42
        inside = (z > front) & (z < 0.42)
        field[k] = np.where(m_vox > 0.5, np.where(inside, -1.0, 1.0), 1.0)
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(-field, level=0.0,
                                               spacing=(1.0 / g, 1.0 / g, 1.0 / g))
    verts = verts - 0.5
    if len(faces) > target_faces:
        try:
            from recon.meshing import decimate

            verts, faces, _ = decimate(verts, faces, target_faces)
        except Exception as exc:                             # pragma: no cover
            print(f"warning: decimation skipped ({exc})", file=sys.stderr)
    return BaselineResult(np.asarray(verts, np.float64), np.asarray(faces, np.int64),
                          time.perf_counter() - t0)


if __name__ == "__main__":
    out = run(sys.argv[1] if len(sys.argv) > 1 else ENGINE / "testdata" / "sneaker.png")
    print(f"baseline B: {len(out.faces)} faces in {out.seconds:.1f}s")
