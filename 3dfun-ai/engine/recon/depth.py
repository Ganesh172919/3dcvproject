"""Stage 2 - monocular relative depth with Depth Anything V2 or MiDaS (ONNX).

The model outputs affine-invariant inverse depth (larger = closer). We convert it
into a normalised *closeness* field in [0, 1] on the working canvas, which the
volumetric stage turns into per-pixel object thickness.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import MODELS_DIR, PipelineConfig
from .runtime import CPU, InferenceSession, runtime_status

log = logging.getLogger(__name__)

_SESSION_CACHE: dict[str, Any] = {}
MIDAS_ID = "midas-v2.1-small"
MIDAS_FILE = "midas_v21_small.onnx"
MIDAS_BYTES = 66764249
DEPTH_ANYTHING_ID = "depth-anything-v2-small-int8"
DEPTH_MODELS = {
    DEPTH_ANYTHING_ID: ("depth_anything_v2_small_quantized.onnx", "Depth Anything V2 Small"),
    MIDAS_ID: (MIDAS_FILE, "MiDaS v2.1 Small"),
}


def midas_available() -> bool:
    path = MODELS_DIR / MIDAS_FILE
    return path.is_file() and path.stat().st_size == MIDAS_BYTES


@dataclass
class DepthResult:
    closeness: np.ndarray      # (S, S) float32 in [0, 1], 1 = nearest
    raw_inverse: np.ndarray    # (S, S) float32 model output
    stats: dict[str, Any]


def _load_session(model_file: str):
    import onnxruntime as ort

    preferred = runtime_status()["preferred"]
    if preferred == CPU and model_file in _SESSION_CACHE:
        return _SESSION_CACHE[model_file]
    path = MODELS_DIR / model_file
    if not path.exists():
        raise FileNotFoundError(f"Depth model not found: {path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(path, so, preferred)
    if preferred == CPU:
        _SESSION_CACHE[model_file] = sess
    return sess


def _infer(sess, rgb: np.ndarray, model: str = DEPTH_ANYTHING_ID) -> np.ndarray:
    if model == MIDAS_ID:
        # The published MiDaS ONNX export takes 256px RGB scaled to [0, 1].
        th = tw = 256
    else:
        h, w = rgb.shape[:2]
        # Depth Anything V2 ViT-S/14 requires H and W to be multiples of 14.
        th = max(14, int(round(cfg_size(h) / 14)) * 14)
        tw = max(14, int(round(cfg_size(w) / 14)) * 14)
    img = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    if model != MIDAS_ID:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
    inp = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0]
    pred = np.asarray(out).squeeze().astype(np.float32)
    return pred


def cfg_size(v: int) -> int:
    """Target inference side length keeping the aspect ratio sane."""
    return int(min(518, max(266, v)))


def estimate_depth(rgb: np.ndarray, mask: np.ndarray, cfg: PipelineConfig) -> DepthResult:
    """Run monocular depth estimation and normalise inside the object mask."""
    model_file, _ = DEPTH_MODELS[cfg.depth_model]
    sess = _load_session(model_file)
    raw = _infer(sess, rgb, cfg.depth_model)
    S = rgb.shape[0]
    inv = cv2.resize(raw, (S, S), interpolation=cv2.INTER_CUBIC)

    vals = inv[mask]
    if vals.size < 32:
        raise ValueError("Object mask too small for depth estimation.")
    lo = float(np.percentile(vals, cfg.depth_percentile_low))
    hi = float(np.percentile(vals, cfg.depth_percentile_high))
    if hi - lo < 1e-6:
        raise ValueError("Depth map is flat; the input has no usable depth cue.")
    closeness = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)

    if cfg.depth_smooth_sigma > 0:
        closeness = cv2.GaussianBlur(closeness, (0, 0), cfg.depth_smooth_sigma)

    if cfg.height_symmetrize:
        mirrored = closeness[:, ::-1]
        w = np.clip(np.abs(closeness - mirrored) * 4.0, 0.0, 0.55)
        closeness = closeness * (1 - w) + mirrored * w
        closeness = np.clip(closeness, 0.0, 1.0)

    # Re-normalise strictly inside the mask, then extrapolate smoothly into the
    # body so that the silhouette rim carries sane thickness values.
    inside = closeness[mask]
    lo2, hi2 = float(inside.min()), float(inside.max())
    norm = (closeness - lo2) / max(hi2 - lo2, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    norm = np.where(mask, norm, 0.0).astype(np.float32)
    filled = _fill_outside(norm, mask)
    return DepthResult(
        closeness=filled.astype(np.float32),
        raw_inverse=inv.astype(np.float32),
        stats={
            "provider": sess.provider,
            "raw_min": float(vals.min()),
            "raw_max": float(vals.max()),
            "p_low": lo,
            "p_high": hi,
            "mask_mean_closeness": float(inside.mean()),
            "mask_std_closeness": float(inside.std()),
        },
    )


def _fill_outside(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Propagate in-mask depth values into the background band (for the rim)."""
    out = field.copy()
    out[~mask] = 0.0
    k = 9
    num = cv2.GaussianBlur(out * mask.astype(np.float32), (k, k), 0)
    den = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    sm = np.divide(num, np.maximum(den, 1e-4))
    out[~mask] = sm[~mask]
    return np.clip(out, 0.0, 1.0)
