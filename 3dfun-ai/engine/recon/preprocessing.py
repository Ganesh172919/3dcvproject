"""Stage 1 - image ingestion, background matting and normalisation.

Produces a square, object-centred RGBA working image plus a binary mask. All
downstream stages operate on this canonical representation.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import MODELS_DIR, PipelineConfig
from .runtime import InferenceSession

log = logging.getLogger(__name__)

IMAGE_MAGIC: dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"RIFF": "image/webp",          # refined below (WEBP has RIFF....WEBP)
    b"GIF8": "image/gif",
    b"BM": "image/bmp",
}
ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp", "image/bmp"}


def sniff_mime(data: bytes) -> str | None:
    """Content-sniff an image buffer (never trust the client-provided MIME)."""
    for magic, mime in IMAGE_MAGIC.items():
        if data.startswith(magic):
            if magic == b"RIFF":
                return "image/webp" if data[8:12] == b"WEBP" else None
            return mime
    return None


def validate_image_bytes(data: bytes, max_mb: int = 20) -> str:
    """Raise ValueError when the payload is not an accepted image."""
    if not data:
        raise ValueError("Uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"Image exceeds the {max_mb} MB limit.")
    mime = sniff_mime(data)
    if mime not in ALLOWED_MIME:
        raise ValueError("Unsupported image type. Use PNG, JPEG, WEBP or BMP.")
    return mime


@dataclass
class PreparedImage:
    rgb: np.ndarray           # (S, S, 3) uint8, object on neutral canvas
    alpha: np.ndarray         # (S, S) float32 in [0, 1]
    mask: np.ndarray          # (S, S) bool
    source_rgb: np.ndarray    # (h, w, 3) uint8 original (EXIF-corrected)
    meta: dict[str, Any]


# --------------------------------------------------------------------------
# matting
# --------------------------------------------------------------------------
def _u2net_alpha(rgb: np.ndarray, providers: dict[str, str] | None = None) -> np.ndarray | None:
    """Salient-object matte with U^2-Net-lite (u2netp) via ONNX Runtime."""
    model_path = MODELS_DIR / "u2netp.onnx"
    if not model_path.exists():
        return None
    try:
        import onnxruntime as ort
    except Exception:  # pragma: no cover - optional dependency
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(model_path, so)
    size = 320
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = ((img - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0]
    if providers is not None:
        providers["matting"] = sess.provider
    pred = np.asarray(out).reshape(out.shape[-2], out.shape[-1]).astype(np.float32)
    pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
    h, w = rgb.shape[:2]
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


def _grabcut_alpha(rgb: np.ndarray) -> np.ndarray:
    """Fallback matte: GrabCut seeded from a border band."""
    h, w = rgb.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb.copy()
    gc = np.full(small.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(2, int(min(small.shape[:2]) * 0.04))
    gc[:border, :] = cv2.GC_BGD
    gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = cv2.GC_BGD
    gc[:, -border:] = cv2.GC_BGD
    inner = border * 3
    gc[inner:-inner, inner:-inner] = cv2.GC_PR_FGD
    gc[gc == cv2.GC_PR_BGD] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except cv2.error:
        log.warning("GrabCut failed; falling back to threshold matte")
        mask = _threshold_alpha(small)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def _threshold_alpha(rgb: np.ndarray) -> np.ndarray:
    """Last-resort matte for images already shot on a plain background."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bg = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
                         rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    bg_med = np.median(bg, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg_med[None, None, :], axis=2)
    thr = max(18.0, float(np.percentile(dist, 55)))
    mask = (dist > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # keep large blobs only
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        keep = np.zeros_like(mask)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size:
            biggest = 1 + int(np.argmax(areas))
            keep[lab == biggest] = 255
            for i in range(1, n):
                if i != biggest and stats[i, cv2.CC_STAT_AREA] > 0.25 * areas.max():
                    keep[lab == i] = 255
        mask = keep
    _ = hsv
    return (mask.astype(np.float32) / 255.0)


def _clean_alpha(alpha: np.ndarray, erode_px: int) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)
    binary = (a > 0.5).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # keep the largest component + any component at least 20 % of its area
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(areas))
        keep = (lab == biggest)
        for i in range(1, n):
            if i != biggest and stats[i, cv2.CC_STAT_AREA] > 0.2 * areas.max():
                keep |= (lab == i)
        binary = keep.astype(np.uint8)
    # fill interior holes
    ff = binary.copy()
    hh, ww = binary.shape
    ffmask = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 1)
    holes = (ff == 0).astype(np.uint8)
    binary = np.clip(binary + holes, 0, 1)
    if erode_px > 0:
        binary = cv2.erode(binary, np.ones((erode_px * 2 + 1,) * 2, np.uint8))
    soft = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 1.0)
    return np.clip(soft, 0.0, 1.0)


def _composite_canvas(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray,
                      cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("No object detected in the image.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    obj_w, obj_h = x1 - x0, y1 - y0
    S = cfg.canvas_size
    pad = cfg.crop_padding_pct
    scale = (S * (1 - 2 * pad)) / max(obj_w, obj_h)
    new_w, new_h = max(1, int(round(obj_w * scale))), max(1, int(round(obj_h * scale)))

    crop_rgb = rgb[y0:y1, x0:x1]
    crop_a = alpha[y0:y1, x0:x1]
    r_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    r_a = cv2.resize(crop_a, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    r_a = np.clip((r_a - 0.35) / 0.3, 0.0, 1.0)  # sharpen the matte ramp

    bg = {"white": 255, "gray": 200, "midgray": 128}[cfg.background]
    canvas = np.full((S, S, 3), bg, dtype=np.float32)
    canvas_a = np.zeros((S, S), dtype=np.float32)
    oy = (S - new_h) // 2
    ox = (S - new_w) // 2
    a3 = r_a[:, :, None]
    canvas[oy:oy + new_h, ox:ox + new_w] = (
        r_rgb.astype(np.float32) * a3 + canvas[oy:oy + new_h, ox:ox + new_w] * (1 - a3)
    )
    canvas_a[oy:oy + new_h, ox:ox + new_w] = r_a
    out_mask = canvas_a > 0.5

    # silhouette must be strictly inside the canvas so the rim can close
    out_mask = cv2.erode(out_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    canvas_a = np.where(out_mask, canvas_a, 0.0).astype(np.float32)

    meta = {
        "crop_box": [int(x0), int(y0), int(w1 := x1), int(y1)],
        "object_size_px": [int(obj_w), int(obj_h)],
        "scale": float(scale),
        "canvas": [int(S), int(S)],
        "placed_box": [int(ox), int(oy), int(ox + new_w), int(oy + new_h)],
        "object_pixel_fraction": float(out_mask.mean()),
        "_w1": w1,
    }
    meta.pop("_w1", None)
    return canvas.astype(np.uint8), canvas_a, out_mask, meta


def prepare_image(data: bytes | str | Path, cfg: PipelineConfig) -> PreparedImage:
    """Decode, EXIF-correct, matte and centre an upload."""
    if isinstance(data, (str, Path)):
        buf = Path(data).read_bytes()
    else:
        buf = data
    mime = validate_image_bytes(buf)
    arr = np.frombuffer(buf, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("The image could not be decoded.")
    if decoded.dtype == np.uint16:
        decoded = (decoded >> 8).astype(np.uint8)
    embedded_alpha = None
    if decoded.ndim == 3 and decoded.shape[2] == 4 and decoded[..., 3].min() < 250:
        embedded_alpha = _apply_exif_orientation(decoded[..., 3], buf).astype(np.float32) / 255.0
    bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR) if decoded.ndim == 2 else decoded[..., :3]
    if bgr is None:
        raise ValueError("The image could not be decoded.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # IMREAD_UNCHANGED leaves the original orientation for explicit EXIF correction.
    rgb = _apply_exif_orientation(rgb, buf)
    h, w = rgb.shape[:2]
    if max(h, w) > 2600:
        s = 2600.0 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if embedded_alpha is not None:
            embedded_alpha = cv2.resize(embedded_alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)

    method = cfg.matte_model
    providers: dict[str, str] = {}
    alpha = embedded_alpha
    if alpha is not None:
        method = "image-alpha"
    elif method == "u2netp":
        alpha = _u2net_alpha(rgb, providers)
        if alpha is None:
            log.warning("u2netp model unavailable - falling back to GrabCut")
            method = "grabcut"
    if alpha is None and method == "grabcut":
        alpha = _grabcut_alpha(rgb)
    if alpha is None:
        alpha = _threshold_alpha(rgb)
        method = "threshold"

    alpha = _clean_alpha(alpha, cfg.alpha_erode_px)
    mask = alpha > 0.5
    if mask.sum() < 0.005 * mask.size:
        raise ValueError("Object occupies too little of the frame; please upload a tighter crop.")
    canvas, canvas_a, canvas_mask, meta = _composite_canvas(rgb, alpha, mask, cfg)
    meta.update({
        "source_size": [int(w), int(h)],
        "mime": mime,
        "matte_method": method,
        "matte_provider": providers.get("matting"),
        "source_bytes": int(len(buf)),
    })
    return PreparedImage(canvas, canvas_a, canvas_mask, rgb, meta)


def _apply_exif_orientation(rgb: np.ndarray, buf: bytes) -> np.ndarray:
    try:
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(buf))
        exif = img.getexif()
        orientation = int(exif.get(274, 1)) if exif else 1
        if orientation == 1:
            return rgb
        transforms = {
            2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }
        return np.array(Image.fromarray(rgb).transpose(transforms[orientation])) if orientation in transforms else rgb
    except Exception:  # pragma: no cover - malformed EXIF
        return rgb
