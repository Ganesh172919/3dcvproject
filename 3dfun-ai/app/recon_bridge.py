"""Bridge between the web layer and the reconstruction engine package.

Keeps the API importable even when the (heavy) vision dependencies are absent,
and centralises upload validation so the API and the worker agree.
"""
from __future__ import annotations

import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

_FALLBACK_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"BM": "image/bmp",
}


def validate_image(data: bytes, max_mb: int = 20) -> tuple[str, int, int]:
    """Content-sniff, size-check and measure an image; returns (mime, w, h)."""
    try:
        from recon.preprocessing import validate_image_bytes  # type: ignore
    except Exception:                                         # pragma: no cover
        validate_image_bytes = None

    if validate_image_bytes is not None:
        mime = validate_image_bytes(data, max_mb)
    else:                                                     # pragma: no cover
        if not data:
            raise ValueError("Uploaded file is empty.")
        if len(data) > max_mb * 1024 * 1024:
            raise ValueError(f"Image exceeds the {max_mb} MB limit.")
        mime = next((m for k, m in _FALLBACK_MAGIC.items() if data.startswith(k)), None)
        if data[8:12] == b"WEBP":
            mime = "image/webp"
        if mime is None:
            raise ValueError("Unsupported image type.")

    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("The image could not be decoded.")
        h, w = img.shape[:2]
        if min(w, h) < 64:
            raise ValueError("Image is too small (minimum 64 px on the short side).")
        if max(w, h) > 8000:
            raise ValueError("Image is too large (maximum 8000 px on the long side).")
        return mime, int(w), int(h)
    except ImportError:                                       # pragma: no cover
        return mime, 0, 0


def engine_available() -> dict:
    """Health descriptor for the /health endpoint."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import trimesh  # noqa: F401
        models = ["depth_anything_v2_small_quantized.onnx", "u2netp.onnx"]
        present = {m: (ENGINE_ROOT / "models" / m).exists() for m in models}
        return {"available": all(present.values()), "models": present,
                "engine_root": str(ENGINE_ROOT)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
