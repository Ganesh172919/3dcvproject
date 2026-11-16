"""Image decoding for the standalone geometry exercises."""
from pathlib import Path

import cv2
import numpy as np


def read_color_image(path: Path | str) -> np.ndarray:
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"Empty input image: {path}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode input image: {path}")
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit or 16-bit image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
