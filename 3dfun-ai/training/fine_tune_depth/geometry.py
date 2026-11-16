"""Camera-frame back-projection for TinyDepth inference and evaluation."""
import numpy as np


def backproject(depth, rows, columns, camera):
    """Convert sampled pixel centers and axial depth to XYZ in camera coordinates."""
    focal = float(camera["focal_px"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    if not np.isfinite([focal, cx, cy]).all() or focal <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal length.")
    z = depth[rows, columns]
    if not np.isfinite(z).all() or (z <= 0).any():
        raise ValueError("The evaluated foreground contains invalid camera depths.")
    return np.stack(((columns + .5 - cx) * z / focal,
                     (cy - rows - .5) * z / focal, z), axis=-1)
