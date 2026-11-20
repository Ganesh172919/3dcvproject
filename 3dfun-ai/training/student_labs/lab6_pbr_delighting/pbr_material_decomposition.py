"""Student Laboratory 6: Physically Based Rendering (PBR) Material Decomposition & Delighting.

Deconstructs a single input photograph into complete game-engine ready PBR texture channels:
1. Bilateral Delighting (isolating intrinsic diffuse Albedo from directional illumination)
2. Tangent-Space Surface Normal Map Synthesis via 2D spatial gradients
3. Specular Roughness Map Estimation via high-frequency luminance variance
4. Metallic Map Estimation
5. Packed ORM Texture Map (Occlusion, Roughness, Metallic)
6. Quantitative decomposition metrics & export.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def decompose_pbr_material(
    image_path: Path | str | None = None,
    filter_radius: int = 5,
    roughness_base: float = 0.45,
    metallic_factor: float = 0.1,
    out_dir: Path | str = "out/student_labs/lab6_pbr",
) -> dict[str, Any]:
    """Performs single-image PBR material decomposition and delighting."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or synthesize input image
    if image_path and Path(image_path).is_file():
        img = Image.open(image_path).convert("RGBA")
    else:
        # Generate 3D render ceramic teapot/sphere patch with directional lighting
        res = 256
        y, x = np.mgrid[-1:1:res*1j, -1:1:res*1j]
        r = np.sqrt(x**2 + y**2)
        mask = (r <= 0.85).astype(np.float32)

        # Base diffuse color (rich indigo-cyan ceramic)
        base_rgb = np.zeros((res, res, 3), dtype=np.float32)
        base_rgb[..., 0] = 0.2 + 0.1 * np.sin(x * 5)
        base_rgb[..., 1] = 0.4 + 0.2 * np.cos(y * 5)
        base_rgb[..., 2] = 0.8

        # Directional light from top-left (L = [0.577, -0.577, 0.577])
        z = np.sqrt(np.maximum(0, 0.85**2 - r**2))
        nx = np.where(mask > 0, x / 0.85, 0.0)
        ny = np.where(mask > 0, y / 0.85, 0.0)
        nz = np.where(mask > 0, z / 0.85, 1.0)
        n_dot_l = np.maximum(0.1, nx * (-0.577) + ny * (-0.577) + nz * 0.577)

        shaded = base_rgb * n_dot_l[..., None]
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([(shaded * 255).clip(0, 255).astype(np.uint8), alpha])
        img = Image.fromarray(rgba)

    # Convert to float numpy arrays
    rgba_np = np.array(img).astype(np.float32) / 255.0
    rgb = rgba_np[..., :3]
    alpha = rgba_np[..., 3] if rgba_np.shape[-1] == 4 else np.ones(rgb.shape[:2], dtype=np.float32)

    # 2. Luminance and Illumination Extraction (Delighting)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    
    # Fast box/spatial smoothing approximation of bilateral illumination
    from scipy.ndimage import uniform_filter, sobel
    illumination = uniform_filter(luminance, size=max(3, filter_radius * 2 + 1))
    illumination = np.maximum(illumination, 0.08)  # prevent division by zero

    # De-lighted Albedo
    mean_illum = np.mean(illumination[alpha > 0.5]) if np.any(alpha > 0.5) else 0.5
    albedo = np.clip(rgb / (illumination[..., None] / max(mean_illum, 0.1)), 0.0, 1.0)
    # Mask out background
    albedo = albedo * alpha[..., None]

    # 3. Tangent-Space Normal Map Generation
    # Compute Sobel gradients of luminance / apparent depth
    grad_x = sobel(luminance, axis=1) * 2.0
    grad_y = sobel(luminance, axis=0) * 2.0

    # Normal vector N = (-dz/dx, -dz/dy, 1) normalized
    norm_factor = np.sqrt(grad_x**2 + grad_y**2 + 1.0)
    norm_x = -grad_x / norm_factor
    norm_y = -grad_y / norm_factor
    norm_z = 1.0 / norm_factor

    # Tangent-space normal mapping: [-1, 1] -> [0, 255]
    normal_map = np.zeros_like(rgb)
    normal_map[..., 0] = (norm_x * 0.5 + 0.5)
    normal_map[..., 1] = (norm_y * 0.5 + 0.5)
    normal_map[..., 2] = (norm_z * 0.5 + 0.5)
    normal_map = normal_map * alpha[..., None]

    # 4. Roughness and Metallic Estimation
    # Local high-frequency variation indicates surface roughness
    high_freq = np.abs(luminance - illumination)
    roughness = np.clip(roughness_base + high_freq * 1.5, 0.05, 0.95) * alpha

    # Metallic factor
    metallic = np.full_like(luminance, metallic_factor) * alpha

    # Ambient Occlusion (cavity estimation via Laplacian)
    ao = np.clip(1.0 - (high_freq * 2.0), 0.2, 1.0) * alpha

    # 5. Packed ORM (Red = Occlusion, Green = Roughness, Blue = Metallic)
    orm = np.dstack([ao, roughness, metallic])

    # 6. Save Artifacts
    albedo_path = out_dir / "albedo.png"
    normal_path = out_dir / "normal.png"
    roughness_path = out_dir / "roughness.png"
    metallic_path = out_dir / "metallic.png"
    orm_path = out_dir / "orm_packed.png"

    Image.fromarray((albedo * 255).astype(np.uint8)).save(albedo_path)
    Image.fromarray((normal_map * 255).astype(np.uint8)).save(normal_path)
    Image.fromarray((roughness * 255).astype(np.uint8)).save(roughness_path)
    Image.fromarray((metallic * 255).astype(np.uint8)).save(metallic_path)
    Image.fromarray((orm * 255).astype(np.uint8)).save(orm_path)

    # 7. Metrics Report
    fg_mask = alpha > 0.5
    metrics = {
        "lab_id": "lab6_pbr",
        "mean_albedo_luminance": round(float(np.mean(luminance[fg_mask])) if np.any(fg_mask) else 0.0, 4),
        "mean_roughness": round(float(np.mean(roughness[fg_mask])) if np.any(fg_mask) else 0.0, 4),
        "mean_metallic": round(float(np.mean(metallic[fg_mask])) if np.any(fg_mask) else 0.0, 4),
        "normal_flatness_ratio": round(float(np.mean(norm_z[fg_mask])) if np.any(fg_mask) else 1.0, 4),
        "artifacts": {
            "albedo": str(albedo_path),
            "normal": str(normal_path),
            "roughness": str(roughness_path),
            "metallic": str(metallic_path),
            "orm_packed": str(orm_path),
        },
    }

    with open(out_dir / "lab6_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    res = decompose_pbr_material()
    print("Lab 6 PBR Material Decomposition Complete:", res)
