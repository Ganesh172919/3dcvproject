"""Specialized Image-to-3D: Orthogonal Space Carving & Visual Hull Construction.

Reconstructs 3D shapes from multi-view silhouettes or orthogonal projections
by carving a 3D voxel grid and extracting a clean isosurface via Marching Cubes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
from skimage import measure


def extract_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Extract clean binary mask from image."""
    if image_bgr.shape[2] == 4:
        mask = (image_bgr[:, :, 3] > 30).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        bg_val = np.median(corners)
        if bg_val > 128:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def carve_visual_hull(
    masks: list[np.ndarray],
    rotations_deg: list[float],
    grid_resolution: int = 64,
) -> np.ndarray:
    """Carve a 3D voxel grid using silhouettes observed from given rotation angles (in degrees).

    Args:
        masks: list of (H, W) binary mask arrays
        rotations_deg: list of azimuth rotation angles around vertical Y axis
        grid_resolution: size of cubic voxel grid (N x N x N)

    Returns:
        voxel_grid: (N, N, N) binary uint8 array (1 = solid, 0 = free space)
    """
    n = grid_resolution
    # Coordinate grid centered at [0, 0, 0], ranging from -1.0 to 1.0
    coords = np.linspace(-1.0, 1.0, n)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    
    # Flatten points: (N^3, 3)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    
    # Occupancy mask: initially all voxels are occupied
    occupied = np.ones(pts.shape[0], dtype=bool)

    for mask, deg in zip(masks, rotations_deg):
        rad = np.radians(deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        # Rotate points around Y axis:
        # X' = X*cos - Z*sin
        # Y' = Y
        # Z' = X*sin + Z*cos
        x_rot = pts[:, 0] * cos_a - pts[:, 2] * sin_a
        y_rot = pts[:, 1]
        
        # Project orthographically onto image coordinates [0, W-1] and [0, H-1]
        h, w = mask.shape
        # Scale [-1, 1] to bounding center
        u = ((x_rot + 1.0) * 0.5 * (w - 1)).astype(np.int32)
        v = (((-y_rot) + 1.0) * 0.5 * (h - 1)).astype(np.int32)

        # Check bounds
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        # Test mask value for points in bounds
        sample_inside = np.zeros(len(pts), dtype=bool)
        sample_inside[in_bounds] = mask[v[in_bounds], u[in_bounds]] > 0
        
        # Intersect with global occupancy
        occupied &= sample_inside

    return occupied.reshape((n, n, n)).astype(np.uint8)


def extract_mesh_marching_cubes(voxel_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert binary voxel grid to smooth triangle mesh via Marching Cubes."""
    # Pad grid with 0 to ensure watertight closed mesh boundaries
    padded = np.pad(voxel_grid.astype(np.float32), pad_width=1, mode="constant", constant_values=0)
    
    # Apply Gaussian smoothing on the scalar field
    from scipy.ndimage import gaussian_filter
    smooth_field = gaussian_filter(padded, sigma=1.0)
    
    verts, faces, _, _ = measure.marching_cubes(smooth_field, level=0.5)
    
    # Shift back padding and normalize to [-1, 1]
    n = voxel_grid.shape[0]
    verts = (verts - 1.0) / (n - 1) * 2.0 - 1.0
    return verts.astype(np.float32), faces.astype(np.int32)


def export_obj(path: Path | str, vertices: np.ndarray, faces: np.ndarray):
    """Export vertices and faces as OBJ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3DCV Specialized Visual Hull Space Carving\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for f_idx in faces:
            f.write(f"f {f_idx[0]+1} {f_idx[1]+1} {f_idx[2]+1}\n")


def build_space_carving(
    image_paths: list[Path | str],
    angles_deg: list[float] | None = None,
    resolution: int = 64,
    output_dir: Path | str = Path("out/specialized_space_carving"),
) -> Path:
    """Run Space Carving visual hull reconstruction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if angles_deg is None:
        if len(image_paths) == 1:
            # Single image mode: use front (0 deg) and 3D render orthogonal profile (90 deg)
            angles_deg = [0.0, 90.0]
        else:
            angles_deg = [i * (360.0 / len(image_paths)) for i in range(len(image_paths))]

    masks = []
    for p in image_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {p}")
        masks.append(extract_mask(bgr))

    if len(masks) == 1 and len(angles_deg) == 2:
        # If single view is provided, simulate orthogonal 90-degree cross-section
        masks.append(masks[0])

    grid = carve_visual_hull(masks, angles_deg, grid_resolution=resolution)
    verts, faces = extract_mesh_marching_cubes(grid)

    stem = Path(image_paths[0]).stem
    out_obj = output_dir / f"{stem}_visual_hull.obj"
    export_obj(out_obj, verts, faces)

    # Save voxel occupancy grid as NPZ for downstream analysis
    np.savez_compressed(output_dir / f"{stem}_voxels.npz", voxels=grid)

    volume = float(np.sum(grid)) / (resolution**3) * 8.0  # Normalized cubic units
    print(f"[SpaceCarving] Extracted {len(verts):,} vertices & {len(faces):,} faces (Volume: {volume:.4f}) -> {out_obj}")
    return out_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", "-i", type=Path, nargs="+", required=True, help="One or more view image paths")
    parser.add_argument("--angles", "-a", type=float, nargs="*", default=None, help="Azimuth rotation angles")
    parser.add_argument("--res", "-r", type=int, default=64, help="Voxel grid resolution (e.g. 64 or 96)")
    parser.add_argument("--out", "-o", type=Path, default=Path("out/specialized_space_carving"), help="Output directory")
    args = parser.parse_args()

    build_space_carving(args.images, args.angles, resolution=args.res, output_dir=args.out)


if __name__ == "__main__":
    main()
