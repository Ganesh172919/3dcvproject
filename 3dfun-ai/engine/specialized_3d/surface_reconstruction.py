"""Specialized Image-to-3D: Depth-Guided SDF Surface Mesh Reconstruction.

Constructs an adaptive 3D Signed Distance Field (SDF) from monocular cues,
extracts the zero-level isosurface via Marching Cubes, applies Taubin surface
smoothing, and exports clean 3D triangle meshes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import measure

if __package__:
    from .images import read_color_image
else:
    from images import read_color_image


def extract_silhouette_and_depth(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment object and generate normalized depth prior."""
    if image_bgr.shape[2] == 4:
        mask = (image_bgr[:, :, 3] > 30).astype(np.uint8) * 255
        rgb = image_bgr[:, :, :3]
    else:
        rgb = image_bgr
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        bg_val = np.median(corners)
        if bg_val > 128:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth = 0.70 * dist + 0.30 * (1.0 - cv2.GaussianBlur(gray, (7, 7), 2.0))
    return mask, np.clip(depth, 0.0, 1.0)


def build_signed_distance_field(
    mask: np.ndarray,
    depth: np.ndarray,
    grid_size: int = 64,
    thickness_factor: float = 0.50,
) -> np.ndarray:
    """Build a 3D Signed Distance Field bounded by silhouette thickness."""
    if grid_size < 2:
        raise ValueError("Grid resolution must be at least two.")
    # Resize mask and depth to grid_size x grid_size
    mask_res = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST) > 0
    depth_res = cv2.resize(depth, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)

    # Compute per-pixel thickness: constant core + depth variation
    t_map = np.clip(0.15 + thickness_factor * depth_res, 0.0, 1.0)

    # Occupancy volume: (grid_size, grid_size, grid_size)
    # Z coordinate runs from -1.0 to 1.0 along depth axis
    z_coords = np.linspace(-1.0, 1.0, grid_size)
    occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    for z_idx, z_val in enumerate(z_coords):
        # Point is inside if it belongs to 2D silhouette and |z| <= t(p)/2
        inside_z = np.abs(z_val) <= (t_map * 0.5)
        occupancy[:, :, z_idx] = mask_res & inside_z
    if not occupancy.any():
        raise ValueError("No occupied voxels. Check the foreground or increase the grid resolution.")

    # Compute Euclidean Signed Distance Transform
    # SDF = EDT(outside) - EDT(inside)
    dist_inside = distance_transform_edt(occupancy)
    dist_outside = distance_transform_edt(~occupancy)
    sdf = dist_outside - dist_inside

    # Smooth the SDF field
    sdf_smooth = gaussian_filter(sdf, sigma=0.8)
    return sdf_smooth


def taubin_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 5, lambda_param: float = 0.5, mu_param: float = -0.53) -> np.ndarray:
    """Apply non-shrinking Taubin smoothing to mesh vertices."""
    # Build vertex adjacency list
    n_verts = len(vertices)
    adjacency = [[] for _ in range(n_verts)]
    for f in faces:
        adjacency[f[0]].extend([f[1], f[2]])
        adjacency[f[1]].extend([f[0], f[2]])
        adjacency[f[2]].extend([f[0], f[1]])

    adj_sets = [np.unique(adj) for adj in adjacency]

    v = vertices.copy()
    for _ in range(iterations):
        # Step 1: Positive Laplacian step with lambda
        laplacian = np.zeros_like(v)
        for i, neighbors in enumerate(adj_sets):
            if len(neighbors) > 0:
                laplacian[i] = np.mean(v[neighbors], axis=0) - v[i]
        v += lambda_param * laplacian

        # Step 2: Negative Laplacian step with mu (to counter shrinkage)
        laplacian = np.zeros_like(v)
        for i, neighbors in enumerate(adj_sets):
            if len(neighbors) > 0:
                laplacian[i] = np.mean(v[neighbors], axis=0) - v[i]
        v += mu_param * laplacian

    return v


def build_surface_mesh(
    image_path: Path | str,
    output_dir: Path | str = Path("out/specialized_surface"),
    resolution: int = 64,
) -> Path:
    """Run full depth-guided SDF surface reconstruction."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = read_color_image(image_path)

    mask, depth = extract_silhouette_and_depth(bgr)
    sdf = build_signed_distance_field(mask, depth, grid_size=resolution)

    # Pad boundaries to ensure watertight surface extraction
    sdf_padded = np.pad(sdf, 1, mode="constant", constant_values=1.0)
    verts, faces, normals, _ = measure.marching_cubes(sdf_padded, level=0.0)

    # Normalize to [-1, 1]
    verts = (verts - 1.0) / (resolution - 1) * 2.0 - 1.0
    verts = taubin_smooth(verts, faces, iterations=4)

    out_obj = output_dir / f"{image_path.stem}_surface.obj"
    with out_obj.open("w", encoding="utf-8") as f:
        f.write("# 3DCV Specialized Depth-Guided SDF Surface Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.5f} {-v[1]:.5f} {v[2]:.5f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.4f} {-n[1]:.4f} {n[2]:.4f}\n")
        for face in faces:
            f1, f2, f3 = face + 1
            f.write(f"f {f1} {f2} {f3}\n")

    print(f"[SurfaceMesh] Extracted {len(verts):,} vertices & {len(faces):,} triangles -> {out_obj}")
    return out_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", "-i", type=Path, required=True, help="Input RGB/RGBA image path")
    parser.add_argument("--res", "-r", type=int, default=64, help="SDF grid resolution")
    parser.add_argument("--out", "-o", type=Path, default=Path("out/specialized_surface"), help="Output directory")
    args = parser.parse_args()

    build_surface_mesh(args.image, args.out, resolution=args.res)


if __name__ == "__main__":
    main()
