"""Specialized Image-to-3D: RGB-D Point Cloud Construction.

Constructs dense 3D point clouds with calibrated pinhole camera unprojection,
surface normal estimation, statistical outlier removal, and color mapping.
Fully standalone and runs on CPU with zero heavy external dependencies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2

if __package__:
    from .images import read_color_image
else:
    from images import read_color_image


def extract_foreground_and_depth(image_bgr: np.ndarray, depth_map: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract binary foreground mask and normalized depth [0, 1]."""
    h, w = image_bgr.shape[:2]

    # Check if image has alpha channel
    if image_bgr.shape[2] == 4:
        alpha = image_bgr[:, :, 3]
        mask = (alpha > 30).astype(np.uint8) * 255
        rgb = image_bgr[:, :, :3]
    else:
        rgb = image_bgr
        # Salient foreground estimation: Otsu thresholding + morphological cleanup
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # Check if background is mostly light or dark
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        bg_val = np.median(corners)
        if bg_val > 128:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Estimate depth if not provided: smooth distance transform + normalized intensity gradient
    if depth_map is not None:
        depth = depth_map.astype(np.float32)
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
    else:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist = dist / dist.max()
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Blend shape-from-shading proxy with distance from silhouette boundary
        depth = 0.65 * dist + 0.35 * (1.0 - cv2.GaussianBlur(gray, (7, 7), 2.0))
        depth = np.clip(depth, 0.0, 1.0)

    return mask, depth


def unproject_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    focal_length: float | None = None,
    depth_scale: float = 1.0,
    depth_offset: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject pixels into 3D camera coordinates (X, Y, Z) with colors and normals.

    Returns:
        points: (N, 3) float32 array of XYZ coordinates
        colors: (N, 3) uint8 array of RGB colors
        normals: (N, 3) float32 array of unit surface normals
    """
    h, w = depth.shape
    if focal_length is None:
        focal_length = max(h, w) * 1.2
    cx, cy = w / 2.0, h / 2.0

    # Physical metric depth Z: depth_offset + depth_scale * (1.0 - depth)
    # Closer points have higher depth score, so smaller distance Z
    z_map = depth_offset + depth_scale * (1.0 - depth)

    # Find valid foreground coordinates
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Foreground mask is empty. Cannot construct point cloud.")

    z = z_map[ys, xs]
    x = (xs + 0.5 - cx) * z / focal_length
    y = (cy - (ys + 0.5)) * z / focal_length
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    # RGB colors
    rgb_clean = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb.shape[2] == 3 else rgb[:, :, :3]
    colors = rgb_clean[ys, xs]

    # Compute surface normals via image spatial gradients
    dz_dx = cv2.Sobel(z_map, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(z_map, cv2.CV_32F, 0, 1, ksize=3)

    nx = -dz_dx[ys, xs] / focal_length
    ny = dz_dy[ys, xs] / focal_length
    nz = np.ones_like(nx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm[norm == 0] = 1.0
    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)

    return points, colors, normals


def filter_statistical_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove statistical outliers based on mean distance to k-nearest neighbors."""
    if len(points) < k_neighbors * 2:
        return points, colors, normals

    # Subsample indices for fast neighbor search if point cloud is very large
    from scipy.spatial import KDTree
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k_neighbors)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    thresh = np.mean(mean_dists) + std_ratio * np.std(mean_dists)
    valid = mean_dists <= thresh
    return points[valid], colors[valid], normals[valid]


def save_ply(path: Path | str, points: np.ndarray, colors: np.ndarray, normals: np.ndarray | None = None):
    """Save point cloud to standard ASCII PLY file format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        if normals is not None:
            for p, norm, c in zip(points, normals, colors):
                f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {norm[0]:.4f} {norm[1]:.4f} {norm[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def build_pointcloud(
    image_path: Path | str,
    output_dir: Path | str,
    depth_path: Path | str | None = None,
    filter_outliers: bool = True,
) -> Path:
    """End-to-end point cloud construction pipeline."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = read_color_image(image_path)

    depth_map = None
    if depth_path is not None:
        depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_map is None or depth_map.shape != bgr.shape[:2]:
            raise ValueError("Supply a readable single-channel depth map with the image dimensions.")

    mask, depth = extract_foreground_and_depth(bgr, depth_map)
    if not mask.any():
        raise ValueError("No foreground detected in image.")
    points, colors, normals = unproject_to_pointcloud(bgr, depth, mask)

    if filter_outliers:
        points, colors, normals = filter_statistical_outliers(points, colors, normals)

    ply_path = output_dir / f"{image_path.stem}_pointcloud.ply"
    save_ply(ply_path, points, colors, normals)

    # Save preview image of estimated depth and mask
    cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), mask)
    depth_viz = (depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_depth_preview.png"), depth_color)

    print(f"[PointCloud] Constructed {len(points):,} 3D points -> {ply_path}")
    return ply_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", "-i", type=Path, required=True, help="Input RGB/RGBA image path")
    parser.add_argument("--depth", "-d", type=Path, default=None, help="Optional precomputed depth map")
    parser.add_argument("--out", "-o", type=Path, default=Path("out/specialized_pointcloud"), help="Output directory")
    args = parser.parse_args()

    build_pointcloud(args.image, args.out, args.depth)


if __name__ == "__main__":
    main()
