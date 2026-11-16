"""Specialized Image-to-3D: Axisymmetric Surface of Revolution Construction.

Extracts the vertical silhouette profile of symmetrical objects (vases, bottles,
glasses, cans, bowls, rockets, columns) and performs full 360-degree rotational
sweeping to produce a watertight textured 3D mesh.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2


def extract_silhouette_profile(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Find the object mask, vertical symmetry axis, and radial profile r(y)."""
    h, w = image_bgr.shape[:2]

    # Mask extraction
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

    # Find bounding box of foreground
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("No foreground detected in image.")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Determine vertical axis of symmetry (center of mass along x)
    axis_x = float(np.mean(xs))

    # Compute radius profile r(y) for each scanline from y_min to y_max
    height_span = y_max - y_min + 1
    y_coords = np.arange(y_min, y_max + 1)
    radii = np.zeros(height_span, dtype=np.float32)

    for idx, y in enumerate(y_coords):
        row_xs = xs[ys == y]
        if len(row_xs) > 0:
            left_r = abs(axis_x - row_xs.min())
            right_r = abs(row_xs.max() - axis_x)
            # Symmetrized radius
            radii[idx] = (left_r + right_r) / 2.0
        else:
            radii[idx] = 0.0

    # Smooth the radius profile along y to reduce segmentation stepping
    kernel_size = max(5, int(height_span * 0.03) | 1)
    radii = cv2.GaussianBlur(radii[:, None], (1, kernel_size), 0).ravel()

    return y_coords, radii, axis_x, mask


def construct_revolution_mesh(
    y_coords: np.ndarray,
    radii: np.ndarray,
    num_radial_steps: int = 48,
    scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Revolve the 1D profile around vertical Y axis to form a 3D triangle mesh.

    Returns:
        vertices: (V, 3) float32
        faces: (F, 3) int32
        normals: (V, 3) float32
        uvs: (V, 2) float32
    """
    n_y = len(y_coords)
    n_theta = num_radial_steps

    # Normalize height and radius to [-1, 1] range
    h_span = y_coords.max() - y_coords.min()
    if h_span == 0:
        h_span = 1.0

    norm_y = (y_coords.min() + y_coords.max()) / 2.0
    y_3d = -(y_coords - norm_y) / (h_span / 2.0)  # Invert Y so up is positive
    r_3d = radii / (h_span / 2.0)

    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # Grid of vertices
    # Shape: (n_y, n_theta, 3)
    vertices = np.zeros((n_y, n_theta, 3), dtype=np.float32)
    uvs = np.zeros((n_y, n_theta, 2), dtype=np.float32)

    for i in range(n_y):
        y_val = y_3d[i]
        r_val = r_3d[i]
        for j, th in enumerate(thetas):
            vertices[i, j, 0] = r_val * np.cos(th)
            vertices[i, j, 1] = y_val
            vertices[i, j, 2] = r_val * np.sin(th)
            uvs[i, j, 0] = j / float(n_theta)
            uvs[i, j, 1] = (y_3d[i] + 1.0) / 2.0

    # Build faces
    faces = []
    for i in range(n_y - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + j_next
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j_next

            # Two triangles per quad
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    # Cap top and bottom with pole vertices to ensure watertight closure
    flat_verts = vertices.reshape(-1, 3)
    flat_uvs = uvs.reshape(-1, 2)

    top_pole = np.array([[0.0, y_3d[0], 0.0]], dtype=np.float32)
    bottom_pole = np.array([[0.0, y_3d[-1], 0.0]], dtype=np.float32)
    top_pole_idx = len(flat_verts)
    bottom_pole_idx = top_pole_idx + 1

    flat_verts = np.vstack([flat_verts, top_pole, bottom_pole])
    flat_uvs = np.vstack([flat_uvs, [[0.5, 1.0]], [[0.5, 0.0]]])

    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        faces.append([top_pole_idx, j, j_next])
        bottom_row = (n_y - 1) * n_theta
        faces.append([bottom_pole_idx, bottom_row + j_next, bottom_row + j])

    faces_arr = np.array(faces, dtype=np.int32)

    # Compute vertex normals
    normals = np.zeros_like(flat_verts)
    v0 = flat_verts[faces_arr[:, 0]]
    v1 = flat_verts[faces_arr[:, 1]]
    v2 = flat_verts[faces_arr[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for idx, f in enumerate(faces_arr):
        normals[f[0]] += face_normals[idx]
        normals[f[1]] += face_normals[idx]
        normals[f[2]] += face_normals[idx]
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    return flat_verts, faces_arr, normals, flat_uvs


def export_obj(path: Path | str, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, uvs: np.ndarray):
    """Export standard Wavefront OBJ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3DCV Specialized Axisymmetric Surface of Revolution\n")
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for face in faces:
            # 1-indexed: v/vt/vn
            f1, f2, f3 = face + 1
            f.write(f"f {f1}/{f1}/{f1} {f2}/{f2}/{f2} {f3}/{f3}/{f3}\n")


def build_revolution(image_path: Path | str, output_dir: Path | str, steps: int = 48) -> Path:
    """Run full axisymmetric surface of revolution."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")

    y_coords, radii, axis_x, mask = extract_silhouette_profile(bgr)
    vertices, faces, normals, uvs = construct_revolution_mesh(y_coords, radii, num_radial_steps=steps)

    out_obj = output_dir / f"{image_path.stem}_revolution.obj"
    export_obj(out_obj, vertices, faces, normals, uvs)

    # Save texture and profile preview
    cv2.imwrite(str(output_dir / f"{image_path.stem}_profile_mask.png"), mask)
    print(f"[Revolution] Revolved {len(vertices):,} vertices & {len(faces):,} triangles -> {out_obj}")
    return out_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", "-i", type=Path, required=True, help="Input RGB/RGBA image path")
    parser.add_argument("--steps", "-s", type=int, default=48, help="Number of radial angular steps")
    parser.add_argument("--out", "-o", type=Path, default=Path("out/specialized_revolution"), help="Output directory")
    args = parser.parse_args()

    build_revolution(args.image, args.out, steps=args.steps)


if __name__ == "__main__":
    main()
