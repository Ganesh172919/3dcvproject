"""Synthetic Data Generator for Depth & Normal Estimation.

Renders multi-angle RGB, 16-bit metric depth, surface normals, and silhouette masks
from 3D OBJ / GLB meshes using Trimesh and vectorised software projection.

Course: 3D Computer Vision Course Project (IIIT Dharwad)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Computes 3x3 rotation matrix for camera orbital pose."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=np.float64)
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ], dtype=np.float64)
    
    return Ry @ Rx


def render_mesh_views(mesh_path: Path, output_dir: Path, num_views: int = 16,
                      resolution: int = 512, distance: float = 1.8) -> int:
    """Renders synthetic views and saves multi-channel ground truth."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    # Normalize mesh to unit cube centered at origin
    vertices = mesh.vertices.copy()
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    if scale > 0:
        vertices = (vertices - center) / scale
    mesh.vertices = vertices

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    normal_dir = output_dir / "normals"
    mask_dir = output_dir / "mask"
    for d in (rgb_dir, depth_dir, normal_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    mesh_name = mesh_path.stem
    views_rendered = 0

    # Ray tracing / z-buffer projection
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.linspace(-15, 30, num_views)

    for i, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
        R = _rotation_matrix(float(azimuth), float(elevation))
        rot_v = vertices @ R.T

        # Perspective camera projection
        focal = resolution * 1.2
        cx, cy = resolution / 2.0, resolution / 2.0

        # Project vertices: camera at (0, 0, distance)
        cam_z = rot_v[:, 2] + distance
        cam_x = rot_v[:, 0]
        cam_y = rot_v[:, 1]

        u = np.round((cam_x / cam_z) * focal + cx).astype(np.int32)
        v = np.round((-cam_y / cam_z) * focal + cy).astype(np.int32)

        # Compute vertex normals
        mesh_rot = mesh.copy()
        mesh_rot.vertices = rot_v
        v_normals = mesh_rot.vertex_normals

        # Rasterize onto image buffer
        depth_map = np.full((resolution, resolution), np.inf, dtype=np.float32)
        normal_map = np.zeros((resolution, resolution, 3), dtype=np.float32)
        mask = np.zeros((resolution, resolution), dtype=np.uint8)

        faces = mesh.faces
        for f in faces:
            pts_u = u[f]
            pts_v = v[f]
            pts_z = cam_z[f]
            pts_n = v_normals[f]

            # Bounding box of triangle
            min_u, max_u = max(0, pts_u.min()), min(resolution - 1, pts_u.max())
            min_v, max_v = max(0, pts_v.min()), min(resolution - 1, pts_v.max())
            if min_u >= max_u or min_v >= max_v:
                continue

            # Rasterize triangle using OpenCV fillPoly for robust polygon fill
            poly = np.stack([pts_u, pts_v], axis=1)
            tri_mask = np.zeros((max_v - min_v + 1, max_u - min_u + 1), dtype=np.uint8)
            poly_shifted = poly - np.array([min_u, min_v])
            cv2.fillConvexPoly(tri_mask, poly_shifted, 1)

            mean_z = float(pts_z.mean())
            mean_n = pts_n.mean(axis=0)
            norm = np.linalg.norm(mean_n)
            if norm > 0:
                mean_n /= norm

            sub_depth = depth_map[min_v:max_v+1, min_u:max_u+1]
            update_mask = (tri_mask == 1) & (mean_z < sub_depth)
            sub_depth[update_mask] = mean_z
            depth_map[min_v:max_v+1, min_u:max_u+1] = sub_depth

            sub_norm = normal_map[min_v:max_v+1, min_u:max_u+1]
            sub_norm[update_mask] = (mean_n + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            normal_map[min_v:max_v+1, min_u:max_u+1] = sub_norm

            mask[min_v:max_v+1, min_u:max_u+1][tri_mask == 1] = 255

        # Create shaded RGB from normal map and directional lighting
        valid = depth_map < np.inf
        depth_finite = np.where(valid, depth_map, 0.0)
        
        # Simple Lambertian shading
        light_dir = np.array([0.4, 0.5, 0.77], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        normals_unpacked = normal_map * 2.0 - 1.0
        shading = np.clip(np.sum(normals_unpacked * light_dir, axis=2), 0.0, 1.0)
        shading = 0.25 + 0.75 * shading  # ambient + diffuse
        
        rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        base_color = np.array([180, 190, 210], dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = np.clip(shading * base_color[c] * (mask / 255.0), 0, 255).astype(np.uint8)

        # 16-bit millimeter depth (0 to 65535 mm)
        depth_16bit = np.clip(depth_finite * 1000.0, 0, 65535).astype(np.uint16)
        normals_uint8 = np.clip(normal_map * 255.0, 0, 255).astype(np.uint8)

        # Save files
        view_id = f"{mesh_name}_view_{i:02d}"
        cv2.imwrite(str(rgb_dir / f"{view_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_dir / f"{view_id}.png"), depth_16bit)
        cv2.imwrite(str(normal_dir / f"{view_id}.png"), cv2.cvtColor(normals_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{view_id}.png"), mask)

        views_rendered += 1

    return views_rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synthetic depth training pairs")
    parser.add_argument("--input_mesh", type=Path, default=Path("benchmarks/data/synthetic/gt/cylinder.obj"))
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("training/data/synthetic_renders"))
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    total_views = 0
    if args.input_dir and args.input_dir.exists():
        meshes = list(args.input_dir.glob("*.obj")) + list(args.input_dir.glob("*.glb"))
        print(f"Rendering {len(meshes)} meshes from {args.input_dir}...")
        for m in meshes:
            n = render_mesh_views(m, args.output_dir, args.num_views, args.resolution)
            total_views += n
            print(f"  * {m.stem}: rendered {n} views")
    elif args.input_mesh and args.input_mesh.exists():
        print(f"Rendering single mesh {args.input_mesh}...")
        total_views = render_mesh_views(args.input_mesh, args.output_dir, args.num_views, args.resolution)
        print(f"Rendered {total_views} views to {args.output_dir}")
    else:
        print("No valid input mesh found.")

    print(f"Synthetic dataset generation complete: {total_views} total views written.")


if __name__ == "__main__":
    main()
