"""Render RGB, perspective z-depth, masks, normals and camera metadata from meshes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh


def rotation(yaw, pitch):
    y, p = map(math.radians, (yaw, pitch))
    return (np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            @ np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]]))


def render_view(vertices, faces, normals, size, focal):
    image = vertices[:, :2] * np.array([1, -1]) / vertices[:, 2:] * focal + size / 2
    depth = np.full((size, size), np.inf)
    normal = np.zeros((size, size, 3))
    for face in faces:
        u, v = image[face].T
        lo = np.maximum(0, np.ceil([u.min() - .5, v.min() - .5]).astype(int))
        hi = np.minimum(size - 1, np.floor([u.max() - .5, v.max() - .5]).astype(int))
        if (lo > hi).any():
            continue
        ax, bx = u[1:] - u[0]
        ay, by = v[1:] - v[0]
        area = ax * by - ay * bx
        if abs(area) < 1e-10:
            continue
        x = np.arange(lo[0], hi[0] + 1)[None, :] + .5 - u[0]
        y = np.arange(lo[1], hi[1] + 1)[:, None] + .5 - v[0]
        b1, b2 = (x * by - y * bx) / area, (ax * y - ay * x) / area
        b0 = 1 - b1 - b2
        inside = (b0 >= -1e-8) & (b1 >= -1e-8) & (b2 >= -1e-8)
        weights = np.stack((b0, b1, b2), axis=-1)[inside] / vertices[face, 2]
        z = 1 / weights.sum(axis=-1)
        yy, xx = np.where(inside)
        yy, xx = yy + lo[1], xx + lo[0]
        nearer = z < depth[yy, xx]
        yy, xx = yy[nearer], xx[nearer]
        depth[yy, xx] = z[nearer]
        n = weights[nearer] @ normals[face]
        normal[yy, xx] = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return depth, normal


def render_mesh_views(mesh_path, output_dir, num_views=8, resolution=96, distance=1.8):
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if mesh.is_empty or mesh.extents.max() <= 0 or num_views < 1 or not 32 <= resolution <= 2048 or not 1.2 <= distance <= 64:
        raise ValueError("Need a valid mesh, >=1 views, resolution 32-2048 and distance 1.2-64.")
    vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) / mesh.extents.max()
    for kind in ("rgb", "depth", "mask", "normals", "cameras"):
        (output_dir / kind).mkdir(parents=True, exist_ok=True)
    light = np.array([.4, .5, -.77])
    light /= np.linalg.norm(light)
    focal = resolution * 1.2
    cameras = []
    for i, (az, el) in enumerate(zip(np.linspace(0, 360, num_views, endpoint=False),
                                     np.linspace(-15, 30, num_views))):
        rot = rotation(az, el)
        depth, normal = render_view(vertices @ rot.T + [0, 0, distance], mesh.faces,
                                    mesh.vertex_normals @ rot.T, resolution, focal)
        mask = np.isfinite(depth)
        if not mask.any():
            raise ValueError(f"No visible pixels for {mesh_path} view {i}")
        shade = .25 + .75 * np.clip(normal @ light, 0, 1)
        name = f"{mesh_path.stem}_view_{i:02d}.png"
        channels = {
            "rgb": (shade[..., None] * np.array([210, 190, 180]) * mask[..., None]).astype(np.uint8),
            "depth": np.rint(np.where(mask, depth, 0) * 1000).astype(np.uint16),
            "mask": mask.astype(np.uint8) * 255,
            "normals": ((normal[..., ::-1] + 1) * 127.5 * mask[..., None]).astype(np.uint8),
        }
        for kind, pixels in channels.items():
            path = output_dir / kind / name
            if not cv2.imwrite(str(path), pixels):
                raise OSError(f"Could not write {path}")
        cameras.append({"image": name, "mesh": mesh_path.stem, "azimuth_deg": float(az),
                        "elevation_deg": float(el), "focal_px": focal, "cx": resolution / 2,
                        "cy": resolution / 2, "world_to_camera_rotation": rot.tolist(),
                        "translation": [0, 0, distance]})
    (output_dir / "cameras" / f"{mesh_path.stem}.json").write_text(
        json.dumps(cameras, indent=2), encoding="utf-8")
    return num_views


def procedural_meshes(folder):
    folder.mkdir(parents=True, exist_ok=True)
    ellipsoid = trimesh.creation.icosphere(2)
    ellipsoid.apply_scale((1, .65, .8))
    shapes = {
        "box": trimesh.creation.box((1, .8, .35)),
        "tall_box": trimesh.creation.box((.58, 1, .54)),
        "cylinder": trimesh.creation.cylinder(.4, .9, 32),
        "wide_cylinder": trimesh.creation.cylinder(.48, .35, 32),
        "cone": trimesh.creation.cone(.47, 1, 32),
        "sphere": trimesh.creation.icosphere(2),
        "ellipsoid": ellipsoid,
        "capsule": trimesh.creation.capsule(height=.74, radius=.29, count=[12, 24]),
    }
    paths = []
    for name, mesh in sorted(shapes.items()):
        target = folder / f"{name}.obj"
        mesh.export(target)
        paths.append(target)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-mesh", "--input_mesh", type=Path)
    source.add_argument("--input-dir", "--input_dir", type=Path)
    source.add_argument("--procedural", action="store_true", help="Generate and render eight reference meshes.")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("out/training_renders"))
    parser.add_argument("--num-views", "--num_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=96)
    args = parser.parse_args()
    if args.procedural:
        meshes = procedural_meshes(args.output_dir / "meshes")
    elif args.input_dir:
        meshes = sorted(m for m in args.input_dir.iterdir() if m.suffix.lower() in {".obj", ".glb"}) if args.input_dir.is_dir() else []
    else:
        meshes = [args.input_mesh] if args.input_mesh.is_file() else []
    if not meshes or len({p.stem for p in meshes}) != len(meshes):
        parser.error("Provide OBJ or GLB input meshes with unique filename stems.")
    for mesh in meshes:
        print(f"{mesh.stem}: {render_mesh_views(mesh, args.output_dir, args.num_views, args.resolution)} views")


if __name__ == "__main__":
    main()
