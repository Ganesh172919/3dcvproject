"""End-to-end image-to-3D reconstruction pipeline (stage orchestration).

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import PipelineConfig, preset_config
from .depth import DEPTH_MODELS, estimate_depth
from .export import export_all
from .meshing import build_mesh
from .preprocessing import prepare_image
from .render import render_preview, render_turntable
from .runtime import inference_hardware
from .texturing import bake
from .volumetrics import build_volume, calibrate_voxel_res

log = logging.getLogger(__name__)

STAGES: list[tuple[str, str]] = [
    ("VALIDATING", "Validating upload"),
    ("PREPROCESSING", "Matting and normalising the reference"),
    ("DEPTH", "Estimating monocular depth"),
    ("VOLUME", "Inflating silhouette to a volume"),
    ("GEOMETRY", "Extracting and cleaning the surface"),
    ("UV", "Unwrapping UVs"),
    ("TEXTURE", "Baking albedo and PBR maps"),
    ("EXPORT", "Writing GLB / OBJ / PLY"),
    ("RENDER", "Rendering previews and turntable"),
]
STAGE_WEIGHTS = {
    "VALIDATING": 1, "PREPROCESSING": 6, "DEPTH": 14, "VOLUME": 12,
    "GEOMETRY": 26, "UV": 8, "TEXTURE": 22, "EXPORT": 5, "RENDER": 6,
}


@dataclass
class ReconstructionResult:
    out_dir: Path
    files: dict[str, Path]
    manifest: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


ProgressFn = Callable[[str, int, str], None]


def _noop(stage: str, progress: int, message: str) -> None:  # pragma: no cover
    return None


def reconstruct(image: bytes | str | Path, out_dir: str | Path,
                cfg: PipelineConfig | None = None,
                quality: str | None = None,
                progress: ProgressFn | None = None,
                asset_name: str = "model") -> ReconstructionResult:
    cfg = cfg or (preset_config(quality) if quality else PipelineConfig())
    progress = progress or _noop
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    warnings: list[str] = []
    done_weight = 0

    def step(stage: str, message: str) -> None:
        nonlocal done_weight
        progress(stage, int(done_weight), message)
        done_weight += STAGE_WEIGHTS.get(stage, 5)

    t0 = time.perf_counter()

    step("VALIDATING", "Validating upload")
    t = time.perf_counter()
    prepared = prepare_image(image, cfg)
    timings["preprocessing"] = time.perf_counter() - t

    progress("PREPROCESSING", int(done_weight), "Object matted and centred")
    cv2.imwrite(str(out / "source_canvas.png"),
                cv2.cvtColor(prepared.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "mask.png"), (prepared.mask.astype(np.uint8) * 255))
    # Educational: Save soft probability map and trimap
    prob_u8 = np.clip(prepared.alpha * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "prob_map.png"), prob_u8)
    trimap = np.zeros_like(prob_u8)
    trimap[prob_u8 > 200] = 255
    trimap[(prob_u8 >= 30) & (prob_u8 <= 200)] = 128
    cv2.imwrite(str(out / "trimap.png"), trimap)
    if prepared.meta["object_pixel_fraction"] < 0.02:
        warnings.append("Object occupies under 2% of the frame; reconstruction may be coarse.")

    t = time.perf_counter()
    step("DEPTH", f"Estimating monocular depth ({DEPTH_MODELS[cfg.depth_model][1]})")
    depth = estimate_depth(prepared.rgb, prepared.mask, cfg)
    timings["depth"] = time.perf_counter() - t
    heat = cv2.applyColorMap((depth.closeness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / "depth.png"), heat)
    cv2.imwrite(str(out / "depth_gray.png"), (depth.closeness * 255).astype(np.uint8))
    depth_16bit = (np.clip(depth.closeness, 0, 1) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(out / "depth_16bit.png"), depth_16bit)

    # Educational: Sobel spatial depth gradients -> 2D Surface Normal Orientation field
    dx = cv2.Sobel(depth.closeness, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth.closeness, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(dx**2 + dy**2 + 0.1**2)
    nx = (-dx / norm + 1.0) * 0.5
    ny = (-dy / norm + 1.0) * 0.5
    nz = (0.1 / norm + 1.0) * 0.5
    normals_rgb = (np.stack([nx, ny, nz], axis=-1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "sobel_gradients.png"), cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "normals.png"), cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))

    # Educational: Camera Ray Unprojection -> 3D Point Cloud (.ply)
    try:
        H, W = prepared.mask.shape
        fx = fy = 0.5 * W / np.tan(0.5 * np.radians(45.0))
        cx, cy = W / 2.0, H / 2.0
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        val_m = prepared.mask & (depth.closeness > 0.05)
        if np.count_nonzero(val_m) == 0:
            val_m = prepared.mask
        u_pts, v_pts = u_coords[val_m], v_coords[val_m]
        z_pts = 2.8 - depth.closeness[val_m] * 1.6
        x_pts = (u_pts - cx) * z_pts / fx
        y_pts = -(v_pts - cy) * z_pts / fy
        c_pts = prepared.rgb[val_m]
        N_pts = len(x_pts)
        if N_pts > 20000:
            sub = N_pts // 20000
            x_pts, y_pts, z_pts, c_pts = x_pts[::sub], y_pts[::sub], z_pts[::sub], c_pts[::sub]
            N_pts = len(x_pts)
        ply_lines = ["ply", "format ascii 1.0", f"element vertex {N_pts}",
                     "property float x", "property float y", "property float z",
                     "property uchar red", "property uchar green", "property uchar blue", "end_header"]
        for pi in range(N_pts):
            ply_lines.append(f"{x_pts[pi]:.4f} {y_pts[pi]:.4f} {z_pts[pi]:.4f} {int(c_pts[pi,0])} {int(c_pts[pi,1])} {int(c_pts[pi,2])}")
        (out / "pointcloud.ply").write_text("\n".join(ply_lines) + "\n", encoding="ascii")
    except Exception as e:
        log.warning("Pointcloud export skipped: %s", e)
    done_weight = 1 + 6

    t = time.perf_counter()
    step("VOLUME", "Building silhouette-constrained volume")
    calibration = calibrate_voxel_res(prepared.mask, depth.closeness, cfg)
    cfg = cfg.with_overrides(voxel_res=calibration["chosen_res"])
    volume = build_volume(prepared.mask, depth.closeness, cfg)
    timings["volume"] = time.perf_counter() - t

    # Educational: Save Orthogonal SDF Cross-Section Slices
    try:
        sdf = volume.sdf
        D, H, W = sdf.shape
        axial = sdf[D // 2, :, :]
        coronal = sdf[:, H // 2, :]
        sagittal = sdf[:, :, W // 2]
        for sname, sarr in [("sdf_axial.png", axial), ("sdf_coronal.png", coronal), ("sdf_sagittal.png", sagittal), ("voxels_slice.png", axial)]:
            s_u8 = np.clip((sarr / 0.3) * 127 + 128, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / sname), cv2.applyColorMap(s_u8, cv2.COLORMAP_TWILIGHT_SHIFTED))
    except Exception as e:
        log.warning("SDF slices export skipped: %s", e)
    done_weight = 1 + 6 + 14

    t = time.perf_counter()
    step("GEOMETRY", "Extracting isosurface and cleaning mesh")
    mesh = build_mesh(volume, cfg)
    timings["geometry"] = time.perf_counter() - t

    def _write_obj(filepath: Path, v: np.ndarray, f: np.ndarray):
        lines = [f"v {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}" for pt in v]
        lines.extend([f"f {face[0]+1} {face[1]+1} {face[2]+1}" for face in f])
        filepath.write_text("\n".join(lines) + "\n", encoding="ascii")

    # Stage 5: Raw Marching Cubes un-smoothed geometry
    if mesh.raw_vertices is not None and mesh.raw_faces is not None:
        try:
            _write_obj(out / "raw_mesh.obj", mesh.raw_vertices, mesh.raw_faces)
        except Exception as e:
            log.warning("Raw mesh obj export skipped: %s", e)

    # Stage 6: Taubin-smoothed and cleaned manifold mesh
    if mesh.validation_vertices is not None and mesh.validation_faces is not None:
        try:
            _write_obj(out / "smoothed.obj", mesh.validation_vertices, mesh.validation_faces)
        except Exception as e:
            log.warning("Smoothed mesh obj export skipped: %s", e)
    done_weight = 1 + 6 + 14 + 12

    progress("UV", int(done_weight), "UV atlas unwrapped")
    # Educational: UV Atlas & Checkerboard visualization
    try:
        uv_size = 512
        uv_canvas = np.zeros((uv_size, uv_size, 3), dtype=np.uint8)
        tile = 32
        for i in range(0, uv_size, tile):
            for j in range(0, uv_size, tile):
                uv_canvas[i:i+tile, j:j+tile] = [215, 220, 230] if ((i // tile + j // tile) % 2 == 0) else [170, 175, 185]
        uvs = (mesh.face_uv * (uv_size - 1)).astype(np.int32)
        for fi in range(0, min(len(uvs) - 2, 3000), 3):
            cv2.polylines(uv_canvas, [uvs[fi:fi+3]], isClosed=True, color=(60, 60, 220), thickness=1)
        cv2.imwrite(str(out / "uv_atlas.png"), uv_canvas)
        cv2.imwrite(str(out / "uv_layout.png"), uv_canvas)
    except Exception as e:
        log.warning("UV atlas render skipped: %s", e)
    done_weight += 8

    t = time.perf_counter()
    step("TEXTURE", "Baking albedo and estimating PBR maps")
    tex = None
    try:
        tex = bake(mesh, prepared.rgb, prepared.alpha, volume.sdf, volume.origin,
                   volume.spacing, mesh.metrics["normalization"], cfg)
    except Exception as exc:  # texturing must never kill geometry
        log.exception("Texture baking failed")
        warnings.append(f"Texture baking failed: {exc}")
    timings["texture"] = time.perf_counter() - t

    if mesh.metrics.get("boundary_edges", 0) > 0:
        warnings.append(f"{mesh.metrics['boundary_edges']} boundary edges remain (not fully watertight).")

    meta = {
        "asset_name": asset_name,
        "pipeline_version": cfg.pipeline_version,
        "config_hash": cfg.config_hash(),
        "quality": cfg.quality,
        "depth_model": cfg.depth_model,
        "matte_method": prepared.meta["matte_method"],
        "face_count": mesh.metrics["faces"],
        "generated_by": "3DCV Lab Group 17",
    }

    t = time.perf_counter()
    step("EXPORT", "Writing model files")
    exported = export_all(out, mesh, tex, meta, cfg)
    timings["export"] = time.perf_counter() - t
    done_weight = 1 + 6 + 14 + 12 + 26 + 8 + 22

    t = time.perf_counter()
    step("RENDER", "Rendering previews")
    previews: dict[str, Any] = {}
    turntable: dict[str, Any] = {}
    if cfg.render_preview:
        previews = render_preview(mesh, tex, out / "preview.png", cfg.render_size)
        progress("RENDER", min(99, done_weight + 3), "Preview rendered")
    if cfg.render_turntable:
        turntable = render_turntable(mesh, tex, out / "turntable", cfg.turntable_frames,
                                     max(224, cfg.render_size // 2))
    timings["render"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0

    # ---- mesh quality metrics -----------------------------------------
    metrics = dict(mesh.metrics)
    metrics.update({
        "texture": tex.stats if tex else None,
        "uv_atlas": "planar-view-projection",
    })
    if tex is None:
        metrics["textured"] = False
    else:
        metrics["textured"] = True

    # watertightness probe through trimesh (independent implementation)
    try:
        import trimesh
        vv = mesh.validation_vertices if mesh.validation_vertices is not None else mesh.vertices
        ff = mesh.validation_faces if mesh.validation_faces is not None else mesh.faces
        tm = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
        metrics["watertight_trimesh"] = bool(tm.is_watertight)
        metrics["volume"] = float(abs(tm.volume)) if tm.is_watertight else None
        metrics["euler_trimesh"] = int(tm.euler_number)
        if tm.is_watertight and tm.volume < 0:
            warnings.append("Inverted normals detected by volume sign.")
        # winding: signed volume positive means outward-facing normals
        if tm.is_watertight:
            metrics["signed_volume"] = float(tm.volume)
    except Exception as exc:  # pragma: no cover
        log.warning("trimesh validation failed: %s", exc)

    manifest = {
        "manifest_version": "1.0",
        "engine": {
            "name": "3DCV Lab Group 17 reconstruction engine",
            "version": cfg.pipeline_version,
            "pipeline": "silhouette-constrained monocular depth inflation + marching cubes",
            "depth_model": cfg.depth_model,
            "matte_model": prepared.meta["matte_method"],
        },
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "input": prepared.meta,
        "volume": volume.stats,
        "resolution_calibration": calibration,
        "mesh": metrics,
        "texture": tex.stats if tex else None,
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        "files": {k: str(v.relative_to(out)) for k, v in exported.files.items()},
        "previews": {k: str(Path(v).relative_to(out)) for k, v in previews.items()},
        "turntable": {k: (str(Path(v).relative_to(out)) if isinstance(v, str) and Path(v).is_absolute() else v)
                      for k, v in turntable.items()},
        "warnings": warnings,
        "limitations": [
            "Single-view reconstruction: surfaces never observed by the camera are "
            "inferred from a depth-driven shell and mirrored texture.",
            "Back-facing texture is a horizontal mirror of the input (symmetric-object assumption).",
            "Roughness/metallic maps are estimated from image statistics, not measured BRDFs.",
        ],
        "hardware": inference_hardware(matting=prepared.meta["matte_provider"], depth=depth.stats["provider"]),
    }
    import json
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress("COMPLETED", 100, "Reconstruction complete")
    return ReconstructionResult(out_dir=out, files=exported.files, manifest=manifest,
                                timings=timings, warnings=warnings)
