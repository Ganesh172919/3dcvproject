"""End-to-end image-to-3D reconstruction pipeline (stage orchestration)."""
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
    if prepared.meta["object_pixel_fraction"] < 0.02:
        warnings.append("Object occupies under 2% of the frame; reconstruction may be coarse.")

    t = time.perf_counter()
    step("DEPTH", f"Estimating monocular depth ({DEPTH_MODELS[cfg.depth_model][1]})")
    depth = estimate_depth(prepared.rgb, prepared.mask, cfg)
    timings["depth"] = time.perf_counter() - t
    heat = cv2.applyColorMap((depth.closeness * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / "depth.png"), heat)
    done_weight = 1 + 6

    t = time.perf_counter()
    step("VOLUME", "Building silhouette-constrained volume")
    calibration = calibrate_voxel_res(prepared.mask, depth.closeness, cfg)
    cfg = cfg.with_overrides(voxel_res=calibration["chosen_res"])
    volume = build_volume(prepared.mask, depth.closeness, cfg)
    timings["volume"] = time.perf_counter() - t
    done_weight = 1 + 6 + 14

    t = time.perf_counter()
    step("GEOMETRY", "Extracting isosurface and cleaning mesh")
    mesh = build_mesh(volume, cfg)
    timings["geometry"] = time.perf_counter() - t
    done_weight = 1 + 6 + 14 + 12

    progress("UV", int(done_weight), "UV atlas unwrapped")
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
        "generated_by": "3DFUN AI",
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
    from .meshing import mesh_metrics
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
            "name": "3DFUN AI reconstruction engine",
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
