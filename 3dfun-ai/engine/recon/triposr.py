"""TripoSR inference using QtMeshEditor's MIT-licensed INT8 ONNX conversion.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR, preset_config
from .export import export_all
from .meshing import MeshResult, mesh_metrics, normalize_transform
from .pipeline import ReconstructionResult, _noop
from .preprocessing import prepare_image
from .render import render_preview, render_turntable
from .runtime import CPU, DIRECTML, InferenceSession, inference_hardware, runtime_status

MODEL_ID = "triposr"
MODEL_REVISION = "01b2f1add36601f964a89fd8a3159919ced4aa40"
MODEL_FILES = {
    "triposr_encoder_int8.onnx": (436065556, "0439dd35121fde195e73f93b0d81113be67ef9c1caa1db6498ae6c16e12fe089"),
    "triposr_decoder.onnx": (180692, "68ab06fb5ed4646c6dc15293ec83de62ba11d50294030795e860bea96cd81706"),
}
MODEL_URL = f"https://huggingface.co/fernandotonon/QtMeshEditor-triposr-onnx/resolve/{MODEL_REVISION}"
RADIUS = 0.87
DENSITY_THRESHOLD = 25.0
CHUNK_SIZE = 4096
GRID_SIZE = {"draft": 64, "standard": 96, "high": 128}


class ReconstructionCancelled(Exception):
    pass


def available() -> bool:
    return all((path := MODELS_DIR / name).is_file() and path.stat().st_size == size
               for name, (size, _) in MODEL_FILES.items())


_TRIPOSR_SESSIONS: dict[tuple[str, str], InferenceSession] = {}


def _session(name: str):
    preferred = runtime_status()["preferred"]
    # These INT8 encoder and point decoder exports fall back on the tested DirectML path.
    # Open CPU sessions directly on DirectML to skip the failing GPU initialization.
    provider = CPU if preferred == DIRECTML else preferred
    key = (name, provider)
    if key in _TRIPOSR_SESSIONS:
        return _TRIPOSR_SESSIONS[key]
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.add_session_config_entry("session.disable_prepacking", "1")
    sess = InferenceSession(MODELS_DIR / name, options, provider)
    _TRIPOSR_SESSIONS[key] = sess
    return sess


def reconstruct_triposr(image: bytes, out: Path, quality: str, options: dict,
                        progress=_noop, cancelled=lambda: False) -> ReconstructionResult:
    from skimage.measure import marching_cubes
    from trimesh import Trimesh

    def report(stage: str, percent: int, message: str) -> None:
        if cancelled():
            raise ReconstructionCancelled()
        progress(stage, percent, message)

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    size = GRID_SIZE[quality]
    cfg = preset_config(
        quality, canvas_size=512, crop_padding_pct=0.075, background="midgray",
        matte_model="u2netp" if options.get("background_removal", True) else "threshold",
        formats=tuple(dict.fromkeys(["glb", *options.get("formats", ["obj", "ply"])])),
        render_preview=options.get("render_preview", True),
        render_turntable=options.get("turntable", False), turntable_frames=12,
    )
    report("PREPROCESSING", 5, "Preparing the reference image")
    prepared = prepare_image(image, cfg)
    cv2.imwrite(str(out / "source_canvas.png"), cv2.cvtColor(prepared.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "mask.png"), prepared.mask.astype(np.uint8) * 255)
    image_tensor = np.ascontiguousarray(prepared.rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    report("DEPTH", 12, "Encoding the image with TripoSR")
    encoder = _session("triposr_encoder_int8.onnx")
    triplane, = encoder.run(["scene_codes"], {"image": image_tensor})
    encoder_provider = encoder.provider
    del encoder
    encoded = time.perf_counter()
    decoder = _session("triposr_decoder.onnx")
    axis = np.linspace(-RADIUS, RADIUS, size, dtype=np.float32)
    density = np.empty(size ** 3, dtype=np.float32)
    last_percent = None
    for offset in range(0, density.size, CHUNK_SIZE):
        percent = 24 + offset * 42 // density.size
        if percent != last_percent and (last_percent is None or percent - last_percent >= 3):
            report("VOLUME", percent, "Sampling the 3D density field")
            last_percent = percent
        indices = np.unravel_index(np.arange(offset, min(offset + CHUNK_SIZE, density.size)), (size,) * 3)
        points = np.column_stack([axis[index] for index in indices])
        values, = decoder.run(["density"], {"scene_codes": triplane, "points": points[None]})
        density[offset:offset + len(points)] = values.ravel()

    report("GEOMETRY", 67, "Extracting the surface")
    if not np.isfinite(density).all() or density.max() <= DENSITY_THRESHOLD:
        raise ValueError("TripoSR could not find a surface. Try a clearer image of one object.")
    # Pad the grid with empty space so a surface at the sampling boundary closes.
    grid = np.pad(density.reshape((size,) * 3), 1)
    spacing = float(axis[1] - axis[0])
    points, faces, _, _ = marching_cubes(grid, DENSITY_THRESHOLD, spacing=(spacing,) * 3,
                                       allow_degenerate=False)
    points = np.asarray(points - RADIUS - spacing, dtype=np.float32)
    report("TEXTURE", 76, "Predicting vertex colours")
    colors = np.empty_like(points)
    for offset in range(0, len(points), CHUNK_SIZE):
        chunk = points[offset:offset + CHUNK_SIZE]
        rgb, = decoder.run(["color"], {"scene_codes": triplane, "points": chunk[None]})
        colors[offset:offset + len(chunk)] = rgb[0]
    np.clip(colors, 0, 1, out=colors)
    hardware = inference_hardware(matting=prepared.meta["matte_provider"], encoder=encoder_provider, decoder=decoder.provider)
    del decoder

    # Rotate -90° around X and +90° around Y for a Y-up, forward-facing glTF.
    vertices, transform = normalize_transform(points[:, [1, 2, 0]] * [-1, 1, -1])
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    topology = Trimesh(vertices=vertices, faces=faces, process=False)
    if topology.volume < 0:
        faces = faces[:, ::-1].copy()
        topology.faces = faces
    metrics = mesh_metrics(vertices, faces)
    metrics.update(watertight_trimesh=bool(topology.is_watertight), normalization=transform,
                   textured=False, vertex_colors=True)
    uv = np.zeros((len(vertices), 2), dtype=np.float32)
    mesh = MeshResult(vertices, faces, uv, uv[faces.ravel()], metrics, vertex_colors=colors,
                      vertex_normals=np.asarray(topology.vertex_normals, dtype=np.float32))
    report("EXPORT", 85, "Writing model files")
    files = export_all(out, mesh, None, {"asset_name": "TripoSR reconstruction", "model": MODEL_ID}, cfg).files

    report("RENDER", 92, "Rendering previews")
    if cfg.render_preview:
        render_preview(mesh, None, out / "preview.png", cfg.render_size)
    if cfg.render_turntable:
        render_turntable(mesh, None, out / "turntable", cfg.turntable_frames, max(224, cfg.render_size // 2))
    config = {"quality": quality, "grid_size": size, "chunk_size": CHUNK_SIZE,
              "model_revision": MODEL_REVISION, **cfg.to_dict()}
    elapsed = time.perf_counter() - started
    manifest = {
        "manifest_version": "1.0", "engine": {"name": "TripoSR", "version": MODEL_REVISION,
        "pipeline": "image encoder, triplane decoder, marching cubes and vertex colours"},
        "config": config, "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "input": prepared.meta, "mesh": metrics, "hardware": hardware,
        "timings_seconds": {"encode_and_preprocess": round(encoded - started, 4), "total": round(elapsed, 4)},
        "files": {key: file.name for key, file in files.items()},
        "limitations": ["TripoSR infers surfaces the camera cannot see.", "Appearance is stored as vertex colours."],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report("COMPLETED", 100, "Reconstruction complete")
    return ReconstructionResult(out, files, manifest)
