"""Reproducible benchmark driver for 3DFUN AI.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

One command runs the whole evaluation and writes every artefact the paper and
the slide deck consume:

    python benchmarks/run_benchmarks.py            # full run
    python benchmarks/run_benchmarks.py --quick    # 4 scenes, 1 repetition

Outputs (all under ``results/``):

    results/raw/runs.csv                 one row per (entry, method, repetition)
    results/raw/runs.json                the same rows as JSON, plus metadata
    results/summary/main_table.csv       mean +- std per method
    results/summary/main_table.json      same, structured
    results/summary/ablation_table.csv   config-vs-config comparison
    results/summary/failures.csv         cases below the quality threshold
    results/logs/<method>_<scene>.log    per-run stage log
    results/environment.json             hardware, library versions, seed, commit

Nothing in this file types a metric by hand: every number is computed here from
the meshes the methods actually produced.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
try:
    import resource
except ImportError:
    resource = None
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
for path in (str(ROOT), str(ENGINE)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:                                                    # imported as a package
    from . import metrics as M
    from .baselines import baseline_tsdf, baseline_visual_hull, baseline_midas, baseline_triposr
except ImportError:                                     # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import metrics as M                                  # noqa: E402
    from baselines import baseline_tsdf, baseline_visual_hull, baseline_midas, baseline_triposr   # noqa: E402

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "results"
SEED = 20261108
METHODS = ("proposed", "tsdf", "visual_hull")


# --------------------------------------------------------------------------- utils
def _rss_mb() -> float:
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(idx)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    """Minimal OBJ so every method leaves the same inspectable artefact."""
    lines = [f"# 3dfun benchmark mesh  vertices={len(v)} faces={len(f)}", "o mesh"]
    lines += [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in np.asarray(v, float)]
    lines += [f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}"
              for a, b, c in np.asarray(f, np.int64)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mesh_stats(v: np.ndarray, f: np.ndarray) -> dict[str, Any]:
    import trimesh
    from engine.recon.meshing import mesh_metrics  # noqa: F401  (keeps parity)

    # OBJ/GLB export stores one position per corner so that UV seams can be drawn
    # without splitting triangles. Weld by exact position (1e-6) to recover the
    # authored topology; this is deterministic and does not depend on the
    # library's tolerance-based merging.
    verts = np.round(np.asarray(v, np.float64), 6)
    uniq, inverse = np.unique(verts, axis=0, return_inverse=True)
    faces = inverse[np.asarray(f, np.int64)]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2]))
    faces = faces[keep]
    welded = trimesh.Trimesh(vertices=uniq, faces=faces, process=False)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    area = float(welded.area)
    raw_mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    return {
        "triangles": int(len(raw_mesh.faces)),
        "vertices": int(len(uniq)),
        "watertight_raw": bool(raw_mesh.is_watertight),
        "watertight": bool(welded.is_watertight),
        "boundary_edges": int((mesh.edges_sorted.shape[0] - mesh.edges_unique.shape[0])),
        "euler": int(mesh.euler_number),
        "surface_area": area,
        "bbox_volume": float(np.prod(mesh.bounds[1] - mesh.bounds[0])),
        "mean_edge": float(mesh.edges_unique_length.mean()) if len(mesh.edges_unique) else 0.0,
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def _versions() -> dict[str, str]:
    mods = ["numpy", "scipy", "cv2", "skimage", "open3d", "trimesh", "onnxruntime",
            "fast_simplification", "matplotlib", "PIL", "pandas"]
    out = {}
    for name in mods:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    return out


def environment() -> dict[str, Any]:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        mem = {line.split(":")[0]: line.split(":")[1].strip() for line in fh.read().splitlines()[:4]}
    with open("/proc/cpuinfo", encoding="utf-8") as fh:
        cpu = [line for line in fh.read().splitlines() if line.startswith("model name")]
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu = ""
    dataset = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu[0].split(":", 1)[1].strip() if cpu else "unknown",
        "cpu_count": os.cpu_count(),
        "memory": mem,
        "gpu": gpu or "none (all runs are CPU-only)",
        "seed": SEED,
        "git_commit": _git_commit(),
        "dataset": {"version": dataset["version"], "checksum": dataset["checksum"],
                    "scenes": dataset["scenes"]},
        "libraries": _versions(),
        "env": {k: v for k, v in os.environ.items()
                if k in {"THREEFUN_PROVIDER", "GENERATION_PROVIDER", "OMP_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED"}},
        "notes": [
            "CPU-only container: GPU-dependent baselines are not evaluated.",
            "Every row in raw/runs.csv comes from an executed run of the method.",
        ],
    }


# ------------------------------------------------------------------- run measures
@dataclass
class RunRow:
    entry_id: str
    split: str
    method: str
    repetition: int
    ok: bool
    seconds: float
    peak_rss_mb: float
    triangles: int
    vertices: int
    watertight: bool
    chamfer: float | None = None
    f_score: float | None = None
    precision: float | None = None
    recall: float | None = None
    iou: float | None = None
    normal_consistency: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    lpips_proxy: float | None = None
    silhouette_iou: float | None = None
    file_mb: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


IMAGE_SIZE = 448       # square canvas used for every rendered-image metric


def _reference_view(image_path: Path) -> np.ndarray:
    """The input photograph, cropped/padded to the square evaluation canvas.

    Photographs are not square while the renderer always produces square
    images, so the input is padded (never stretched) with its own border colour
    and then resized. The same canvas is used for every method.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    side = max(h, w)
    border = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3),
                             img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    canvas = np.full((side, side, 3), fill, np.uint8)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top:top + h, left:left + w] = img
    return cv2.resize(canvas, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _render_match(v: np.ndarray, f: np.ndarray, size: int,
                  rotate_deg: float = 0.0) -> np.ndarray:
    """Render a mesh from the canonical (PCA-aligned) reference viewpoint.

    Both the prediction and the reference surface are placed in their own
    canonical pose before rendering, so the comparison is pose-invariant and
    only the reconstructed shape is being scored.
    """
    from recon.meshing import MeshResult
    from recon.render import render_view

    vn = M.ict_normalize(v)
    if rotate_deg:
        theta = np.radians(rotate_deg)
        rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        vn = vn @ rot.T
    mesh = MeshResult(vertices=vn.astype(np.float32), faces=f.astype(np.int32),
                      vertex_uv=np.zeros((len(vn), 2), np.float32),
                      face_uv=np.zeros((len(f) * 3, 2), np.float32),
                      metrics={}, validation_vertices=vn.astype(np.float32),
                      validation_faces=f.astype(np.int32))
    # Single-sided per-pixel rasterisation drops triangles whose winding faces
    # away from the camera, which looks like holes and fissures in an otherwise
    # closed mesh. The flat (two-sided) rasteriser gives the solid silhouette the
    # comparison needs; it is applied identically to every method.
    return render_view(mesh, None, size, azimuth_deg=0.0, elevation_deg=8.0,
                       mode="clay", supersample=2, per_pixel=False)


def _silhouette(img: np.ndarray) -> np.ndarray:
    """Object mask of a rendered or photographed image.

    The renderer paints the object on a dark gradient, the photographs sit on a
    light background, so the mask is built from local contrast against the
    image border rather than from an absolute threshold.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    border = np.concatenate([grey[:5].ravel(), grey[-5:].ravel(),
                             grey[:, :5].ravel(), grey[:, -5:].ravel()])
    background = float(np.median(border))
    spread = float(np.percentile(border, 90) - np.percentile(border, 10)) + 1.0
    diff = np.abs(grey - background)
    mask = (diff > max(10.0, 0.45 * diff.max())).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # keep only the connected component nearest the image centre
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        centre = np.array([grey.shape[1] / 2, grey.shape[0] / 2])
        areas = stats[1:, cv2.CC_STAT_AREA]
        dists = np.linalg.norm(centroids[1:] - centre, axis=1)
        score = areas / (dists + 1.0)
        keep = 1 + int(np.argmax(score))
        mask = (labels == keep).astype(np.uint8)
    return mask.astype(np.float32)


# ------------------------------------------------------------------ method runners
def run_proposed(image: Path, out_dir: Path, quality: str, config: dict | None = None,
                 log: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    import recon

    if config:
        from recon.config import preset_config
        cfg = preset_config(quality).with_overrides(**config)
    else:
        cfg = None
    t0 = time.perf_counter()
    res = recon.reconstruct(image, out_dir, cfg=cfg, quality=None if cfg else quality)
    seconds = time.perf_counter() - t0
    if log is not None:
        for stage, took in res.timings.items():
            log.append(f"  stage {stage:14s} {took:8.3f}s")
    import trimesh

    mesh = trimesh.load(out_dir / res.files["obj"], force="mesh", process=False)
    manifest = res.manifest
    return (np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64),
            {"seconds": seconds, "manifest": manifest,
             "file_mb": sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6,
             "stage_timings": res.timings, "warnings": res.warnings})


def run_baseline(method: str, image: Path, out_dir: Path, log: list[str] | None = None):
    t0 = time.perf_counter()
    if method == "tsdf":
        res = baseline_tsdf.run(image)
    elif method == "visual_hull":
        res = baseline_visual_hull.run(image)
    elif method == "midas":
        res = baseline_midas.run(image)
    elif method == "triposr":
        res = baseline_triposr.run(image)
    else:  # pragma: no cover
        raise ValueError(method)
    seconds = time.perf_counter() - t0
    if log is not None:
        log.append(f"  {method}: {len(res.faces)} faces in {res.seconds:.2f}s")
    np.save(out_dir / f"{method}_vertices.npy", res.vertices)
    np.save(out_dir / f"{method}_faces.npy", res.faces)
    _write_obj(out_dir / "model.obj", res.vertices, res.faces)
    size = (out_dir / f"{method}_vertices.npy").stat().st_size + \
           (out_dir / f"{method}_faces.npy").stat().st_size
    return res.vertices.astype(np.float64), res.faces.astype(np.int64), {
        "seconds": seconds, "file_mb": size / 1e6}


def evaluate_run(entry: dict, method: str, v: np.ndarray, f: np.ndarray,
                 seconds: float, extra: dict, repetition: int) -> RunRow:
    image_path = DATA / entry["image"]
    ref_img = _reference_view(image_path)
    size = IMAGE_SIZE
    ref_mask = _silhouette(ref_img)
    # The reference pose is unknown for a single photograph, so the render is
    # scored at the yaw that best matches the input silhouette. The same search
    # is applied to every method, so the comparison stays fair.
    best = None
    probe_size = 192                        # cheap silhouette search grid
    ref_small = cv2.resize(ref_mask, (probe_size, probe_size))
    for rotate in range(0, 360, 30):
        candidate = _render_match(v, f, probe_size, rotate_deg=rotate)
        score = M.silhouette_iou(_silhouette(candidate), ref_small)
        if best is None or score > best[0]:
            best = (score, rotate)
    render = _render_match(v, f, size, rotate_deg=best[1])
    stats = _mesh_stats(v, f)
    row = RunRow(entry_id=entry["id"], split=entry["split"], method=method,
                 repetition=repetition, ok=True, seconds=seconds,
                 peak_rss_mb=_rss_mb(), **{k: stats[k] for k in
                                           ("triangles", "vertices", "watertight")})
    row.file_mb = float(extra.get("file_mb") or 0.0)
    row.psnr = M.psnr(render, ref_img)
    row.ssim = M.ssim(render, ref_img)
    row.lpips_proxy = M.lpips_proxy(render, ref_img)
    row.silhouette_iou = M.silhouette_iou(_silhouette(render), _silhouette(ref_img))
    if entry.get("gt_mesh"):
        gt_v, gt_f = _load_obj(DATA / entry["gt_mesh"])
        scores = M.evaluate_geometry(v, f, gt_v, gt_f, samples=20000, tau=0.01, seed=SEED)
        row.chamfer = scores.chamfer
        row.f_score = scores.f_score
        row.precision = scores.precision
        row.recall = scores.recall
        row.iou = scores.iou
        row.normal_consistency = scores.normal_consistency
    row.extra = {
        "stage_timings": extra.get("stage_timings") or {},
        "warnings": (extra.get("warnings") or [])[:4],
        "file_mb": extra.get("file_mb"),
    }
    return row


# ------------------------------------------------------------------------ driver
def load_dataset() -> dict:
    return json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))


def _load_progress(path: Path) -> list[RunRow]:
    """Rows already written by an interrupted run (append-only JSONL)."""
    rows: list[RunRow] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(RunRow(**json.loads(line)))
        except TypeError:
            continue
    return rows


def run(quick: bool = False, repetitions: int = 3, quality: str = "draft",
        methods: tuple[str, ...] = METHODS, force: bool = False,
        quiet: bool = False, resume: bool = True) -> dict:
    _seed_everything(SEED)
    dataset = load_dataset()
    entries = [e for e in dataset["entries"] if e["split"] == "synthetic" or e.get("gt_mesh")]
    # all entries are evaluated; the real photographs simply have no geometry GT
    entries = dataset["entries"]
    if quick:
        keep = {"synthetic/box", "synthetic/sphere", "synthetic/stairs", "real/sneaker"}
        entries = [e for e in entries if e["id"] in keep]
        repetitions = 1

    raw_dir = RESULTS / "raw" / "meshes"
    log_dir = RESULTS / "logs"
    for d in (raw_dir, log_dir, RESULTS / "summary"):
        d.mkdir(parents=True, exist_ok=True)
    if force:
        for stale in raw_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

    progress_path = RESULTS / "raw" / "runs.jsonl"
    if force and progress_path.exists():
        progress_path.unlink()
    rows: list[RunRow] = _load_progress(progress_path) if resume else []
    done = {(r.entry_id, r.method, r.repetition) for r in rows if r.ok}
    progress_times = {(r.entry_id, r.method, r.repetition): r.seconds
                      for r in _load_progress(progress_path) if r.ok and r.seconds}
    if rows and not quiet:
        print(f"resuming: {len(done)} runs already recorded", flush=True)

    for entry in entries:
        image = DATA / entry["image"]
        for method in methods:
            for rep in range(1, repetitions + 1):
                log: list[str] = [f"{entry['id']} | {method} | rep {rep} | {quality}"]
                if (entry["id"], method, rep) in done:
                    continue
                row: RunRow
                run_dir = raw_dir / f"{entry['id'].replace('/', '_')}__{method}__r{rep}"
                run_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                try:
                    cached = run_dir / "model.obj"
                    if cached.exists() and not force:
                        # Reuse the mesh this exact configuration already produced. The
                        # wall clock is carried over from the run that produced it, so a
                        # resumed benchmark keeps the original timing instead of a NaN.
                        v, f = _load_obj(cached)
                        previous = progress_times.get((entry["id"], method, rep))
                        if previous is None:
                            import json as _json
                            _m = run_dir / "manifest.json"
                            if _m.exists():
                                previous = (_json.loads(_m.read_text(encoding="utf-8"))
                                            .get("timings_seconds", {}).get("total"))
                        extra = {"seconds": previous if previous else float("nan"),
                                 "reused": True,
                                 "file_mb": sum(p.stat().st_size for p in run_dir.rglob("*")
                                                if p.is_file()) / 1e6}
                        log.append("  reused cached mesh (delete results/raw/meshes for a "
                                   "fresh timing)")
                    elif method == "proposed":
                        v, f, extra = run_proposed(image, run_dir, quality, log=log)
                    else:
                        v, f, extra = run_baseline(method, image, run_dir, log=log)
                    row = evaluate_run(entry, method, v, f, extra["seconds"], extra, rep)
                    log.append(f"  seconds={row.seconds:.2f} peak={row.peak_rss_mb:.0f}MB "
                               f"tris={row.triangles}")
                    if row.chamfer is not None:
                        log.append(f"  chamfer={row.chamfer:.5f} f={row.f_score:.4f} "
                                   f"iou={row.iou:.4f} nc={row.normal_consistency:.4f}")
                except Exception as exc:                                 # noqa: BLE001
                    row = RunRow(entry_id=entry["id"], split=entry["split"], method=method,
                                 repetition=rep, ok=False, seconds=time.perf_counter() - t0,
                                 peak_rss_mb=_rss_mb(), triangles=0, vertices=0,
                                 watertight=False, error=f"{type(exc).__name__}: {exc}")
                    log.append("  FAILED\n" + traceback.format_exc())
                rows.append(row)
                (log_dir / f"{entry['id'].replace('/', '_')}__{method}__r{rep}.log").write_text(
                    "\n".join(log), encoding="utf-8")
                with open(progress_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(row)) + "\n")
                if not quiet:
                    status = "ok " if row.ok else "ERR"
                    cd = f"{row.chamfer:.4f}" if row.chamfer is not None else "   -  "
                    print(f"[{status}] {entry['id']:22s} {method:12s} r{rep} "
                          f"{row.seconds:6.2f}s cd={cd} ssim={row.ssim or 0:.3f}", flush=True)
                del row
                gc.collect()

    rows = [r for r in rows if r.ok] or rows
    summary = summarise(rows)
    ablation = run_ablation(quality=quality, entries=entries, repetitions=1,
                            quiet=quiet) if not quick else []
    sweep = run_thickness_sweep(quality=quality, entries=entries,
                                quiet=quiet) if not quick else []
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quality": quality,
        "repetitions": repetitions,
        "quick": quick,
        "methods": list(methods),
        "entries": [e["id"] for e in entries],
        "summary": summary,
        "ablation": ablation,
        "thickness_sweep": sweep,
        "rows": [asdict(r) for r in rows],
    }
    write_outputs(out, rows)
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def summarise(rows: list[RunRow]) -> list[dict[str, Any]]:
    fields = ["chamfer", "f_score", "iou", "normal_consistency", "psnr", "ssim",
              "lpips_proxy", "silhouette_iou", "seconds", "peak_rss_mb", "triangles",
              "file_mb"]
    out = []
    for method in METHODS:
        method_rows = [r for r in rows if r.method == method and r.ok]
        if not method_rows:
            continue
        syn = [r for r in method_rows if r.chamfer is not None]
        real = [r for r in method_rows if r.chamfer is None]
        rec: dict[str, Any] = {
            "method": method,
            "n_runs": len(method_rows),
            "n_synthetic": len(syn),
            "n_real": len(real),
            "watertight_pct": 100.0 * sum(r.watertight for r in method_rows) / len(method_rows),
        }
        for field_name in fields:
            pooled = [getattr(r, field_name) for r in method_rows]
            mean, std = _mean_std(pooled)
            rec[f"{field_name}_mean"] = mean
            rec[f"{field_name}_std"] = std
            if field_name != "seconds":
                geo = [getattr(r, field_name) for r in syn]
                g_mean, g_std = _mean_std(geo)
                rec[f"{field_name}_syn_mean"] = g_mean
                rec[f"{field_name}_syn_std"] = g_std
        out.append(rec)
    return out


# ------------------------------------------------------------------------ ablation
# Config-vs-config comparisons only need geometry, so the expensive preview and
# turntable rendering is switched off: the same knobs, measured ~10x faster.
ABLATION_BASE = {"render_preview": False, "render_turntable": False, "turntable_frames": 0}

ABLATION_CONFIGS = [
    ("full", {}),
    # no constant slab core (f = 0): the rim is only closed if the depth field
    # happens to reach zero there, so holes/open rims appear
    ("no_slab_core", {"thickness_floor_ratio": 0.0}),
    # depth no longer modulates the thickness (g = 0): constant-thickness extrusion
    ("depth_flat_slab", {"thickness_depth_gain": 0.0}),
    # thinner solids overall: t_max scaled to 35% of the silhouette width
    ("thin_inflation", {"thickness_scale": 0.35}),
    # no Taubin smoothing: raw marching-cubes staircase
    ("no_taubin_smoothing", {"taubin_iters": 0}),
    # coarse volumetric grid instead of the calibrated resolution
    ("coarse_grid", {"voxel_res": 64, "target_faces": 6000}),
]


THICKNESS_SWEEP = [0.25, 0.35, 0.42, 0.55, 0.70, 0.85, 1.00, 1.15]


def run_thickness_sweep(quality: str, entries: list[dict], quiet: bool = False,
                        force: bool = True) -> list[dict[str, Any]]:
    """Measure Chamfer / F-score / IoU as the inflation thickness is scaled.

    ``thickness_scale`` multiplies t_max, i.e. how deep the solid is relative to
    the silhouette width; 1.0 is the preset used by every table in the paper.
    """
    syn = [e for e in entries if e.get("gt_mesh")]
    out = []
    for scale in THICKNESS_SWEEP:
        cd, fs, iou, secs = [], [], [], []
        for entry in syn:
            run_dir = RESULTS / "raw" / "sweep" / f"ts{scale:.2f}__{entry['id'].replace('/', '_')}"
            cached = run_dir / "model.obj"
            if force and cached.exists():
                cached.unlink()
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                override = {**ABLATION_BASE, "thickness_scale": scale}
                if cached.exists():
                    v, f = _load_obj(cached)
                    extra = {"seconds": float("nan")}
                else:
                    v, f, extra = run_proposed(DATA / entry["image"], run_dir, quality,
                                               config=override)
                gt_v, gt_f = _load_obj(DATA / entry["gt_mesh"])
                scores = M.evaluate_geometry(v, f, gt_v, gt_f, samples=20000, tau=0.01, seed=SEED)
                cd.append(scores.chamfer); fs.append(scores.f_score)
                iou.append(scores.iou); secs.append(extra["seconds"])
            except Exception as exc:                                  # noqa: BLE001
                if not quiet:
                    print(f"  sweep {scale} {entry['id']} failed: {exc}")
        if not cd:
            continue
        secs_arr = np.asarray(secs, dtype=float)
        timed = secs_arr[np.isfinite(secs_arr)]
        rec = {"thickness_scale": scale, "n": len(cd), "n_timed": int(timed.size),
               "chamfer_mean": float(np.mean(cd)), "chamfer_std": float(np.std(cd)),
               "f_score_mean": float(np.mean(fs)), "f_score_std": float(np.std(fs)),
               "iou_mean": float(np.mean(iou)), "iou_std": float(np.std(iou)),
               "seconds_mean": float(timed.mean()) if timed.size else float("nan"),
               "seconds_std": float(timed.std()) if timed.size else float("nan")}
        out.append(rec)
        if not quiet:
            print(f"  sweep thickness_scale={scale:.2f}  cd={rec['chamfer_mean']:.5f} "
                  f"f={rec['f_score_mean']:.4f} iou={rec['iou_mean']:.4f} "
                  f"{rec['seconds_mean']:.2f}s", flush=True)
    _write_csv(RESULTS / "summary" / "thickness_sweep.csv", out)
    return out


def run_ablation(quality: str, entries: list[dict], repetitions: int = 1,
                 quiet: bool = False, force: bool = True) -> list[dict[str, Any]]:
    """Config-vs-config comparison of the proposed pipeline (same engine, same data)."""
    syn = [e for e in entries if e.get("gt_mesh")]
    if not syn:
        return []
    results = []
    for name, overrides in ABLATION_CONFIGS:
        cd, fs, iou, secs, ssim = [], [], [], [], []
        for entry in syn:
            run_dir = RESULTS / "raw" / "ablation" / f"{name}__{entry['id'].replace('/', '_')}"
            if force:
                shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                override = {**ABLATION_BASE, **overrides}
                v, f, extra = run_proposed(DATA / entry["image"], run_dir, quality,
                                           config=override)
                gt_v, gt_f = _load_obj(DATA / entry["gt_mesh"])
                scores = M.evaluate_geometry(v, f, gt_v, gt_f, samples=20000, tau=0.01,
                                             seed=SEED)
                cd.append(scores.chamfer)
                fs.append(scores.f_score)
                iou.append(scores.iou)
                secs.append(extra["seconds"])
                ref = _reference_view(DATA / entry["image"])
                ssim.append(M.ssim(_render_match(v, f, ref.shape[0]), ref))
            except Exception as exc:                                  # noqa: BLE001
                if not quiet:
                    print(f"  ablation {name} {entry['id']} failed: {exc}")
        rec = {"config": name, "overrides": overrides, "n": len(cd)}
        rec["chamfer_mean"], rec["chamfer_std"] = _mean_std(cd)
        rec["f_score_mean"], rec["f_score_std"] = _mean_std(fs)
        rec["iou_mean"], rec["iou_std"] = _mean_std(iou)
        rec["ssim_mean"], rec["ssim_std"] = _mean_std(ssim)
        rec["seconds_mean"], rec["seconds_std"] = _mean_std(secs)
        results.append(rec)
        if not quiet:
            print(f"  ablation {name:22s} cd={rec['chamfer_mean']:.5f} "
                  f"f={rec['f_score_mean']:.3f} iou={rec['iou_mean']:.3f}", flush=True)
    return results


# ------------------------------------------------------------------------ writers
def write_outputs(out: dict, rows: list[RunRow]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(environment(), indent=2),
                                              encoding="utf-8")
    (RESULTS / "raw" / "runs.json").write_text(
        json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2)
        + "\n", encoding="utf-8")
    fields = list(asdict(rows[0]).keys()) if rows else []
    with open(RESULTS / "raw" / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            rec = asdict(r)
            rec["extra"] = json.dumps(rec["extra"])
            rec.pop("manifest", None)
            writer.writerow(rec)
    _write_summary_csv(out["summary"], RESULTS / "summary" / "main_table.csv")
    (RESULTS / "summary" / "main_table.json").write_text(
        json.dumps(out["summary"], indent=2), encoding="utf-8")
    if out["ablation"]:
        _write_ablation_csv(out["ablation"], RESULTS / "summary" / "ablation_table.csv")
        (RESULTS / "summary" / "ablation_table.json").write_text(
            json.dumps(out["ablation"], indent=2), encoding="utf-8")
    (RESULTS / "summary" / "thickness_sweep.json").write_text(
        json.dumps(out.get("thickness_sweep") or [], indent=2), encoding="utf-8")
    _write_per_scene(rows, RESULTS / "summary" / "per_scene.csv")
    _write_failures(rows, RESULTS / "summary" / "failures.csv")
    _write_timings(rows, RESULTS / "summary" / "stage_timings.csv")


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_summary_csv(summary: list[dict], path: Path) -> None:
    if not summary:
        return
    fields = list(summary[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _write_ablation_csv(ablation: list[dict], path: Path) -> None:
    fields = ["config", "overrides", "n", "chamfer_mean", "chamfer_std", "f_score_mean",
              "f_score_std", "iou_mean", "iou_std", "ssim_mean", "ssim_std",
              "seconds_mean", "seconds_std"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in ablation:
            writer.writerow({k: json.dumps(rec[k]) if k == "overrides" else rec.get(k)
                             for k in fields})


def _write_per_scene(rows: list[RunRow], path: Path) -> None:
    """Per (scene, method) means: the table the report and slides quote per scene."""
    grouped: dict[tuple[str, str], list[RunRow]] = {}
    for r in rows:
        grouped.setdefault((r.entry_id, r.method), []).append(r)
    out = []
    for (entry_id, method), items in sorted(grouped.items()):
        ok = [i for i in items if i.ok]
        if not ok:
            out.append({"entry_id": entry_id, "method": method, "n": 0, "chamfer": None,
                        "f_score": None, "iou": None, "ssim": None, "seconds": None,
                        "watertight_pct": 0.0})
            continue
        def mean(attr):
            vals = [getattr(i, attr) for i in ok if getattr(i, attr) is not None]
            return float(np.mean(vals)) if vals else None
        out.append({
            "entry_id": entry_id, "method": method, "n": len(ok),
            "chamfer": mean("chamfer"), "f_score": mean("f_score"), "iou": mean("iou"),
            "normal_consistency": mean("normal_consistency"), "ssim": mean("ssim"),
            "silhouette_iou": mean("silhouette_iou"), "seconds": mean("seconds"),
            "peak_rss_mb": mean("peak_rss_mb"), "triangles": mean("triangles"),
            "file_mb": mean("file_mb"),
            "watertight_pct": 100.0 * sum(i.watertight for i in ok) / len(ok),
        })
    _write_csv(path, out)


def _write_failures(rows: list[RunRow], path: Path) -> None:
    """Runs whose geometry or silhouette quality falls below a stated threshold."""
    fields = ["entry_id", "method", "repetition", "reason", "chamfer", "f_score",
              "silhouette_iou", "seconds"]
    bad = []
    for r in rows:
        reasons = []
        if not r.ok:
            reasons.append(f"exception: {r.error}")
        else:
            if r.chamfer is not None and r.chamfer > 0.06:
                reasons.append("chamfer above 0.06")
            if r.f_score is not None and r.f_score < 0.50:
                reasons.append("F-score below 0.50")
            if r.silhouette_iou is not None and r.silhouette_iou < 0.55:
                reasons.append("silhouette IoU below 0.55")
            if not r.watertight:
                reasons.append("mesh not watertight")
        if reasons:
            bad.append({"entry_id": r.entry_id, "method": r.method,
                        "repetition": r.repetition, "reason": "; ".join(reasons),
                        "chamfer": r.chamfer, "f_score": r.f_score,
                        "silhouette_iou": r.silhouette_iou, "seconds": r.seconds})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bad)
    (path.with_suffix(".json")).write_text(json.dumps(bad, indent=2), encoding="utf-8")


def _write_timings(rows: list[RunRow], path: Path) -> None:
    """Per-stage wall clock for the proposed pipeline (mean +- std over runs)."""
    stages: dict[str, list[float]] = {}
    for r in rows:
        if r.method != "proposed" or not r.ok:
            continue
        for stage, took in (r.extra.get("stage_timings") or {}).items():
            stages.setdefault(stage, []).append(float(took))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "mean_s", "std_s", "n"])
        writer.writeheader()
        for stage, values in stages.items():
            mean, std = _mean_std(values)
            writer.writerow({"stage": stage, "mean_s": f"{mean:.4f}",
                             "std_s": f"{std:.4f}", "n": len(values)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the 3DFUN AI benchmark suite.")
    ap.add_argument("--quick", action="store_true", help="4 scenes, 1 repetition")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--quality", default="draft", choices=["draft", "standard", "high"])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--force", action="store_true", help="clear cached meshes first")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = run(quick=args.quick, repetitions=args.repetitions, quality=args.quality,
              methods=tuple(m.strip() for m in args.methods.split(",") if m.strip()),
              force=args.force, quiet=args.quiet)
    print("\nsummary written to", RESULTS / "summary" / "main_table.csv")
    for rec in out["summary"]:
        print(f"  {rec['method']:12s} n={rec['n_runs']:3d} "
              f"chamfer={rec['chamfer_mean']:.5f}"
              f"+-{rec['chamfer_std']:.5f} f={rec['f_score_mean']:.3f} "
              f"iou={rec['iou_mean']:.3f} ssim={rec['ssim_mean']:.3f} "
              f"t={rec['seconds_mean']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
