# 3DFUN AI — architecture

This document explains what the system is made of, why the boundaries are where they are, and
which invariants each layer is responsible for. It is written to be read before the code, and it
tells you which file to open next.

Team and guide are listed in [`AUTHORS.md`](AUTHORS.md); measurements live in
[`report.md`](report.md) and the paper.

---

## 1. The three layers

```
   browser                         server process                     disk
┌──────────────┐   HTTPS/SSE   ┌──────────────────────┐   calls   ┌──────────────┐
│ web/ studio  │ ────────────► │ app/ (FastAPI)       │ ────────► │ engine/recon │
│ vanilla ES   │ ◄──────────── │ routers → services   │ ◄──────── │ NumPy/SciPy  │
│ modules      │   event stream│ SQLite + storage     │  manifest │ ONNX Runtime │
└──────────────┘               └──────────────────────┘           └──────────────┘
```

The important boundary is between `app/` and `engine/`. The service never touches mesh data; it
hands the engine an image path and a configuration, receives a directory of artefacts plus a
manifest, and stores both. That is what makes the engine replaceable — and it is also why the
benchmark harness can drive the *same* reconstruction code the product runs.

### `engine/` — the reconstruction engine

| File | Responsibility |
|---|---|
| `recon/config.py` | `PipelineConfig` dataclass, quality presets (`draft`, `standard`, `high`), `config_hash()` |
| `recon/preprocessing.py` | load → U²-Net matte → erode/threshold → crop with padding → square canvas |
| `recon/depth.py` | Depth Anything V2 inference, percentile normalisation, smoothing, symmetry prior |
| `recon/volumetrics.py` | slab-thickness field, occupancy grid, EDT signed distance field, grid calibration |
| `recon/meshing.py` | marching cubes, island removal, Taubin smoothing, mesh metrics |
| `recon/texturing.py` | planar view-projection atlas, back-projected albedo, de-lighting, AO/roughness/metallic/normal |
| `recon/render.py` | two rasterisers (per-pixel z-buffer and fast flat), preview/turntable output |
| `recon/export.py` | GLB, glTF-separate, OBJ+MTL, PLY, STL writers |
| `recon/pipeline.py` | stage orchestration, timings, manifest assembly |

Invariants the engine guarantees, stated once and tested:

1. **Watertight by construction.** The thickness field is floored at `min_thickness` on every
   silhouette pixel, so the occupancy volume is closed and `marching_cubes` returns a closed
   surface. Nothing downstream fills holes.
2. **Silhouette containment.** Occupancy is only defined inside the silhouette prism, so the
   isosurface cannot extend outside the projected outline.
3. **Reproducibility.** Same image + same `PipelineConfig` ⇒ same `config_hash` and the same
   triangle count. The grid resolution is *calibrated* by a 64³ probe, never sampled randomly.
4. **Self-description.** Every run writes `manifest.json`: config hash, per-stage timings, mesh
   statistics (faces, vertices, boundary/non-manifold edges, Euler characteristic, watertight
   flag), texture size, UV coverage and the limitations that apply to that run.

### `app/` — the service

| Module | Responsibility |
|---|---|
| `main.py` | app factory, router mounting under `/api/v1`, static hosting of `web/`, security headers |
| `routers/` | 8 routers: `auth`, `uploads`, `generations`, `assets`, `gallery`, `billing`, `admin`, `system` |
| `state_machine.py` | the only legal generation transitions (`queued → processing → completed/failed/cancelled`) |
| `jobs.py` | job queue, worker loop, SSE event emission with sequence numbers and replay |
| `events.py` | event log, heartbeats, last-event-id replay |
| `credits.py` | immutable ledger: reserve → commit or refund, never an in-place balance edit |
| `auth.py`, `security.py` | JWT access tokens, Argon2id hashing, session records, MFA hooks |
| `ratelimit.py` | per-route and per-identity limits |
| `providers.py` | provider adapters: local engine, generic HTTP, mock |
| `storage.py` | content-addressed local storage with SHA-256 names |
| `recon_bridge.py` | the single call site that invokes the engine |
| `serializers.py` | `flatten_metrics()` and friends: dataclass → JSON for the UI |

Design rules that the service enforces:

* A generation cannot change state through any path other than `state_machine.transition()`.
* Credits are reserved *before* the engine starts and refunded on failure; the ledger is
  append-only, so an audit is a `SELECT`, not a reconstruction of history.
* Progress is streamed, not polled, but the SSE endpoint replays missed events by `id`, so a
  reconnect never loses a stage.
* Nothing in the API imports NumPy mesh code at module import time, so the API boots without the
  engine's heavier dependencies being warm.

### `web/` — the studio

Dependency-free ES modules (no build step, no bundler): `web/static/js/app.js` is the SPA,
`app.css` the design system, `web/app/index.html` the shell. The viewer is three.js r160 from a
local copy. Pages: landing, studio, dashboard, assets, asset detail, gallery, admin, benchmark,
settings, login.

The landing page and the `/app/benchmark` page read `GET /api/v1/results/summary`, which serves
the *same* `results/summary/*.csv` files that the paper and deck are generated from. The site
cannot drift from the paper without the verifier failing.

---

## 2. Request lifecycle of one reconstruction

```
POST /api/v1/uploads            → image stored, SHA-256 addressed, EXIF stripped
POST /api/v1/generations        → estimate credits → reserve → create row (queued) → enqueue
GET  /api/v1/generations/{id}/events   (SSE)
        queued → processing (stage, progress) → completed
                            ↘ failed → refund → retry available
engine/recon                    → writes model.glb/obj/ply, textures, previews, manifest.json
GET  /api/v1/assets/{id}/download      → signed, short-lived, content-type checked
```

Failure semantics: an exception inside the engine fails the *job*, refunds the *credits*, and
keeps the *artefacts* that were already written so the failure can be debugged from the manifest.
The API never returns a stack trace to a client; it returns a reference code that appears in
`results/logs/` and in the server log.

---

## 3. The benchmark harness as a second front end

`benchmarks/run_benchmarks.py` is a peer of `app/`, not a script inside `engine/`:

```
dataset (benchmarks/data/dataset.json + images + reference meshes)
   ↓
for each image × method × repetition:
      proposed  → engine/recon (draft preset)
      tsdf      → benchmarks/baselines/baseline_tsdf.py
      visual_hull → benchmarks/baselines/baseline_visual_hull.py
   ↓
metrics (benchmarks/metrics.py): Chamfer, F-score, IoU, normal consistency,
                                 silhouette IoU, PSNR, SSIM, LPIPS proxy
   ↓
results/raw/{runs.csv, runs.jsonl, meshes/}, results/logs/, results/environment.json
   ↓
results/summary/{main_table, per_scene, ablation_table, thickness_sweep, stage_timings, failures}.csv
   ↓
scripts/{make_figures, generate_paper, generate_slides, generate_report}.py → paper/, presentations/, docs/
```

Two properties make it a benchmark rather than a demo:

* **Isolation.** All three methods share the ONNX depth graph, the target triangle budget and the
  volumetric resolution, so the comparison measures the *formulation*, not the front-end.
* **Resumability.** Each finished run is appended to `results/raw/runs.jsonl`. An interrupted run
  resumes; a cached mesh is reused with its original wall clock, so a resumed benchmark never
  mixes a fresh mesh with a fabricated timing.

---

## 4. Why these choices

| Decision | Reason |
|---|---|
| classical geometry + one learned depth network | the evaluation machine has 2 CPU cores and no GPU; anything else could not be run, and unrunnable baselines are reported as *not evaluated* rather than quoted |
| slab thickness floored at a constant core | makes watertightness structural instead of a repair step; measured in the ablation (`no_slab_core` costs the most) |
| occupancy intersected with the silhouette prism | removes the classic depth-back-projection leak outside the outline |
| grid resolution calibrated by a probe mesh | decimating afterwards breaks watertightness; changing the grid does not |
| planar view-projection atlas | exact on the camera-facing half, one bake, no chart packing; grazing angles stretch, which is documented |
| XML-free frontend, no build step | the studio must run from `./run.sh` with no toolchain; a bundler would add a failure mode for the reviewer |
| results-driven documents | every number in the paper, deck and report is generated from `results/`; the verifier fails if a printed number does not trace back |

---

## 5. Where to look for what

* "Why is the mesh watertight?" → `engine/recon/volumetrics.py` (`min_thickness`, occupancy) and
  `engine/recon/meshing.py` (marching cubes on a closed field).
* "How is the score computed?" → `benchmarks/metrics.py`, then `docs/BENCHMARK.md`.
* "How do I add a provider?" → `app/providers.py` (three adapters already there).
* "How do I reproduce a number?" → `docs/BENCHMARK.md` §6 and `paper/` Appendix A.
* "Why did a run fail?" → `results/summary/failures.csv`, then that run's log in `results/logs/`.
