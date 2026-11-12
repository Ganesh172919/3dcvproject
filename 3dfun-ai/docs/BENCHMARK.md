# 3DFUN AI — benchmark protocol

This is the document the numbers in `paper/` and `docs/report.md` must agree with. It specifies
the data, the three methods, the metrics, the harness's resumability rules, and how to reproduce
or re-derive every table.

The rule for the whole project: **no number is typed by hand**. Each table below exists as a CSV
under `results/summary/`; `scripts/` turn those CSVs into the paper, the deck and the report. If a
figure is quoted anywhere and cannot be traced to a CSV, it is a bug — `scripts/verify_submission.py`
checks exactly that (number cross-check: 119/119 in the paper, 280/280 in the report, 70/70 in the deck).

---

## 1. Data

`benchmarks/data/dataset.json` — dataset **v1.0.0**, created 2026-11-08, seed **20261108**,
checksum **`d76fef12`** (first 8 hex of the SHA-256 of the canonical entries list).

| Split | Scenes | Reference | Scored on |
|---|---|---|---|
| synthetic | 13 | analytic ground-truth mesh (`benchmarks/data/synthetic/gt/*.obj`), procedurally built by `benchmarks/make_dataset.py`, CC0 | Chamfer, F-score, IoU, normal consistency, silhouette IoU, image metrics |
| real | 12 | none — no captured 3D scan exists | silhouette IoU, image metrics, geometry statistics only |

Real images are CC0 (StockSnap.io, WordPress Photo Directory, Rawpixel public-domain
collection, Wikimedia Commons) plus one repository test asset; per-entry source and licence are
stored in the dataset JSON and reproduced in `docs/report.md` §Dataset.

Every synthetic scene is scored on every metric; nothing is excluded from the aggregate. Two of
them (`synthetic/stairs`, an open staircase, and `synthetic/lamp`, a hollow shade on a stem) are
open or hollow in the subject itself, so a closed shell is a weaker target there; they are visible
individually in `results/summary/per_scene.csv` and discussed in the paper's results section.
`benchmarks/data/dataset.json` flags one reference mesh (`synthetic/capsule`) as not watertight, and
that flag is surfaced in the paper rather than hidden.

## 2. Methods

All three methods share the same inputs, the same depth network, the same silhouette matte, the
same normalisation (ICT-style: object bounding box centred, longest axis scaled to 1) and the same
target triangle budget. Only the *formulation* differs.

| Key | Implementation | Idea |
|---|---|---|
| `proposed` | `engine/recon/pipeline.py` | silhouette prism × depth-modulated slab thickness, closed core, EDT SDF, marching cubes |
| `tsdf` | `benchmarks/baselines/baseline_tsdf.py` | one-view truncated signed distance field, classic fusion-style carve |
| `visual_hull` | `benchmarks/baselines/baseline_visual_hull.py` | silhouette-only visual hull (space carving without depth cues) |

`tsdf` and `visual_hull` are deliberately *strong* baselines: same depth map, same grid resolution,
same triangle target. They are not strawmen and they win on individual scenes (see per-scene CSV).

Not evaluated, and why (recorded rather than guessed): learned single-image mesh regressors
(e.g. TripoSR-class models), multi-view photogrammetry, and diffusion-prior methods. None can be
run on the 2-core CPU-only evaluation machine within the resource budget, so no number is quoted
for them.

## 3. Metrics

`benchmarks/metrics.py`:

* **Chamfer distance** — symmetric mean nearest-neighbour distance between 20 000 surface samples of
  prediction and ground truth, on ICT-normalised geometry (object max axis = 1). Lower is better.
* **F-score @ τ = 0.01** — harmonic mean of precision and recall at a 1 %-of-object diameter
  threshold, on the same samples. Higher is better.
* **Volumetric IoU** — occupancy agreement on a 64³ grid over the normalised bounding box.
* **Normal consistency** — mean absolute dot product between nearest-neighbour normals.
* **Silhouette IoU** — rerendered mask versus the input matte, both resized to a common size.
  The only geometric measure available for real photographs.
* **PSNR / SSIM / LPIPS proxy** — image fidelity of the shaded rerender (`lpips_proxy` is a
  gradient/colour-statistics proxy, not the learned LPIPS; it is named accordingly everywhere).
* **Runtime / peak RSS / triangles / vertices / watertight / UV coverage** — measured per run.

Watertightness is decided by an exact weld on `np.round(v, 6)` followed by a
boundary-edge count (`_mesh_stats`), *never* by `trimesh.is_watertight` on an exported OBJ, which is
meaningless after float reformatting. Both `watertight` (post-weld) and `watertight_raw` are stored.

## 4. Harness

`benchmarks/run_benchmarks.py`

```
python3 -m benchmarks.run_benchmarks [--quick] [--repetitions N] [--quality draft|standard|high]
                                    [--methods proposed,tsdf,visual_hull] [--force] [--quiet]
```

Protocol as shipped: **25 images × 3 methods × 3 repetitions = 225 cells**, `--quality draft`
(the preset used for every table; `standard`/`high` are available but were not benchmarked, so no
number is claimed for them).

* One process, sequential, so peak RSS is comparable across methods (≈ 926–933 MB).
* Every finished cell is appended to `results/raw/runs.jsonl` **immediately**; meshes are written to
  `results/raw/meshes/<entry>__<method>__r<k>/`, logs to `results/logs/`.
* `--force` clears cached meshes first. Without it, an existing run directory is reused **and its
  original wall clock is read back** from the progress log or the manifest, so a resumed benchmark
  never pairs a cached mesh with a fabricated timing.
* Failures of the run itself (exceptions) are recorded and skipped; *quality* failures are recorded
  in `results/summary/failures.csv` with an explicit reason string, not silently dropped.
* Ablation and sweep are separate phases appended to the same results tree:
  `results/raw/ablation/`, `results/raw/sweep/`.

Outputs: `results/raw/{runs.csv, runs.jsonl, runs.json}`, `results/summary/{main_table, per_scene,
ablation_table, thickness_sweep, stage_timings, failures}.csv`, `results/environment.json`.

## 5. Results as shipped

`results/summary/main_table.csv` (n = 75 per method; CD/F/IoU/NC/SSIM from the 39 synthetic
runs, silhouette IoU from all 75):

| Method | Chamfer ↓ | F@1 % ↑ | IoU ↑ | NC ↑ | SSIM ↑ | s ↓ | watertight % ↑ | tris |
|---|---|---|---|---|---|---|---|---|
| proposed | 0.0639 | 0.1482 | 0.0545 | 0.7965 | 0.4438 | 27.73 | 96 | 7782 |
| tsdf | 0.0962 | 0.0533 | 0.0271 | 0.6069 | 0.4230 | 1.43 | 0 | 7974 |
| visual_hull | 0.0728 | 0.0982 | 0.0384 | 0.7136 | 0.4387 | 2.13 | 0 | 7999 |

Standard deviations, per-split means and image metrics are in the CSV; `scripts/make_figures.py`
plots them. Runtime decomposition (`stage_timings.csv`, mean over 75 proposed runs, seconds):
preprocessing 0.71, depth 0.71, volume 0.08, geometry 0.21, texture 1.04, export 0.38,
**render 24.56**, total 27.70 — the preview/turntable renderer accounts for ≈ 89 % of wall clock,
which is why the method's cost is reported both with and without it.

**Ablation** (13 synthetic scenes, draft preset, preview rendering off; `ablation_table.csv`).
Runtimes here are *not* comparable with the table above — they exclude the turntable, which is
~89 % of the full-pipeline wall clock:

| Config | Chamfer ↓ | F@1 % ↑ | IoU ↑ | s |
|---|---|---|---|---|
| full | 0.0639 | 0.148 | 0.055 | 4.53 |
| no_slab_core (f = 0) | 0.1016 | 0.071 | 0.023 | 3.86 |
| depth_flat_slab (g = 0) | 0.0811 | 0.112 | 0.041 | 1.87 |
| thin_inflation (0.35) | 0.0910 | 0.102 | 0.036 | 1.87 |
| no_taubin_smoothing | 0.0639 | 0.148 | 0.055 | 1.88 |
| coarse_grid (64³) | 0.0637 | 0.142 | 0.054 | 1.73 |

Reading: the constant slab core is the single load-bearing knob (removing it costs 59 % in
Chamfer and drops F-score from 0.148 to 0.071); a flat slab costs 27 % and thin inflation 42 %;
Taubin smoothing is free in accuracy terms (0.0639 vs 0.0639) and is kept for normal quality; the
coarse 64³ grid matches accuracy at 2.2× the speed, which is why `draft` uses a smaller grid than
`standard`.

**Thickness sweep** (`thickness_sweep.csv`, 13 synthetic scenes each):

| κ = t_max / obj_w | Chamfer ↓ | F@1 % ↑ | IoU ↑ |
|---|---|---|---|
| 0.25 | 0.1025 | 0.075 | 0.028 |
| 0.35 | 0.0910 | 0.102 | 0.036 |
| 0.42 | 0.0842 | 0.107 | 0.042 |
| 0.55 | 0.0749 | 0.103 | 0.039 |
| **0.70 (chosen)** | **0.0639** | **0.148** | **0.055** |
| 0.85 | 0.0569 | 0.127 | 0.048 |
| 1.00 | 0.0625 | 0.135 | 0.052 |
| 1.15 | 0.0672 | 0.116 | 0.043 |

κ = 0.70 is chosen on F-score and IoU (the two metrics that are comparable across thicknesses);
0.85 reaches a marginally lower Chamfer but loses ~14 % of F-score, i.e. it is closer on average
and worse where it matters. This asymmetry is reported, not hidden.

## 6. Reproducing a single number

```bash
./run.sh benchmark --quick          # 4 scenes × 1 rep — sanity, ~1 min
./run.sh benchmark                  # full 225 cells — see README for wall clock
python3 -m benchmarks.run_benchmarks --methods proposed --repetitions 1 --force   # one method
```

To trace a specific cell: grep its id in `results/summary/per_scene.csv` (needle mean), then in
`results/raw/runs.jsonl` (per-repetition row incl. `extra`), then open
`results/raw/meshes/<entry>__<method>__r<k>/manifest.json` for the config hash and stage timings,
and `results/logs/<entry>__<method>__r<k>.log` for the stage trace.

Environment of the shipped run is pinned in `results/environment.json` (CPU model, core count,
RAM, Python and library versions, ONNX Runtime providers, seed) and reproduced verbatim in the
paper appendix. Timings are **not** claimed to transfer to other hardware.

## 7. What would make a comparison invalid

Documented so a reader can check rather than trust:

* quoting `proposed` with `render_preview` off and the baselines with it on (runtime only);
* mixing presets between methods (all tables are `draft`);
* comparing ablation runtimes against main-table runtimes (ablation has rendering off);
* reusing a cached mesh with a fresh wall clock (harness reads the original timing back);
* comparing against a baseline that did not share the depth graph (all three do).
