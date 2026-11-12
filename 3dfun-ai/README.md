# 3DFUN AI — An End-to-End Image-to-3D Generation Framework

Single photograph in, watertight UV-textured 3D mesh out — with a web studio, a REST API,
a reproducible benchmark suite and an IEEE paper generated from the measurements.

**3D Computer Vision course project** · Indian Institute of Information Technology, Dharwad

| # | Student | Roll |
|---|---------|------|
| 1 | J Ganesh | 23BDS024 |
| 2 | M Jagadeshwar | 23BDS033 |
| 3 | B Varshith | 23BDS011 |
| 4 | Pranav Kashyap | 23BCS098 |
| 5 | C Sai Aravind | 23BDS075 |

**Guide:** Dr. Shirshendu Layek, Assistant Professor, Dept. of Data Science and Artificial
Intelligence (DSAI), IIIT Dharwad

See [`AUTHORS.md`](AUTHORS.md) for the per-area contribution table.

---

## What this repository is

Two things live here, and they share the same reconstruction code:

1. **A working product** — a FastAPI service plus a hand-written single-page web studio where
   you upload a photo, watch the eight pipeline stages progress live, orbit the resulting mesh
   in a WebGL viewer and download GLB / glTF / OBJ+MTL / PLY / STL.
2. **A measurement harness** — `benchmarks/` builds a 25-image dataset (13 analytically-defined
   scenes with reference meshes, 12 real CC0 photographs), runs the proposed pipeline and two
   classical baselines on exactly the same inputs, and writes every number the paper and the
   slide deck use into `results/`.

Nothing in the paper is typed by hand. `make paper` re-reads `results/`; if a row is missing
the paper prints `not evaluated` with the reason instead of inventing a figure.

### The reconstruction method in one paragraph

The object is matted with U²-Net, centred on a square canvas, and Depth Anything V2 (int8 ONNX,
CPU) predicts a relative-depth *closeness* field. Each silhhouette pixel is then inflated along
the viewing axis into a slab whose thickness is a monotone function of closeness, bounded by the
object's own outline — so the surface cannot leak outside the silhouette, and the rim stays
closed, which makes the extracted isosurface watertight by construction. The occupancy grid is
converted to a signed distance field, meshed with marching cubes, Taubin-smoothed, unwrapped with
a planar view-projection atlas, and textured by back-projecting the input image (albedo
de-lighting plus estimated AO / roughness / metallic / normal maps).

---

## Quick start

### Windows PowerShell

From the project folder, install Python dependencies, verify and download the model weights, and start the local app:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Install -Seed
```

The launcher uses Python 3.12. It can install Python through `uv` if available. Otherwise install Python 3.12 with the Windows `py` launcher first. `-Install` installs dependencies and verifies all five ONNX files (about 535 MB total). Use `-InstallModels` to download or repair just the weights. Open [the local lab](http://127.0.0.1:8010/) to rotate or download a built-in example. Open the Studio and sign in with **Use demo account**, or register your own account. Select a reference photo and an engine, then click **Generate model**. Stop the server with Ctrl+C. For later starts, run `powershell -ExecutionPolicy Bypass -File .\run.ps1`. Use `-Port` to change the local port.

### GPU inference

On Windows, start the app with one of these commands after stopping the current instance:

```powershell
# DirectX 12 GPU, including Intel Iris Xe, AMD and NVIDIA
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device directml

# NVIDIA GPU with CUDA 12.8 and cuDNN 9.x available to ONNX Runtime
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device cuda
```

The launcher creates a separate environment for each GPU runtime. The CPU launch remains available with `-Device cpu`. On Linux, install `requirements-cuda.txt` in a separate environment and launch with `RECON_DEVICE=cuda ./run.sh` after installing the matching NVIDIA libraries.

Inference uses the chosen ONNX provider when available and retries on the CPU if GPU setup or inference fails. Meshing, baking, and export run on the CPU. For CUDA, `RECON_CUDA_MEMORY_MB` caps the ONNX allocator arena at 2,048 MiB by default. The model details and dashboard show which provider handled each run. The GPU provider can assign individual unsupported operations to the CPU.

### Learning workflow

Use **Upload your image** on the homepage or the upload area in Studio. After choosing a model and quality, generate the mesh and inspect it in the viewer. On **Dashboard → Experiment notebook**, select the photo to see runs grouped by identical image content. Use **Run this image again** to hold the input constant, change one setting, and export the measurements as CSV. These are measurements of your local runs.

### Gemini chat

Create a key in [Google AI Studio](https://aistudio.google.com/apikey). In this project's `.env` file, set:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Open **Chat** and select **Check setup** after saving `.env`. The server picks up the edited key immediately. The key stays on the server. Chat messages go to Gemini, and reconstruction runs on the local machine. You can omit `GEMINI_MODEL` to use the default.

### macOS and Linux

```bash
# 1. dependencies (Python 3.12 recommended)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. install the verified weights and start the product
python scripts/fetch_models.py --all
./run.sh --seed
#    → http://localhost:8000        landing page
#    → http://localhost:8000/app/   web studio
#    → http://localhost:8000/docs   OpenAPI
```

Demo credentials created by the seed (development only):

| Account | Email | Password |
|---------|-------|----------|
| admin | `admin@3dfun.dev` | `DemoPassword123!` |
| user | `demo@3dfun.dev` | `DemoPassword123!` |

### Reconstruct one image from the command line

```bash
python3 engine/run_recon.py engine/testdata/sneaker.png --quality standard --out out/sneaker
ls out/sneaker     # model.glb model.obj model.mtl model.ply basecolor.png normal.png
                   # orm.png ao.png roughness.png metallic.png preview.png turntable.mp4
                   # manifest.json
```

`manifest.json` records the config hash, every stage timing, the mesh metrics and the
limitations that apply to that run.

### Reproduce the benchmarks and the paper

```bash
make benchmark     # 25 images × 3 methods × 3 repetitions → results/
make figures       # charts + qualitative grids → results/figures/
make paper         # paper/main.pdf, every number read from results/
make slides        # presentations/3DFUN-AI-Project.pptx (16:9, speaker notes)
make report        # docs/report.md (+ PDF), file inventory, acceptance checklist
make test          # pytest
make verify        # end-to-end check: tests, paper compile, number cross-check
make all           # figures + paper + slides + report in one go
```

Expected wall clock on the two-core CPU container used for the paper: `make benchmark`
≈ 40 min with preview and turntable rendering on (≈ 15 min if you set
`render_preview: false` in `engine/recon/config.py`), `make figures` ≈ 8 s, `make paper`
≈ 15 s, `make slides` ≈ 3 s. `make benchmark QUICK=1` runs 4 scenes × 1 repetition in
≈ 3 min as a smoke test.

Expected terminal output of a benchmark run looks like (real lines from the run that produced
the paper):

```
[ok ] synthetic/box          proposed     r1  31.29s cd=0.0206 ssim=0.499
[ok ] synthetic/box          tsdf         r1   1.43s cd=0.1233 ssim=0.461
[ok ] synthetic/box          visual_hull  r1   2.31s cd=0.0588 ssim=0.492
  ablation full                   cd=0.06392 f=0.148 iou=0.055
  sweep thickness_scale=0.70  cd=0.06392 f=0.1482 iou=0.0545 3.73s
summary written to results/summary/main_table.csv
```

The run is resumable: every finished run is appended to `results/raw/runs.jsonl`, so an
interrupted benchmark picks up where it stopped. `--force` deletes the cache and re-runs
everything, which is what you want for a fresh timing measurement.

---

## Layout

```
3dfun-ai/
├── engine/            reconstruction engine + ONNX weights + CLI
│   ├── models/        depth_anything_v2_small_quantized.onnx, u2netp.onnx  (see Models)
│   ├── recon/         config · preprocessing · depth · volumetrics · meshing
│   │                  texturing · render · export · pipeline
│   ├── run_recon.py   command-line entry point
│   └── testdata/      sample photographs
├── app/               FastAPI service (auth, uploads, generations, assets, gallery,
│                      credits, billing, admin, jobs, SSE events, rate limits)
├── web/               single-page studio (vanilla ES modules, three.js r160), landing page
├── benchmarks/        dataset builder, metrics, baselines, benchmark driver
├── tests/             API, generation-lifecycle and engine tests
├── paper/             IEEEtran LaTeX source + figures generated from results/
├── presentations/     generated 16:9 deck + speaker notes
├── tools/             ui_audit.py (headless screenshot + UI/UX audit)
├── docs/              architecture, API notes, screenshots, final checklists
├── results/           raw runs, summaries, figures, environment.json (generated)
└── Makefile
```

## Models

The course method uses two small ONNX graphs. Download them once with `make models` or the Windows launcher:

| Model | File | Size | Licence | Weights |
|-------|------|------|---------|---------|
| Depth Anything V2 Small (int8) | `engine/models/depth_anything_v2_small_quantized.onnx` | 27 MB | Apache-2.0 | [ONNX conversion](https://huggingface.co/onnx-community/depth-anything-v2-small) |
| U²-Net portable | `engine/models/u2netp.onnx` | 4.6 MB | Apache-2.0 | [U²-Net release](https://github.com/xuebinqin/U-2-Net) |

The Studio offers three CPU reconstruction options:

| Option | Method | Weights |
|-------|--------|---------|
| TripoSR INT8 | Learned 3D density and vertex colours. Draft samples a 64³ grid in chunks. | [Pinned QtMeshEditor ONNX export](https://huggingface.co/fernandotonon/QtMeshEditor-triposr-onnx/tree/01b2f1add36601f964a89fd8a3159919ced4aa40), 436 MB plus decoder (MIT) |
| Depth Anything V2 | Relative depth with the course silhouette pipeline and baked texture atlas. | 27 MB, as listed above |
| MiDaS Small | Relative depth from a different estimator with the same silhouette pipeline. | [Official MiDaS v2.1 Small ONNX release](https://github.com/isl-org/MiDaS/releases/tag/v2_1), 67 MB (MIT) |

Run `python scripts/fetch_models.py --all --check` to check the local files against their SHA-256 digests. The CPU build of each model needs no dedicated VRAM. GPU builds use the same weights. The Chair, Teapot and Robot references come from [TripoSR's GitHub examples](https://github.com/VAST-AI-Research/TripoSR/tree/107cefdc244c39106fa830359024f6a2f1c78871/examples). Their public GLBs are generated outputs from this local project.

## Published benchmark scope

The submitted paper evaluates the depth-and-silhouette method against two classical baselines: a TSDF fusion of the same monocular depth map
(`benchmarks/baselines/baseline_tsdf.py`) and a visual hull with a depth-derived back plane
(`benchmarks/baselines/baseline_visual_hull.py`). TripoSR and MiDaS were added to the Studio after those measurements.

## Known limitations of the course method

* Single-view inference: geometry behind the object is inferred, not observed. Concavities that
  the silhouette cannot express (a handle seen edge-on) are filled rather than carved.
* No learned 3D prior: the engine cannot invent unseen detail and does not read text prompts.
* CPU-only: a standard run takes ~20 s at draft quality on two cores; render (turntable) dominates.
* Chamfer / F-score / IoU are computed against the reference meshes of the analytically-defined
  split only. Real photographs have no metric ground truth and are scored on runtime, memory,
  topology and self-consistency between input and re-rendered silhouettes.

Full list with measurements: `paper/` §Limitations, `docs/report.md` §7 and
`results/summary/failures.csv` (one row per run that crossed the stated thresholds).

## Documentation map

| Document | What it answers |
|---|---|
| [`docs/index.md`](docs/index.md) · [`mkdocs.yml`](mkdocs.yml) | the full interactive Diátaxis documentation portal |
| [`docs/report.md`](docs/report.md) · [`docs/report.pdf`](docs/report.pdf) | the full project report, generated from `results/` |
| [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md) | step-by-step onboarding tutorial from clone to 3D WebGL render |
| [`docs/tutorials/gpu-acceleration.md`](docs/tutorials/gpu-acceleration.md) | DirectML (Windows) and CUDA (Linux) hardware acceleration tutorial |
| [`docs/how-to/fine-tune-depth.md`](docs/how-to/fine-tune-depth.md) | how-to guide for domain-specific depth fine-tuning and SILog loss |
| [`docs/how-to/deploy-docker.md`](docs/how-to/deploy-docker.md) | production multi-stage container deployment with Docker Compose |
| [`docs/how-to/export-3d-printing.md`](docs/how-to/export-3d-printing.md) | exporting watertight meshes and slicing STLs for FDM/SLA 3D printing |
| [`docs/reference/api-spec.md`](docs/reference/api-spec.md) | REST API endpoints, JWT authentication, and SSE event schema |
| [`docs/reference/configuration.md`](docs/reference/configuration.md) | environment variables, meshing budgets, and inflation hyperparameters |
| [`docs/reference/metrics.md`](docs/reference/metrics.md) | mathematical metric definitions (CD, F-Score, IoU, NC, SSIM) |
| [`docs/explanations/volumetric-inflation.md`](docs/explanations/volumetric-inflation.md) | mathematical theory of constrained occupancy fields vs. TSDF |
| [`docs/explanations/quantization-strategy.md`](docs/explanations/quantization-strategy.md) | Vision Transformer INT8 post-training quantization and accuracy bounds |
| [`docs/adr/0001-fastapi-and-vanilla-spa.md`](docs/adr/0001-fastapi-and-vanilla-spa.md) | ADR 0001: FastAPI Backend and Vanilla JS SPA rationale |
| [`docs/adr/0002-onnx-runtime-int8.md`](docs/adr/0002-onnx-runtime-int8.md) | ADR 0002: ONNX Runtime INT8 CPU Inference rationale |
| [`docs/adr/0003-sdf-over-pointclouds.md`](docs/adr/0003-sdf-over-pointclouds.md) | ADR 0003: Signed Distance Fields for 2-manifold watertightness |
| [`docs/adr/0004-analytic-synthetic-benchmark.md`](docs/adr/0004-analytic-synthetic-benchmark.md) | ADR 0004: Procedural Analytic Benchmark rationale |
| [`docs/adr/0005-taubin-smoothing.md`](docs/adr/0005-taubin-smoothing.md) | ADR 0005: Dual-step Taubin non-shrinking smoothing rationale |
| [`training/README.md`](training/README.md) | training suite: synthetic renderer, depth fine-tuner, ONNX exporter, Optuna tuner |
| [`presentations/DEFENSE_FAQ.md`](presentations/DEFENSE_FAQ.md) | 15 academic defense questions and counter-arguments |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how the engine, service and studio fit together |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | the measurement protocol, metric definitions and how to read the tables |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | running it locally, configuring it from the environment, and serving it behind a process manager and proxy |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | the errors this project actually hit, and the fix |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | what was executed for this submission, with the command and the output |
| [`docs/FILE_INVENTORY.md`](docs/FILE_INVENTORY.md) | every tracked file, its kind and its line count |
| [`docs/ACCEPTANCE_CHECKLIST.md`](docs/ACCEPTANCE_CHECKLIST.md) | the submission checklist, ticked from the files on disk |
| [`AUTHORS.md`](AUTHORS.md) | team, guide, per-area contributions, citation block |
| [`paper/main.pdf`](paper/main.pdf) | the IEEEtran paper (8 pages) |
| [`presentations/3DFUN-AI-Project.pptx`](presentations/3DFUN-AI-Project.pptx) + [`speaker_notes.md`](presentations/speaker_notes.md) | the 15-slide deck |

## Verification

`make verify` runs 30 checks and is the only acceptance gate that matters. On this machine it
prints `30 passed, 0 warnings, 0 failed`, covering: team and guide in every artefact; no secrets
and no weights in git; raw runs, logs and an environment record for all 225 runs; three
repetitions per configuration; summary means that reproduce from the raw rows; every figure the
paper references; a paper that compiles with **0 undefined references and 0 overfull boxes** in
8 pages; a 16:9 deck of 15 slides with nothing outside the canvas, clipped or without speaker
notes; and every number printed in the paper, the deck and the report matched back to
`results/` (119/119, 70/70, 280/280). `scripts/audit_renders.py` additionally rasterises every
page of the paper and the report for a visual pass.

## Licence

Code: MIT (`LICENSE`). Dataset: CC0-1.0 for the procedurally-defined scenes; the photographs are
CC0-1.0 works whose authors, hosts and source links are listed in
`benchmarks/data/ATTRIBUTION.md`.
