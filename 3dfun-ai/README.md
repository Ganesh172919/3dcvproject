# 3DFUN AI — An End-to-End Image-to-3D Generation Framework

A local research prototype for reconstructing 3D meshes from one photograph. It includes a web
studio, REST API, benchmark suite and IEEE-format paper.

**3D Computer Vision course project** · Indian Institute of Information Technology, Dharwad

## Project team

| Student | Roll number | Department |
|---|---|---|
| J Ganesh | 23BDS024 | Data Science and Artificial Intelligence |
| M Jagadeshwar | 23BDS033 | Data Science and Artificial Intelligence |
| B Varshith | 23BDS011 | Data Science and Artificial Intelligence |
| Pranav Kashyap | 23BCS098 | Computer Science and Engineering |
| C Sai Aravind | 23BDS075 | Data Science and Artificial Intelligence |

**Guide:** Dr. Shirshendu Layek, Assistant Professor, Dept. of Data Science and Artificial
Intelligence (DSAI), IIIT Dharwad

See [`AUTHORS.md`](AUTHORS.md) for the per-area contribution table.

## Course submission

The Group 17 archive has the four-page IEEE report, captioned demo, source, sample data and recorded results. Follow the [packaging and verification guide](docs/how-to/academic-packaging-and-verification.md) to build it. The [earlier mesh benchmark paper](paper/main.pdf) and [unsigned originality form](docs/CERTIFICATE_OF_ORIGINALITY.md) are also available.

## What this repository is

Two things live here, and they share the same reconstruction code:

1. **A working product** — a FastAPI service plus a hand-written single-page web studio where
   you upload a photo, watch the eight pipeline stages progress live, orbit the resulting mesh
   in a WebGL viewer and download GLB / glTF / OBJ+MTL / PLY / STL.
2. **A measurement harness** — `benchmarks/` builds a 25-image dataset (13 analytically-defined
   scenes with reference meshes, 12 real CC0 photographs), runs the proposed pipeline and two
   classical baselines on exactly the same inputs, and writes every number the paper and the
   course report uses into `results/`.

The Group 17 course report uses the recorded mesh benchmark and a separate TinyDepth experiment. See [the course README](submission/README.md) for the archive layout and [the training guide](training/README.md) for reproducible training.

### The reconstruction method in one paragraph

The object is matted with U²-Net and centred on a square canvas. Depth Anything V2 (int8 ONNX)
predicts a relative-depth *closeness* field. Each silhouette pixel is then inflated along
the viewing axis into a slab whose thickness is a monotone function of closeness, bounded by the
sampled outline. The pipeline pads the occupancy grid and checks mesh closure after extraction and cleanup. The grid is
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

Inference uses the chosen ONNX provider when available and retries on the CPU if GPU setup or inference fails. DirectML runs the depth estimators and background matting. The TripoSR INT8 encoder and decoder use CPU in the DirectML build. Meshing, baking, and export run on the CPU. For CUDA, `RECON_CUDA_MEMORY_MB` caps the ONNX allocator arena at 2,048 MiB by default. The model details and dashboard show the providers used in each run. The GPU provider can assign individual unsupported operations to the CPU.

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

### Reproduce the research experiments

```bash
make install-dev
make models
make test
```

Use [the training guide](training/README.md) to reproduce TinyDepth and its held-out evaluation. The historical full benchmark uses 25 images. In a source checkout with the dataset, run `make benchmark` or run `make benchmark QUICK=1` for a shorter pass. Use a separate working copy to preserve the saved results. The course ZIP includes the recorded tables and two sample images; `python benchmarks/make_dataset.py` regenerates the synthetic reference split and uses any real photos already in the optional photo cache. Photo sources are in [the attribution table](benchmarks/ATTRIBUTION.md).

To rebuild the Group 17 report with Tectonic, run `make report` after the course figures are available. The archive includes the editable LaTeX and the figures used in the PDF.
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
├── submission/        course report, demo and archive tools
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

## Documentation

- [Local project handbook](docs/handbook/README.md) and [documentation index](docs/index.md).
- [Architecture](docs/ARCHITECTURE.md), [REST API](docs/reference/api-spec.md), and [troubleshooting](docs/TROUBLESHOOTING.md).
- [Recorded benchmark protocol and results](docs/BENCHMARK.md) and [TinyDepth training](training/README.md).
- [Course report source](submission/report.tex) and [archive build instructions](docs/how-to/academic-packaging-and-verification.md).

## Licence

Code: MIT (`LICENSE`). Dataset: CC0-1.0 for the procedural scenes. The [attribution table](benchmarks/ATTRIBUTION.md) lists photograph sources and licenses.
