# 3DFUN AI — Group 17

A local 3D computer vision project for IIIT Dharwad. Upload a single object image in the Studio to reconstruct and inspect a textured 3D mesh. The separate TinyDepth program trains on known synthetic geometry and exports a visible-surface point cloud.

**Group 17 project contributors:** J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011), Pranav Kashyap (23BCS098), and C Sai Aravind (23BDS075). See [project attribution](code/AUTHORS.md) for their roles. Project contact: 23bds024@iiitdwd.ac.in. Confirm the official submission roster and required emails before uploading.

## Install and reconstruct one image

Extract the archive, open a terminal in the folder containing this README and use Python 3.12. Create a clean environment and activate it:

```sh
python -m venv .venv
```

On Windows PowerShell use `.\.venv\Scripts\Activate.ps1`. On macOS/Linux use `source .venv/bin/activate`. The commands below work in either shell after activation:

```sh
python -m pip install -r requirements.txt
cd code
python scripts/fetch_models.py
python engine/run_recon.py ../sample_data/sneaker.png --quality draft --out out/quick-demo
```

If you already use uv, `uv venv --python 3.12 .venv` followed by `uv pip install -r requirements.txt` from the archive root is another install option. Activate that environment before running the demo.

Open `code/out/quick-demo/model.glb` in a GLB viewer. The output folder also has preview images and `manifest.json` (runtime and mesh diagnostics). Use `../sample_data/box.png` for the second input. The box is a project-generated reference. Its compact OBJ and manifest sit beside the samples for calibration:

```sh
python -m training.calibration.optimize_parameters --dataset ../sample_data/manifest.json --trials 2 --output out/calibration.json
```

The one-box run validates the geometry code path. A manifest with at least three distinct reference objects is needed for a fit/holdout comparison. Recorded benchmark and training tables are in `code/results/`. The package contains two image inputs; generated data, caches, weights, user databases, and local secrets are excluded.

## Start the website

Keep the terminal in `code/`:

```sh
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Open [the local Studio](http://127.0.0.1:8010/), register a local account, choose the Depth model and Draft, and upload an input. The prototype library includes saved results. Install optional providers with `python scripts/fetch_models.py --all` and restart the server. The CPU path was verified. For GPU and optional Gemini setup, see the [local setup guide](code/docs/handbook/local-setup.md). A 4 GB CUDA card was unavailable for testing.

The [course report](report.pdf) and [captioned demo](demo.mp4) accompany this README. The [benchmark slides](code/presentations/3DFUN-AI-Project.pptx) cover the earlier mesh evaluation.

## Selected documentation

The [project handbook](code/docs/handbook/README.md) links the local setup guide, architecture notes, operations runbook, and research reproducibility guide. Once the server is running, open the [handbook page](http://127.0.0.1:8010/project-docs) to browse them.

## Train and evaluate

From `code/`, use `python -m training.pipeline --device cpu` to print the training plan. It executes only when you pass `--run`. The [training guide](code/training/README.md) covers optional PyTorch dependencies, custom meshes, ablation, and held-out point-cloud metrics. The archive includes the measured eight-view point-cloud results for TinyDepth, the SILog-only ablation, and constant depth in [recorded evidence](code/results/course17/pointcloud_summary.json). TinyDepth is a separate learning experiment. The Studio's single-photo mesh still uses pretrained relative depth and a volumetric prior. The `code/submission/` folder includes the IEEE manuscript source and archive builder.

## Upstream credit and weight downloads

| Upstream | What this project uses or adds |
|---|---|
| [U²-Net](https://github.com/xuebinqin/U-2-Net) / [public U²-NetP ONNX release](https://github.com/danielgatis/rembg/releases/tag/v0.0.0) | Frozen foreground model. We add mask cleanup and a canonical square canvas. |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) / [public Small INT8 ONNX weights](https://huggingface.co/onnx-community/depth-anything-v2-small) | Frozen relative-depth model. We convert its output to bounded per-pixel thickness, build a distance field, mesh it and export textures. |
| [MiDaS](https://github.com/isl-org/MiDaS) / [public v2.1 Small weights](https://github.com/isl-org/MiDaS/releases/tag/v2_1) | Alternative frozen depth model using the same post-processing. |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) / [public ONNX conversion](https://huggingface.co/fernandotonon/QtMeshEditor-triposr-onnx) | Alternative direct image-to-3D model. We add local ONNX inference, mesh extraction and Studio integration. |

The download script checks published artifacts against the project's expected hashes. The discussion in the report distinguishes local results from upstream results.

