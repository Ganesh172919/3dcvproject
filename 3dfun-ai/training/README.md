# Training TinyDepth

TinyDepth predicts camera-axis depth. The mesh renderer normalizes each object's longest side to 1 and stores depth in 16-bit PNGs at 1,000 counts per normalized unit. The trained model exports visible-surface point clouds from a supplied mask and known camera intrinsics.

From the source directory (inside `code/` in the archive), install the base requirements. For training and ONNX export, add:

```sh
python -m pip install -r requirements-training.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
```

For NVIDIA CUDA, use a compatible PyTorch build from the [PyTorch installation guide](https://pytorch.org/get-started/locally/) in place of the CPU build. The reference run used CPU. GPU usage on a 4 GB card has not been measured.

Print the plan for rendering, training, ONNX export, and held-out 3D evaluation:

```sh
python -m training.pipeline --device cpu --ablation
```

The command previews the stages and creates no run directory. Add `--run` when you want to execute them. Use `--views 3 --size 64 --epochs 2 --output-dir out/tinydepth-quick` for a short run. Choose a fresh output directory each time.

## Use your own meshes

Put at least three distinct OBJ or GLB files in `meshes/`, with unique filename stems. One split file can keep similar shapes together. For example, for `mug.obj`, `jar.glb`, `toy-car.glb`, and `bottle.obj`, a file named `mesh-split.json` could contain:

```json
{"train": ["mug", "jar"], "val": ["toy-car"], "test": ["bottle"]}
```

```sh
python -m training.pipeline --mesh-dir meshes --mesh-split mesh-split.json --output-dir out/custom-depth --device cpu --ablation
```

Assign each stem to exactly one split. Omit `--mesh-split` to use the seeded split. Keep copies or variants of the same shape in one split. To use existing renders, replace `--mesh-dir meshes` with `--data-dir out/training_renders`. That directory must follow the project renderer layout, with matching `rgb/`, `depth/`, `mask/`, and `cameras/` labels. Names use `MESH_view_NN.png`. Use `--grad-weight` to change the full model's gradient loss weight, which defaults to 0.5.

The pipeline checks the label files and writes `split.json` before fitting. `experiment.json` records stage status and commands. The trainer selects a checkpoint by validation MAE and records held-out MAE, RMSE, AbsRel, and δ1 along with a training-set median baseline in `model/training_history.json`. The exporter checks FP32 ONNX Runtime predictions against PyTorch.

The 3D evaluator reconstructs camera-frame points from a shared sample of valid pixels for the prediction, reference, and constant-depth baseline. It reports symmetric unsquared Chamfer and F-score at a distance of 0.01 normalized units by default. Use `--tau` and `--max-points` to set the threshold and sampling limit for both models. Results are averaged over views within each mesh and then across meshes. See `evaluation/per_view.csv`, `evaluation/summary.json`, and the two example PLYs. The [recorded CPU run](../results/course17/pointcloud_summary.json) used the eight held-out cylinder views. TinyDepth with gradients achieved Chamfer 0.0980 and F-score 0.0456, compared with 0.1358 and 0.0211 for constant depth.

With `--ablation`, the SILog-only run reuses the same split, seed, and evaluation settings. Its scores are in `silog_only/evaluation/`. `comparison.json` shows per-mesh and mean differences after checking that the two runs used matched views and metric settings.

To recheck a saved checkpoint's 2D metrics, use `python -m training.fine_tune_depth.train_depth --data-dir out/tinydepth-experiment/dataset --evaluate out/tinydepth-experiment/model/best_model.pth --device cpu`. To run SDF calibration on the separate mesh pipeline, fetch the pretrained weights and use `python -m training.calibration.optimize_parameters --dataset PATH/manifest.json`. A single scene checks calibration. At least three independent reference scenes enable a fit and holdout split.
