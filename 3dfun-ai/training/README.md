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

The command above previews the steps and creates no run directory. Add `--run` to execute all stages when you are ready. Use `--views 3 --size 64 --epochs 2 --output-dir out/tinydepth-quick` for a short check or `--mesh-dir PATH` to supply at least three OBJ or GLB files with unique stems. Choose a fresh output directory for each run. The optional ablation trains and exports a second checkpoint with SILog alone and writes its 3D scores to `silog_only/evaluation/` using the same test views and metric settings.

Each experiment records its commands and stage status in `experiment.json`. All views of a mesh stay in one split. The trainer selects a checkpoint by validation MAE and records held-out MAE, RMSE, AbsRel, and δ1 along with a training-set median baseline in `model/training_history.json`. The exporter verifies FP32 ONNX Runtime predictions against PyTorch.

The 3D evaluator reconstructs camera-frame points from a shared sample of valid pixels for the prediction, reference, and constant-depth baseline. It reports symmetric unsquared Chamfer and F-score at a distance of 0.01 normalized units by default. Use `--tau` and `--max-points` to set the threshold and sampling limit for both models. Results are averaged over views within each mesh and then across meshes. See `evaluation/per_view.csv`, `evaluation/summary.json`, and the two example PLYs. The [recorded CPU run](../results/course17/pointcloud_summary.json) used the eight held-out cylinder views. TinyDepth with gradients achieved Chamfer 0.0980 and F-score 0.0456, compared with 0.1358 and 0.0211 for constant depth.

To recheck a saved checkpoint's 2D metrics, use `python -m training.fine_tune_depth.train_depth --data-dir out/tinydepth-experiment/dataset --evaluate out/tinydepth-experiment/model/best_model.pth --device cpu`. To run SDF calibration on the separate mesh pipeline, fetch the pretrained weights and use `python -m training.calibration.optimize_parameters --dataset PATH/manifest.json`. A single scene checks calibration. At least three independent reference scenes enable a fit and holdout split.
