# Train TinyDepth on custom geometry

TinyDepth predicts visible camera-axis depth on 3D render RGB images. Training uses foreground masks and a mesh-level split. The [point-cloud explanation](../explanations/tinydepth-and-pointcloud-evaluation.md) describes the camera geometry and the measured metrics.

Install the base and training requirements from the project source directory. Add the same PyTorch CPU build used in the training README.

```sh
python -m pip install -r requirements.txt -r requirements-training.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
```

Prepare a directory containing at least three OBJ or GLB meshes with distinct file stems. Preview the run plan and then start training.

```sh
python -m training.pipeline --mesh-dir path/to/meshes --device cpu --ablation
python -m training.pipeline --mesh-dir path/to/meshes --device cpu --ablation --run --output-dir out/custom-depth
```

Omit `--mesh-dir` to use the procedural scene set. Set `--views`, `--size` and `--epochs` to adjust run length. Each experiment uses a new output directory to preserve previous measurements.

Review `model/training_history.json` and `evaluation/per_view.csv`. The export stage produces `model/tinydepth.onnx` in FP32 and tests numerical agreement with the PyTorch checkpoint. It also exports example reference and predicted point clouds in PLY format. The Studio uses its own pretrained depth model for complete mesh reconstruction.
