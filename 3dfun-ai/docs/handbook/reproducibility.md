# Research and reproducibility

## Read the existing evidence

- `report.pdf` at the archive root gives the experiment setup, results, ablations and failure cases.
- `code/results/summary/` contains the recorded reconstruction tables. `code/results/raw/runs.csv` contains the per-run records. `code/results/dataset_metadata.json` records the cohort size and dataset revision in the archive.
- `code/results/course17/` contains the TinyDepth and SILog-only training histories, held-out cloud scores for both, and the course figures.

## Inspect a training plan

From `code/`, the command below prints the stages without creating files:

```sh
python -m training.pipeline --device cpu --ablation
```

The plan renders RGB, depth, masks and camera metadata and saves a mesh-level split in `split.json`. It selects a checkpoint on validation MAE and exports it to ONNX. The point-cloud evaluator uses matched foreground pixels and camera intrinsics to compare the predicted and reference surfaces. It records Chamfer distance and F-score per view and aggregates across held-out meshes. With `--ablation`, the two runs share a split and metric settings. `comparison.json` records the paired scores and differences.

Use `--mesh-dir` with at least three different OBJ or GLB meshes, or `--data-dir` to reuse the output of the project renderer. Supply `--mesh-split` with a JSON map of `train`, `val`, and `test` mesh stems to keep variants together. See `training/README.md` for an example, the environment setup, and the explicit execution flag. The shipped box and sneaker are for the reconstruction demo.

The website keeps the three small training lessons on the Student Labs page. Install the PyTorch dependencies from the training README before starting those lessons. The metrics lab runs on the base CPU install. Its seed and sample count make the noisy-sphere comparison repeatable. The evaluator also accepts actual meshes and point clouds on the command line. Compare geometry in a common coordinate frame when using your own files.

## Review an extracted package

Follow the archive README from a new directory. Confirm that the CLI produces a GLB from `sample_data/`, the demo plays for its full duration, and the IEEE report includes the author block, seven numbered sections, references, and contributions. Use the approved group details when submitting.

**Evidence** `submission/package.py` · `submission/report.tex` · `training/pipeline.py` · `training/evaluate_clouds.py`
