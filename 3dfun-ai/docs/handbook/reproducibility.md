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

The plan renders RGB, depth, masks and camera metadata and splits by mesh identity. It selects a checkpoint on the validation split and exports it to ONNX. The point-cloud evaluator uses matched foreground pixels and camera intrinsics to compare predicted depth and a constant baseline against the reference depth. It records Chamfer distance and F-score per view and aggregates across held-out meshes. With `--ablation`, it uses the same split and metric settings for SILog alone.

Use `--mesh-dir` with at least three different OBJ or GLB meshes to plan a custom dataset. Use `--tau` and `--max-points` to set the 3D F-score threshold and point sample cap. The `training/README.md` explains optional dependencies and the explicit execution flag. The shipped box and sneaker are for the reconstruction demo.

## Review an extracted package

Follow the archive README from a new directory. Confirm that the CLI produces a GLB from `sample_data/`, the demo plays for its full duration, and the IEEE report includes the author block, seven numbered sections, references, and contributions. Use the approved group details when submitting.

**Evidence** `submission/package.py` · `submission/report.tex` · `training/pipeline.py` · `training/evaluate_clouds.py`
