# Mesh benchmark

The recorded evaluation used 25 scenes, three methods, and three repeated CPU runs per method and scene. The 13 procedural scenes have reference meshes. The 12 real-image scenes contribute to runtime, rendering and topology measurements. This historical cohort is described by `results/dataset_metadata.json` and `benchmarks/ATTRIBUTION.md`.

## Methods and metrics

- **Depth + SDF (Course Method - Group 17)** uses a silhouette and pretrained relative depth to construct a bounded volume. Marching cubes extracts the surface, which is cleaned, decimated and textured.
- **TSDF-style baseline** builds a surface from a single estimated depth map.
- **Visual-hull baseline** extrudes the silhouette to a depth-derived back plane.

Geometry metrics use 20,000 sampled points per mesh after normalization and inertia alignment. Chamfer is the symmetric mean nearest-neighbor distance. F-score uses a threshold of 0.01 normalized object units. IoU measures occupied sample bins in a shared 64³ grid. See the [metric definitions](reference/metrics.md).

| Method | Chamfer ↓ | F-score ↑ | Surface IoU ↑ | Closed | Mean wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Depth + SDF (Course Method - Group 17) | **0.0639** | **0.148** | **0.055** | **96%** | 27.73 s |
| TSDF-style | 0.0962 | 0.053 | 0.027 | 0% | 1.43 s |
| Visual hull | 0.0728 | 0.098 | 0.038 | 0% | 2.13 s |

Geometry columns use the 3D render split. Closure and wall time include all 25 scenes. The comparisons describe the measured pipeline implementations, which do different amounts of preview and texture work.

- `results/raw/runs.csv`, `results/summary/main_table.csv` and `results/summary/per_scene.csv`.
- `results/summary/ablation_table.csv`, `results/summary/thickness_sweep.csv` and `results/summary/failures.csv`.
- `results/environment.json`. This cohort predates the TripoSR, MiDaS and TinyDepth experiments.

## Run the benchmark

Use a separate working copy if you wish to preserve the recorded tables.

```sh
python scripts/fetch_models.py
python benchmarks/make_dataset.py
python benchmarks/run_benchmarks.py --quick
```

Run without `--quick` for the cohort available locally. The dataset builder generates all 13 analytic scenes and includes the sneaker and any compatible images already in the optional photo cache. See the attribution table for the historical photo sources. Use `--force` for new timings when the output has cached runs.

TinyDepth training and camera-frame point-cloud evaluation use a different protocol. Follow the `training/README.md` and consult `results/course17/pointcloud_summary.json`.
