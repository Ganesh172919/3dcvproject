# Project file map

| Path | Contents |
| --- | --- |
| `app/` | API routing, storage, authentication, queues and generation lifecycle. |
| `engine/recon/` | Model inference, volumetric reconstruction, meshing and export. |
| `engine/specialized_3d/` | Optional point cloud, surface of revolution, visual hull and SDF exercises. |
| `web/` | Landing page, Studio, vendored Three.js modules and saved examples. |
| `benchmarks/` | Analytic dataset builder, classical baselines, and mesh metrics. |
| `training/` | Synthetic rendering, TinyDepth training, ONNX export, cloud evaluation, and volume calibration. |
| `tests/` | Pytest suite and Playwright browser smoke test. |
| `results/` | Recorded CPU benchmark, depth and cloud results, and report figures. |
| `submission/` | Course report source, demo generator, archive builder and verifier. |
| `paper/` | Earlier IEEE-format mesh benchmark paper and bibliography. |

Use `git ls-files` for the tracked source inventory. To inspect the deliverable use `python -m zipfile -l out/course17/Group17_3dcv_project.zip`.
