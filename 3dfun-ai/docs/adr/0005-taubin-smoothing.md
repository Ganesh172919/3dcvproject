# ADR 0005: Taubin mesh smoothing

## Decision

Use Open3D's Taubin filter in `engine/recon/meshing.py` to reduce voxel staircase artifacts. The filter alternates smoothing and expansion passes to limit shrinkage.

## Consequences

The filter moves vertices and can change the enclosed volume or soften small features. The `results/summary/ablation_table.csv` records measured geometry metrics for the full pipeline and the run without Taubin smoothing.

The recorded ablation gives Chamfer 0.06394 without Taubin smoothing and 0.06392 with it. The change is small on the 13 synthetic scenes. The motivation is reducing visible grid artifacts. See [the benchmark](../BENCHMARK.md).
