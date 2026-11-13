# ADR 0005: Taubin mesh smoothing

## Decision

Use Open3D's Taubin filter in [mesh cleanup](../../engine/recon/meshing.py) to reduce voxel staircase artifacts. The filter alternates smoothing and expansion passes to limit shrinkage.

## Consequences

The filter moves vertices and can change the enclosed volume or soften small features. The [ablation table](../../results/summary/ablation_table.csv) records measured geometry metrics for the full pipeline and the run without Taubin smoothing.
