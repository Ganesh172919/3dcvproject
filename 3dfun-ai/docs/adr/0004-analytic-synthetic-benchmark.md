# ADR 0004: Procedural Analytic Benchmark with Closed Ground Truth Geometry

## Status
Accepted

## Context
Benchmarking 3D reconstruction algorithms is fraught with evaluation ambiguities:
1. Scanning real-world physical objects with LiDAR or structured-light scanners introduces sensor noise, occluded blind spots, and alignment calibration errors.
2. Popular datasets like ShapeNet or Objaverse contain thousands of "dirty" CAD models with self-intersecting faces, open planar sheets, and missing normals, corrupting ground-truth Chamfer distance and volumetric IoU calculations.

## Decision
We engineered a dual-track benchmark suite:
1. **Analytic Synthetic Dataset (13 Procedural Scenes):** Geometric primitives (spheres, cylinders, cones, tori, capsules, ellipsoids) with exact closed mathematical surface formulations and zero topological noise.
2. **Real Photograph Dataset (12 CC0-1.0 Physical Objects):** Real-world items evaluating operational metrics (latency, memory, visual SSIM, watertightness) where metric ground truth cannot be honestly claimed.

## Consequences
### Positive:
- **Known reference geometry:** Chamfer distance and F-score are estimated from sampled points. Volumetric IoU is estimated on a voxel grid.
- **Academic Honesty:** Avoids claiming artificial metric precision on uncalibrated consumer photographs.
- **Lightweight Repository:** Procedural generation scripts require $<2$ MB of storage, avoiding multi-gigabyte dataset downloads.

### Negative:
- Primitive geometric shapes do not capture high-frequency organic textures found in complex photorealistic scans.
