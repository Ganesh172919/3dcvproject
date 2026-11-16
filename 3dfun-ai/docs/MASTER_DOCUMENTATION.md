# 3DFUN AI technical handbook

3DFUN AI is a local 3D Computer Vision project from IIIT Dharwad. The Studio reconstructs an isolated object from one photograph. The TinyDepth workflow trains on synthetic data with known cameras and evaluates visible depth and point clouds.

## Reconstruct a textured mesh

1. The engine orients, crops, and centers a source photo. It uses a provided alpha mask or runs U²-NetP followed by mask cleanup.
2. Depth Anything V2 Small assigns relative inverse-depth ranks to foreground pixels. The engine normalizes the usable range and uses a bounded thickness function to add volume inside the outline.
3. A signed distance field represents the volume. Marching cubes extracts its zero surface. The engine removes small components, smooths the surface, and simplifies the mesh.
4. A planar atlas maps the source photo to the front and estimates the texture for other surfaces. The CLI and Studio support GLB, OBJ, and PLY export. The manifest records the effective configuration, timings, providers, and geometry checks.

See [the architecture](ARCHITECTURE.md), [volumetric inflation](explanations/volumetric-inflation.md), and [Studio quickstart](tutorials/quickstart.md). The separate [geometry exercises](how-to/specialized-3d-reconstruction.md) illustrate back-projection, visual hulls, and surfaces of revolution.

## Technical evidence

The [mesh benchmark](BENCHMARK.md) compares the depth-and-silhouette method with TSDF-style and visual-hull baselines across 25 scenes. Mesh-based metrics use the 13 synthetic scenes with reference geometry. The method produced closed surfaces in 72 of 75 runs. The thin plate and ring show why closure and silhouette agreement need geometry metrics alongside them.

The [TinyDepth report](explanations/tinydepth-and-pointcloud-evaluation.md) describes a 24,369-parameter network trained on procedural RGB and depth pairs. The eight withheld cylinder views produced mean point-cloud Chamfer of 0.0980, compared with 0.1358 for a constant-depth predictor. The evaluation assumes a known focal length and foreground mask.

The calibration tool in `training/calibration/optimize_parameters.py` samples thickness and smoothing settings. It selects a candidate on fitting scenes by Chamfer distance and evaluates the baseline and selected candidate on held-out scenes where available. It measures the volume stage before cleanup and texturing. The command in the course archive README runs calibration on the supplied box.

## Service and submission

- [REST API reference](reference/api-spec.md) and [generation lifecycle](reference/database-and-state-machine.md).
- [Environment settings](reference/configuration.md), [execution providers](tutorials/gpu-acceleration.md), and [local operations](handbook/operations.md).
- [Package the report, video, and source](how-to/academic-packaging-and-verification.md) and [troubleshoot common failures](TROUBLESHOOTING.md).
