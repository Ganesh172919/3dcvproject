# Constrained volumetric inflation

A single image shows the visible outline and appearance. The reconstruction uses these observations to infer a volume, including the hidden side.

The depth-based pipeline converts relative depth to a closeness field \(d\) within the mask. Each occupied pixel receives thickness

\[
t=t_{\min}+(t_{\max}-t_{\min})\operatorname{clip}(f+g\,d^\gamma,0,1).
\]

The silhouette bounds the occupied voxels in the image plane. The depth term adds relief, and the constant term supplies thickness where the closeness estimate is low. The implementation computes a signed distance field from Euclidean distance transforms on both sides of the occupied volume.

Positive field values along the padded grid boundary keep the zero crossing away from that boundary. Marching cubes extracts the surface, then cleanup removes small components, smooths the mesh, and attempts to fill small holes. Interpolation and smoothing can shift the projected outline. The pipeline measures closure after cleanup and before splitting vertices for UV mapping.

The [implementation](../../engine/recon/volumetrics.py) and [mesher](../../engine/recon/meshing.py) contain the exact steps. The [benchmark protocol](../BENCHMARK.md) describes how the result is scored against TSDF and visual-hull baselines.
