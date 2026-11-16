# Specialized Image-to-3D Constructions

This module provides 4 specialized, self-contained 3D computer vision reconstruction pipelines from single/multi-view images. All methods run on standard CPU without requiring discrete GPUs or gigabyte-sized neural checkpoints.

---

## Capabilities

| Script | Specialized 3D Method | Outputs | Best Suited For |
|---|---|---|---|
| `pointcloud_construction.py` | Calibrated RGB-D Backprojection | `.ply` (Dense Point Cloud + Colors + Unit Normals) | Surface point geometry, inspection, depth unprojection |
| `revolution_construction.py` | 360° Axisymmetric Surface of Revolution | `.obj` (Watertight Revolved Mesh + UVs + Normals) | Rotational symmetric objects (vases, bottles, cans, glasses, bowls) |
| `space_carving_construction.py` | Orthogonal Visual Hull / Space Carving | `.obj` (Polygon Mesh) + `.npz` (3D Voxel Grid) | Multi-view silhouettes, volumetric occupancy carving |
| `surface_reconstruction.py` | Depth-Guided Signed Distance Field (SDF) | `.obj` (Watertight Triangle Mesh + Taubin Smoothing) | Organic objects, closed solid meshes, shape prior inflation |

---

## Quick Start / Ready to Run

Run any specialized method directly, or use the master runner `run_specialized.py`:

```sh
# Run all 4 specialized constructions sequentially on a sample image:
python specialized_3d/run_specialized.py --image 3dfun-ai/engine/testdata/sneaker.png --mode all --out out/specialized_demo

# Run only dense Point Cloud reconstruction:
python specialized_3d/pointcloud_construction.py --image 3dfun-ai/engine/testdata/sneaker.png --out out/pcd_demo

# Run only Surface of Revolution:
python specialized_3d/revolution_construction.py --image 3dfun-ai/benchmarks/data/synthetic/images/box.png --out out/rev_demo

# Run only Space Carving (Visual Hull):
python specialized_3d/space_carving_construction.py --images 3dfun-ai/benchmarks/data/synthetic/images/box.png --res 64 --out out/carving_demo

# Run only Depth-Guided SDF Surface Mesh:
python specialized_3d/surface_reconstruction.py --image 3dfun-ai/engine/testdata/sneaker.png --res 64 --out out/surface_demo
```
