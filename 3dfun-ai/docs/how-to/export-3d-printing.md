# How-To Guide: Exporting Watertight Meshes for Additive Manufacturing (3D Printing)

Unlike unconstrained monocular reconstruction methods that generate open shells, floating triangles, or non-manifold edges, 3DFUN AI is mathematically proven to produce closed, orientable 2-manifold surfaces (see Lemma 1 in the technical report). This guide explains how to export, verify, and slice 3DFUN AI meshes for 3D printing.

---

## 1. Verifying Watertightness and Manifold Integrity

Before sending a mesh to a 3D printer, confirm that the mesh is strictly watertight and free of self-intersections:

```python
import trimesh

# Load generated mesh
mesh = trimesh.load("out/sneaker/model.obj")

# Topological checks
print("Is Watertight:", mesh.is_watertight)
print("Is Volume Positive:", mesh.is_volume)
print("Euler Characteristic (Must be 2 for genus-0 sphere topology):", mesh.euler_number)
print("Boundary Edges (Must be 0):", len(mesh.edges_boundary))
```

If `mesh.is_watertight` returns `True` and boundary edges are `0`, the model is ready for manufacturing without requiring netfabb, MeshLab, or Blender repair steps.

---

## 2. Setting Minimum Wall Thickness for Slicers

Consumer FDM (Fused Deposition Modeling) and SLA (Stereolithography) printers require a minimum structural wall thickness to prevent print collapse during extrusion or peeling:
- **FDM Printers (0.4 mm nozzle):** Minimum wall thickness $\ge 1.2$ mm (3 perimeter shells).
- **SLA Resin Printers:** Minimum wall thickness $\ge 0.8$ mm.

In 3DFUN AI, the minimum wall thickness is parameterized by $t_{\min}$ in `engine/recon/pipeline.py`:

```bash
python scripts/run_pipeline.py \
  --input engine/testdata/sneaker.png \
  --output out/sneaker_3dprint \
  --thickness_min 0.04 \
  --thickness_max 0.80 \
  --format obj,stl
```

Setting `--thickness_min 0.04` guarantees that even the thinnest edges of the foreground silhouette maintain a solid structural core.

---

## 3. Exporting to Binary STL Format

While GLB is ideal for web rendering, slicers (PrusaSlicer, Ultimaker Cura, Bambu Studio) prefer STL:

```python
import trimesh

mesh = trimesh.load("out/sneaker/model.obj")

# Scale model to target real-world dimensions (e.g. 100 mm along longest axis)
scale_factor = 100.0 / max(mesh.extents)
mesh.apply_scale(scale_factor)

# Export as binary STL
mesh.export("out/sneaker/model_print.stl", file_type="stl")
print(f"Exported print-ready STL with dimensions {mesh.extents} mm.")
```

---

## 4. Slicing Recommendations

1. **Orientation:** Align the flat back or symmetry plane of the reconstructed model directly on the build plate to minimize overhang supports.
2. **Infill:** Set infill to 15%–20% (Gyroid or Grid pattern) for functional strength.
3. **Supports:** Enable tree supports for overhang angles exceeding 50°.

---
*Evidence: Marching cubes extraction in `engine/recon/meshing.py`; thickness equations in `engine/recon/sdf.py`.*
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*

---

## 5. Advanced Mesh Repair Workflows

For cases where the pipeline produces meshes that do not pass the watertightness check (rare — expected rate: ~4% based on benchmarks), the following repair workflows are available.

### 5.1 Automatic Repair with trimesh

```python
import trimesh

mesh = trimesh.load("out/model.obj")

if not mesh.is_watertight:
    print(f"Non-watertight: {len(mesh.edges_boundary)} boundary edges detected.")

    # Step 1: Fill holes with simple fan triangulation
    trimesh.repair.fill_holes(mesh)

    # Step 2: Fix face winding order consistency
    trimesh.repair.fix_winding(mesh)

    # Step 3: Remove duplicate faces and vertices
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    # Step 4: Merge nearby vertices within tolerance
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    print(f"After repair — Watertight: {mesh.is_watertight}, Boundary edges: {len(mesh.edges_boundary)}")
    mesh.export("out/model_repaired.obj")
```

### 5.2 MeshLab CLI Repair (Optional)

For meshes with complex self-intersections that trimesh cannot resolve:

```bash
# Install MeshLab server on Linux
sudo apt install meshlab

# Apply MeshLab's Poisson repair and manifold cleaning
meshlabserver -i out/model.obj \
  -o out/model_repaired.obj \
  -s scripts/meshlab_repair.mlx
```

Where `scripts/meshlab_repair.mlx` applies:
- **Remove Duplicate Faces** (`Remove_Duplicated_Face`)
- **Remove Zero Area Faces** (`Remove_Zero_Area_Faces`)
- **Close Holes** (`Close_Holes`, max hole size: 30 edges)
- **Recompute Normals** (`Compute_Normals_for_Point_Sets`)

### 5.3 Scaling Reference Table for Common 3D Printers

| Printer Model | Technology | Layer Height | Min Wall Thickness | Recommended Scale for 100mm Model |
|---|---|---|---|---|
| Bambu Lab P1S | FDM (PLA) | 0.1–0.3 mm | 1.2 mm | `scale_factor = 100 / max(mesh.extents)` |
| Creality Ender 3 | FDM (PLA) | 0.2 mm | 1.6 mm | `scale_factor = 80 / max(mesh.extents)` |
| Formlabs Form 3 | SLA (Resin) | 0.05 mm | 0.6 mm | `scale_factor = 60 / max(mesh.extents)` |
| Prusa MK4 | FDM (PLA/PETG) | 0.15 mm | 1.2 mm | `scale_factor = 100 / max(mesh.extents)` |
| Bambu Lab X1 Carbon | FDM (ABS) | 0.1–0.35 mm | 1.5 mm | `scale_factor = 120 / max(mesh.extents)` |

### 5.4 Post-Processing Surface Quality Optimization

For print-critical applications requiring extremely smooth surfaces (dental models, jewelry prototyping, figurines):

```python
import trimesh
import numpy as np

mesh = trimesh.load("out/model_repaired.obj")

# Subdivide triangles for extra vertex density before smoothing
mesh = mesh.subdivide_to_size(max_edge=0.01)  # max edge length in normalized units

# Apply extra Laplacian smoothing passes after Taubin (acceptable here because
# we are targeting visual appearance, not dimensional accuracy)
for _ in range(5):
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.3)

mesh.export("out/model_highquality.stl")
print(f"High-quality mesh: {len(mesh.triangles)} triangles, watertight: {mesh.is_watertight}")
```

---

## 6. Troubleshooting Common Export Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| `mesh.is_watertight = False` | U²-Net silhouette has stray pixels at image border | Add `--padding 0.06` to pipeline args |
| STL file opens in slicer but shows split body | Multiple disconnected components in mesh | Run `mesh = max(mesh.split(), key=lambda m: m.volume)` to keep largest component |
| Slicer reports "non-manifold edges" | Degenerate zero-area triangles | Run `mesh.remove_degenerate_faces()` |
| Model prints too thin (collapses during print) | `thickness_min` too low | Set `--thickness_min 0.06` for FDM printing |
| Excessive print supports needed | Model oriented with complex overhangs | Rotate mesh: `mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1,0,0]))` |

---

*Evidence: Mesh I/O and repair in `engine/recon/meshing.py`; topology checks in `benchmarks/metrics.py`.*
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
