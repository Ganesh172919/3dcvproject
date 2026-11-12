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
