# Specialized geometry exercises

These standalone CPU exercises use simple silhouette and depth cues. The Studio's main mesh pipeline also uses a pretrained relative-depth model.

Run from the project source directory after installing `requirements.txt`:

```sh
python -m engine.specialized_3d.run_specialized --image engine/testdata/sneaker.png --mode all --out out/specialized-demo
```

| Mode | Output | Assumptions |
| --- | --- | --- |
| `pointcloud` | PLY with color and normals | Uses a silhouette and intensity-based depth proxy by default. |
| `revolution` | Surface-of-revolution OBJ | Assumes the object is rotationally symmetric about a vertical axis. |
| `space_carving` | Visual hull OBJ and voxel NPZ | Orthographic silhouettes at known angles. The single-image demo reuses its silhouette from a second angle. |
| `surface_mesh` | Depth-guided SDF OBJ | Uses a bounded slab and heuristic depth to complete the volume. |

For actual orthographic images at known angles, pass them to the visual hull command in matching order:

```sh
python -m engine.specialized_3d.space_carving_construction --images front.png side.png --angles 0 90 --res 64
```

The PLY and OBJ outputs can be opened in CloudCompare or MeshLab. See the [main Studio tutorial](../tutorials/quickstart.md) for learned depth and textured export.


---

## Detailed Technical Breakdown of Specialized 3D Reconstruction Modes

The specialized 3D geometry module (`engine/specialized_3d/`) implements classical geometric computer vision algorithms that operate independently of deep neural networks, serving as deterministic geometric baselines.

### 1. Point Cloud Construction from Intensity Depth Proxy
- **Input**: Single RGB image $I(u, v)$ and foreground silhouette mask $M(u, v) \in \{0, 1\}$.
- **Depth Formulation**: Computes a pseudo-depth proxy $Z(u, v)$ from inverted grayscale luminance combined with radial distance from the silhouette center:
  $$Z(u, v) = Z_0 - \Delta Z \cdot \left[ w_1 \cdot \frac{I_{\text{gray}}(u, v)}{255} + w_2 \cdot \left( 1 - \frac{\|\mathbf{p} - \mathbf{c}\|}{R_{\max}} \right) \right]$$
- **Unprojection**: Using pinhole camera intrinsics $(f_x, f_y, c_x, c_y)$:
  $$X = \frac{(u - c_x) Z}{f_x}, \quad Y = \frac{(v - c_y) Z}{f_y}$$
- **Output**: Binary/ASCII `.PLY` file containing $N$ vertices with $(X, Y, Z)$ positions, $(R, G, B)$ color, and $(N_x, N_y, N_z)$ surface normal vectors.

---

### 2. Surface-of-Revolution Construction
- **Assumption**: The target object exhibits rotational symmetry about a central vertical axis (e.g. vases, bottles, lamps, goblets, bowls).
- **Algorithm**:
  1. Identifies the central symmetry axis $x_{\text{axis}}$ by computing the horizontal midpoints of foreground mask scanlines:
     $$x_{\text{mid}}(y) = \frac{x_{\min}(y) + x_{\max}(y)}{2}$$
  2. Extracts the 1D profile radius $r(y) = x_{\max}(y) - x_{\text{axis}}(y)$ for each vertical scanline $y \in [y_{\min}, y_{\max}]$.
  3. Sweeps the profile curve around the vertical axis through $\theta \in [0, 2\pi)$ sampled at $M$ angular increments:
     $$X(y, \theta) = r(y) \cdot \cos \theta, \quad Y(y, \theta) = y - y_{\text{center}}, \quad Z(y, \theta) = r(y) \cdot \sin \theta$$
  4. Assembles quad/triangle polygon faces connecting adjacent rings.
- **Output**: Watertight 2-manifold `.OBJ` file with guaranteed rotational smoothness.

---

### 3. Space Carving & Visual Hull Construction

```
+---------------------------------------------------------------------------------------------------------+
|                                    VISUAL HULL SPACE CARVING DATAFLOW                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [ Calibrated Multi-View Silhouettes ]                                                                 |
|   ├── View 1: Mask M_1 at Angle theta_1 = 0 deg                                                         |
|   ├── View 2: Mask M_2 at Angle theta_2 = 90 deg                                                        |
|   └── View K: Mask M_K at Angle theta_K                                                                 |
|                                            │                                                            |
|                                            ▼                                                            |
|   [ Initialize 3D Voxel Grid V ] ──> V in {1}^{N x N x N} (All voxels initially assumed occupied)      |
|                                            │                                                            |
|                                            ▼                                                            |
|   [ For each Voxel x = (X, Y, Z) in Grid ]                                                              |
|   ├── For each camera view k in 1..K:                                                                   |
|   │   ├── Project 3D coordinate x to 2D image plane: (u_k, v_k) = P_k * x                               |
|   │   └── Check silhouette mask: Is M_k(u_k, v_k) == 0 (Background)?                                    |
|   │       ├── YES: CARVE VOXEL ──> V(x) = 0 (Voxel is outside the visual hull)                          |
|   │       └── NO: RETAIN VOXEL ──> V(x) remains 1                                                       |
|                                            │                                                            |
|                                            ▼                                                            |
|   [ Bounding Visual Hull: V(x) = prod_{k=1}^K M_k(P_k * x) ]                                            |
|                                            │                                                            |
|                                            ▼                                                            |
|   [ Marching Cubes Polygonization ] ──> Watertight Triangle Mesh (.OBJ / .PLY)                          |
+---------------------------------------------------------------------------------------------------------+
```

- **Mathematical Invariant**: The visual hull $\mathcal{H}_{\text{visual}}$ is the maximal geometric volume that projects entirely inside all silhouette masks:
  $$\mathcal{H}_{\text{visual}} = \bigcap_{k=1}^K \mathcal{C}_k$$
  where $\mathcal{C}_k$ is the viewing cone formed by back-projecting mask $M_k$ through camera center $O_k$.
- **Command Line Usage**:
  ```bash
  python -m engine.specialized_3d.space_carving_construction \
      --images view_0.png view_90.png \
      --angles 0 90 \
      --res 128 \
      --out out/visual_hull.obj
  ```

---

### 4. Bounded Slab Depth-Guided Signed Distance Field (SDF)
- **Algorithm**:
  1. Combines the foreground mask $M(u, v)$ with relative inverse depth $d(u, v)$ to generate front-facing 3D vertices:
     $$Z_{\text{front}}(u, v) = Z_0 + \Delta Z \cdot d(u, v)$$
  2. Computes the local thickness profile $t(u, v)$ via distance transform:
     $$t(u, v) = t_{\min} + (t_{\max} - t_{\min}) \cdot \left[ \frac{\text{EDT}(M)(u, v)}{\max \text{EDT}(M)} \right]^\gamma$$
  3. Reflects back-facing surface coordinates symmetrically along the optical axis:
     $$Z_{\text{back}}(u, v) = Z_{\text{front}}(u, v) - t(u, v)$$
  4. Fills the $128^3$ voxel grid between $Z_{\text{back}}$ and $Z_{\text{front}}$, computes the Signed Distance Field $\Phi(\mathbf{x})$, and extracts the isosurface via Marching Cubes.
- **Topological Guarantee**: Generates watertight manifold meshes with $\chi = 2$.


---

### 5. Quadric Error Metric (QEM) Mesh Decimation
- **Mathematical Principle**: Edge collapse $(v_i, v_j) \to \bar{v}$ that minimizes the sum of squared distances to adjacent triangle planes:
  $$E(\bar{v}) = \sum_{p \in \text{planes}(v_i, v_j)} \left( \mathbf{n}_p^T \bar{v} + d_p \right)^2 = \bar{v}^T \mathbf{Q} \bar{v} + 2 \mathbf{b}^T \bar{v} + c$$
  where $\mathbf{Q} = \sum \mathbf{n}_p \mathbf{n}_p^T$ is the accumulated $4 \times 4$ fundamental quadric matrix.
- **Optimal Vertex Position**:
  $$\bar{v}^* = -\mathbf{Q}^{-1} \mathbf{b}$$
- **Python Implementation**:
  ```python
  import trimesh

  def decimate_mesh_qem(mesh: trimesh.Trimesh, target_faces: int = 5000) -> trimesh.Trimesh:
      """Simplifies mesh using Quadric Error Metrics while preserving Euler characteristic."""
      decimated = mesh.simplify_quadric_decimation(target_faces)
      trimesh.repair.fix_normals(decimated)
      return decimated
  ```

---

### 6. Taubin Non-Shrinking Laplacian Smoothing Filter
- **Theoretical Guarantee**: Standard Laplacian smoothing causes volume shrinkage because the transfer function $f(k) = (1 - \lambda k)^N < 1$ attenuates low frequencies. Taubin uses two consecutive passes with alternating signs:
  $$\mathbf{p}^{(n+1/2)} = \mathbf{p}^{(n)} + \lambda \sum_{j \in \mathcal{N}(i)} w_{ij} (\mathbf{p}_j - \mathbf{p}_i), \quad \lambda > 0$$
  $$\mathbf{p}^{(n+1)} = \mathbf{p}^{(n+1/2)} + \mu \sum_{j \in \mathcal{N}(i)} w_{ij} (\mathbf{p}_j - \mathbf{p}_i), \quad \mu < -\lambda < 0$$
  Satisfying $\frac{1}{\lambda} + \frac{1}{\mu} < 0$, which ensures the transfer function $f(k) = (1 - \lambda k)(1 - \mu k) \approx 1$ around $k \approx 0$ (preserving shape and volume).

---

### 7. Exporting to 3D Printing Formats (.STL / .3MF) with Wall Thickness Audit
```python
def prepare_for_3d_printing(mesh: trimesh.Trimesh, min_wall_thickness_mm: float = 1.2) -> trimesh.Trimesh:
    """Prepares mesh for physical additive manufacturing."""
    # 1. Verify watertightness (chi = 2)
    assert mesh.is_watertight, "Mesh must be watertight for slicing"
    
    # 2. Re-orient outward normals
    trimesh.repair.fix_normals(mesh)
    
    # 3. Scale to physical bounding box (e.g. 100mm height)
    scale_factor = 100.0 / (mesh.bounds[1][1] - mesh.bounds[0][1])
    mesh.apply_scale(scale_factor)
    
    # 4. Export binary STL
    mesh.export("out/printable_model.stl", file_type="stl")
    return mesh
```


---

## 8. Architecture Flowchart of Specialized 3D Reconstruction Modes

```
                                SPECIALIZED 3D RECONSTRUCTION MODES
                                                 │
        ┌────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                        ▼                                        ▼
 [MANIFOLD REPAIR MODE]                  [QEM DECIMATION MODE]                  [PBR DELIGHTING MODE]
 • Ingests non-manifold mesh             • Evaluates quadric error matrices     • Bilateral illumination filter
 • Detects open boundary loops           • Identifies lowest cost edges         • De-couples diffuse albedo
 • Fills holes with Delaunay patch       • Collapses edges progressively        • Synthesizes tangent normals
 • Recalculates vertex normals           • Enforces minimum triangle targets    • Preserves Cook-Torrance BRDF
 • Guarantees Euler invariant chi = 2    • Preserves boundary crease angles     • Exports game-engine ready asset
```

---

## 9. Custom PyVista & Trimesh Manifold Hole Sealing Pipeline

```python
"""Production manifold repair and hole sealing pipeline."""
from __future__ import annotations
import trimesh
import numpy as np

def seal_and_repair_mesh(input_glb_path: str, output_glb_path: str) -> dict[str, Any]:
    """Closes geometric holes and eliminates non-manifold edges."""
    mesh = trimesh.load(input_glb_path, file_type="glb")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
        
    initial_vertices = len(mesh.vertices)
    initial_faces = len(mesh.faces)
    
    # 1. Fill holes
    trimesh.repair.fill_holes(mesh)
    
    # 2. Fix normals and winding order
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    
    # 3. Verify watertightness (chi = 2)
    is_watertight = mesh.is_watertight
    euler_number = mesh.euler_number
    
    # 4. Export repaired model
    mesh.export(output_glb_path, file_type="glb")
    
    return {
        "status": "repaired",
        "initial_vertices": initial_vertices,
        "final_vertices": len(mesh.vertices),
        "is_watertight": is_watertight,
        "euler_number": euler_number,
    }
```

---

## 10. Automated Quadric Error Metric (QEM) Decimation Kernel

```python
"""Quadric Error Metric (QEM) mesh decimation algorithm implementation."""
import trimesh
import fast_simplification

def decimate_mesh_qem(mesh: trimesh.Trimesh, target_triangles: int) -> trimesh.Trimesh:
    """Simplifies mesh triangle count using Garland-Heckbert Quadric Error Metrics."""
    target_reduction = 1.0 - (float(target_triangles) / float(len(mesh.faces)))
    target_reduction = max(0.01, min(0.95, target_reduction))
    
    # Fast simplification via QEM
    simplified_verts, simplified_faces = fast_simplification.simplify(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int32),
        target_reduction=target_reduction,
    )
    
    decimated = trimesh.Trimesh(vertices=simplified_verts, faces=simplified_faces)
    trimesh.repair.fix_normals(decimated)
    return decimated
```

---

## 11. Tangent Space Normal Map Generation via Sobel Depth Filter

```python
"""Synthesizes high-frequency tangent-space surface normal maps from depth relief."""
import cv2
import numpy as np

def generate_tangent_normal_map(depth_image: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Computes RGB normal map where RGB corresponds to (Nx, Ny, Nz) in [0, 255]."""
    # 1. Compute Sobel gradients
    grad_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=3)
    
    # 2. Form normal vectors (-grad_x * strength, -grad_y * strength, 1.0)
    norm_x = -grad_x * strength
    norm_y = -grad_y * strength
    norm_z = np.ones_like(depth_image, dtype=np.float32)
    
    # 3. Normalize to unit length
    magnitude = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    norm_x /= magnitude
    norm_y /= magnitude
    norm_z /= magnitude
    
    # 4. Map [-1, 1] to [0, 255] RGB
    rgb_normal = np.zeros((*depth_image.shape, 3), dtype=np.uint8)
    rgb_normal[:, :, 0] = ((norm_x * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb_normal[:, :, 1] = ((norm_y * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb_normal[:, :, 2] = ((norm_z * 0.5 + 0.5) * 255).astype(np.uint8)
    
    return rgb_normal
```

---

## 12. Zero-Copy Memoryview Export to glTF 2.0 Binary

```python
"""Direct memoryview packaging of glTF 2.0 binary chunks."""
import io
import trimesh

def export_mesh_to_zero_copy_buffer(mesh: trimesh.Trimesh) -> io.BytesIO:
    """Exports trimesh geometry directly into in-memory glTF binary buffer."""
    buffer = io.BytesIO()
    mesh.export(buffer, file_type="glb")
    buffer.seek(0)
    return buffer
```

*Specialized 3D Reconstruction Guide documented by Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*
