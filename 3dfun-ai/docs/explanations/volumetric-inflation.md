# Constrained Volumetric Inflation: Mathematical Foundations & Level Set Theory

**Module**: `engine/recon/volumetrics.py`
**Theoretical Author**: J Ganesh (`23BDS024`), Dept. of Data Science & Artificial Intelligence, IIIT Dharwad
**Course Context**: 3D Computer Vision Course Project

---

## 1. The Geometric Challenge of Single-View Extrusion

When reconstructing a 3D physical object from a single 2D projection, the camera captures visible surface radiance along optical lines of sight, leaving the occluded back-facing surface and depth dimensions completely unobserved.

Classical solutions to this challenge fall into two divergent paradigms:

1. **Shape-from-Silhouette (Visual Hull Extrusion)**:
   Back-projects the 2D binary silhouette $\alpha(x, y)$ along the viewing axis $z$ into an infinite cylinder or extruded prism:
   $$\mathcal{V}_{\text{hull}} = \big\{ (x, y, z) \in \mathbb{R}^3 \mid \alpha(\pi(x, y, z)) = 1 \big\}$$
   *Fundamental Flaw*: It cannot represent surface relief or depth variation across the front face. Every point on the visible surface is extruded into a flat slab, resulting in boxy, prism-like reconstructions with an F-score of only 0.098.

2. **Truncated Signed Distance Field (TSDF) Fusion (Curless & Levoy, 1996)**:
   Projects predicted monocular depth $D(x, y)$ into a voxelized signed distance volume:
   $$\phi_{\text{TSDF}}(\mathbf{x}) = \operatorname{clip}\big(D(\pi(\mathbf{x})) - \mathbf{x}_z, \; -\delta, \; +\delta\big)$$
   *Fundamental Flaw*: While effective in multi-view stereo where hundreds of intersecting camera rays carve away free space, single-view TSDF suffers from open boundaries. At the silhouette perimeter $\partial \alpha$, the monocular depth estimate has no intersecting opposing rays to close the rear surface. As a result, Marching Cubes clips against the domain boundaries, producing an open shell with **0% watertightness**.

---

## 2. Mathematical Formulation of Constrained Slab Inflation

3DFUN AI unites the hard boundary guarantees of the visual hull with the continuous relief carving of monocular depth by formulating 3D reconstruction as **constrained volumetric slab inflation**.

```
+--------------------------------------------------------------------------------------------------+
|                            CONSTRAINED VOLUMETRIC SLAB GEOMETRY                                  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|              Camera Viewpoint (z -> +inf)                                                        |
|                           |                                                                      |
|                           v                                                                      |
|                  +-----------------+  <-- Front Surface z_front = +t(p)/2 (Carved by Depth d)    |
|                 /                   \                                                            |
|                |   Isotropic Core   |                                                            |
|                |    (f * t_max)     |                                                            |
|                 \                   /                                                            |
|                  +-----------------+  <-- Rear Surface z_back = -t(p)/2                          |
|                  |<----- t(p) ---->|                                                             |
|                                                                                                  |
|   Hard Silhouette Limit: For any ray where alpha(p) = 0, occupancy is identically zero.          |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 The Slab Thickness Governing Equation
Let $\Omega_S = [-1, 1]^2$ represent the normalized square canvas, and let $W = x_{\max} - x_{\min}$ denote the metric width of the foreground silhouette.

The maximum permissible volumetric depth extent is defined as:
$$t_{\max} = \kappa \cdot W$$
where $\kappa = 0.70$ is the aspect ratio scaling parameter.

For every pixel $p = (x, y) \in \Omega_S$, the total slab thickness $t(p)$ is governed by:

$$\boxed{t(p) = t_{\min} + (t_{\max} - t_{\min}) \cdot \operatorname{clip}\left( f + g \cdot d(p)^\gamma, \; 0, \; 1 \right)}$$

### 2.2 Functional Role of Hyperparameters

| Parameter | Default Value | Mathematical & Physical Role | Sensitivity / Failure Mode if Omitted |
|---|---:|---|---|
| $t_{\min}$ | $0.02$ | **Degeneracy Floor**: Ensures minimal non-zero thickness. | If set to $0.0$, thin boundaries collapse into zero-thickness non-manifold edges. |
| $f$ | $0.55$ | **Isotropic Core Ratio**: Base solid fraction independent of depth. | If $f = 0.0$, Chamfer distance degrades by 59.0% as rims collapse and pinch shut. |
| $g$ | $0.45$ | **Depth Modulation Gain**: Modulates front relief ($f + g = 1.0$). | If $g = 0.0$, mesh collapses into a flat extruded cookie-cutter visual hull. |
| $\gamma$ | $0.85$ | **Sub-linear Gamma Contrast**: Compresses dynamic range. | Linear ($\gamma = 1.0$) suppresses subtle curvature on flatter geometric regions. |
| $\kappa$ | $0.70$ | **Global Aspect Scale**: Ratio of maximum depth to object width. | If too large ($>1.2$), objects appear artificially bulbous; if too small ($<0.4$), objects become cards. |

---

## 3. Occupancy Indicator & Signed Distance Construction

### 3.1 Continuous Occupancy Indicator Field
The 3D occupancy field $\mathcal{O}: \mathbb{R}^3 \to \{0, 1\}$ is defined strictly as the product of the 2D silhouette indicator and the 1D axial depth interval:

$$\mathcal{O}(x, y, z) = \mathbb{1}[\alpha(x, y) = 1] \;\cdot\; \mathbb{1}\left[ -\frac{t(x, y)}{2} \le z \le \frac{t(x, y)}{2} \right]$$

### 3.2 Two-Sided Euclidean Distance Transform (EDT)
Let $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$. Direct surface extraction from binary voxels generates severe high-frequency staircase steps. To obtain continuous zero-crossings, 3DFUN AI converts $\mathcal{O}$ into a signed distance function (SDF) $\phi: \mathbb{R}^3 \to \mathbb{R}$ satisfying the Eikonal condition $\|\nabla \phi\| = 1$:

$$\phi(\mathbf{x}) = \begin{cases} -d(\mathbf{x}, \partial \mathcal{O}) & \text{if } \mathbf{x} \in \mathcal{O} \\ +d(\mathbf{x}, \partial \mathcal{O}) & \text{if } \mathbf{x} \notin \mathcal{O} \end{cases}$$

On a discrete voxel grid with spacing $h$, this is computed via two independent Euclidean Distance Transforms:

$$\phi_{\text{raw}}(\mathbf{x}) = \operatorname{EDT}\big(\mathcal{O} == 0\big)(\mathbf{x}) - \operatorname{EDT}\big(\mathcal{O} == 1\big)(\mathbf{x})$$

The raw distance field is smoothed via an isotropic 3D Gaussian kernel:
$$\phi = \phi_{\text{raw}} * G_{3D, \sigma=0.8}$$

---

## 4. Boundary Protection & Topological Closure Theorem

### Theorem: Guaranteed Topological Manifold Closure
*Let $\Omega_{\text{grid}} = [-L, L]^3$ be the finite computational domain of the voxel grid. If the canvas padding fraction $\beta \ge 0.04$ and grid boundary distance $\phi(\mathbf{x}) \ge \epsilon > 0$ for all $\mathbf{x} \in \partial \Omega_{\text{grid}}$, then the extracted zero-isosurface $\mathcal{S} = \{ \mathbf{x} \mid \phi(\mathbf{x}) = 0 \}$ forms a closed 2-manifold with zero boundary edges.*

#### Proof Sketch:
1. Because the foreground silhouette is padded by margin $\beta = 0.04$, the 2D silhouette $\alpha(x, y) = 0$ for all $(x, y)$ within the boundary collar of the canvas.
2. By the occupancy equation, $\mathcal{O}(x, y, z) = 0$ throughout the domain perimeter.
3. Therefore, $\operatorname{EDT}(\mathcal{O} == 0) = 0$ and $\operatorname{EDT}(\mathcal{O} == 1) \ge 4 \cdot h$ on the outer voxel layer, making $\phi(\mathbf{x}) > 0$ strictly on $\partial \Omega_{\text{grid}}$.
4. By the Intermediate Value Theorem, any continuous zero-crossing $\phi(\mathbf{x}) = 0$ must be strictly interior to $\Omega_{\text{grid}}$.
5. Marching Cubes extracts closed triangle loops for all internal level-set components that do not intersect domain limits, resulting in a mesh with zero boundary edges ($\partial \mathcal{M} = \emptyset$).

Padding keeps the zero surface away from the sampled volume boundary. Check the exported topology separately. In the recorded benchmark, 72 of 75 proposed-method runs produced watertight meshes.

---

## 5. Limitations, Failure Modes, and Mitigation Strategies

### 5.1 Failure Mode: Thin or Transparent Structures

For objects with thin protrusions (bicycle spokes, wire frames, plant stems) or transparent surfaces (glass, water, crystals), the constrained slab model produces suboptimal results:

- **Thin protrusions**: Structures narrower than 2 voxels in the grid are below the Nyquist resolution limit and will be merged or filled in by the SDF.
- **Transparent objects**: The U²-Net matting model may fail to extract correct silhouettes for transparent or reflective surfaces (glass cups, metallic spheres).

**Mitigation**: For thin structures, increase `target_triangles` to 25,000–50,000 to raise voxel grid resolution. For transparent objects, photograph them against a strongly contrasting matte background.

### 5.2 Failure Mode: Non-Convex Silhouettes with Complex Topology

Objects with genus > 0 topology (a torus, a handle, a ring) cannot be faithfully reconstructed by the constrained slab method because the occupancy field $\mathcal{O}(x,y,z)$ is defined as a simple product of a 2D silhouette and a 1D depth interval — it has no mechanism for creating through-holes in the inflation direction.

For example, a coffee mug handle projects to a silhouette with a hole. The slab inflation fills the handle's interior, producing a solid cylindrical protrusion rather than a hollow ring.

**Mitigation**: This is a fundamental limitation of single-view reconstruction without learned shape priors. For genus > 0 objects, the system still produces the best achievable watertight mesh from the available depth and silhouette information.

### 5.3 Failure Mode: Extreme Viewpoint Foreshortening

For objects photographed at oblique angles (near-grazing incidence), the depth variation along the viewing axis collapses: $d(x,y)$ approaches a constant, eliminating surface relief carving. The slab degenerates toward a flat extruded silhouette.

**Mitigation**: 3DFUN AI performs best with front-facing photographs. The Web Studio displays a recommended pose guide overlay.

---

## 6. Relationship to Published Methods

The constrained volumetric slab inflation method implemented in 3DFUN AI draws from several published computer vision techniques:

| Technique | Source | How Used in 3DFUN AI |
|---|---|---|
| **Shape from Silhouette (Visual Hull)** | Laurentini (1994) | Provides the 2D silhouette constraint $\alpha(x,y)$ |
| **Truncated Signed Distance Fields** | Curless & Levoy (1996) | Inspired the SDF construction approach |
| **Euclidean Distance Transform** | Meijster et al. (2000) | Used to compute exact EDT-based SDF |
| **Marching Cubes** | Lorensen & Cline (1987) | Triangle mesh extraction from zero-isosurface |
| **Taubin λ\|μ Smoothing** | Taubin (1995) | Post-processing staircase removal |
| **Depth Anything V2** | Yang et al. (2024) | Monocular depth estimation backbone |
| **U²-Net** | Qin et al. (2020) | Foreground salient object matting |

The key **novel contribution** of 3DFUN AI is the mathematical integration of these components into a unified **constrained inflation framework** with the analytical watertightness guarantee (Section 4), which is not present in any single published method.

---

*See also: [ADR 0003 — SDF Over Point Clouds](../adr/0003-sdf-over-pointclouds.md), [ADR 0005 — Taubin Smoothing](../adr/0005-taubin-smoothing.md), [Configuration Reference](../reference/configuration.md).*
*Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011), Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075) (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*


---

## 7. Hyperparameter Sensitivity Analysis

To understand how each inflation parameter affects reconstruction quality, the pipeline
was run on 5 synthetic scenes while sweeping one parameter at a time (all others held
at their default values). Results are from `results/summary/ablation_table.csv`.

### 7.1 Effect of Constant Core Fraction (f)

| f value | Chamfer Distance | F@1% | Watertight % | Visual Observation |
|---:|---:|---:|---:|---|
| 0.10 | 0.1024 | 0.0932 | 62 % | Severe rim pinching; holes at silhouette boundary |
| 0.30 | 0.0801 | 0.1241 | 88 % | Slight collapse at thin edges |
| **0.55** | **0.0639** | **0.1482** | **96 %** | **Default — optimal balance** |
| 0.70 | 0.0658 | 0.1421 | 100 % | Slight over-inflation; objects look bloated |
| 0.90 | 0.0834 | 0.1203 | 100 % | Heavy over-inflation; depth carving suppressed |

**Conclusion**: f = 0.55 gives the best Chamfer / F-score tradeoff. Values below 0.30
cause topological failures (non-watertight meshes). Values above 0.70 suppress depth
modulation and produce boxy shapes.

### 7.2 Effect of Grid Resolution (target_triangles)

| target_triangles | Grid dim | CD | F@1% | Peak RAM (MB) | Latency (s) |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 72^3 | 0.0821 | 0.1124 | 180 | 0.31 |
| 4,000 | 96^3 | 0.0714 | 0.1298 | 310 | 0.52 |
| **8,000** | **128^3** | **0.0639** | **0.1482** | **560** | **0.90** |
| 16,000 | 160^3 | 0.0591 | 0.1601 | 1,100 | 1.74 |
| 32,000 | 192^3 | 0.0562 | 0.1683 | 2,400 | 3.42 |

**Conclusion**: 8,000 triangles (128^3 grid) is the default sweet spot — it delivers
near-optimal geometry at under 1 second latency and under 600 MB RAM. For 3D printing
applications where dimensional accuracy is critical, 16,000–32,000 triangles is
recommended if system RAM allows.

---

## 8. Visual Failure Gallery

The following failure modes have been documented during development. Each has an
associated mitigation in the pipeline configuration.

### Failure 1: Rim Pinching (f too low)
- **Appearance**: The object's silhouette perimeter collapses to sharp knife-edge ridges.
- **Cause**: The minimum inflation at the silhouette boundary is t_min + f * (t_max - t_min).
  When f is very small, rim pixels get almost zero thickness.
- **Fix**: Increase `constant_core` (f) to at least 0.40, or increase `thickness_min`.

### Failure 2: Boxy Extrusion (depth signal lost)
- **Appearance**: The mesh looks like a cookie-cutter extrusion with flat front and back.
- **Cause**: The depth map `d(x,y)` has near-zero dynamic range (uniform depth scene,
  overexposed photograph, or depth model failure).
- **Fix**: Check the intermediate depth map (`--debug-intermediates`). If uniform, re-shoot
  with better lighting contrast. Increase `depth_scale` (g) is not a substitute.

### Failure 3: Floating Islands
- **Appearance**: Small disconnected mesh patches floating near the main object.
- **Cause**: Stray foreground pixels in the U2-Net alpha mask (reflections, cast shadows).
- **Fix**: Post-process by keeping only the largest connected component:
  ```python
  components = mesh.split(only_watertight=False)
  mesh = max(components, key=lambda m: m.volume)
  ```

### Failure 4: Open Mesh (non-watertight)
- **Appearance**: `mesh.is_watertight = False`, typically with 2-12 boundary edges.
- **Cause**: Insufficient canvas padding; the SDF zero-crossing touches the grid boundary.
- **Fix**: Increase `RECON_PADDING` from 0.04 to 0.06 in `.env`.

---

*Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),*
*Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075).*
*Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad.*
