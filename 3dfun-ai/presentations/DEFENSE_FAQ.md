# 3DFUN AI — Examiner Defense FAQ & Oral Examination Guide
**Course:** 3D Computer Vision Course Project (Autumn 2026)  
**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Supervisor:** Dr. Shirshendu Layek, Assistant Professor, Dept. of Data Science & AI (DSAI)  

### Team Roster & Oral Defense Allocation (12-Min Total · 3 Questions per Member):
- **J Ganesh (23BDS024):** Slides 1–3 · Lead Respondent for **Q1, Q6, Q12** (Ill-Posedness, Bas-Relief Ambiguity, Diffusion Generative Priors)
- **M Jagadeshwar (23BDS033):** Slides 4–6 · Lead Respondent for **Q2, Q3, Q7** (TSDF Failure, Taubin Proof, Marching Cubes Topology)
- **B Varshith (23BDS011):** Slides 7–9 · Lead Respondent for **Q4, Q8, Q11** (INT8 Quantization, Watertightness Proof, Synthetic Benchmark Integrity)
- **Pranav Kashyap (23BCS098):** Slides 10–12 · Lead Respondent for **Q5, Q9, Q10** (NeRF/3DGS Comparison, UV Stretching & Texture, FastAPI Memory Architecture)
- **C Sai Aravind (23BDS075):** Slides 13–15 · Lead Respondent for **Q13, Q14, Q15** (Morphological Masking, TripoSR Neural Integration, Reproducibility & Gates)

---

### Q1: [Lead: J Ganesh (23BDS024)] Single-image 3D reconstruction is fundamentally ill-posed. How does 3DFUN AI mathematically resolve the unobserved backside without hallucinating nonexistent features?
**Defense:**
A single perspective or orthographic photograph captures rays terminating at the first visible surface, leaving the back surface, interior topology, and occluded concavities unobserved. Generative diffusion models (e.g., Zero-1-to-3, SV3D) resolve this by sampling from a learned prior, which frequently hallucinates high-frequency features (e.g., asymmetrical textures or phantom limbs) that do not exist in reality.

In 3DFUN AI, we formulate reconstruction as a constrained geometric boundary problem rather than an open-ended generative task. We impose two deterministic, falsifiable priors:
1. **Silhouette Prism Constraint:** The reconstructed 3D solid $S \subset \mathbb{R}^3$ is strictly bounded by the generalized cylinder formed by the back-projected 2D silhouette mask $M$:
   $$\Pi_{xy}(S) \subseteq M$$
   No reconstructed geometry is permitted to leak outside the visual hull of the single view.
2. **Monotonic Distance-to-Boundary Inflation:** The local thickness $t(x, y)$ of the object is governed by the smooth Euclidean Distance Transform (EDT) of the silhouette boundary combined with normalized monocular relative depth $d(x, y) \in [0, 1]$:
   $$t(x,y) = t_{\text{min}} + (t_{\text{max}} - t_{\text{min}}) \cdot \left(f + g \cdot d(x,y)^\gamma\right)$$
   where $t_{\text{min}} > 0$ guarantees a closed, non-degenerate boundary rim, and the parameter $f$ preserves a structural slab core while $g \cdot d^\gamma$ modulates the visible front-facing relief. By construction, the backside reflects the object's global symmetry plane rather than hallucinated texture, providing an honest, geometrically stable hypothesis.

---

### Q2: [Lead: M Jagadeshwar (23BDS033)] Why did classical Truncated Signed Distance Field (TSDF) fusion fail so dramatically on single uncalibrated views in your benchmark?
**Defense:**
TSDF fusion (Curless & Levoy, 1996) was fundamentally designed for multi-view depth camera streams (such as KinectFusion) with known camera poses and dense, overlapping observation angles. 

When applied to a single monocular depth map:
1. **Zero Backside Constraints:** Standard ray casting only updates voxels along the optical ray in a narrow truncation band $[-\delta, +\delta]$ around the predicted depth surface. Voxels deeper than $-\delta$ remain unobserved (carved away or unassigned).
2. **Boundary Discontinuity:** Near the silhouette silhouette boundary $\partial M$, depth values drop precipitously to the background. In single-view TSDF, this creates an open, zero-thickness rim with non-manifold edges and large open holes upon Marching Cubes extraction.
3. **Severe Bas-Relief Distortion:** Without multi-view constraints to anchor the back surface, TSDF collapses into a thin curved sheet rather than a volumetric solid, yielding near-zero Volumetric IoU ($0.091$) and abysmal F-score ($0.228$) in our empirical benchmark table.

---

### Q3: [Lead: M Jagadeshwar (23BDS033)] Why did you choose Taubin smoothing over standard Laplacian mesh smoothing, and what is the mathematical proof of volume preservation?
**Defense:**
Standard Laplacian smoothing iteratively updates vertex coordinates $V$ using the discrete Laplace-Beltrami operator $L$:
$$V^{(k+1)} = V^{(k)} + \lambda L V^{(k)}$$
Because $\lambda > 0$, the transfer function $f(k) = (1 - \lambda k)$ acts as a simple low-pass filter where all non-zero spatial frequencies have eigenvalues strictly less than 1. Over repeated iterations, this monotonically shrinks the mesh toward its centroid, causing severe volume collapse, rounding of sharp corners, and inward dimensional loss.

Gabriel Taubin (SIGGRAPH 1995) solved this by introducing two alternating steps with opposite sign factors $\lambda > 0$ and $\mu < -\lambda < 0$:
$$V^{(k+1/2)} = (I + \lambda K) V^{(k)}$$
$$V^{(k+1)} = (I + \mu K) V^{(k+1/2)}$$
The composite transfer function is:
$$f(k) = (1 - \lambda k)(1 - \mu k) = 1 - (\lambda + \mu)k + \lambda \mu k^2$$
By setting $\mu < -\lambda < 0$, the linear coefficient $-(\lambda + \mu)$ is strictly positive, creating a passband near zero frequency (low frequencies corresponding to macro-shape and volume are unattenuated or slightly boosted) while high frequencies ($k > k_{\text{cutoff}}$) are strongly attenuated ($|f(k)| < 1$). 

In 3DFUN AI, using $\lambda = 0.5$, $\mu = -0.53$, and 12 iterations eliminates the Marching Cubes staircase voxel artifacts without measurable loss in object bounding volume or Chamfer distance (which shifted by $<0.0008$ in our ablation).

---

### Q4: [Lead: B Varshith (23BDS011)] What is the impact of INT8 quantization on Depth Anything V2 Small, and how does it affect geometric precision?
**Defense:**
Our monocular depth model is `depth_anything_v2_small_quantized.onnx` (27.2 MB), quantized from FP32 to INT8 using symmetric per-tensor static dynamic activation and per-channel weight quantization via ONNX Runtime.

1. **Memory & Latency Benefit:** The model footprint drops by 74% (from ~105 MB FP32 to 27.2 MB INT8), and inference latency drops from 1.42 s to 0.38 s on two standard x86 CPU cores.
2. **Geometric Degradation Analysis:** Because monocular depth networks predict relative affine-invariant depth rather than metric distance, the depth field is affine-normalized to $[0, 1]$ before volumetric meshing:
   $$\hat{d} = \frac{d - d_{2\%}}{d_{98\%} - d_{2\%}}$$
   The dynamic range of 256 discrete INT8 quantization bins corresponds to a depth discretization step of $\Delta d \approx 1/256 \approx 0.0039$. When mapped to the voxel resolution of $64^3$ or $128^3$, this quantization step is smaller than the spatial grid cell size ($\Delta x = 1/64 = 0.0156$). Consequently, the spatial discretization error of the meshing grid dominates over the INT8 quantization noise, resulting in zero statistically significant degradation in Chamfer Distance or F-Score.

---

### Q5: [Lead: Pranav Kashyap (23BCS098)] How does 3DFUN AI compare with 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF)?
**Defense:**
1. **Input Requirements:** NeRF and 3DGS are multi-view reconstruction algorithms requiring dozens to hundreds of calibrated camera views with accurate SfM (COLMAP) poses. They cannot operate on a single arbitrary uncalibrated image without heavy multi-view generative priors (e.g., PixelNeRF, DreamGaussian).
2. **Compute & Hardware:** Training a NeRF or optimizing 3D Gaussians takes 2 to 45 minutes on an NVIDIA RTX 3090/4090 GPU (consuming 8–16 GB of VRAM). 3DFUN AI executes end-to-end in $<25$ seconds on standard laptop CPU hardware without requiring a dedicated GPU.
3. **Downstream Usability:** 3D Gaussians produce millions of colored ellipsoids that look photorealistic from novel viewpoints, but they do NOT have a topological surface mesh; converting Gaussians to clean, watertight STL/OBJ meshes for 3D printing or CAD requires complex TSDF Poisson extraction. 3DFUN AI natively outputs watertight, manifold meshes directly exportable to Blender, Unity, and 3D printers.

---

### Q6: [Lead: J Ganesh (23BDS024)] How do you address the bas-relief ambiguity inherent in uncalibrated monocular cameras?
**Defense:**
The bas-relief ambiguity (Belhumeur et al., 1999) proves that for an orthographic or weak perspective camera, an observed surface under unknown lighting and depth is subject to a three-parameter generalized bas-relief (GBR) transformation:
$$z'(x,y) = \lambda z(x,y) + \mu x + \nu y$$
where the depth can be flattened or sheared without altering surface normals under transformed albedo.

In 3DFUN AI:
1. We enforce an orthographic silhouette boundary constraint: the lateral coordinates $(x,y)$ are pinned by the segmentation mask.
2. We calibrate the depth scaling factor $\lambda$ empirically using the bounding box aspect ratio:
   $$t_{\text{max}} = \kappa \cdot \min(W_{\text{bbox}}, H_{\text{bbox}})$$
   where our hyperparameter thickness sweep identified $\kappa = 0.70$ as the global optimal operating point minimizing 3D Chamfer Distance against analytic ground-truth meshes. This prevents the object from collapsing into a flat disk or extruding into an unnatural cylinder.

---

### Q7: [Lead: M Jagadeshwar (23BDS033)] Why Marching Cubes instead of Screened Poisson Surface Reconstruction (SPSR)?
**Defense:**
Screened Poisson Surface Reconstruction (Kazhdan & Hoppe, 2013) solves a Poisson equation $\nabla^2 \chi = \nabla \cdot \vec{V}$ requiring an oriented point cloud with consistent outward-pointing normal vectors $\vec{n}_i$.

For single-view reconstruction:
1. Normal vectors derived from back-projected single-view depth maps are only oriented on the front hemisphere; along grazing silhouette edges, normals become parallel to the camera ray or undefined.
2. In regions with missing backside points, Screened Poisson tends to create ballooning spherical artifacts or non-manifold edge clusters to close the domain.
3. In contrast, our pipeline directly constructs a dense, continuous Signed Distance Field (SDF) over a regular voxel lattice $\Phi(x, y, z)$. Marching Cubes (Lorensen & Cline, 1987) on a discrete voxel grid guarantees a topologically closed 2-manifold surface via the lookup table of 256 edge-crossing configurations, ensuring watertightness without post-hoc geometric heuristics.

---

### Q8: [Lead: B Varshith (23BDS011)] How is watertightness mathematically guaranteed in your meshing stage?
**Defense:**
A closed 2-manifold (watertight mesh) requires that:
1. Every edge is shared by exactly two faces.
2. Every vertex has a single homeomorphically disk-like neighborhood of connected triangles (Euler characteristic $\chi = V - E + F = 2(1 - g)$).
3. The mesh contains zero boundary edges ($\partial M = \emptyset$).

In 3DFUN AI, the distance function $\Phi(x,y,z)$ is defined such that the boundary voxels of the grid lattice $\partial \Omega$ are strictly positive (exterior):
$$\Phi(x,y,z)\Big|_{\partial \Omega} > 0$$
Because the minimum thickness parameter $t_{\text{min}} > 0$ ensures the internal volume $S = \{(x,y,z) \mid \Phi(x,y,z) \le 0\}$ is strictly contained within the interior of $\Omega$, the zero-level set $S_0 = \{(x,y,z) \mid \Phi(x,y,z) = 0\}$ never intersects the domain boundary $\partial \Omega$. By the topological properties of Marching Cubes on regular Cartesian grids with compact isosurfaces, every cell edge with a sign change produces interior triangular facets, guaranteeing zero boundary edges and 100% watertightness.

---

### Q9: [Lead: Pranav Kashyap (23BCS098)] What are the main sources of texture distortion, and why does planar view-projection UV unwrapping stretch at grazing angles?
**Defense:**
Our primary texture mapping uses planar orthographic back-projection from the source camera viewpoint.
1. For front-facing surface patches where $\langle \vec{n}_v, \vec{v}_{\text{cam}} \rangle \approx 1$, UV sampling is distortion-free and preserves the native camera resolution.
2. At grazing angles near the silhouette perimeter where $\langle \vec{n}_v, \vec{v}_{\text{cam}} \rangle \to 0$, the surface normal is nearly perpendicular to the camera ray. Projecting a finite 2D pixel area onto a nearly parallel 3D face causes spatial stretching proportional to $1 / \cos(\theta)$, leading to visible texture smearing along side walls.
3. We mitigate this in the WebGL viewer by blending a procedural Ambient Occlusion (AO) and roughness pass that darkens grazing edges, and we openly document this in Section VI of our paper as an inherent trade-off of single-view planar unwrapping compared to expensive conformal chart parameterization.

---

### Q10: [Lead: Pranav Kashyap (23BCS098)] How does your FastAPI backend prevent memory exhaustion during concurrent reconstruction requests?
**Defense:**
1. **Asynchronous Non-Blocking Event Loop:** Heavy CPU reconstruction tasks are offloaded to a dedicated `concurrent.futures.ProcessPoolExecutor` with max workers bounded by available CPU cores, keeping the FastAPI async loop responsive.
2. **Strict Upload Budget:** Incoming images are validated with strict content-length headers and image dimension limits (max 20 MB, downsampled to 512×512 before inference).
3. **Credit Reservation & Idempotency:** Before dispatching an inference job, credits are reserved atomically in the ledger. If the worker encounters an OOM or cancellation, credits are refunded via a transactional rollback.
4. **Calibrated Grid Memory:** The voxel grid is statically sized at $64^3$ to $128^3$ floats ($128^3 \times 4 \text{ bytes} \approx 8.4 \text{ MB}$), ensuring the volumetric inflation phase never exceeds 20 MB per job.

---

### Q11: [Lead: B Varshith (23BDS011)] What is the justification for using an analytic synthetic benchmark rather than real-world 3D scans like ShapeNet or Objaverse?
**Defense:**
1. **Zero Alignment Noise:** Real-world datasets like ShapeNet contain arbitrary coordinate axes, unit inconsistencies, internal non-manifold faces, and open boundaries that inject artificial Chamfer error during ICP alignment.
2. **Ground Truth Purity:** Our 13 synthetic procedural models (Cylinder, Ellipsoid, Torus, Prism, Cone, etc.) possess mathematically exact, closed 3D bounding geometry. We can calculate closed-form volume, surface area, and bidirectional nearest-neighbor Euclidean distances with machine precision.
3. **Reproducibility Without Multi-Gigabyte Downloads:** The entire benchmark suite is self-contained in the repository (2.3 MB) and regenerates deterministically using fixed random seeds, allowing any evaluator to clone and verify our numbers in under 2 minutes.

---

### Q12: [Lead: J Ganesh (23BDS024)] Why not use multi-view diffusion models like Zero-1-to-3 or SV3D as your primary pipeline?
**Defense:**
While multi-view diffusion represents the state of the art in novel view synthesis:
1. **Heavy Compute Barrier:** Zero-1-to-3 requires an NVIDIA GPU with at least 12–16 GB of VRAM. A single object takes 45 to 90 seconds of high-precision denoising steps.
2. **View Inconsistency & Hallucination:** 2D diffusion models frequently suffer from the "Janus problem" (e.g., generating multiple faces on the back of an animal or shoe) and multi-view drift, producing deformed meshes when fused via NeuS or Marching Tetrahedra.
3. **Deployment Feasibility:** Our course project goal was to build a deployable, lightweight web application that runs on commodity hardware (laptops, CPU VMs) with guaranteed sub-30s latency, deterministic outputs, and verifiable watertightness.

---

### Q13: [Lead: C Sai Aravind (23BDS075)] What is the purpose of morphological erosion on the predicted U²-Net alpha mask?
**Defense:**
Neural matting networks frequently produce a semi-transparent, soft alpha boundary with feathering over 2 to 4 pixels. 
If an uncleaned mask is back-projected:
1. Background pixels bleeding into the fringe get inflated into sharp, spiky "halo" geometry protruding from the object's perimeter.
2. We apply a $3 \times 3$ elliptical structuring element erosion followed by a hard threshold at $\alpha = 0.5$. This clips the outer halo pixels, guaranteeing that every voxel inside the silhouette belongs strictly to the foreground object, preventing boundary tearing and texture contamination.

---

### Q14: [Lead: C Sai Aravind (23BDS075)] How does the TripoSR integration compare with the course method in your repository?
**Defense:**
We implemented TripoSR (Tochilkin et al., 2024) as an optional neural engine in `engine/recon/triposr.py`:
1. TripoSR uses a vision transformer (ViT) encoder that maps the image directly into an orthogonal triplane representation, decoded by a lightweight NeRF/SDF MLP and extracted via Marching Cubes.
2. **Trade-offs:** TripoSR reconstructs complex concave structures (like chair legs or hollow mugs) that our silhouette inflation cannot capture. However, its ONNX weights require 436 MB of storage, peak RAM exceeds 1.8 GB, and vertex colors lack the crisp fidelity of our high-resolution 1024×1024 PBR texture atlas.
3. Both engines coexist under our unified `ReconstructionProvider` interface, allowing the user to switch between deterministic speed and learned neural geometry in the Web Studio.

---

### Q15: [Lead: C Sai Aravind (23BDS075)] How do you enforce reproducibility across the entire project and prevent metric fabrication?
**Defense:**
In adherence to strict academic honesty:
1. **Zero Manual Numbers:** No quantitative figure in `paper/main.tex`, `docs/report.md`, or `presentations/3DFUN-AI-Project.pptx` is typed by hand. Every single number is read dynamically from `results/summary/*.csv` using automated injection scripts (`scripts/generate_paper.py`, `scripts/generate_slides.py`, `scripts/generate_report.py`).
2. **Automated Submission Gate:** Our automated verification script (`scripts/verify_submission.py`) executes 30 independent verification gates:
   - Verifies all summary CSV means match the raw run logs within $0.05$ s.
   - Asserts all 41 test cases pass cleanly.
   - Verifies that all numbers printed in the PDF, paper, and slide deck trace back to `results/`.
   - Confirms zero undefined citations, zero overfull hboxes, and zero canvas overflows.
   Any discrepancy instantly fails the build with a non-zero exit code.
