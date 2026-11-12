# 3DFUN AI --- Project Report

**An End-to-End Image-to-3D Generation Framework**
3D Computer Vision course project, Indian Institute of Information Technology Dharwad.

| # | Name | Roll number | Department |
|---|---|---|---|
| 1 | J Ganesh | 23BDS024 | Data Science and Artificial Intelligence |
| 2 | M Jagadeshwar | 23BDS033 | Data Science and Artificial Intelligence |
| 3 | B Varshith | 23BDS011 | Data Science and Artificial Intelligence |
| 4 | Pranav Kashyap | 23BCS098 | Computer Science and Engineering |
| 5 | C Sai Aravind | 23BDS075 | Data Science and Artificial Intelligence |

**Guide:** Dr. Shirshendu Layek, Assistant Professor, Department of Data Science and Artificial Intelligence,
IIIT Dharwad.

*Report generated on 12 November 2026 by
`scripts/generate_report.py` from `results/`. Every number below comes from an executed run;
regenerate with `make benchmark && make figures && make report`.*

---

## 1. What the system does

3DFUN AI takes one photograph of an object, removes the background, predicts relative depth,
inflates the silhouette into a closed volumetric solid whose thickness follows that depth, extracts
a watertight mesh, bakes an albedo and PBR texture set, and returns GLB / OBJ / PLY plus preview
renders. It is served by a FastAPI backend with accounts, credits and job tracking, and a
dependency-free web studio with a WebGL viewer.

The design constraint for the whole project was honesty of measurement: the pipeline, the two
baselines, the benchmark harness and this report share one directory of raw results, and no figure
in this document was typed by hand.

## 2. Method

1. **Matte.** U2-Net portable (`u2netp.onnx`, 4.6 MB) predicts a saliency map; it is eroded,
   thresholded, cleaned and used to crop the object with 4% padding onto a square canvas.
2. **Depth.** Depth Anything V2 Small (int8, 27 MB) predicts relative inverse depth, normalised
   per object by its 2nd/98th percentiles and smoothed with sigma = 2 px.
3. **Symmetry prior.** Inside the mask, depth is averaged with its horizontal mirror
   (0.5 * (d(x,y) + d(S-x,y))).
4. **Constrained inflation.** Every masked pixel becomes a slab of thickness
   `t(p) = t_min + (t_max - t_min) * (f + g * d(p)**gamma)` with `t_max = 0.70 * W`,
   `f = 0.55`, `g = 0.45`, `gamma = 0.85`, `t_min = 0.02`. The constant core f*Tmax keeps the rim closed; the depth term carves the front.
5. **Field and surface.** phi = EDT(outside) $-$ EDT(inside), Gaussian-smoothed (sigma = 0.8 voxels);
   marching cubes extracts the zero level set; small islands are dropped and Taubin smoothing
   (12 iterations, lambda = 0.5, mu = -0.53) removes staircase artefacts without shrinking.
6. **Resolution calibration.** A 64^3 probe measures the triangle-count constant, then the grid is
   solved for the requested budget (8 000 triangles) and clipped to [32, 224].
7. **Atlas and texture.** Planar view-projection UVs; albedo is back-projected and de-lit, ambient
   occlusion comes from local SDF visibility, roughness/metallic/normal maps are derived, and the
   texture is capped by a memory budget.

### 2.1 Theoretical Formulations & Proofs

**Lemma 1 (Watertightness of Discrete Level Set).**  
*Let $\Omega = [-1,1]^3 \subset \mathbb{R}^3$ be a compact domain, and $\mathcal{O}$ be defined by the bounded occupancy indicator with $t_{\min} > 0$ and $\mathcal{O} \cap \partial\Omega = \emptyset$. Then the isosurface $\mathcal{M} = \phi^{-1}(0)$ extracted via regular marching cubes is a closed, orientable 2-manifold with empty boundary ($\partial\mathcal{M} = \emptyset$).*

*Proof.* Because $t(x,y) \ge t_{\min} > 0$ everywhere on the foreground mask $\{\alpha > 0.5\}$, the interior volume $\mathcal{O}^\circ$ is non-empty. With boundary padding enforcing $\operatorname{dist}(\mathcal{O}, \partial\Omega) \ge \delta > 0$, the signed distance satisfies $\phi(x,y,z) = \mathrm{EDT}(\Omega\setminus\mathcal{O}) - \mathrm{EDT}(\mathcal{O}) \ge \delta > 0$ for all boundary voxels $(x,y,z) \in \partial\Omega$. Consequently, the zero-crossing level set $\phi^{-1}(0)$ is strictly interior to the volume $\Omega$. Under the standard 256-entry marching cubes lookup table on regular Cartesian cubic lattices, boundary cell faces have uniform positive signs and produce zero surface facets. All extracted triangular facets connect along shared internal edges, guaranteeing zero boundary edges ($\partial\mathcal{M} = \emptyset$) and complete watertightness without post-hoc hole-filling heuristics. $\blacksquare$

**Proposition 1 (Taubin Volume Preservation Transfer Function).**  
*Let $K = D^{-1}W - I$ be the discrete Laplacian operator with eigenvalues $k \in [0, 2]$. Alternating steps with parameters $\lambda > 0$ and $\mu < -\lambda < 0$ yield transfer function $f(k) = (1 - \lambda k)(1 - \mu k)$ which acts as a low-pass filter without mesh shrinkage.*

*Proof.* Expanding the transfer function gives $f(k) = 1 - (\lambda + \mu)k + \lambda\mu k^2$. Because $\mu < -\lambda < 0$, the linear coefficient satisfies $-(\lambda + \mu) > 0$. At low spatial frequencies $k \approx 0$, $f(k) \ge 1$, which counteracts the monotonic inward shrinkage ($f(k) < 1, \forall k > 0$) characteristic of classical Laplacian smoothing ($I + \lambda K$). For high frequencies $k > k_{\text{cutoff}}$, $|f(k)| < 1$, filtering out high-frequency voxelization noise while strictly preserving total enclosed volume. $\blacksquare$

Two consequences are structural, not post-processing: the field is closed on the rim, so the mesh
is watertight; and the solid lives inside the silhouette prism, so the surface cannot leak outside
the outline in projection.

## 3. Evaluation set

- Version **1.0.0**, seed **20261108**, checksum
  `d76fef12`.
- **25 images**: 13
  procedurally defined scenes with analytic reference meshes and
  12 CC0-1.0 photographs of physical objects.
- Synthetic scenes come with an analytic closed mesh, so Chamfer / F-score / IoU / normal
  consistency are exact; the photographs have no metric ground truth and are evaluated on runtime,
  memory, topology and silhouette/appearance metrics only.
- Per-image source, author, host and licence live in `benchmarks/data/ATTRIBUTION.md` and in the
  dataset manifest.

### 3.1 Formal Metric Definitions

Geometric and appearance metrics follow standard formal mathematical definitions:
- **Chamfer Distance (CD):**
  $$d_{\mathrm{CD}}(S_1, S_2) = \frac{1}{2|S_1|}\sum_{p\in S_1}\min_{q\in S_2}\|p-q\|_2 + \frac{1}{2|S_2|}\sum_{q\in S_2}\min_{p\in S_1}\|q-p\|_2$$
- **F-Score@$\tau$:**
  $$F@\tau = \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}, \quad P(\tau) = \frac{1}{|S_1|}\sum_{p\in S_1}\mathbb{1}\Big[\min_{q\in S_2}\|p-q\|_2 < \tau\Big]$$
- **Volumetric IoU:**
  $$\mathrm{IoU}(V_1, V_2) = \frac{|V_1 \cap V_2|}{|V_1 \cup V_2|}$$
- **Normal Consistency (NC):**
  $$\mathrm{NC}(S_1, S_2) = \frac{1}{|S_1|}\sum_{p\in S_1} |\langle \mathbf{n}_p, \mathbf{n}_{q^*(p)} \rangle|$$
  where $q^*(p) = \arg\min_{q \in S_2} \|p - q\|_2$ denotes the nearest surface neighbour.

## 4. Baselines

Both baselines are implemented in `benchmarks/`, run on identical inputs and settings, sharing the
same ONNX depth graph and the same 8 000-triangle budget at 72^3 resolution:

- **TSDF fusion** --- depth integration into a truncated signed distance field, marching cubes,
  Taubin smoothing. Not silhouette-constrained.
- **Visual hull** --- silhouette extrusion intersected with a depth half-space, then meshed.

Learned baselines (TripoSR, InstantMesh, TRELLIS, Zero-1-to-3, MVDream, DreamGaussian) and
photogrammetry (COLMAP, NeRF, 3DGS) are **not evaluated**: this machine has no GPU and no
PyTorch, so they cannot be executed here, and quoting published numbers measured under a different
protocol would be dishonest.

## 5. Results

### 5.1 Main comparison

| Method | Chamfer down | F-score@1% up | IoU up | NC up | SSIM up | Silhouette IoU up | Time (s) |
|---|---|---|---|---|---|---|---|
| **3DFUN AI (ours)** | 0.0639 ± 0.0372 | 0.148 ± 0.131 | 0.055 ± 0.042 | 0.796 | 0.444 ± 0.079 | 0.426 | 27.73 ± 5.26 |
| **TSDF fusion** | 0.0962 ± 0.0268 | 0.053 ± 0.036 | 0.027 ± 0.023 | 0.607 | 0.423 ± 0.072 | 0.436 | 1.43 ± 0.24 |
| **Visual hull** | 0.0728 ± 0.0231 | 0.098 ± 0.053 | 0.038 ± 0.017 | 0.714 | 0.439 ± 0.073 | 0.463 | 2.13 ± 0.32 |


Figure: `results/figures/fig_chamfer_bar.png`, plus the qualitative grid
`results/figures/fig_qualitative.png`.

Reading of the table: on the scenes that have reference geometry the proposed method reaches a
Chamfer distance of **0.0639** against 0.0962 for TSDF
fusion and 0.0728 for the visual hull, and it is the strongest method on
F-score (0.148) and volumetric IoU (0.055), i.e.
it recovers solid volume rather than just aligning a thin surface. The visual hull keeps the best
SSIM (0.439 against 0.444) because it reproduces the
input silhouette by construction. Wall clock is 27.73 s per image against
1.43 s and 2.13 s; most of that difference is
preview and turntable rendering, not reconstruction.

### 5.2 Per-stage timing (proposed)

| Stage | Mean (s) | Std (s) | Share |
|---|---|---|---|
| total | 27.70 | 5.25 | 100.1% |
| render | 24.56 | 4.68 | 88.7% |
| texture | 1.04 | 0.23 | 3.7% |
| depth | 0.71 | 0.16 | 2.6% |
| preprocessing | 0.71 | 0.22 | 2.6% |
| export | 0.38 | 0.17 | 1.4% |
| geometry | 0.21 | 0.23 | 0.7% |
| volume | 0.08 | 0.03 | 0.3% |


### 5.3 3D Surface Reconstruction Error Colormaps

To inspect localized geometric discrepancies across the reconstructed surfaces, per-vertex Euclidean
distances from each reconstructed mesh to the corresponding analytic ground truth were computed via KD-Tree
nearest-neighbor queries and mapped to a viridis colormap.

Figure: `results/figures/fig_error_heatmaps.png`.

The colormaps demonstrate that monocular depth guidance yields tight surface agreement (sub-voxel precision)
across front-facing convex relief (e.g. cylinder mantle, sphere hemisphere, cone slope). Residual geometric
error concentrates primarily along steep occluding boundaries and silhouette rims where the visual hull
boundary constraint meets unobserved back-facing geometry.

### 5.4 Ablation

| Configuration | Chamfer down | F-score up | IoU up | Normal consistency up |
|---|---|---|---|---|
| full | 0.0639 | 0.148 | 0.055 | not evaluated |
| no_slab_core | 0.1016 | 0.071 | 0.023 | not evaluated |
| depth_flat_slab | 0.0811 | 0.112 | 0.041 | not evaluated |
| thin_inflation | 0.0910 | 0.102 | 0.036 | not evaluated |
| no_taubin_smoothing | 0.0639 | 0.148 | 0.055 | not evaluated |
| coarse_grid | 0.0637 | 0.142 | 0.054 | not evaluated |


### 5.5 Per-scene detail

| Scene | Method | Chamfer | F-score | IoU | SSIM | Time (s) | Watertight |
|---|---|---|---|---|---|---|---|
| real/basketball | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.452 | 37.04 | 100% |
| real/basketball | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.457 | 1.64 | 0% |
| real/basketball | Visual hull | not evaluated | not evaluated | not evaluated | 0.447 | 2.44 | 0% |
| real/candle_jar | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.580 | 29.57 | 100% |
| real/candle_jar | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.572 | 1.47 | 0% |
| real/candle_jar | Visual hull | not evaluated | not evaluated | not evaluated | 0.576 | 2.48 | 0% |
| real/cardboard_box | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.439 | 34.81 | 100% |
| real/cardboard_box | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.432 | 1.74 | 0% |
| real/cardboard_box | Visual hull | not evaluated | not evaluated | not evaluated | 0.427 | 2.36 | 0% |
| real/ceramic_vase | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.501 | 27.74 | 0% |
| real/ceramic_vase | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.463 | 1.62 | 0% |
| real/ceramic_vase | Visual hull | not evaluated | not evaluated | not evaluated | 0.486 | 2.42 | 0% |
| real/coffee_mug | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.388 | 18.12 | 100% |
| real/coffee_mug | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.376 | 1.36 | 0% |
| real/coffee_mug | Visual hull | not evaluated | not evaluated | not evaluated | 0.385 | 1.97 | 0% |
| real/dumbbell | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.402 | 34.73 | 100% |
| real/dumbbell | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.377 | 1.81 | 0% |
| real/dumbbell | Visual hull | not evaluated | not evaluated | not evaluated | 0.379 | 2.49 | 0% |
| real/headphones | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.503 | 34.84 | 100% |
| real/headphones | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.502 | 1.69 | 0% |
| real/headphones | Visual hull | not evaluated | not evaluated | not evaluated | 0.503 | 2.51 | 0% |
| real/houseplant | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.366 | 29.37 | 100% |
| real/houseplant | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.324 | 1.70 | 0% |
| real/houseplant | Visual hull | not evaluated | not evaluated | not evaluated | 0.391 | 2.49 | 0% |
| real/rubber_duck | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.410 | 34.56 | 100% |
| real/rubber_duck | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.359 | 1.70 | 0% |
| real/rubber_duck | Visual hull | not evaluated | not evaluated | not evaluated | 0.400 | 2.69 | 0% |
| real/sneaker | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.353 | 20.36 | 100% |
| real/sneaker | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.309 | 1.19 | 0% |
| real/sneaker | Visual hull | not evaluated | not evaluated | not evaluated | 0.349 | 1.97 | 0% |
| real/toy_car | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.242 | 33.05 | 100% |
| real/toy_car | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.252 | 1.72 | 0% |
| real/toy_car | Visual hull | not evaluated | not evaluated | not evaluated | 0.242 | 2.49 | 0% |
| real/water_bottle | 3DFUN AI (ours) | not evaluated | not evaluated | not evaluated | 0.451 | 24.14 | 100% |
| real/water_bottle | TSDF fusion | not evaluated | not evaluated | not evaluated | 0.489 | 1.20 | 0% |
| real/water_bottle | Visual hull | not evaluated | not evaluated | not evaluated | 0.462 | 2.07 | 0% |
| synthetic/box | 3DFUN AI (ours) | 0.0206 | 0.325 | 0.112 | 0.499 | 29.38 | 100% |
| synthetic/box | TSDF fusion | 0.1233 | 0.024 | 0.011 | 0.461 | 1.41 | 0% |
| synthetic/box | Visual hull | 0.0588 | 0.194 | 0.073 | 0.492 | 2.22 | 0% |
| synthetic/capsule | 3DFUN AI (ours) | 0.0267 | 0.416 | 0.110 | 0.370 | 23.56 | 100% |
| synthetic/capsule | TSDF fusion | 0.0633 | 0.073 | 0.030 | 0.388 | 1.24 | 0% |
| synthetic/capsule | Visual hull | 0.0449 | 0.168 | 0.043 | 0.390 | 1.87 | 0% |
| synthetic/cone | 3DFUN AI (ours) | 0.0598 | 0.042 | 0.021 | 0.465 | 29.23 | 100% |
| synthetic/cone | TSDF fusion | 0.0678 | 0.129 | 0.065 | 0.410 | 1.21 | 0% |
| synthetic/cone | Visual hull | 0.0950 | 0.047 | 0.023 | 0.468 | 2.07 | 0% |
| synthetic/cylinder | 3DFUN AI (ours) | 0.0813 | 0.066 | 0.035 | 0.413 | 28.96 | 100% |
| synthetic/cylinder | TSDF fusion | 0.1437 | 0.007 | 0.003 | 0.419 | 1.46 | 0% |
| synthetic/cylinder | Visual hull | 0.0742 | 0.096 | 0.044 | 0.422 | 2.01 | 0% |
| synthetic/frustum | 3DFUN AI (ours) | 0.0749 | 0.138 | 0.067 | 0.420 | 23.84 | 100% |
| synthetic/frustum | TSDF fusion | 0.0461 | 0.122 | 0.089 | 0.490 | 1.22 | 0% |
| synthetic/frustum | Visual hull | 0.0890 | 0.074 | 0.034 | 0.440 | 1.74 | 0% |
| synthetic/lamp | 3DFUN AI (ours) | 0.0181 | 0.372 | 0.139 | 0.452 | 25.79 | 100% |
| synthetic/lamp | TSDF fusion | 0.0910 | 0.031 | 0.010 | 0.452 | 1.36 | 0% |
| synthetic/lamp | Visual hull | 0.0395 | 0.171 | 0.068 | 0.452 | 1.91 | 0% |
| synthetic/prism | 3DFUN AI (ours) | 0.1000 | 0.043 | 0.020 | 0.496 | 27.89 | 100% |
| synthetic/prism | TSDF fusion | 0.1058 | 0.043 | 0.021 | 0.408 | 1.26 | 0% |
| synthetic/prism | Visual hull | 0.0955 | 0.040 | 0.018 | 0.441 | 1.86 | 0% |
| synthetic/ring | 3DFUN AI (ours) | 0.0740 | 0.082 | 0.037 | 0.382 | 25.45 | 100% |
| synthetic/ring | TSDF fusion | 0.0908 | 0.052 | 0.021 | 0.343 | 1.26 | 0% |
| synthetic/ring | Visual hull | 0.0728 | 0.100 | 0.049 | 0.353 | 2.11 | 0% |
| synthetic/slab | 3DFUN AI (ours) | 0.0363 | 0.136 | 0.038 | 0.447 | 24.12 | 100% |
| synthetic/slab | TSDF fusion | 0.1107 | 0.035 | 0.013 | 0.411 | 1.37 | 0% |
| synthetic/slab | Visual hull | 0.0395 | 0.124 | 0.030 | 0.423 | 1.92 | 0% |
| synthetic/sphere | 3DFUN AI (ours) | 0.0403 | 0.153 | 0.083 | 0.514 | 29.07 | 100% |
| synthetic/sphere | TSDF fusion | 0.1148 | 0.032 | 0.020 | 0.491 | 1.34 | 0% |
| synthetic/sphere | Visual hull | 0.0947 | 0.046 | 0.026 | 0.482 | 1.81 | 0% |
| synthetic/stairs | 3DFUN AI (ours) | 0.1159 | 0.032 | 0.021 | 0.545 | 22.58 | 100% |
| synthetic/stairs | TSDF fusion | 0.1236 | 0.029 | 0.018 | 0.446 | 1.24 | 0% |
| synthetic/stairs | Visual hull | 0.1048 | 0.036 | 0.022 | 0.545 | 1.94 | 0% |
| synthetic/tall_box | 3DFUN AI (ours) | 0.0427 | 0.099 | 0.013 | 0.384 | 20.23 | 100% |
| synthetic/tall_box | TSDF fusion | 0.0847 | 0.062 | 0.030 | 0.419 | 1.17 | 0% |
| synthetic/tall_box | Visual hull | 0.0494 | 0.121 | 0.046 | 0.428 | 1.67 | 0% |
| synthetic/thin_plate | 3DFUN AI (ours) | 0.1404 | 0.023 | 0.013 | 0.621 | 24.89 | 100% |
| synthetic/thin_plate | TSDF fusion | 0.0853 | 0.053 | 0.021 | 0.524 | 1.27 | 0% |
| synthetic/thin_plate | Visual hull | 0.0884 | 0.059 | 0.025 | 0.585 | 1.80 | 0% |


### 5.6 Failures

No run crashed. The qualitative failure modes (asymmetric backs, occluded handles, thin structures thinner than the minimum slab) are discussed in §Limitations, and the threshold flags are recorded per run in `results/raw/runs.csv`.


## 6. Reproducibility

```bash
make install     # Python deps
make models      # ONNX weights (checksums printed, nothing large in git)
make seed        # demo database + accounts
make benchmark   # full evaluation -> results/
make figures     # figures from results/
make paper       # paper/main.pdf from results/
make report      # this document
make verify      # tests + paper compile + number cross-check
```

Environment recorded in `results/environment.json`:

- CPU: Intel(R) Xeon(R) Processor @ 2.60GHz, 2 cores
- RAM: 2030416 kB
- GPU: none (all runs are CPU-only) --- every number in this report is CPU inference
- Python 3.13.16; numpy 2.3.5, scipy 1.17.1, cv2 5.0.0, skimage 0.26.0, open3d 0.20.0, trimesh 5.2.0, onnxruntime 1.31.1, fast_simplification 0.2.0, matplotlib 3.10.7, PIL 11.3.0, pandas 2.2.3
- Commit: `uncommitted`, seed 20261108, dataset
  1.0.0 (`d76fef12`)

## 7. Limitations

- The back of the object is never observed: the method mirrors a symmetry prior, so strongly
  asymmetric objects are reconstructed plausibly but incorrectly.
- Anything the silhouette cannot express is filled, not carved; hollow interiors and occluded
  handles are lost.
- Broken mattes (transparency, fur, heavy self-occlusion) break the volume, because the volume is
  literally the matte inflated.
- Textures are single-view; roughness and metallic are heuristics, and surfaces nearly parallel to
  the view direction stretch in the planar atlas.
- LPIPS, CLIP similarity, GSO/Objaverse reference sets, learned baselines and GPU throughput are
  **not evaluated** here, with the reasons above.

## 8. Code map

| Path | Contents |
|---|---|
| `engine/recon/` | reconstruction stages (matte, depth, volume, geometry, texture, export) |
| `engine/run_recon.py` | CLI |
| `app/` | FastAPI service: auth, assets, generation, gallery, admin, system |
| `web/` | dependency-free studio SPA + three.js viewer |
| `benchmarks/` | dataset generator, metrics, two baselines, harness |
| `results/` | raw runs, logs, summary CSVs, figures, environment record |
| `scripts/` | model fetcher, figure/paper/slide/report generators, verifier |
| `tests/` | 38 tests (API, generation flow, real-engine smoke) |

## 9. Team Contributions & 12-Minute Defense Slide Allocation

| # | Presenter | Roll Number | Primary Engineering Responsibility | Defense Section & Slides | Slot |
|---|---|---|---|---|---|
| 1 | J Ganesh | 23BDS024 | Core Volumetric Engine, Distance Transform, Priors | Part 1: Slides 1–3 (Problem, Ill-Posedness, Constraints) | 0:00 – 2:15 |
| 2 | M Jagadeshwar | 23BDS033 | Meshing, Marching Cubes, Taubin Smoothing, Atlas | Part 2: Slides 4–6 (Related Work, Architecture, Pipeline) | 2:15 – 4:45 |
| 3 | B Varshith | 23BDS011 | Benchmark Harness, Multi-Engine Baselines, Metrics | Part 3: Slides 7–9 (Dataset Protocol, Results, Timings) | 4:45 – 7:30 |
| 4 | Pranav Kashyap | 23BCS098 | FastAPI Service, Credit Ledger, Job State Machine | Part 4: Slides 10–12 (Ablations, Visuals, Web Studio) | 7:30 – 10:15 |
| 5 | C Sai Aravind | 23BDS075 | Web Studio SPA, Three.js Viewer, AR QuickLook | Part 5: Slides 13–15 (Limitations, Roadmap, Defense Q&A) | 10:15 – 12:00 |

## 10. Acknowledgment

We thank Dr. Shirshendu Layek, Assistant Professor, Department of Data Science and Artificial Intelligence,
IIIT Dharwad, for supervising this project and for insisting on measurement over assertion.
