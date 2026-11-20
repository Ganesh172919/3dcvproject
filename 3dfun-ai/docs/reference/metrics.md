# Reference: Quantitative Evaluation Metrics & Protocol

This document provides formal mathematical definitions, implementation references, and interpretation standards for the 3D computer vision metrics used throughout the 3DCV Lab Group 17 benchmark harness.

---

## 1. Geometric Accuracy Metrics (3D Render Ground Truth)

Because physical photographs lack exact ground truth meshes, geometric distance metrics are evaluated exclusively on 3D render scenes with analytic reference geometry (`benchmarks/data/3D render/gt/`).

### 1.1 Chamfer Distance (CD)
Measures the mean symmetric Euclidean distance between point sets sampled uniformly from the reconstructed surface $S_1$ and ground truth surface $S_2$:
$$d_{\mathrm{CD}}(S_1, S_2) = \frac{1}{2|S_1|}\sum_{p \in S_1}\min_{q \in S_2}\|p - q\|_2 + \frac{1}{2|S_2|}\sum_{q \in S_2}\min_{p \in S_1}\|q - p\|_2$$
- **Units:** Dimensionless (normalized bounding box where $L_{\max} = 1.0$).
- **Sample Count:** $|S_1| = |S_2| = 10\,000$ points sampled via surface-area-weighted face sampling.
- **Interpretation:** Lower is better ($0.0$ indicates perfect geometric coincidence).

---

### 1.2 F-Score@$\tau$
Precision and recall at a defined distance threshold $\tau$:
$$P(\tau) = \frac{1}{|S_1|}\sum_{p \in S_1}\mathbb{1}\left[\min_{q \in S_2}\|p - q\|_2 < \tau\right], \quad R(\tau) = \frac{1}{|S_2|}\sum_{q \in S_2}\mathbb{1}\left[\min_{p \in S_1}\|q - p\|_2 < \tau\right]$$
$$F@\tau = \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}$$
- **Thresholds Evaluated:** $\tau \in \{0.5\%, 1.0\%, 2.0\%\}$ of the bounding box diagonal.
- **Interpretation:** Higher is better (range $[0, 1]$).

---

### 1.3 Volumetric Intersection-over-Union (IoU)
Evaluates solid enclosed volume alignment by rasterizing both meshes onto a regular $128^3$ voxel occupancy grid:
$$\mathrm{IoU}(V_1, V_2) = \frac{|V_1 \cap V_2|}{|V_1 \cup V_2|}$$
- **Interpretation:** Higher is better. Distinguishes volumetric solids from hollow or degenerate flat surface sheets.

---

### 1.4 Normal Consistency (NC)
Evaluates high-frequency surface orientation agreement:
$$\mathrm{NC}(S_1, S_2) = \frac{1}{|S_1|}\sum_{p \in S_1} |\langle \mathbf{n}_p, \mathbf{n}_{q^*(p)} \rangle|$$
where $q^*(p) = \arg\min_{q \in S_2} \|p - q\|_2$ is the nearest neighbor on the reference surface, and $\mathbf{n}$ denotes unit surface normals.

---

## 2. Appearance & Perceptual Metrics

### 2.1 Structural Similarity Index (SSIM)
Evaluates image-space visual fidelity by re-rendering the reconstructed 3D mesh from the primary input camera viewpoint and comparing it to the masked input photograph:
$$\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

---

## 3. Computational & Operational Metrics

| Metric | Symbol | Unit | Measurement Method |
|---|---|---|---|
| Latency | $T_{\text{wall}}$ | seconds | Wall-clock elapsed time measured across 3 repetitions. |
| Peak Memory | $\text{RSS}_{\text{peak}}$ | megabytes | Peak resident set size tracked via `psutil` / `getrusage`. |
| Watertightness | $W_{\text{pct}}$ | percentage | Percentage of test runs producing zero boundary edges ($\partial\mathcal{M} = \emptyset$). |

---
*Evidence: Metric computation implementations in `benchmarks/metrics.py`.*
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*

---

## 4. Metric Reporting Standards and Reproducibility Requirements

For benchmark results to be considered reproducible and academically honest, the following reporting standards are enforced by the `benchmarks/run_benchmarks.py` harness:

### 4.1 Mandatory Reporting Fields

Each benchmark result entry in `results/summary/benchmark_results.csv` must include:

| Field | Type | Description |
|---|---|---|
| `scene_id` | string | Identifier of the benchmark scene (e.g., `sphere_r0.3`) |
| `method` | string | Reconstruction method (`proposed`, `baseline_hull`, `baseline_tsdf`) |
| `chamfer_distance` | float | $d_{CD}$ (lower is better, normalized units) |
| `fscore_1pct` | float | $F@1\%$ (higher is better, range [0,1]) |
| `fscore_2pct` | float | $F@2\%$ (higher is better, range [0,1]) |
| `iou_128` | float | Volumetric IoU at 128³ grid (higher is better, range [0,1]) |
| `normal_consistency` | float | NC (higher is better, range [0,1]) |
| `is_watertight` | bool | Whether boundary edge count is 0 |
| `latency_sec` | float | End-to-end wall-clock reconstruction time |
| `peak_rss_mb` | float | Peak resident set size in megabytes |
| `n_triangles` | int | Triangle count of output mesh |
| `execution_provider` | string | ONNX execution provider used (`CPUExecutionProvider`, etc.) |
| `cpu_model` | string | CPU model string from `platform.processor()` |
| `onnx_runtime_version` | string | ONNX Runtime version string |
| `timestamp` | ISO-8601 | UTC timestamp of the run |

### 4.2 Sampling Protocol for Point Cloud Metrics

Chamfer Distance and F-Score are sensitive to the number and distribution of surface samples. 3DCV Lab Group 17 uses **uniform area-weighted face sampling** via trimesh:

```python
import trimesh
import numpy as np

def sample_mesh_surface(mesh: trimesh.Trimesh, n_points: int = 10_000) -> np.ndarray:
    """
    Sample points uniformly from mesh surface weighted by triangle area.
    Returns (n_points, 3) float32 array of 3D coordinates.
    """
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)
```

**Critical**: Both reconstructed and ground-truth point clouds must use the same `n_points = 10,000` and the same normalization (bounding box scaled to unit diagonal).

### 4.3 Normalization Convention

All geometric metrics are computed in a **normalized coordinate frame** where the bounding box diagonal of the ground-truth mesh equals 1.0:

```python
def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to unit bounding-box diagonal, centered at origin."""
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / bbox_diag)
    return mesh
```

The same scale factor applied to the ground-truth mesh is applied to the reconstructed mesh before computing any metrics.

---

## 5. Metric Computation Implementation Details

### 5.1 Chamfer Distance Implementation

```python
from scipy.spatial import cKDTree
import numpy as np

def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """
    Bidirectional Chamfer Distance between two point sets.

    Args:
        pts1: (N, 3) reconstructed surface samples
        pts2: (M, 3) ground-truth surface samples
    Returns:
        Scalar Chamfer Distance (lower is better)
    """
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    # Forward: each reconstructed point → nearest GT point
    d12, _ = tree2.query(pts1, k=1, workers=-1)
    # Backward: each GT point → nearest reconstructed point
    d21, _ = tree1.query(pts2, k=1, workers=-1)

    return (d12.mean() + d21.mean()) / 2.0
```

### 5.2 F-Score Implementation

```python
def fscore(pts1: np.ndarray, pts2: np.ndarray, threshold: float = 0.01) -> float:
    """
    F-Score at distance threshold.

    Args:
        threshold: Distance threshold in normalized units (0.01 = 1% of bounding diagonal)
    """
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    d12, _ = tree2.query(pts1, k=1)
    d21, _ = tree1.query(pts2, k=1)

    precision = (d12 < threshold).mean()
    recall    = (d21 < threshold).mean()

    if precision + recall < 1e-8:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
```

---

## 6. Interpreting Benchmark Results

### Understanding the Baselines

The 3DCV Lab Group 17 benchmark compares three methods:

| Method | Description | Expected CD | Expected $F@1\%$ |
|---|---|---|---|
| `baseline_hull` | Pure visual hull (silhouette extrusion) | ~0.0812 | ~0.098 |
| `baseline_tsdf` | Unpadded TSDF without inflation | ~0.0754 | ~0.112 |
| `proposed` | 3DCV Lab Group 17 constrained slab inflation | **~0.0639** | **~0.148** |

Per-scene results and differences between methods are recorded in `results/summary/per_scene.csv`.

---

*Evidence: Metric implementations in `benchmarks/metrics.py`; harness in `benchmarks/run_benchmarks.py`.*
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
