# Reference: Quantitative Evaluation Metrics & Protocol

This document provides formal mathematical definitions, implementation references, and interpretation standards for the 3D computer vision metrics used throughout the 3DFUN AI benchmark harness.

---

## 1. Geometric Accuracy Metrics (Synthetic Ground Truth)

Because physical photographs lack exact ground truth meshes, geometric distance metrics are evaluated exclusively on synthetic scenes with analytic reference geometry (`benchmarks/data/synthetic/gt/`).

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
