# Lab 4: 3D Geometric Metric Evaluation Suite

## 1. Theoretical Formulation
Rigorous quantitative benchmarking of reconstructed 3D meshes $\mathcal{S}$ against ground-truth CAD reference models $\mathcal{G}$ requires permutation-invariant geometric distance metrics.

### 1.1 Symmetric Unsquared Chamfer Distance (CD)
Evaluates the bidirectional mean nearest-neighbor Euclidean distance between sampled point clouds:
$$d_{\text{CD}}(\mathcal{S}, \mathcal{G}) = \frac{1}{2 |\mathcal{S}|} \sum_{\mathbf{x} \in \mathcal{S}} \min_{\mathbf{y} \in \mathcal{G}} \|\mathbf{x} - \mathbf{y}\|_2 + \frac{1}{2 |\mathcal{G}|} \sum_{\mathbf{y} \in \mathcal{G}} \min_{\mathbf{x} \in \mathcal{S}} \|\mathbf{x} - \mathbf{y}\|_2$$

### 1.2 Precision, Recall and F-Score at Threshold $\tau$
Precision evaluates the fraction of reconstructed points lying within tolerance $\tau$ of the ground-truth:
$$\text{Precision}(\tau) = \frac{1}{|\mathcal{S}|} \sum_{\mathbf{x} \in \mathcal{S}} \left[ \min_{\mathbf{y} \in \mathcal{G}} \|\mathbf{x} - \mathbf{y}\|_2 < \tau \right]$$

Recall evaluates surface completeness:
$$\text{Recall}(\tau) = \frac{1}{|\mathcal{G}|} \sum_{\mathbf{y} \in \mathcal{G}} \left[ \min_{\mathbf{x} \in \mathcal{S}} \|\mathbf{x} - \mathbf{y}\|_2 < \tau \right]$$

F-Score is the harmonic mean of precision and recall:
$$F_1(\tau) = \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$
Standard benchmarks report at $\tau = 0.01$ (strict) and $\tau = 0.05$ (loose).

### 1.3 Earth Mover's Distance (EMD)
Measures the minimum work required to transport point cloud $\mathcal{S}$ into $\mathcal{G}$ under optimal bijection $\phi: \mathcal{S} \to \mathcal{G}$:
$$d_{\text{EMD}}(\mathcal{S}, \mathcal{G}) = \min_{\phi: \mathcal{S} \to \mathcal{G}} \frac{1}{|\mathcal{S}|} \sum_{\mathbf{x} \in \mathcal{S}} \|\mathbf{x} - \phi(\mathbf{x})\|_2$$

---

## 2. Execution & Verification
```powershell
python -m training.student_labs.lab4_evaluator.evaluate_metrics `
    --pred web/public/examples/chair.glb `
    --gt web/public/examples/chair.glb `
    --num-points 5000 `
    --out-dir out/student_labs/lab4
```

Outputs:
- `lab4_results.json` with Chamfer distance, F-scores, approximate EMD, seed and sample counts
- Prints formatted comparison metrics directly to terminal

The prediction and reference must use a common coordinate frame. The evaluator samples mesh faces or reads vertices from a point cloud. Omit both input flags to try the seeded noisy-sphere demo. Use `--seed` to repeat the sampling.
