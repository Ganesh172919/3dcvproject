# Lab 5: Feed-Forward Triplane Generator (LRM / TripoSR Mini)

## Academic Context
- **Course**: 3D Computer Vision (Group 17) · IIIT Dharwad
- **Supervisor**: Dr. Shirshendu Layek
- **Paradigm**: Feed-Forward Large Reconstruction Model (LRM) with Orthogonal Feature Triplanes

---

## 1. Pedagogical Objectives
1. Understand how high-capacity generative 3D systems (e.g. TripoSR, InstantMesh, LRM) achieve single-second 3D generation without iterative gradient descent at test-time.
2. Master the mathematical projection from a 2D image feature tensor into three orthogonal 3D spatial feature planes:
   $$T_{xy} \in \mathbb{R}^{C \times H_p \times W_p}, \quad T_{xz} \in \mathbb{R}^{C \times H_p \times W_p}, \quad T_{yz} \in \mathbb{R}^{C \times H_p \times W_p}$$
3. Implement differentiable bilinear sampling using `torch.nn.functional.grid_sample`.
4. Train a coordinate MLP decoder $g_\theta(\mathbf{f}(\mathbf{x}))$ and extract a continuous 3D polygon mesh via Marching Cubes.
5. Export the trained model to ONNX for CPU / DirectML runtime execution.

---

## 2. Mathematical Formulation

### 2.1 Coordinate Triplane Bilinear Sampling
Given a query point $\mathbf{x} = (x, y, z) \in [-1, 1]^3$, we project $\mathbf{x}$ onto the three orthogonal coordinate planes:
$$\pi_{xy}(\mathbf{x}) = (x, y), \quad \pi_{xz}(\mathbf{x}) = (x, z), \quad \pi_{yz}(\mathbf{x}) = (y, z)$$

The feature vector at $\mathbf{x}$ is formed by concatenating the bilinearly interpolated features:
$$\mathbf{f}(\mathbf{x}) = \left[ T_{xy}\left(\pi_{xy}(\mathbf{x})\right) \,\|\, T_{xz}\left(\pi_{xz}(\mathbf{x})\right) \,\|\, T_{yz}\left(\pi_{yz}(\mathbf{x})\right) \right] \in \mathbb{R}^{3C}$$

### 2.2 Occupancy Prediction
The coordinate MLP $g_\theta: \mathbb{R}^{3C} \to \mathbb{R}$ evaluates the occupancy probability:
$$p(\mathbf{x}) = \sigma\left(g_\theta(\mathbf{f}(\mathbf{x}))\right) \in [0, 1]$$

Trained using binary cross-entropy loss:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p(\mathbf{x}_i) + (1 - y_i) \log(1 - p(\mathbf{x}_i)) \right]$$

---

## 3. How to Run via CLI

```bash
python training/student_labs/lab5_triplane_lrm/train_triplane_lrm.py
```
