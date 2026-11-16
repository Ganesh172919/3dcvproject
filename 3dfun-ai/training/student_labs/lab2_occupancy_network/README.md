# Lab 2: Implicit Neural Representation & Occupancy Network (MLP-SDF)

## 1. Theoretical Formulation
An implicit neural representation represents 3D solid geometry as the level set of a continuous neural coordinate field:
$$f_\theta: \mathbb{R}^3 \to [0, 1] \quad (\text{Occupancy})$$
$$\Phi_\theta: \mathbb{R}^3 \to \mathbb{R} \quad (\text{Signed Distance Field})$$

### 1.1 Fourier Positional Encoding
Standard MLPs suffer from spectral bias, struggling to learn high-frequency geometric contours. We map spatial coordinates $\mathbf{x} = (x, y, z)$ through sinusoidal Fourier features:
$$\gamma(\mathbf{x}) = \left[ \mathbf{x}, \sin(2^0 \pi \mathbf{x}), \cos(2^0 \pi \mathbf{x}), \dots, \sin(2^{F-1} \pi \mathbf{x}), \cos(2^{F-1} \pi \mathbf{x}) \right]$$

### 1.2 Binary Cross-Entropy Loss
The network is trained on points sampled uniformly and near surface boundaries:
$$\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

### 1.3 Isosurface Extraction via Marching Cubes
At test time, the trained implicit field is queried across an $N^3$ voxel grid (e.g. $64^3$ or $128^3$). The zero-level set $\mathcal{S} = \{ \mathbf{x} \in \mathbb{R}^3 \mid \Phi(\mathbf{x}) = 0.5 \}$ is polygonized using Marching Cubes into a closed 2-manifold triangle mesh.

---

## 2. Supported Analytical CAD Primitives
Students can train models on four mathematical primitives:
- **Torus:** $(R - \sqrt{x^2 + y^2})^2 + z^2 \le r^2$ ($R=0.5, r=0.2$)
- **Sphere:** $x^2 + y^2 + z^2 \le r^2$ ($r=0.6$)
- **Cylinder:** $x^2 + y^2 \le r^2 \land |z| \le h/2$
- **Cube:** $\max(|x|, |y|, |z|) \le s$

---

## 3. Execution & Verification
```powershell
# Train on a Torus
python -m training.student_labs.lab2_occupancy_network.train_occupancy_sdf --primitive torus --epochs 10 --out-dir out/student_labs/lab2

# Train on a Sphere
python -m training.student_labs.lab2_occupancy_network.train_occupancy_sdf --primitive sphere --epochs 10 --out-dir out/student_labs/lab2
```

Outputs:
- `occupancy_mesh.obj`: Extracted polygonal 3D model
- `model.pth`: Trained implicit MLP weights
- `lab2_results.json`: Training loss curve and volumetric accuracy
