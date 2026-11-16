# Lab 3: 3D Gaussian Splatting (3DGS) Fundamentals & Optimization

## 1. Theoretical Formulation
3D Gaussian Splatting represents 3D scenes as a collection of anisotropic 3D Gaussian primitives $\mathcal{G}_i$:
$$\mathcal{G}_i(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \mu_i)^T \Sigma_i^{-1} (\mathbf{x} - \mu_i) \right)$$

### 1.1 Covariance Matrix Factorization
To guarantee positive semi-definiteness during gradient descent, covariance matrix $\Sigma$ is parameterized by a scaling matrix $S = \text{diag}(s_x, s_y, s_z)$ and rotation quaternion $\mathbf{q} \to R$:
$$\Sigma = R S S^T R^T$$

### 1.2 Differentiable 2D Projection
Under affine camera projection with viewing transformation $W$ and Jacobian $J$, the 2D projected covariance $\Sigma'$ on the image plane is:
$$\Sigma' = J W \Sigma W^T J^T$$

### 1.3 Alpha-Compositing Blending
Color along viewing rays is accumulated across sorted depth order:
$$C = \sum_{i=1}^K \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$
where $\alpha_i$ is evaluated from the 2D projected Gaussian footprint.

---

## 2. Laboratory Implementation: Mini Gaussian Splatter
In `train_single_view_splat.py`, students optimize 3D Gaussians from a reference image using:
- **Learnable Parameters:** Centers $\mu \in \mathbb{R}^{N \times 3}$, Log-scales $\mathbf{s} \in \mathbb{R}^{N \times 3}$, Logit-opacities $\alpha \in \mathbb{R}^{N \times 1}$, Colors $\mathbf{c} \in \mathbb{R}^{N \times 3}$
- **Objective:** Mean Squared Error (MSE) + Structural Similarity (SSIM) image reconstruction loss

---

## 3. Execution & Verification
```powershell
python -m training.student_labs.lab3_3dgs_mini.train_single_view_splat --num-points 800 --iterations 30 --out-dir out/student_labs/lab3
```

Outputs:
- `gaussians.ply`: Exported 3D Gaussian point cloud with positions, colors, and scales
- `rendered_reconstruction.png`: 2D rendered splat image
- `lab3_results.json`: Optimization loss history and PSNR metrics
