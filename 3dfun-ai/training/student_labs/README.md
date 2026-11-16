# 3DFUN AI — Student Educational Laboratory Curriculum
**Department of Data Science & Artificial Intelligence · IIIT Dharwad**  
**Course:** 3D Computer Vision · **Supervisor:** Dr. Shirshendu Layek  

Welcome to the 3DFUN AI Student Educational Curriculum. These laboratory modules are engineered to demystify single-view and neural 3D computer vision. Each module contains runnable, standalone PyTorch code with synthetic datasets, closed-form loss formulations, validation metrics, and export checkpoints.

---

## Laboratory Overview

```
+---------------------------------------------------------------------------------------------------------+
|                                    STUDENT EDUCATIONAL LAB MODULES                                      |
+---------------------------------------------------------------------------------------------------------+
| Lab 1: Monocular Relative Depth Fine-Tuning (TinyDepth)                                                 |
|        - Architecture: Lightweight Conv/Transformer with Skip Connections                               |
|        - Losses: SILog loss + Multiscale Spatial Gradient loss + Surface Normal Consistency loss        |
|        - Target: Train & export ONNX runtime checkpoint                                                 |
|                                                                                                         |
| Lab 2: Implicit Neural Representation & Occupancy Network (MLP-SDF)                                     |
|        - Architecture: Continuous coordinate MLP with Sinusoidal Fourier Encodings                      |
|        - Analytical Ground Truth: CAD Primitives (Torus, Sphere, Cylinder, Cube)                        |
|        - Isosurface: Marching Cubes zero-level set extraction -> OBJ 3D Model                           |
|                                                                                                         |
| Lab 3: 3D Gaussian Splatting (3DGS) Fundamentals                                                        |
|        - Representation: Position mu, Covariance Sigma = R S S^T R^T, Opacity alpha, Color c            |
|        - Ray Projection: Differentiable 2D footprint splatting and alpha compositing                    |
|        - Export: Point Cloud .PLY export with Gaussian radii                                            |
|                                                                                                         |
| Lab 4: 3D Geometric Metric Evaluation Suite                                                             |
|        - Formulations: Symmetric Chamfer Distance, Earth Mover's Distance (EMD), Precision, Recall     |
|        - Thresholds: F-Score @ tau = 0.01 and tau = 0.05                                                |
|        - Tooling: Fast KD-Tree nearest-neighbor surface point cloud comparison                          |
+---------------------------------------------------------------------------------------------------------+
```

---

## Quickstart: Running the Labs

From the project root:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 1. Run Lab 1: Train TinyDepth Monocular Depth Estimator
python -m training.student_labs.lab1_depth_finetuning.train_student_depth --epochs 5 --lr 0.001

# 2. Run Lab 2: Train Continuous Occupancy MLP on a Torus CAD primitive
python -m training.student_labs.lab2_occupancy_network.train_occupancy_sdf --primitive torus --epochs 10

# 3. Run Lab 3: Optimize 3D Gaussians from a reference view
python -m training.student_labs.lab3_3dgs_mini.train_single_view_splat --num-points 800 --iterations 30

# 4. Run Lab 4: Benchmark two meshes with Chamfer Distance & F-Score
python -m training.student_labs.lab4_evaluator.evaluate_metrics --pred web/public/examples/chair.glb --gt web/public/examples/chair.glb
```

---

## Academic Assessment & Evaluation Criteria

Students are graded on:
1. Convergence of Scale-Invariant Logarithmic (SILog) loss curves.
2. Understanding of non-shrinking Taubin smoothing vs. standard Laplacian smoothing.
3. Verification of 2-manifold closed surfaces ($\chi = V - E + F = 2$ for spherical genus 0).
4. Rigorous baseline comparisons against TSDF Fusion and Visual Hull.
