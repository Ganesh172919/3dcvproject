# Lab 1: Monocular Relative Depth Fine-Tuning (TinyDepth)

## 1. Theoretical Formulation
Single-view depth estimation predicts continuous per-pixel metric depth $Z(u, v) \in \mathbb{R}^+$ from a single RGB camera observation $I(u, v) \in \mathbb{R}^{H \times W \times 3}$.

During training, we minimize a three-part composite loss:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{SILog}} + \beta \mathcal{L}_{\text{grad}} + \gamma \mathcal{L}_{\text{normal}}$$

### 1.1 Scale-Invariant Logarithmic (SILog) Loss
$$\mathcal{L}_{\text{SILog}} = \sqrt{\frac{1}{N}\sum_i d_i^2 - \frac{\lambda}{N^2}\left(\sum_i d_i\right)^2}$$
where $d_i = \ln(y_i) - \ln(y_i^*)$ and $\lambda \in [0, 1]$ controls the degree of scale invariance (typically $\lambda = 0.85$).

### 1.2 Multiscale Gradient Loss
Enforces sharp discontinuity boundaries across silhouette edges:
$$\mathcal{L}_{\text{grad}} = \sum_s \frac{1}{N_s} \sum_{i} \left( \left|\nabla_x d_i^s\right| + \left|\nabla_y d_i^s\right| \right)$$

### 1.3 Surface Normal Consistency Loss
Derives surface normal vector fields $\mathbf{n}(u, v)$ from spatial finite differences:
$$\mathbf{n}(u, v) = \frac{(-\frac{\partial Z}{\partial x}, -\frac{\partial Z}{\partial y}, 1)^T}{\sqrt{(\frac{\partial Z}{\partial x})^2 + (\frac{\partial Z}{\partial y})^2 + 1}}$$
$$\mathcal{L}_{\text{normal}} = \frac{1}{N} \sum_i (1 - \cos(\mathbf{n}_{\text{pred}}, \mathbf{n}_{\text{gt}}))$$

---

## 2. Model Architecture: TinyDepth
This lab uses a four-stage convolutional encoder-decoder with skip connections. The toy train and validation sets share primitive families. The mesh-disjoint experiment in the course report uses the separate [TinyDepth pipeline](../../README.md).
- **Input:** Normalized RGB image $(B, 3, H, W)$
- **Encoder:** Four convolution blocks downsampling spatial resolution ($S, S/2, S/4, S/8$)
- **Decoder:** Transposed convolution upsampling blocks with concatenation skip connections
- **Head:** Softplus activation ensuring strictly positive metric depth $Z > 0$

---

## 3. Execution & Verification
```powershell
python -m training.student_labs.lab1_depth_finetuning.train_student_depth --epochs 5 --lr 0.001 --out-dir out/student_labs/lab1
```

The script exports:
- `tinydepth_student.pth`: PyTorch model weights & training history
- `tinydepth_student.onnx`: Standalone ONNX runtime deployment checkpoint
- `lab1_results.json`: Validation MAE, RMSE, and $\delta_1 < 1.25$ metrics
