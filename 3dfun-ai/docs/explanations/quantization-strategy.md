# Explanation: Post-Training INT8 Quantization Strategy for Vision Transformers

Modern single-image 3D pipelines often rely on massive feed-forward neural networks or diffusion models requiring 16 GB+ of VRAM and high-end discrete GPUs. 3DFUN AI takes an alternative path: delivering fast, client-side capable reconstruction on ordinary consumer hardware. This document explains the theory, implementation, and empirical verification of our post-training INT8 quantization strategy.

---

## 1. Why Quantize Vision Transformers?

Depth Anything V2 Small utilizes a Vision Transformer (ViT-S) backbone with 24.8 million parameters. In standard single-precision floating point (FP32), storing these weights requires $\approx 98$ MB of RAM, and computing multi-head self-attention introduces substantial memory bandwidth bottlenecks on standard CPUs lacking high-bandwidth memory (HBM).

### Advantages of 8-Bit Integer (INT8) Quantization:
1. **$3.6\times$ Storage & Memory Compression:** Model footprint drops from 98 MB to 27 MB, fitting comfortably into CPU L3 caches and mobile storage budgets.
2. **AVX2 / VNNI Vector Acceleration:** Modern x86-64 CPUs feature Vector Neural Network Instructions (VNNI), executing 4 INT8 multiply-accumulate operations in a single clock cycle.
3. **Bandwidth Reduction:** Reducing memory traffic by $75\%$ significantly decreases cache miss penalties during self-attention token projections.

---

## 2. Dynamic vs. Static Range Calibration

Quantization maps continuous real numbers $x \in [\alpha, \beta]$ to discrete 8-bit signed integers $q \in [-128, 127]$:
$$q = \text{round}\left(\frac{x}{S}\right) + Z, \quad x \approx S \cdot (q - Z)$$
where $S$ represents the scale factor and $Z$ represents the zero-point offset.

### Our Strategy:
- **Weights (Static Asymmetric Quantization):** Static scale and zero-point parameters are computed offline per channel across the weight tensors.
- **Activations (Dynamic Symmetric Quantization):** Because intermediate token activations in attention layers can exhibit sudden outliers, activation dynamic ranges are calculated dynamically per inference batch. This prevents clipping artifacts without requiring extensive offline calibration datasets.

---

## 3. Numerical Error & Ablation Verification

To ensure that quantization does not degrade downstream 3D geometric accuracy, we evaluated the reconstruction quality between the unquantized FP32 model and our INT8 ONNX graph across the 13 reference synthetic scenes:

| Model Precision | Model Size (MB) | Mean Depth Latency (s) | Chamfer Distance | F-Score@1.0% | Normal Consistency |
|---|---|---|---|---|---|
| FP32 (Full Precision) | 98.4 MB | 1.12 s | 0.0381 | 0.842 | 0.891 |
| FP16 (Half Precision) | 49.2 MB | 0.58 s | 0.0382 | 0.841 | 0.890 |
| **INT8 (Dynamic, Ours)** | **27.3 MB** | **0.35 s** | **0.0384** | **0.839** | **0.888** |

### Key Observation:
INT8 quantization yields a $3.2\times$ reduction in inference latency with negligible geometric degradation: Chamfer Distance increases by less than $0.8\%$, and Normal Consistency drops by less than $0.3\%$. The downstream bilateral inflation and Taubin filtering absorb residual micro-quantization noise.

---
*Evidence: ONNX model graphs in `engine/models/`; benchmark validation in `benchmarks/run_benchmarks.py`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
