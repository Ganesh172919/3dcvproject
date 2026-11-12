# Tutorial: GPU Acceleration via DirectML & CUDA in ONNX Runtime

While 3DFUN AI defaults to highly optimized CPU execution via OpenMP and INT8 quantization, developers on Windows or Linux workstations with dedicated GPUs can enable hardware acceleration using ONNX Runtime execution providers.

---

## 1. Execution Provider Overview

ONNX Runtime supports pluggable hardware execution providers (EPs):
- **CPU (Default):** Runs cross-platform using multithreaded AVX2/AVX-512 vector instructions.
- **DirectML (`DmlExecutionProvider`):** Microsoft DirectX 12 hardware acceleration supporting any modern GPU on Windows (NVIDIA, AMD Radeon, Intel Arc, and integrated Iris Xe).
- **CUDA / TensorRT (`CUDAExecutionProvider`):** Native NVIDIA hardware acceleration for Linux and Windows workstations.

---

## 2. Enabling DirectML Acceleration (Windows)

DirectML provides broad hardware compatibility without requiring proprietary CUDA toolkits.

### Step 1: Install the DirectML Package
In your active virtual environment:

```powershell
pip install onnxruntime-directml
```

### Step 2: Configure Environment or Code
To instruct the inference engine to request DirectML:

```python
import onnxruntime as ort

providers = [
    ('DmlExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession("engine/models/depth_anything_v2_vits.onnx", providers=providers)
print("Active execution providers:", session.get_providers())
```

---

## 3. Enabling CUDA Acceleration (NVIDIA Workstations)

For Linux or Windows setups equipped with CUDA 11.8 or 12.x and cuDNN 8.x+:

### Step 1: Install ONNX Runtime with CUDA Support
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### Step 2: Verification Script
Verify that your GPU device is detected:

```python
import onnxruntime as ort

print("Available providers:", ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers(), "CUDA EP not found!"

session = ort.InferenceSession("engine/models/depth_anything_v2_vits.onnx", providers=['CUDAExecutionProvider'])
print("Session initialized successfully on CUDA.")
```

---

## 4. Performance Benchmarking Across Providers

Inference latency on Depth Anything V2 Small ($518 \times 518$ input resolution):

| Hardware / Provider | Execution Precision | Mean Latency (ms) | Speedup Factor |
|---|---|---|---|
| Intel Core i7-12700H (CPU) | INT8 Quantized | 348 ms | $1.0\times$ (Baseline) |
| AMD Radeon RX 6800M (DirectML) | FP16 / INT8 | 72 ms | $4.8\times$ |
| NVIDIA GeForce RTX 3080 (CUDA) | FP16 / TensorRT | 38 ms | $9.1\times$ |
| NVIDIA RTX 4090 (TensorRT) | FP16 Graph Optimized | 19 ms | $18.3\times$ |

---

## 5. Fallback Mechanism

3DFUN AI includes automatic fallback in `engine/recon/depth.py`: if `CUDAExecutionProvider` or `DmlExecutionProvider` fails to allocate video memory or encounters driver incompatibility, it logs a warning and falls back to `CPUExecutionProvider` without crashing the reconstruction pipeline.

---
*Evidence: Model loading logic in `engine/recon/depth.py` and `benchmarks/run_benchmarks.py`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
