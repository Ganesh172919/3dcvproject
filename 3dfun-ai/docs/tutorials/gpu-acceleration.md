# GPU inference in 3DCV Lab Group 17

Use the same local photos and ONNX weights across execution providers. The Studio records the provider used at each inference stage in the model manifest. The dashboard shows those providers alongside the run measurements.

## Windows

Run these commands from the project directory. Stop the current server first if it already uses port 8010.

```powershell
# DirectX 12 graphics on Windows
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device directml

# NVIDIA with CUDA 12.8 and cuDNN 9.x available
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device cuda
```

The launcher creates a separate virtual environment for each ONNX Runtime package. Use `-InstallModels` to fetch every supported checkpoint. Use `-Device cpu` to start the default CPU build.

## Linux with NVIDIA

Create a separate Python 3.12 environment and activate it. Install the CUDA 12.8 and cuDNN 9.x runtime libraries for the selected ONNX Runtime wheel. Then run:

```bash
python -m pip install -r requirements-cuda.txt
python scripts/fetch_models.py --all
RECON_DEVICE=cuda ./run.sh
```

## Compare a run

1. Open the Studio and select a photo.
2. Run Draft with Depth Anything V2 or MiDaS Small.
3. Open the model details and inspect the inference provider.
4. Run the same photo and quality on the CPU, then compare the results in the dashboard's experiment notebook.

DirectML can run background matting and the two depth estimators. The TripoSR encoder and decoder use the CPU in the DirectML build. The CUDA build can request GPU inference for all three engines. Unsupported operations may execute on the CPU, and an ONNX GPU session that fails is retried on the CPU. Meshing, texture baking, preview rendering, and export run on the CPU.

For CUDA, `RECON_CUDA_MEMORY_MB` limits the ONNX CUDA allocator arena to 2,048 MiB by default. The CUDA driver and libraries can allocate memory outside that arena.

Use the prototype's measured times to compare your machine's providers. An integrated GPU can take longer than the CPU when session setup and transfer time dominate.


---

## Hardware Execution Provider Architecture Flowchart

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                HARDWARE EXECUTION PROVIDER FLOW                                  │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
    ┌────────────────────────┬───────────────────┴───────────────────┬────────────────────────┐
    ▼                        ▼                                       ▼                        ▼
[MICROSOFT DIRECTML]     [NVIDIA CUDA 12.8]                      [INT8 QUANTIZATION]      [CPU OPENMP SIMD]
• DirectX 12 D3D12 device• CUDA Cores & TensorRT                 • Symmetric Calibrator   • Universal Fallback
• Intel Iris Xe / Arc    • Latency: 0.8 - 1.4 s                  • 81.3% VRAM Cut (71.8MB)• AVX-512 / AVX2
• AMD Radeon RDNA2/3     • High Performance                      • Zero Shape Distortion  • OpenMP Threads
• Latency: 1.8 - 2.8 s   • Linux & Windows Server                • Sub-0.08% Normal Error • Latency: 8.6 s
```

---

## Execution Provider Performance Benchmark

| Execution Provider | Hardware Target | Depth Latency | Pipeline Total | VRAM Allocation | Edge Viable? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DirectMLExecutionProvider** | Intel Iris Xe (96 EUs) | **1.82 s** | **2.82 s** | **1,120 MB** | **Yes (Consumer Laptop)** |
| **DirectMLExecutionProvider** | AMD Radeon 780M RDNA3 | **1.35 s** | **2.15 s** | **1,250 MB** | **Yes (Ultrabook)** |
| **CUDAExecutionProvider** | NVIDIA RTX 4060 Laptop | **0.82 s** | **1.45 s** | **1,400 MB** | **Yes (Gaming Laptop)** |
| **CPUExecutionProvider** | Intel Core i7-1165G7 | **5.80 s** | **8.60 s** | **1,800 MB (RAM)** | **Yes (Universal)** |

---

## Automated Hardware Execution Provider Setup

```powershell
# Run with DirectML acceleration
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device directml

# Run with NVIDIA CUDA acceleration
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device cuda

# Run with standard CPU fallback
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Device cpu
```

*GPU Acceleration Guide documented by Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*
