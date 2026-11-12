# ADR 0002: ONNX Runtime INT8 CPU Inference Over PyTorch/CUDA

## Status
Accepted

## Context
Deploying deep learning vision backbones (U2-Net for foreground matting and Depth Anything V2 for relative depth) typically requires PyTorch, torchvision, and NVIDIA CUDA libraries. A complete PyTorch + CUDA virtual environment exceeds 4.5 GB of disk space and demands a high-end dedicated GPU with compatible driver versions. Academic computer lab environments and student laptops frequently lack dedicated GPUs or have locked driver permissions.

## Decision
We standardized all neural network inference on **ONNX Runtime** using post-training **INT8 quantization**:
1. All PyTorch models are converted to static computation graphs in the open ONNX format.
2. Models are quantized using dynamic activation range calibration to 8-bit signed integers.
3. ONNX Runtime executes multithreaded CPU inference using OpenMP, AVX2, and AVX-512 VNNI instructions.

## Consequences
### Positive:
- **Zero Heavy Dependencies:** The entire Python virtual environment is $<300$ MB, installable in seconds via `uv` or `pip`.
- **Universal Hardware Execution:** Runs on standard x86-64 and ARM64 CPUs (including macOS Apple Silicon and Raspberry Pi 5).
- **Sub-Second Latency:** Monocular depth inference executes in $\approx 350$ ms on consumer CPUs, making interactive web reconstruction practical without cloud GPU clusters.

### Negative:
- Slightly reduced float precision (mean relative depth discrepancy $\le 0.8\%$ compared to FP32).
