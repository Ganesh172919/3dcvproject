# ADR 0002: ONNX Runtime for local inference

## Decision

Use ONNX Runtime and downloaded, checksum-verified weights. The course benchmark uses the INT8 Depth Anything V2 Small graph and U²-Net portable. CPU is the default execution provider. The launcher also supports DirectML and CUDA environments.

## Consequences

The same inference entry points can run on local CPU installations. Accelerator support depends on the runtime and model operators. Per-run manifests record the providers used. See the [GPU tutorial](../tutorials/gpu-acceleration.md) and [quantization explanation](../explanations/quantization-strategy.md).
