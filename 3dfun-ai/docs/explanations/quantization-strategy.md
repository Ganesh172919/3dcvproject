# Quantized depth weights

The Studio downloads a 27.3 MB INT8 ONNX export of Depth Anything V2 Small from the upstream conversion repository. The model downloader accepts the graph only if it matches the recorded SHA-256 hash. U²-NetP is downloaded separately for foreground masking.

An INT8 representation stores a quantized approximation of a real value using a scale and, depending on the scheme, a zero point. Lower-precision weights can reduce storage and memory bandwidth. The execution provider, quantized operators, and data movement determine the effective inference time.

The [recorded mesh benchmark](../BENCHMARK.md) uses one quantized model. The timing and geometry measurements describe that pipeline. A local run can record active providers in its manifest, as described in the [GPU tutorial](../tutorials/gpu-acceleration.md). TinyDepth uses a separate FP32 export with numerical agreement checks against its PyTorch checkpoint.
