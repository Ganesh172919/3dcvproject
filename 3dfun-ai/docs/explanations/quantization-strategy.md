# Quantized ONNX models

The course pipeline runs Depth Anything V2 Small using the INT8 ONNX artifact listed in [the model fetcher](../../scripts/fetch_models.py). The downloaded depth graph is about 27 MB. The fetcher checks its SHA-256 digest before installation.

Quantization represents tensors with integers and scale factors. It can reduce model storage and memory traffic. The effect on inference time depends on the operators, ONNX provider, and hardware. The reported [stage timings](../../results/summary/stage_timings.csv) use the bundled INT8 depth model. The saved benchmark evaluates changes to the reconstruction pipeline and contains no comparison across FP32, FP16, and INT8 models.

For a precision comparison, use equivalent exports from the same model checkpoint and keep the images, quality, provider, and rendering options fixed. Report both depth-stage time and downstream mesh metrics on the same reference scenes.
