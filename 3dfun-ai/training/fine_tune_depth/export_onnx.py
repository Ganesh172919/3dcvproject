"""Export a TinyDepth checkpoint and verify ONNX Runtime parity."""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

if __package__:
    from .model import load_model
else:
    from model import load_model


def export(source, output):
    model, state = load_model(source)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        dummy = torch.rand((1, 3, state["size"], state["size"]))
        probe = torch.rand((2, 3, state["size"] + 3, state["size"] + 5))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, dummy, str(output), input_names=["rgb"], output_names=["depth"],
                          dynamic_axes={name: {0: "batch", 2: "height", 3: "width"}
                                        for name in ("rgb", "depth")}, opset_version=17, dynamo=False)
        onnx.checker.check_model(str(output))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(output), opts, providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            for tensor in (dummy, probe):
                actual = session.run(["depth"], {"rgb": tensor.numpy()})[0]
                np.testing.assert_allclose(actual, model(tensor).numpy(), rtol=1e-4, atol=1e-5)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(f"Verified PyTorch/ONNX Runtime parity: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
