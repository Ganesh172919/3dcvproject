"""Build the course report's reconstruction comparison and held-out depth figure."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/course17"
PUBLIC = ROOT / "web/public/examples"
RECORDED = ROOT / "results/course17"


def compare(output):
    rows = [
        ("Sneaker", PUBLIC / "source-sneaker.png", PUBLIC / "three-quarter.png", PUBLIC / "side.png"),
        ("Rubber duck", PUBLIC / "rubber-duck-input.png", PUBLIC / "rubber-duck-preview.png",
         RECORDED / "duck_side.png"),
        ("Ring", PUBLIC / "ring-input.png", PUBLIC / "ring-preview.png",
         PUBLIC / "ring-midas-preview.png"),
    ]
    fig, ax = plt.subplots(3, 3, figsize=(7.12, 5.1))
    for i, (name, *paths) in enumerate(rows):
        for j, src in enumerate(paths):
            axis = ax[i, j]
            axis.imshow(plt.imread(src))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if i == 0:
                axis.set_title(("Input image", "Depth Anything V2 mesh", "Second view / method")[j], fontsize=9)
            if j == 0:
                axis.set_ylabel(name, fontsize=8)
    fig.tight_layout(pad=.35)
    fig.savefig(output / "output_comparison.png", dpi=180)
    plt.close(fig)


def depth_figure(experiment, output):
    data = experiment / "dataset"
    model = experiment / "model/tinydepth.onnx"
    view = "cylinder_view_01.png"
    rgb = plt.imread(data / "rgb" / view)[..., :3]
    truth = cv2.imread(str(data / "depth" / view), cv2.IMREAD_UNCHANGED) * .001
    valid = truth > 0
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    pred = session.run(None, {"rgb": rgb.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    hist = json.loads((experiment / "model/training_history.json").read_text(encoding="utf-8"))
    other = json.loads((experiment / "silog_only/training_history.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.08, 2.4), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 2))
    for i, (image, label) in enumerate(((rgb, "Held-out RGB"), (truth, "Reference depth"), (pred, "Predicted depth"))):
        a = fig.add_subplot(grid[0, i])
        plotted = a.imshow(np.ma.masked_where(~valid, image) if i else image,
                           cmap="viridis", vmin=1.2 if i else None, vmax=1.9 if i else None)
        a.set_title(label, fontsize=8)
        a.axis("off")
        if i == 2:
            cb = fig.colorbar(plotted, ax=a, shrink=.53, location="bottom",
                              pad=.03, ticks=[1.2, 1.9])
            cb.ax.tick_params(labelsize=7)
    a = fig.add_subplot(grid[0, 3])
    for row, label in ((hist, "SILog + gradients"), (other, "SILog only")):
        a.plot([e["epoch"] for e in row["history"]], [e["val"]["mae"] for e in row["history"]], label=label)
    a.set_xlabel("Epoch", fontsize=8)
    a.set_ylabel("Validation MAE", fontsize=8)
    a.tick_params(labelsize=7)
    a.grid(alpha=.25)
    a.legend(fontsize=7, loc="upper right")
    fig.savefig(output / "tinydepth_figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="Completed training.pipeline run with --ablation")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    compare(args.output)
    depth_figure(args.experiment, args.output)
