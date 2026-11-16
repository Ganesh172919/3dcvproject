"""Plan or explicitly run a local TinyDepth experiment from meshes to evaluated 3D clouds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def pipeline_steps(args):
    root = args.output_dir
    data = root / "dataset"
    model = root / "model"
    source = ["--input-dir", str(args.mesh_dir)] if args.mesh_dir else ["--procedural"]
    stages = [("render", [
        "training.dataset_gen.synthetic_renderer", *source, "--output-dir", str(data),
        "--num-views", str(args.views), "--resolution", str(args.size),
    ])]
    train = [
        "training.fine_tune_depth.train_depth", "--data-dir", str(data), "--epochs", str(args.epochs),
        "--size", str(args.size), "--batch-size", str(args.batch_size), "--device", args.device,
        "--seed", str(args.seed), "--threads", str(args.threads), "--lr", str(args.lr),
    ]
    stages.append(("train", [*train, "--output-dir", str(model)]))
    stages.extend(export_and_evaluation(args, model, root / "evaluation"))
    if args.ablation:
        ablation = root / "silog_only"
        stages.append(("silog_ablation", [
            *train, "--output-dir", str(ablation), "--grad-weight", "0",
        ]))
        stages.extend(export_and_evaluation(args, ablation, ablation / "evaluation", "_ablation"))
    return stages


def export_and_evaluation(args, model, evaluation, suffix=""):
    onnx = model / "tinydepth.onnx"
    return [
        (f"export{suffix}", [
            "training.fine_tune_depth.export_onnx", "--source", str(model / "best_model.pth"),
            "--output", str(onnx),
        ]),
        (f"evaluate_3d{suffix}", [
            "training.evaluate_clouds", "--data-dir", str(args.output_dir / "dataset"),
            "--model", str(onnx), "--history", str(model / "training_history.json"),
            "--output-dir", str(evaluation), "--tau", str(args.tau), "--max-points", str(args.max_points),
        ]),
    ]


def write_manifest(output_dir, manifest):
    temp = output_dir / "experiment.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(output_dir / "experiment.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, help="At least three OBJ or GLB meshes with unique stems")
    parser.add_argument("--output-dir", type=Path, default=Path("out/tinydepth-experiment"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--size", type=int, default=96, help="Square render and network input size")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--tau", type=float, default=.01, help="3D F-score distance in normalized units")
    parser.add_argument("--max-points", type=int, default=4096, help="Maximum sampled points per test view")
    parser.add_argument("--ablation", action="store_true", help="Train, export and evaluate a SILog-only model")
    parser.add_argument("--run", action="store_true", help="Execute the printed plan and create output files")
    args = parser.parse_args()
    if min(args.views, args.epochs, args.batch_size, args.threads) < 1 or not 32 <= args.size <= 512:
        parser.error("Use positive views, epochs, batch size and threads, with size 32-512.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed <= 2**32 - 1:
        parser.error("Use a positive, finite learning rate and a seed in [0, 2**32 - 1].")
    if not math.isfinite(args.tau) or args.tau <= 0 or args.max_points < 16:
        parser.error("Use a positive F-score distance and at least 16 points per view.")
    if args.mesh_dir:
        if not args.mesh_dir.is_dir():
            parser.error(f"Mesh directory not found: {args.mesh_dir}")
        stems = [path.stem.casefold() for path in args.mesh_dir.iterdir()
                 if path.is_file() and path.suffix.lower() in {".obj", ".glb"}]
        if len(stems) < 3 or len(set(stems)) != len(stems):
            parser.error("Provide at least three mesh files with unique stems.")
    stages = pipeline_steps(args)
    for label, stage in stages:
        print(f"{label}: {shlex.join(['python', '-m', *stage])}", flush=True)
    if not args.run:
        return
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new --output-dir to preserve previous results.")
    manifest = {"schema_version": 1, "requested_device": args.device, "complete": False,
                "stages": [{"name": label, "args": stage, "status": "pending"} for label, stage in stages]}
    write_manifest(args.output_dir, manifest)
    for stage in manifest["stages"]:
        stage["status"] = "running"
        write_manifest(args.output_dir, manifest)
        try:
            subprocess.run([sys.executable, "-m", *stage["args"]], check=True)
        except (subprocess.CalledProcessError, OSError, KeyboardInterrupt):
            stage["status"] = "failed"
            write_manifest(args.output_dir, manifest)
            raise
        stage["status"] = "complete"
        write_manifest(args.output_dir, manifest)
    manifest["complete"] = True
    write_manifest(args.output_dir, manifest)
    print(f"Experiment recorded in {args.output_dir / 'experiment.json'}")


if __name__ == "__main__":
    main()
