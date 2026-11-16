"""Measure held-out visible point clouds and export a predicted/reference PLY pair."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial import cKDTree
import trimesh

from training.fine_tune_depth.geometry import backproject


def geometry_scores(points, reference, tau):
    forward = cKDTree(reference).query(points)[0]
    backward = cKDTree(points).query(reference)[0]
    precision = float((forward <= tau).mean())
    recall = float((backward <= tau).mean())
    return {"chamfer": float((forward.mean() + backward.mean()) / 2),
            "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate(data_dir, session, history, output_dir, max_points, tau):
    names = history["splits"]["test"]
    if not names or len(set(names)) != len(names):
        raise ValueError("The history must contain distinct held-out views.")
    constant = float(history["constant_depth"])
    size = int(history["size"])
    if not 16 <= size <= 512:
        raise ValueError("The history has an invalid TinyDepth input size.")
    cameras = {}
    rows = []
    example = None
    for name in sorted(names):
        stem, _, view = Path(name).stem.rpartition("_view_")
        if Path(name).name != name or not stem or not view.isdecimal() or not name.endswith(".png"):
            raise ValueError(f"Invalid held-out view name: {name}")
        pixels = cv2.imread(str(data_dir / "rgb" / name), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(data_dir / "depth" / name), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(data_dir / "mask" / name), cv2.IMREAD_GRAYSCALE)
        if (pixels is None or depth is None or mask is None or depth.ndim != 2
                or depth.dtype != np.uint16 or pixels.shape[:2] != depth.shape or mask.shape != depth.shape):
            raise ValueError(f"Missing or inconsistent RGB, depth or mask for {name}.")
        if stem not in cameras:
            camera_file = data_dir / "cameras" / f"{stem}.json"
            cameras[stem] = {row["image"]: row for row in json.loads(camera_file.read_text(encoding="utf-8"))}
        if name not in cameras[stem]:
            raise ValueError(f"Missing intrinsics for {name}.")
        camera = cameras[stem][name]
        rgb = pixels[..., ::-1]
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        inputs = resized.transpose(2, 0, 1)[None].astype(np.float32) / 255
        prediction = session.run(["depth"], {"rgb": inputs})[0][0, 0]
        if prediction.shape != (size, size):
            raise ValueError(f"ONNX depth dimensions do not match {name}.")
        if prediction.shape != depth.shape:
            prediction = cv2.resize(prediction, depth.shape[::-1], interpolation=cv2.INTER_LINEAR)
        yy, xx = np.where((depth > 0) & (mask > 0))
        if yy.size < 16:
            raise ValueError(f"Too few valid depth pixels for {name}.")
        if yy.size > max_points:
            ids = np.random.default_rng(17).choice(yy.size, size=max_points, replace=False)
            yy, xx = yy[ids], xx[ids]
        reference = backproject(depth.astype(np.float32) / 1000, yy, xx, camera)
        predicted = backproject(prediction, yy, xx, camera)
        baseline = backproject(np.full(depth.shape, constant, dtype=np.float32), yy, xx, camera)
        scores = {"tinydepth": geometry_scores(predicted, reference, tau),
                  "constant": geometry_scores(baseline, reference, tau)}
        rows.append({"mesh": stem, "view": name, "points": int(len(yy)),
                     **{f"{method}_{metric}": value for method, result in scores.items()
                        for metric, value in result.items()}})
        if example is None:
            example = (name, predicted, reference, rgb[yy, xx])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_view.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        groups.setdefault(row["mesh"], []).append(row)
    metrics = [key for key in rows[0] if key.startswith(("tinydepth_", "constant_"))]
    per_mesh = {stem: {key: float(np.mean([row[key] for row in group])) for key in metrics}
                for stem, group in groups.items()}
    name, predicted, reference, colors = example
    for filename, points in (("predicted.ply", predicted), ("reference.ply", reference)):
        trimesh.points.PointCloud(points, colors=colors).export(output_dir / filename)
    summary = {"schema_version": 1, "coordinate_frame": "camera", "distance_units": "normalized world units",
               "fscore_threshold": tau, "max_points_per_view": max_points,
               "views": len(rows), "example_view": name, "per_mesh": per_mesh,
               "mean_by_mesh": {key: float(np.mean([row[key] for row in per_mesh.values()])) for key in metrics}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["mean_by_mesh"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "history", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--tau", type=float, default=.01)
    args = parser.parse_args()
    if args.max_points < 16 or not np.isfinite(args.tau) or args.tau <= 0:
        parser.error("Use at least 16 points and a finite positive F-score threshold.")
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir to preserve previous results.")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model), opts, providers=["CPUExecutionProvider"])
    try:
        history = json.loads(args.history.read_text(encoding="utf-8"))
        evaluate(args.data_dir, session, history, args.output_dir, args.max_points, args.tau)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
