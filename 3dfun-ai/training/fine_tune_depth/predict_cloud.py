"""Predict depth with the trained ONNX model, then back-project a visible RGB point cloud."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import trimesh

if __package__:
    from .geometry import backproject
else:
    from geometry import backproject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True, help="Renderer camera JSON for this view")
    parser.add_argument("--output", type=Path, default=Path("out/point_cloud"))
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None or bgr.shape[:2] != mask.shape or not mask.any():
        parser.error("Use a nonempty mask with the same dimensions as the RGB input.")
    records = json.loads(args.cameras.read_text(encoding="utf-8"))
    camera = next((r for r in records if r["image"] == args.image.name), None)
    if camera is None:
        parser.error(f"Camera entry for {args.image.name} is missing.")
    rgb = bgr[..., ::-1]
    inputs = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    depth = session.run(["depth"], {"rgb": inputs})[0][0, 0]
    yy, xx = np.where(mask > 0)
    points = backproject(depth, yy, xx, camera)
    args.output.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points, colors=rgb[yy, xx]).export(args.output / "points.ply")
    depth_png = np.where(mask > 0, np.clip(np.rint(depth * 1000), 0, 65535), 0).astype(np.uint16)
    if not cv2.imwrite(str(args.output / "depth.png"), depth_png):
        raise OSError("Could not save the depth PNG.")
    print(f"Wrote {len(points)} camera-frame points to {args.output / 'points.ply'}")


if __name__ == "__main__":
    main()
