"""Calibrate the actual SDF mesh against reference geometry with held-out scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from benchmarks.metrics import evaluate_geometry
from engine.recon.config import preset_config
from engine.recon.depth import estimate_depth
from engine.recon.meshing import marching_cubes
from engine.recon.preprocessing import prepare_image
from engine.recon.volumetrics import build_volume


def load_scenes(dataset_file, config):
    entries = json.loads(dataset_file.read_text(encoding="utf-8"))["entries"]
    root = dataset_file.parent
    scenes = []
    for entry in sorted((row for row in entries if row.get("gt_mesh")), key=lambda row: row["id"]):
        prepared = prepare_image(root / entry["image"], config)
        close = estimate_depth(prepared.rgb, prepared.mask, config).closeness
        gt = trimesh.load(root / entry["gt_mesh"], force="mesh")
        scenes.append((entry["id"], prepared.mask, close, gt))
    if not scenes or len(scenes) == 2 or len({scene[0] for scene in scenes}) != len(scenes):
        raise ValueError("Provide one verification scene or at least three distinct 3D render scenes.")
    return scenes


def score(scenes, config, overrides, samples, seed):
    cfg = config.with_overrides(**overrides)
    result = []
    for name, mask, depth, gt in scenes:
        volume = build_volume(mask, depth, cfg)
        try:
            v, f, _ = marching_cubes(volume.sdf, volume.spacing, volume.origin)
        except (RuntimeError, ValueError):
            return None
        scores = evaluate_geometry(v, f, gt.vertices, gt.faces, samples=samples, seed=seed)
        result.append({"scene": name, "chamfer": scores.chamfer, "f_score": scores.f_score})
    return {"chamfer": float(np.mean([r["chamfer"] for r in result])), "scenes": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="JSON manifest with images and reference meshes")
    parser.add_argument("--output", type=Path, default=Path("out/calibration.json"))
    parser.add_argument("--trials", type=int, default=8, help="Includes the unmodified baseline")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.trials < 1 or args.samples < 500:
        parser.error("Use at least one trial and at least 500 surface samples.")
    cfg = preset_config("draft", canvas_size=196, voxel_res=48)
    rng = np.random.default_rng(args.seed)
    scenes = load_scenes(args.dataset, cfg)
    order = rng.permutation(len(scenes))
    n = max(1, len(scenes) // 4) if len(scenes) >= 3 else 0
    train, test = [scenes[i] for i in order[n:]], [scenes[i] for i in order[:n]]
    trace = []
    for step in range(args.trials):
        floor = float(rng.uniform(.35, .75)) if step else cfg.thickness_floor_ratio
        overrides = {} if not step else {
            "thickness_scale": float(rng.uniform(.4, 1.05)),
            "thickness_floor_ratio": floor,
            "thickness_depth_gain": 1 - floor,
            "inflation_power": float(rng.uniform(.6, 1.4)),
            "field_smooth_sigma": float(rng.uniform(.45, 1.2)),
        }
        measured = score(train, cfg, overrides, args.samples, args.seed)
        trace.append({"overrides": overrides, "fit": measured})
        print(f"trial {step + 1}/{args.trials}: {measured['chamfer']:.5f}" if measured else f"trial {step + 1}: no mesh", flush=True)
    valid = [row for row in trace if row["fit"] is not None]
    if not valid:
        parser.error("No candidate produced every fit mesh.")
    best = min(valid, key=lambda row: row["fit"]["chamfer"])
    result = {"protocol": "depth-driven SDF and Marching Cubes before post-processing",
              "seed": args.seed, "surface_samples": args.samples, "base_config": cfg.to_dict(),
              "fit_scenes": [s[0] for s in train], "held_out_scenes": [s[0] for s in test],
              "best_overrides": best["overrides"], "trials": trace,
              "held_out_baseline": score(test, cfg, {}, args.samples, args.seed) if test else None,
              "held_out_selected": score(test, cfg, best["overrides"], args.samples, args.seed) if test else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
