"""Reconstruct a mesh from one photograph."""
import argparse
import json
import time

if __package__:
    from .recon import preset_config, reconstruct
else:
    from recon import preset_config, reconstruct

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--out", default="out")
    parser.add_argument("--quality", choices=("draft", "standard", "high"), default="standard")
    parser.add_argument("--faces", type=int)
    parser.add_argument("--voxels", type=int)
    args = parser.parse_args()
    overrides = {}
    if args.faces is not None:
        overrides["target_faces"] = args.faces
    if args.voxels is not None:
        overrides["voxel_res"] = args.voxels
    config = preset_config(args.quality, **overrides)
    started = time.perf_counter()

    def progress(stage, pct, msg):
        print(f"[{pct:3d}%] {stage:14s} {msg}", flush=True)

    result = reconstruct(args.image, args.out, config, progress=progress)
    print(json.dumps({
        "manifest": str(result.out_dir / "manifest.json"),
        "faces": result.manifest["mesh"]["faces"],
        "watertight": result.manifest["mesh"].get("watertight_trimesh"),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
