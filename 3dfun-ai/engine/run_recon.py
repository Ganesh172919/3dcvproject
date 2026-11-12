"""CLI: python run_recon.py <image> --out DIR [--quality standard]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from recon import reconstruct, preset_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--out", default="out"); ap.add_argument("--quality", default="standard")
    ap.add_argument("--faces", type=int, default=None); ap.add_argument("--voxels", type=int, default=None)
    a = ap.parse_args()
    over = {}
    if a.faces: over["target_faces"] = a.faces
    if a.voxels: over["voxel_res"] = a.voxels
    cfg = preset_config(a.quality, **over)
    t=time.time()
    def prog(stage, pct, msg): print(f"[{pct:3d}%] {stage:14s} {msg}", flush=True)
    res = reconstruct(a.image, a.out, cfg, progress=prog)
    print(json.dumps({"manifest": str(res.out_dir/"manifest.json"),
                      "faces": res.manifest["mesh"]["faces"],
                      "watertight": res.manifest["mesh"].get("watertight_trimesh"),
                      "seconds": round(time.time()-t,1)}, indent=2))
if __name__ == "__main__": main()
