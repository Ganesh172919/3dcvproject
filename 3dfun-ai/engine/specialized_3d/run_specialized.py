"""Master CLI: Specialized Image-to-3D Constructions Runner.

Executes specialized 3D reconstruction pipelines:
1. 'pointcloud'   : Dense RGB-D backprojection + surface normals + outlier filtering (.ply)
2. 'revolution'   : Axisymmetric 360-degree rotational surface of revolution (.obj)
3. 'space_carving': Orthogonal visual hull voxel carving + Marching Cubes (.obj, .npz)
4. 'surface_mesh' : Depth-guided Signed Distance Field (SDF) + Taubin smoothing (.obj)
5. 'all'          : Runs all 4 specialized pipelines sequentially.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

if __package__:
    from .pointcloud_construction import build_pointcloud
    from .revolution_construction import build_revolution
    from .space_carving_construction import build_space_carving
    from .surface_reconstruction import build_surface_mesh
else:
    from pointcloud_construction import build_pointcloud
    from revolution_construction import build_revolution
    from space_carving_construction import build_space_carving
    from surface_reconstruction import build_surface_mesh

MODES = ("pointcloud", "revolution", "space_carving", "surface_mesh", "all")


def run_specialized_pipeline(
    image_path: Path | str,
    mode: str = "all",
    output_dir: Path | str = Path("out/specialized_recon"),
    resolution: int = 64,
) -> dict[str, Path]:
    """Execute chosen specialized image-to-3D construction."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Target image does not exist: {image_path}")
    if mode not in MODES:
        raise ValueError(f"Unknown reconstruction mode: {mode}")
    if mode in {"space_carving", "surface_mesh", "all"} and resolution < 2:
        raise ValueError("Grid resolution must be at least two.")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    start_time = time.time()
    print("============================================================")
    print(" 3DCV SPECIALIZED IMAGE-TO-3D CONSTRUCTION")
    print(f" Input Image: {image_path.resolve()}")
    print(f" Mode       : {mode}")
    print(f" Output Dir : {output_dir.resolve()}")
    print("============================================================")

    if mode in ("pointcloud", "all"):
        t0 = time.time()
        pcd_out = output_dir / "pointcloud"
        pcd_path = build_pointcloud(image_path, pcd_out)
        results["pointcloud"] = pcd_path
        print(f"-> Point Cloud completed in {time.time() - t0:.2f}s")

    if mode in ("revolution", "all"):
        t0 = time.time()
        rev_out = output_dir / "revolution"
        rev_path = build_revolution(image_path, rev_out)
        results["revolution"] = rev_path
        print(f"-> Surface of Revolution completed in {time.time() - t0:.2f}s")

    if mode in ("space_carving", "all"):
        t0 = time.time()
        sc_out = output_dir / "space_carving"
        sc_path = build_space_carving([image_path], resolution=resolution, output_dir=sc_out)
        results["space_carving"] = sc_path
        print(f"-> Space Carving completed in {time.time() - t0:.2f}s")

    if mode in ("surface_mesh", "all"):
        t0 = time.time()
        sm_out = output_dir / "surface_mesh"
        sm_path = build_surface_mesh(image_path, sm_out, resolution=resolution)
        results["surface_mesh"] = sm_path
        print(f"-> Surface Mesh completed in {time.time() - t0:.2f}s")

    total_time = time.time() - start_time
    print("============================================================")
    print(f" All requested specialized 3D tasks finished in {total_time:.2f}s!")
    for k, v in results.items():
        print(f" - {k.upper():<14}: {v.resolve()}")
    print("============================================================")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", "-i", type=Path, required=True, help="Input RGB/RGBA image path")
    parser.add_argument(
        "--mode", "-m",
        choices=MODES,
        default="all",
        help="Specialized 3D construction technique to apply (default: all)",
    )
    parser.add_argument("--res", "-r", type=int, default=64, help="Grid resolution for volumetric methods")
    parser.add_argument("--out", "-o", type=Path, default=Path("out/specialized_recon"), help="Output directory")
    args = parser.parse_args()

    try:
        run_specialized_pipeline(args.image, mode=args.mode, output_dir=args.out, resolution=args.res)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
