"""Record a TinyDepth split and check that every selected view has its labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

SPLITS = ("train", "val", "test")


def mesh_views(folder):
    groups = {}
    for path in sorted((folder / "rgb").glob("*.png")):
        mesh, marker, view = path.stem.rpartition("_view_")
        if not marker or not mesh or not view.isascii() or not view.isdecimal():
            raise ValueError(f"{path.name}: expected MESH_view_NN.png")
        groups.setdefault(mesh, []).append(path.name)
    if len(groups) < 3:
        raise ValueError("The dataset needs at least three mesh stems for train, validation and test.")
    if len({name.casefold() for name in groups}) != len(groups):
        raise ValueError("Use mesh stems that are unique even on a case-insensitive filesystem.")
    return groups


def seeded_mesh_split(stems, seed):
    stems = sorted(stems)
    random.Random(seed).shuffle(stems)
    holdout = max(1, len(stems) // 5)
    return {"train": stems[2 * holdout:], "val": stems[holdout:2 * holdout], "test": stems[:holdout]}


def view_splits(groups, meshes):
    if not isinstance(meshes, dict) or set(meshes) != set(SPLITS):
        raise ValueError("The mesh split must have exactly three keys: train, val and test.")
    assigned = []
    for split in SPLITS:
        names = meshes[split]
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError(f"{split} must contain a non-empty list of mesh stems.")
        assigned.extend(names)
    if len(set(assigned)) != len(assigned):
        raise ValueError("Assign each mesh once to prevent overlapping views across splits.")
    unknown = sorted(set(assigned) - set(groups))
    missing = sorted(set(groups) - set(assigned))
    if unknown or missing:
        raise ValueError(f"Mesh split does not match dataset. Unknown: {unknown}. Unassigned: {missing}.")
    return {split: [view for mesh in meshes[split] for view in groups[mesh]] for split in SPLITS}


def write_split(folder, output, seed, assignment=None):
    groups = mesh_views(folder)
    meshes = (json.loads(assignment.read_text(encoding="utf-8")) if assignment
              else seeded_mesh_split(groups, seed))
    views = view_splits(groups, meshes)
    for mesh, names in groups.items():
        path = folder / "cameras" / f"{mesh}.json"
        cameras = {row["image"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
        for name in names:
            for kind in ("depth", "mask"):
                if not (folder / kind / name).is_file():
                    raise ValueError(f"Missing {kind} label for {name}.")
            camera = cameras.get(name, {})
            if not {"focal_px", "cx", "cy"} <= camera.keys():
                raise ValueError(f"Missing camera intrinsics for {name}.")
            focal, cx, cy = (float(camera[key]) for key in ("focal_px", "cx", "cy"))
            if focal <= 0 or not all(math.isfinite(value) for value in (focal, cx, cy)):
                raise ValueError(f"Invalid camera intrinsics for {name}.")
    record = {"schema_version": 1, "seed": seed, "meshes": meshes, "views": views}
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--assignment", type=Path, help="JSON with train, val and test lists of mesh stems")
    args = parser.parse_args()
    try:
        write_split(args.data_dir, args.output, args.seed, args.assignment)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
