"""Compare TinyDepth and its loss ablation on the same held-out views."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_evaluation(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with (folder / "per_view.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_view = {(row["mesh"], row["view"]): row for row in rows}
    if not rows or len(rows) != len(by_view) or summary["views"] != len(rows):
        raise ValueError(f"{folder}: missing or duplicate holdout views.")
    return summary, by_view


def compare(full_dir, ablation_dir):
    full, full_views = read_evaluation(full_dir)
    ablation, ablation_views = read_evaluation(ablation_dir)
    settings = ("coordinate_frame", "distance_units", "fscore_threshold", "max_points_per_view")
    if full_views.keys() != ablation_views.keys() or any(full[key] != ablation[key] for key in settings):
        raise ValueError("The two evaluations must use the same test views and metric settings.")

    by_mesh = {}
    for key, row in full_views.items():
        other = ablation_views[key]
        if any(row[field] != other[field] for field in ("points", "constant_chamfer", "constant_f_score")):
            raise ValueError(f"Reference sample or baseline changed for {key[1]}.")
        by_mesh.setdefault(key[0], []).append((row, other))

    def scores(pairs):
        result = {}
        for metric in ("chamfer", "f_score"):
            full_score = mean(float(row[f"tinydepth_{metric}"]) for row, _ in pairs)
            ablation_score = mean(float(other[f"tinydepth_{metric}"]) for _, other in pairs)
            result[metric] = {"full": full_score, "silog_only": ablation_score,
                              "constant_depth": mean(float(row[f"constant_{metric}"]) for row, _ in pairs),
                              "full_minus_silog": full_score - ablation_score}
        return result

    per_mesh = {mesh: scores(pairs) for mesh, pairs in by_mesh.items()}
    averages = {metric: {key: mean(mesh[metric][key] for mesh in per_mesh.values())
                         for key in per_mesh[next(iter(per_mesh))][metric]}
                for metric in ("chamfer", "f_score")}
    return {"schema_version": 1, **{key: full[key] for key in settings}, "views": len(full_views),
            "per_mesh": per_mesh, "mean_by_mesh": averages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.full, args.ablation)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
