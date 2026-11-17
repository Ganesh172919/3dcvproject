"""Lab 3/4: 3D Metric Evaluation Suite (Chamfer Distance, EMD, F-Score).

Course: 3D Computer Vision · Student Educational Laboratory
Theoretical Foundation:
1. Symmetric Unsquared Chamfer Distance (CD):
    CD(S_1, S_2) = 1/(2 |S_1|) sum_{x in S_1} min_{y in S_2} ||x - y||_2
                 + 1/(2 |S_2|) sum_{y in S_2} min_{x in S_1} ||x - y||_2

2. Precision, Recall & F-Score at threshold tau:
    Precision(tau) = 1/|S_1| sum_{x in S_1} [ min_{y in S_2} ||x - y|| < tau ]
    Recall(tau)    = 1/|S_2| sum_{y in S_2} [ min_{x in S_1} ||x - y|| < tau ]
    F1(tau)        = 2 * (Precision * Recall) / (Precision + Recall)

3. Earth Mover's Distance (EMD):
    EMD(S_1, S_2) = min_{phi: S_1 -> S_2} 1/|S_1| sum_{x in S_1} ||x - phi(x)||_2
"""
from __future__ import annotations

# Re-export from evaluator implementation
from ..lab4_evaluator.evaluate_metrics import (
    sample_mesh_surface,
    compute_chamfer_and_fscore,
    compute_emd_approx as approximate_earth_movers_distance,
    run_evaluation,
)

__all__ = ["sample_mesh_surface", "compute_chamfer_and_fscore", "approximate_earth_movers_distance", "run_evaluation"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lab: 3D Metric Evaluator")
    parser.add_argument("--pred", type=str, default=None)
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./out/evaluation")
    args = parser.parse_args()
    res = run_evaluation(pred_path=args.pred, gt_path=args.gt, out_dir=args.out_dir)
    print("Symmetric Chamfer Distance:", res["chamfer_distance"]["symmetric"])
    print("F-Score (tau=0.01):", res["f_scores"]["tau_0.01"]["f_score"])
