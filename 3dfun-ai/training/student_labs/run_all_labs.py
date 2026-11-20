"""3DCV Lab Group 17 — Master Student Laboratory Curriculum Runner.

Executes all 6 PyTorch student laboratories sequentially and generates
a consolidated evaluation report for student coursework submission.

Department of Data Science & Artificial Intelligence, IIIT Dharwad
Course: 3D Computer Vision · Supervisor: Dr. Shirshendu Layek
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("student_labs_master")

OUT_DIR = Path("out/student_labs_master")


def run_all_labs() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    results = {}

    log.info("=" * 70)
    log.info("3DCV Lab Group 17 — Executing All 6 PyTorch Student Laboratories")
    log.info("=" * 70)

    # Lab 1: Depth Fine-Tuning
    log.info("[1/6] Running Lab 1: Monocular Relative Depth Fine-Tuning...")
    try:
        from training.student_labs.lab1_depth_finetuning.train_student_depth import train_student_depth
        res1 = train_student_depth(epochs=2, lr=0.002, out_dir=OUT_DIR / "lab1")
        results["lab1_depth"] = {"status": "success", "metrics": res1}
    except Exception as exc:
        log.warning("Lab 1 skipped/failed: %s", exc)
        results["lab1_depth"] = {"status": "error", "error": str(exc)}

    # Lab 2: Occupancy Network MLP
    log.info("[2/6] Running Lab 2: Implicit Neural Representation & Occupancy Network...")
    try:
        try:
            from training.student_labs.lab2_sdf_occupancy.train_sdf_occupancy import train_occupancy_network
        except ImportError:
            from training.student_labs.lab2_occupancy_network.train_occupancy_sdf import train_occupancy_network
        res2 = train_occupancy_network(primitive="torus", epochs=5, hidden_dim=64, lr=0.005, out_dir=OUT_DIR / "lab2")
        results["lab2_occupancy"] = {"status": "success", "metrics": res2}
    except Exception as exc:
        log.warning("Lab 2 skipped/failed: %s", exc)
        results["lab2_occupancy"] = {"status": "error", "error": str(exc)}

    # Lab 3: 3D Gaussian Splatting (3DGS)
    log.info("[3/6] Running Lab 3: 3D Gaussian Splatting Fundamentals...")
    try:
        from training.student_labs.lab3_3dgs_mini.train_single_view_splat import train_gaussian_splat
        res3 = train_gaussian_splat(num_points=400, iterations=15, lr=0.02, out_dir=OUT_DIR / "lab3")
        results["lab3_3dgs"] = {"status": "success", "metrics": res3}
    except Exception as exc:
        log.warning("Lab 3 skipped/failed: %s", exc)
        results["lab3_3dgs"] = {"status": "error", "error": str(exc)}

    # Lab 4: 3D Metric Evaluation Suite
    log.info("[4/6] Running Lab 4: 3D Geometric Metric Evaluation Suite...")
    try:
        from training.student_labs.lab4_evaluator.evaluate_metrics import run_evaluation
        res4 = run_evaluation(out_dir=OUT_DIR / "lab4", num_points=1000, seed=42)
        results["lab4_evaluator"] = {"status": "success", "metrics": res4}
    except Exception as exc:
        log.warning("Lab 4 skipped/failed: %s", exc)
        results["lab4_evaluator"] = {"status": "error", "error": str(exc)}

    # Lab 5: Feed-Forward Triplane Generator (LRM Mini)
    log.info("[5/6] Running Lab 5: Feed-Forward Triplane Generator (LRM Mini)...")
    try:
        from training.student_labs.lab5_triplane_lrm.train_triplane_lrm import train_triplane_lrm
        res5 = train_triplane_lrm(epochs=3, lr=0.005, primitive="torus", plane_res=16, out_dir=OUT_DIR / "lab5")
        results["lab5_triplane"] = {"status": "success", "metrics": res5}
    except Exception as exc:
        log.warning("Lab 5 skipped/failed: %s", exc)
        results["lab5_triplane"] = {"status": "error", "error": str(exc)}

    # Lab 6: PBR Material Decomposition & Delighting
    log.info("[6/6] Running Lab 6: PBR Material Decomposition & Delighting...")
    try:
        from training.student_labs.lab6_pbr_delighting.pbr_material_decomposition import decompose_pbr_material
        res6 = decompose_pbr_material(filter_radius=5, roughness_base=0.45, metallic_factor=0.1, out_dir=OUT_DIR / "lab6")
        results["lab6_pbr"] = {"status": "success", "metrics": res6}
    except Exception as exc:
        log.warning("Lab 6 skipped/failed: %s", exc)
        results["lab6_pbr"] = {"status": "error", "error": str(exc)}

    total_time = round(time.time() - start_time, 2)
    summary = {
        "institution": "IIIT Dharwad · Department of Data Science & AI",
        "course": "3D Computer Vision · Group 17",
        "supervisor": "Dr. Shirshendu Layek",
        "total_execution_time_seconds": total_time,
        "completed_labs": len([k for k, v in results.items() if v["status"] == "success"]),
        "total_labs": 6,
        "results": results,
    }

    report_path = OUT_DIR / "consolidated_lab_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 70)
    log.info("All Laboratories Finished in %.2fs. Report: %s", total_time, report_path)
    log.info("=" * 70)
    return summary


if __name__ == "__main__":
    run_all_labs()
