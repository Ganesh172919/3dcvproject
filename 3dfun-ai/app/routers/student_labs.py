"""Student educational 3D computer vision laboratory router.

Allows students to inspect curriculum labs, run training modules, view loss curves,
and retrieve metrics directly from the 3DFUN AI platform.
"""
from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import credits
from ..auth import get_current_user
from ..config import ROOT, settings
from ..db import CreditReason, StudentLabRun, User, get_db, new_id, utcnow
from ..schemas import ApiError, ok

router = APIRouter(prefix="/student-labs", tags=["student-labs"])

LABS_CATALOG = [
    {
        "id": "lab1_depth",
        "title": "Lab 1: Monocular Relative Depth Fine-Tuning",
        "subtitle": "Lightweight CNN/Transformer TinyDepth with SILog, Gradient & Normal Consistency Losses",
        "math_formula": r"\mathcal{L} = \alpha \mathcal{L}_{\text{SILog}} + \beta \mathcal{L}_{\text{grad}} + \gamma \mathcal{L}_{\text{normal}}",
        "default_config": {"epochs": 4, "lr": 0.001, "alpha": 1.0, "beta": 0.5, "gamma": 0.2},
        "description": "Train and fine-tune a lightweight depth estimator. Calculates scale-invariant logarithmic loss, multiscale boundary gradients, and normal consistency, outputting MAE, RMSE, and delta1 < 1.25 metrics with ONNX export.",
        "cost_credits": 10,
    },
    {
        "id": "lab2_occupancy",
        "title": "Lab 2: Implicit Neural Representation & Occupancy Network",
        "subtitle": "Continuous 3D MLP field f_theta(x, y, z) -> [0, 1] with Marching Cubes",
        "math_formula": r"\mathcal{L}_{\text{BCE}} = -\sum [y_i \log p_i + (1 - y_i) \log (1 - p_i)], \quad \Phi(x) = \min_{y \in \partial \Omega} \|x - y\|",
        "default_config": {"primitive": "torus", "epochs": 10, "lr": 0.002, "hidden_dim": 128},
        "description": "Learn a continuous 3D implicit representation of CAD primitives (torus, sphere, cylinder, cube) using sinusoidal positional encodings and extract the zero-level set via Marching Cubes.",
        "cost_credits": 10,
    },
    {
        "id": "lab3_3dgs",
        "title": "Lab 3: 3D Gaussian Splatting (3DGS) Fundamentals",
        "subtitle": "Differentiable 2D ray projection and alpha-compositing optimization",
        "math_formula": r"\Sigma' = J W \Sigma W^T J^T, \quad C = \sum_i c_i \alpha_i \prod_{j < i} (1 - \alpha_j)",
        "default_config": {"num_points": 800, "iterations": 25, "lr": 0.02},
        "description": "Optimize 3D Gaussians (means, covariance matrices, and opacities) through differentiable 2D footprint splatting and alpha blending to reconstruct reference images, exporting PLY point clouds.",
        "cost_credits": 10,
    },
    {
        "id": "lab4_evaluator",
        "title": "Lab 4: 3D Geometric Metric Evaluation Suite",
        "subtitle": "Symmetric Chamfer Distance, EMD, and F-Score at thresholds tau=0.01, 0.05",
        "math_formula": r"\text{CD}(S_1, S_2) = \frac{1}{2|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x - y\| + \frac{1}{2|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|",
        "default_config": {"thresholds": [0.01, 0.05]},
        "description": "Rigorously benchmark reconstructed 3D meshes and point clouds against ground truth models using KD-Tree nearest neighbor queries and Earth Mover's Distance approximations.",
        "cost_credits": 5,
    },
]


@router.get("")
def list_labs():
    return ok({"labs": LABS_CATALOG})


@router.get("/runs")
def list_runs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    runs = db.execute(
        select(StudentLabRun).where(StudentLabRun.user_id == user.id).order_by(StudentLabRun.created_at.desc())
    ).scalars().all()
    return ok([
        {
            "id": r.id,
            "lab_id": r.lab_id,
            "name": r.name,
            "status": r.status,
            "metrics": r.metrics,
            "artifacts": r.artifacts,
            "created_at": r.created_at,
            "completed_at": r.completed_at,
        }
        for r in runs
    ])


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.execute(
        select(StudentLabRun).where(StudentLabRun.id == run_id, StudentLabRun.user_id == user.id)
    ).scalar_one_or_none()
    if run is None:
        raise ApiError("RUN_NOT_FOUND", "Student lab run not found.", status=404)
    return ok({
        "id": run.id,
        "lab_id": run.lab_id,
        "name": run.name,
        "status": run.status,
        "config": run.config,
        "metrics": run.metrics,
        "logs": run.logs,
        "artifacts": run.artifacts,
        "created_at": run.created_at,
        "completed_at": run.completed_at,
    })


@router.get("/runs/{run_id}/artifacts/{filename}")
def download_lab_artifact(
    run_id: str,
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from fastapi.responses import FileResponse
    run = db.execute(
        select(StudentLabRun).where(StudentLabRun.id == run_id, StudentLabRun.user_id == user.id)
    ).scalar_one_or_none()
    if run is None:
        raise ApiError("RUN_NOT_FOUND", "Student lab run not found.", status=404)
    file_path = ROOT / "out" / "student_labs" / run_id / filename
    if not file_path.is_file():
        raise ApiError("FILE_NOT_FOUND", f"Artifact '{filename}' not found.", status=404)
    return FileResponse(file_path, filename=filename)


def _execute_lab_background(run_id: str, lab_id: str, cfg: dict[str, Any]):
    from ..db import SessionLocal
    with SessionLocal() as db:
        run = db.get(StudentLabRun, run_id)
        if not run:
            return
        run.status = "running"
        db.commit()

        out_dir = ROOT / "out" / "student_labs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if lab_id == "lab1_depth":
                from training.student_labs.lab1_depth_finetuning.train_student_depth import train_student_depth
                epochs = int(cfg.get("epochs", 4))
                lr = float(cfg.get("lr", 1e-3))
                res = train_student_depth(epochs=epochs, lr=lr, out_dir=out_dir)
                run.metrics = {
                    "final_loss": res.get("final_loss"),
                    "mae": res.get("final_mae"),
                    "delta1": res.get("final_delta1"),
                    "epochs": epochs,
                }
                run.artifacts = {
                    "checkpoint": res.get("checkpoint"),
                    "onnx": res.get("onnx"),
                    "results_json": str(out_dir / "lab1_results.json"),
                }
                run.logs = f"Training completed successfully in {epochs} epochs.\nFinal MAE: {res.get('final_mae'):.4f}, delta1: {res.get('final_delta1'):.3f}"

            elif lab_id in ("lab2_occupancy", "lab2_sdf"):
                try:
                    from training.student_labs.lab2_sdf_occupancy.train_sdf_occupancy import train_occupancy_network
                except ImportError:
                    from training.student_labs.lab2_occupancy_network.train_occupancy_sdf import train_occupancy_network
                primitive = str(cfg.get("primitive", "torus"))
                epochs = int(cfg.get("epochs", 8))
                res = train_occupancy_network(primitive=primitive, epochs=epochs, out_dir=out_dir)
                run.metrics = {
                    "final_loss": res.get("final_loss"),
                    "accuracy": res.get("final_accuracy"),
                    "iou": res.get("final_iou"),
                    "vertices": res.get("vertices"),
                    "faces": res.get("faces"),
                }
                run.artifacts = {
                    "mesh": res.get("mesh_path"),
                    "results_json": str(out_dir / "lab2_results.json"),
                }
                run.logs = f"Implicit occupancy field trained for {primitive}.\nIoU: {res.get('final_iou'):.3f}, Accuracy: {res.get('final_accuracy')*100:.1f}%"

            elif lab_id == "lab3_3dgs":
                from training.student_labs.lab3_3dgs_mini.train_single_view_splat import train_gaussian_splat
                num_pts = int(cfg.get("num_points", 800))
                iters = int(cfg.get("iterations", 25))
                res = train_gaussian_splat(num_points=num_pts, iterations=iters, out_dir=out_dir)
                run.metrics = {
                    "num_gaussians": num_pts,
                    "final_loss": res.get("final_loss"),
                    "final_psnr": res.get("final_psnr"),
                }
                run.artifacts = {
                    "render_png": res.get("render_path"),
                    "ply_cloud": res.get("ply_path"),
                    "results_json": str(out_dir / "lab3_results.json"),
                }
                run.logs = f"3D Gaussian Splatting complete ({num_pts} Gaussians).\nFinal PSNR: {res.get('final_psnr'):.2f} dB"

            elif lab_id in ("lab3_metrics_evaluator", "lab4_evaluator"):
                try:
                    from training.student_labs.lab3_metrics_evaluator.evaluate_metrics import run_evaluation
                except ImportError:
                    from training.student_labs.lab4_evaluator.evaluate_metrics import run_evaluation
                res = run_evaluation(out_dir=out_dir)
                run.metrics = {
                    "chamfer_symmetric": res.get("chamfer_distance", {}).get("symmetric"),
                    "f_score_tau_01": res.get("f_scores", {}).get("tau_0.01", {}).get("f_score"),
                    "f_score_tau_05": res.get("f_scores", {}).get("tau_0.05", {}).get("f_score"),
                    "emd": res.get("earth_movers_distance"),
                }
                run.artifacts = {
                    "results_json": str(out_dir / "lab4_results.json"),
                }
                run.logs = f"3D Evaluation benchmarks computed.\nChamfer Distance: {run.metrics.get('chamfer_symmetric'):.5f}, EMD: {run.metrics.get('emd'):.5f}"

            run.status = "completed"
            run.completed_at = utcnow()
        except Exception as exc:
            run.status = "failed"
            run.logs = f"Lab execution failed: {exc}"
            run.completed_at = utcnow()
        finally:
            db.commit()


@router.post("/{lab_id}/run")
def trigger_lab_run(
    lab_id: str,
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lab_info = next((item for item in LABS_CATALOG if item["id"] == lab_id), None)
    if not lab_info:
        raise ApiError("UNKNOWN_LAB", f"Lab '{lab_id}' does not exist.", status=404)

    cost = lab_info["cost_credits"]
    wallet = credits.get_wallet(db, user.id)
    if credits.available(wallet.balance, wallet.reserved) < cost:
        raise ApiError("INSUFFICIENT_CREDITS", f"You need {cost} credits to launch this student training run.", status=402)

    # Debit credits for lab run
    run_id = new_id("lbr")
    credits.grant(db, user.id, -cost, reason=CreditReason.STUDENT_LAB,
                  idempotency_key=f"labrun:{run_id}", meta={"lab_id": lab_id})

    merged_config = dict(lab_info["default_config"])
    if isinstance(payload.get("config"), dict):
        merged_config.update(payload["config"])

    run = StudentLabRun(
        id=run_id,
        user_id=user.id,
        lab_id=lab_id,
        name=f"{lab_info['title']} Run",
        status="queued",
        config=merged_config,
    )
    db.add(run)
    db.commit()

    # Launch background training execution thread
    threading.Thread(
        target=_execute_lab_background,
        args=(run_id, lab_id, merged_config),
        daemon=True,
        name=f"lab-runner-{run_id}",
    ).start()

    return ok({
        "run_id": run_id,
        "lab_id": lab_id,
        "status": "queued",
        "credits_charged": cost,
        "balance": wallet.balance - cost,
        "message": f"{lab_info['title']} queued successfully. You can monitor progress in the Student Laboratory tab.",
    })
