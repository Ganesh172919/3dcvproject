"""Student educational 3D computer vision laboratory router.

Allows students to inspect curriculum labs, run training modules, view loss curves,
and retrieve metrics directly from the 3DFUN AI platform.
"""
from __future__ import annotations

import logging
import threading
from importlib.util import find_spec
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import credits
from ..auth import get_current_user
from ..config import ROOT
from ..db import CreditReason, StudentLabRun, User, get_db, new_id, utcnow
from ..schemas import ApiError, ok

router = APIRouter(prefix="/student-labs", tags=["student-labs"])
log = logging.getLogger(__name__)
lab_slot = threading.Lock()


class LabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DepthConfig(LabConfig):
    epochs: int = Field(4, ge=1, le=20)
    lr: float = Field(0.001, gt=0, le=0.1)
    alpha: float = Field(1.0, ge=0, le=10)
    beta: float = Field(0.5, ge=0, le=10)
    gamma: float = Field(0.2, ge=0, le=10)


class OccupancyConfig(LabConfig):
    primitive: Literal["torus", "sphere", "cylinder", "cube"] = "torus"
    epochs: int = Field(10, ge=1, le=30)
    hidden_dim: int = Field(128, ge=16, le=256)
    lr: float = Field(0.002, gt=0, le=0.1)


class SplatConfig(LabConfig):
    num_points: int = Field(800, ge=100, le=3000)
    iterations: int = Field(25, ge=1, le=100)
    lr: float = Field(0.02, gt=0, le=0.1)


class EvaluationConfig(LabConfig):
    thresholds: tuple[Literal[0.01], Literal[0.05]] = (0.01, 0.05)
    num_points: int = Field(4000, ge=16, le=100_000)
    seed: int = Field(17, ge=0, le=2**32 - 1)


CONFIG_TYPES = {
    "lab1_depth": DepthConfig, "lab2_occupancy": OccupancyConfig,
    "lab3_3dgs": SplatConfig, "lab4_evaluator": EvaluationConfig,
}

LABS_CATALOG = [
    {
        "id": "lab1_depth",
        "title": "Lab 1: Monocular Relative Depth Fine-Tuning",
        "subtitle": "Convolutional encoder-decoder with log-depth, gradient and normal losses",
        "math_formula": r"\mathcal{L} = \alpha \mathcal{L}_{\text{SILog}} + \beta \mathcal{L}_{\text{grad}} + \gamma \mathcal{L}_{\text{normal}}",
        "description": "Fit a depth estimator on generated shapes. Inspect validation MAE, RMSE and delta1. The course TinyDepth experiment uses a separate mesh-disjoint split.",
        "cost_credits": 10,
    },
    {
        "id": "lab2_occupancy",
        "title": "Lab 2: Implicit Neural Representation & Occupancy Network",
        "subtitle": "Continuous 3D MLP field f_theta(x, y, z) -> [0, 1] with Marching Cubes",
        "math_formula": r"\mathcal{L}_{\text{BCE}} = -\sum [y_i \log p_i + (1 - y_i) \log (1 - p_i)], \quad \Phi(x) = \min_{y \in \partial \Omega} \|x - y\|",
        "description": "Learn a continuous 3D implicit representation of CAD primitives (torus, sphere, cylinder, cube) using sinusoidal positional encodings and extract the zero-level set via Marching Cubes.",
        "cost_credits": 10,
    },
    {
        "id": "lab3_3dgs",
        "title": "Lab 3: 3D Gaussian Splatting (3DGS) Fundamentals",
        "subtitle": "Fit circular Gaussian footprints to one image",
        "math_formula": r"\Sigma' = J W \Sigma W^T J^T, \quad C = \sum_i c_i \alpha_i \prod_{j < i} (1 - \alpha_j)",
        "description": "Project 3D points to one camera and fit colors, radii and opacities with an L1 image loss. Export the rendered image and colored point centers.",
        "cost_credits": 10,
    },
    {
        "id": "lab4_evaluator",
        "title": "Lab 4: 3D Geometric Metric Evaluation Suite",
        "subtitle": "Symmetric Chamfer Distance, EMD, and F-Score at thresholds tau=0.01, 0.05",
        "math_formula": r"\text{CD}(S_1, S_2) = \frac{1}{2|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x - y\| + \frac{1}{2|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|",
        "description": "Rigorously benchmark reconstructed 3D meshes and point clouds against ground truth models using KD-Tree nearest neighbor queries and Earth Mover's Distance approximations.",
        "cost_credits": 5,
    },
]


@router.get("")
def list_labs():
    has_torch = find_spec("torch") is not None
    return ok({"labs": [
        {**lab, "default_config": CONFIG_TYPES[lab["id"]]().model_dump(),
         "available": lab["id"] == "lab4_evaluator" or has_torch}
        for lab in LABS_CATALOG
    ]})


def _artifact_names(run: StudentLabRun) -> dict[str, str]:
    return {label: path.replace("\\", "/").rsplit("/", 1)[-1]
            for label, path in (run.artifacts or {}).items() if path}


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
            "artifacts": _artifact_names(r),
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
        "artifacts": _artifact_names(run),
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
    run = db.execute(
        select(StudentLabRun).where(StudentLabRun.id == run_id, StudentLabRun.user_id == user.id)
    ).scalar_one_or_none()
    if run is None:
        raise ApiError("RUN_NOT_FOUND", "Student lab run not found.", status=404)
    folder = (ROOT / "out" / "student_labs" / run_id).resolve()
    file_path = (folder / filename).resolve()
    if filename not in _artifact_names(run).values() or file_path.parent != folder or not file_path.is_file():
        raise ApiError("FILE_NOT_FOUND", f"Artifact '{filename}' not found.", status=404)
    return FileResponse(file_path, filename=filename)


def _fail_lab(db, run, cost):
    run.status = "failed"
    run.logs = "The lab stopped before it finished. The credits were returned. Check the server log for details."
    run.completed_at = utcnow()
    credits.grant(db, run.user_id, cost, reason=CreditReason.REFUND,
                  idempotency_key=f"labrun-refund:{run.id}", meta={"lab_id": run.lab_id}, commit=False)
    db.commit()


def _execute_lab_background(run_id: str, lab_id: str, cfg: dict[str, Any], cost: int):
    try:
        _execute_lab(run_id, lab_id, cfg, cost)
    finally:
        lab_slot.release()


def _execute_lab(run_id: str, lab_id: str, cfg: dict[str, Any], cost: int):
    from ..db import SessionLocal
    with SessionLocal() as db:
        run = db.get(StudentLabRun, run_id)
        if not run:
            return
        run.status = "running"
        db.commit()

        out_dir = ROOT / "out" / "student_labs" / run_id

        try:
            if lab_id == "lab1_depth":
                from training.student_labs.lab1_depth_finetuning.train_student_depth import train_student_depth
                epochs = cfg["epochs"]
                res = train_student_depth(**cfg, out_dir=out_dir)
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
                primitive = cfg["primitive"]
                res = train_occupancy_network(**cfg, out_dir=out_dir)
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
                num_pts = cfg["num_points"]
                res = train_gaussian_splat(**cfg, out_dir=out_dir)
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
                from training.student_labs.lab4_evaluator.evaluate_metrics import run_evaluation
                res = run_evaluation(out_dir=out_dir, num_points=cfg["num_points"], seed=cfg["seed"])
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
            db.commit()
        except Exception:
            log.exception("Student lab %s failed", run_id)
            db.rollback()
            _fail_lab(db, run, cost)


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

    try:
        config = CONFIG_TYPES[lab_id].model_validate(payload.get("config", {})).model_dump(mode="json")
    except ValidationError as exc:
        details = [{"field": ".".join(map(str, err["loc"])), "issue": err["msg"]} for err in exc.errors()]
        raise ApiError("INVALID_LAB_CONFIG", "Check the lab settings.", status=422, details=details) from exc
    if lab_id != "lab4_evaluator" and find_spec("torch") is None:
        raise ApiError("LAB_DEPENDENCY_MISSING", "Install PyTorch using the training guide first.", status=503)
    if not lab_slot.acquire(blocking=False):
        raise ApiError("LAB_BUSY", "A lab is already running. Try again when it finishes.", status=409)

    cost = lab_info["cost_credits"]
    run_id = new_id("lbr")
    try:
        wallet = credits.grant(db, user.id, -cost, reason=CreditReason.STUDENT_LAB,
                               idempotency_key=f"labrun:{run_id}", meta={"lab_id": lab_id}, commit=False)
        balance = credits.available(wallet.balance, wallet.reserved)
        run = StudentLabRun(id=run_id, user_id=user.id, lab_id=lab_id,
                            name=lab_info["title"], status="queued", config=config)
        db.add(run)
        db.commit()
        try:
            threading.Thread(target=_execute_lab_background, args=(run_id, lab_id, config, cost),
                             daemon=True, name=f"lab-runner-{run_id}").start()
        except RuntimeError as exc:
            _fail_lab(db, run, cost)
            raise ApiError("LAB_START_FAILED", "The lab could not start. Credits were returned.", status=503) from exc
    except Exception:
        lab_slot.release()
        raise
    return ok({
        "run_id": run_id,
        "lab_id": lab_id,
        "status": "queued",
        "credits_charged": cost,
        "balance": balance,
        "message": "Lab queued. Watch its status under Recent runs.",
    })
