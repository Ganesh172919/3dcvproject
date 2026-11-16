"""Generation endpoints: estimate, create, list, detail, cancel, retry, events."""
from __future__ import annotations


from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .. import credits, events, metrics
from ..auth import get_current_user, owned_generation
from ..config import settings
from ..db import (AuditLog, Generation, GenerationEvent, GenerationStatus, Provider,
                  Upload, User, get_db, utcnow)
from .. import providers
from ..jobs import runner
from ..ratelimit import enforce, generation_limiter
from ..schemas import (ApiError, CreateGenerationRequest, EstimateRequest, ok)
from ..serializers import generation_public
from ..state_machine import assert_transition

router = APIRouter(prefix="/generations", tags=["generations"])


def _queue_position(db: Session) -> int:
    return db.execute(select(func.count()).select_from(Generation).where(
        Generation.status.in_([GenerationStatus.QUEUED]))).scalar_one()


def _pick_provider(db: Session, privacy_mode: bool, engine: str | None = None) -> str:
    """Weighted provider routing with health fallback (recorded in metadata)."""
    if engine:
        return engine
    if settings.generation_provider and settings.generation_provider not in {"local", ""} \
            and settings.environment in {"test", "development"}:
        # A single explicit provider wins in dev/test (e.g. GENERATION_PROVIDER=mock).
        return settings.generation_provider
    rows = db.execute(select(Provider).where(Provider.enabled.is_(True))
                      .order_by(Provider.weight.desc())).scalars().all()
    if not rows:
        return settings.generation_provider
    if privacy_mode:
        no_retention = [p for p in rows if "no-retention" in (p.notes or "")]
        if no_retention:
            return no_retention[0].slug
    for provider in rows:
        if provider.healthy:
            return provider.slug
    return rows[0].slug


@router.post("/estimate")
def estimate(payload: EstimateRequest, user: User = Depends(get_current_user),
             db: Session = Depends(get_db)):
    provider_slug = _pick_provider(db, privacy_mode=False, engine=payload.options.engine)
    provider = providers.get_provider(provider_slug)
    cost = provider.estimate_cost(payload.quality, payload.options.model_dump())
    wallet = credits.get_wallet(db, user.id)
    return ok({
        "credits": cost.credits,
        "balance": wallet.balance,
        "available": credits.available(wallet.balance, wallet.reserved),
        "estimated_seconds": cost.estimated_seconds,
        "provider": cost.provider,
        "model": getattr(provider, "model_name", settings.generation_model),
        "queue_position": _queue_position(db),
    })


@router.post("")
def create_generation(payload: CreateGenerationRequest, request: Request,
                      user: User = Depends(get_current_user),
                      db: Session = Depends(get_db),
                      idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    enforce(generation_limiter, user.id, "generation")
    if not payload.accept_terms:
        raise ApiError("TERMS_REQUIRED", "You must accept the generation terms.", status=422)

    key = payload.idempotency_key or idempotency_key
    if key:
        existing = db.execute(select(Generation).where(
            Generation.user_id == user.id,
            Generation.idempotency_key == key)).scalar_one_or_none()
        if existing is not None:
            return ok(generation_public(existing))

    upload = db.get(Upload, payload.upload_id)
    if upload is None or upload.user_id != user.id:
        raise ApiError("UPLOAD_NOT_FOUND", "Upload the reference image first.", status=404)

    provider_slug = _pick_provider(db, payload.privacy_mode, payload.options.engine)
    # resolve through the module attribute so provider routing stays patchable
    provider = providers.get_provider(provider_slug)
    cost = provider.estimate_cost(payload.quality, payload.options.model_dump())

    gen = Generation(
        user_id=user.id, upload_id=upload.id, provider_slug=provider_slug,
        model=getattr(provider, "model_name", settings.generation_model),
        quality=payload.quality, preset=payload.preset,
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        options=payload.options.model_dump(), privacy_mode=payload.privacy_mode,
        status=GenerationStatus.QUEUED, stage="QUEUED", progress=0,
        idempotency_key=key, queue_position=_queue_position(db) + 1,
        config={"provider_reason": "weighted-routing", "estimated_credits": cost.credits},
    )
    db.add(gen)
    db.flush()
    credits.reserve(db, user.id, gen, cost.credits)     # raises 402 when short
    db.add(AuditLog(actor_id=user.id, action="generation.create", target_type="generation",
                    target_id=gen.id, ip=request.client.host if request.client else ""))
    db.commit()
    events.record_event(db, gen.id, "generation.queued", "QUEUED", 0,
                        {"creditsReserved": cost.credits, "provider": provider_slug})
    metrics.inc("generation_created_total", quality=payload.quality, provider=provider_slug)
    gen.queue_position = 0
    db.commit()
    runner.enqueue(gen.id)
    return ok(generation_public(gen))


@router.get("")
def list_generations(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                     page: int = 1, page_size: int = 20, status: str | None = None,
                     q: str | None = None):
    page = max(1, page)
    page_size = min(max(page_size, 1), 100)
    stmt = select(Generation).where(Generation.user_id == user.id)
    count_stmt = select(func.count()).select_from(Generation).where(Generation.user_id == user.id)
    if status:
        try:
            wanted = GenerationStatus(status.upper())
        except ValueError:
            raise ApiError("INVALID_FILTER", f"Unknown status '{status}'.", status=422)
        stmt = stmt.where(Generation.status == wanted)
        count_stmt = count_stmt.where(Generation.status == wanted)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(func.lower(Generation.prompt).like(like))
        count_stmt = count_stmt.where(func.lower(Generation.prompt).like(like))
    total = db.execute(count_stmt).scalar_one()
    rows = db.execute(stmt.options(selectinload(Generation.asset)).order_by(Generation.created_at.desc())
                      .offset((page - 1) * page_size).limit(page_size)).scalars().all()
    source_ids = {g.upload_id for g in rows if g.upload_id}
    uploads = {u.id: u for u in db.scalars(select(Upload).where(
        Upload.id.in_(source_ids), Upload.user_id == user.id,
    ))} if source_ids else {}
    return ok({"items": [generation_public(g, reference=uploads.get(g.upload_id)) for g in rows], "total": total,
               "page": page, "page_size": page_size})


@router.get("/{generation_id}")
def get_generation(generation_id: str, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    return ok(generation_public(gen))


@router.post("/{generation_id}/cancel")
def cancel_generation(generation_id: str, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    if gen.status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED,
                      GenerationStatus.OPTIMIZING, GenerationStatus.RENDERING_PREVIEW):
        raise ApiError("NOT_CANCELLABLE", "This generation already finished.", status=409)
    gen.cancel_requested = True
    if gen.status in (GenerationStatus.QUEUED, GenerationStatus.VALIDATING):
        credits.release(db, gen.user_id, gen)
        gen.status = GenerationStatus.CANCELLED
        gen.stage = "CANCELLED"
        gen.completed_at = utcnow()
        db.commit()
        events.record_event(db, gen.id, "generation.cancelled", "CANCELLED", gen.progress)
    else:
        gen.status = GenerationStatus.CANCEL_REQUESTED
        db.commit()
        events.record_event(db, gen.id, "generation.cancel_requested", "CANCEL_REQUESTED",
                            gen.progress)
    return ok(generation_public(gen))


@router.post("/{generation_id}/retry")
def retry_generation(generation_id: str, request: Request,
                     user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    if gen.status not in (GenerationStatus.FAILED, GenerationStatus.CANCELLED):
        raise ApiError("NOT_RETRYABLE", "Only failed or cancelled jobs can be retried.", status=409)
    cost = providers.get_provider(gen.provider_slug).estimate_cost(gen.quality, gen.options or {})
    assert_transition(gen.status, GenerationStatus.QUEUED)
    gen.status = GenerationStatus.QUEUED
    gen.stage = "QUEUED"
    gen.progress = 0
    gen.error_code = None
    gen.error_message = None
    gen.cancel_requested = False
    gen.completed_at = None
    credits.reserve(db, gen.user_id, gen, cost.credits)
    events.record_event(db, gen.id, "generation.queued", "QUEUED", 0, {"retry": True})
    db.add(AuditLog(actor_id=user.id, action="generation.retry", target_type="generation",
                    target_id=gen.id))
    db.commit()
    runner.enqueue(gen.id)
    return ok(generation_public(gen))


@router.get("/{generation_id}/events")
def list_events(generation_id: str, after: int = 0, limit: int = 200,
                user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    rows = db.execute(select(GenerationEvent).where(
        GenerationEvent.generation_id == gen.id, GenerationEvent.sequence > after)
        .order_by(GenerationEvent.sequence.asc()).limit(min(limit, 500))).scalars().all()
    from ..events import serialize
    return ok([serialize(r) for r in rows])


@router.get("/{generation_id}/attempts")
def list_attempts(generation_id: str, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    gen = owned_generation(db, generation_id, user)
    return ok([{
        "id": a.id, "attempt_no": a.attempt_no, "status": a.status,
        "provider": a.provider_slug, "error_code": a.error_code,
        "error_message": a.error_message, "metrics": a.metrics,
        "started_at": a.started_at, "finished_at": a.finished_at,
        "duration_ms": a.duration_ms,
    } for a in gen.attempts])


@router.get("/{generation_id}/stages")
def get_generation_stages(generation_id: str, user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """Fetch structured intermediate representations for student inspection across all 8 pipeline stages."""
    gen = owned_generation(db, generation_id, user)
    asset = gen.asset
    manifest = asset.manifest if asset else {}
    files = asset.files if asset else {}

    def get_url(filename: str) -> str | None:
        if filename in files:
            return f"/api/v1/assets/{asset.id}/download/{filename}"
        for k, v in files.items():
            if k.endswith(filename):
                return f"/api/v1/assets/{asset.id}/download/{filename}"
        return None

    stages = [
        {
            "stage_index": 1,
            "id": "preprocessing",
            "name": "1. Preprocessing & Alpha Matting",
            "description": "Raw 2D input -> Saliency Boundary Probability Map -> Trimap -> Alpha Cutoff Segmentation",
            "math_formula": r"\mathcal{L}_{\text{BCE}} = -\sum_{i} [y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i)]",
            "primary_asset": get_url("source_canvas.png") or get_url("mask.png"),
            "assets": {
                "source": get_url("source_canvas.png"),
                "mask": get_url("mask.png"),
                "prob_map": get_url("prob_map.png"),
                "trimap": get_url("trimap.png"),
            },
            "parameters": {"alpha_cutoff": 0.5, "feather_radius": 2, "padding": "12%"},
            "telemetry": manifest.get("input", {}),
        },
        {
            "stage_index": 2,
            "id": "depth",
            "name": "2. Monocular Relative Depth Estimation",
            "description": "Depth Anything V2 / MiDaS relative inverse depth field & 2D Sobel spatial gradients",
            "math_formula": r"Z(u, v) = \frac{1}{\alpha \cdot d(u, v) + \beta}, \quad \nabla d = \left( \frac{\partial d}{\partial x}, \frac{\partial d}{\partial y} \right)",
            "primary_asset": get_url("depth.png") or get_url("depth_gray.png"),
            "assets": {
                "color_map": get_url("depth.png"),
                "grayscale": get_url("depth_gray.png"),
                "depth_16bit": get_url("depth_16bit.png"),
                "normals": get_url("normals.png"),
                "sobel_normals": get_url("sobel_gradients.png"),
            },
            "parameters": {"model": gen.model, "depth_range": "[0, 1]", "sobel_ksize": 3},
            "telemetry": manifest.get("engine", {}),
        },
        {
            "stage_index": 3,
            "id": "unprojection",
            "name": "3. Camera Ray Unprojection & Point Cloud",
            "description": "Back-projecting 2D (u, v) pixels into 3D camera coordinate space (X, Y, Z) via intrinsic matrix K",
            "math_formula": r"X = \frac{(u - c_x) Z}{f_x}, \quad Y = \frac{(v - c_y) Z}{f_y}, \quad Z = \text{depth}(u, v)",
            "primary_asset": get_url("pointcloud.ply"),
            "assets": {"pointcloud_ply": get_url("pointcloud.ply")},
            "parameters": {"fov_deg": 45.0, "focal_length": 180.0, "density": "100%"},
            "telemetry": {"point_count": 20000, "coordinate_system": "Right-handed (+X right, +Y up, +Z out)"},
        },
        {
            "stage_index": 4,
            "id": "volume",
            "name": "4. Volumetric Inflation & Signed Distance Fields (SDF)",
            "description": "Silhouette-constrained 3D voxel density matrix Phi(x, y, z) with Euclidean distance transform",
            "math_formula": r"\Phi(\mathbf{x}) = \min_{\mathbf{y} \in \partial \Omega} \|\mathbf{x} - \mathbf{y}\| \cdot \text{sgn}(\mathbf{x})",
            "primary_asset": get_url("sdf_axial.png"),
            "assets": {
                "voxels_slice": get_url("voxels_slice.png"),
                "axial_slice": get_url("sdf_axial.png"),
                "coronal_slice": get_url("sdf_coronal.png"),
                "sagittal_slice": get_url("sdf_sagittal.png"),
            },
            "parameters": {"voxel_resolution": "128³", "closeness_power": 1.4},
            "telemetry": manifest.get("volume", {}),
        },
        {
            "stage_index": 5,
            "id": "geometry",
            "name": "5. Isosurface Extraction via Marching Cubes",
            "description": "Lorensen & Cline lookup table triangle polygonization with Euler topology verification",
            "math_formula": r"\mathcal{S} = \{ \mathbf{x} \in \mathbb{R}^3 \mid \Phi(\mathbf{x}) = 0.0 \}, \quad \chi = V - E + F",
            "primary_asset": get_url("raw_mesh.obj") or get_url("model.glb"),
            "assets": {"raw_mesh_obj": get_url("raw_mesh.obj"), "mesh_glb": get_url("model.glb"), "mesh_obj": get_url("model.obj")},
            "parameters": {"isovalue": 0.0, "lookup_tables": "256 configurations (15 canonical)"},
            "telemetry": manifest.get("mesh", {}),
        },
        {
            "stage_index": 6,
            "id": "smoothing",
            "name": "6. Mesh Post-Processing & Smoothing",
            "description": "Taubin dual-pass non-shrinking smoothing vs Laplacian smoothing with Quadric Error Metric (QEM)",
            "math_formula": r"\mathbf{v}' = (\mathbf{I} + \mu \mathbf{L})(\mathbf{I} + \lambda \mathbf{L}) \mathbf{v}, \quad \lambda > 0, \mu < -\lambda",
            "primary_asset": get_url("smoothed.obj") or get_url("model.glb"),
            "assets": {"smoothed_obj": get_url("smoothed.obj"), "model_glb": get_url("model.glb")},
            "parameters": {"taubin_lambda": 0.5, "taubin_mu": -0.53, "iterations": 4},
            "telemetry": {"smoothing_algorithm": "Taubin filter", "faces": asset.polygons if asset else 0},
        },
        {
            "stage_index": 7,
            "id": "uv",
            "name": "7. UV Parameterization & Seam Packing",
            "description": "Planar/conformal unwrap of 3D triangular surface onto 2D texture coordinate atlas (u, v)",
            "math_formula": r"(u, v) \in [0, 1]^2, \quad \iint \left( \left\|\frac{\partial \mathbf{x}}{\partial u}\right\|^2 + \left\|\frac{\partial \mathbf{x}}{\partial v}\right\|^2 \right) du\,dv",
            "primary_asset": get_url("uv_layout.png") or get_url("uv_atlas.png"),
            "assets": {"uv_layout_png": get_url("uv_layout.png"), "uv_atlas_png": get_url("uv_atlas.png")},
            "parameters": {"atlas_resolution": "1024x1024", "padding_pixels": 4},
            "telemetry": {"atlas_type": manifest.get("mesh", {}).get("uv_atlas", "planar-view-projection")},
        },
        {
            "stage_index": 8,
            "id": "texture",
            "name": "8. PBR Texture Back-Projection & Baking",
            "description": "Color back-projection with Lambertian de-lighting, AO hemisphere integration, and normal perturbation",
            "math_formula": r"L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) (\mathbf{n} \cdot \omega_i) d\omega_i",
            "primary_asset": get_url("albedo.png") or get_url("basecolor.png") or get_url("preview.png"),
            "assets": {
                "albedo": get_url("albedo.png") or get_url("basecolor.png"),
                "basecolor": get_url("basecolor.png"),
                "normal": get_url("normal.png"),
                "orm": get_url("orm.png"),
                "ao": get_url("ao.png"),
                "roughness": get_url("roughness.png"),
                "metallic": get_url("metallic.png"),
            },
            "parameters": {"pbr_workflow": "Metallic-Roughness", "delighting": True},
            "telemetry": manifest.get("texture", {}),
        },
    ]

    return ok({"generation_id": gen.id, "stages": stages})
