"""ORM -> API payload serialisers (single source for every response shape)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import Asset, CreditTransaction, Generation, Upload, User
from .storage import storage
from .state_machine import is_retryable

DOWNLOAD_LABELS = {
    "glb": "GLB",
    "gltf": "glTF bundle",
    "obj": "OBJ bundle",
    "ply": "PLY (binary, with normals)",
    "stl": "STL (geometry only)",
    "png": "PNG texture atlas",
    "mp4": "Turntable video",
    "gif": "Turntable GIF",
}


def _asset_files(asset: Asset) -> dict[str, str]:
    """flat {relative path: storage key} for the asset (rebuilt when absent)."""
    keys: dict[str, str] = getattr(asset, "_file_keys", {}) or {}
    if keys:
        return keys
    stored = asset.files or {}
    if all(isinstance(v, str) and "/" in v for v in stored.values()):
        return dict(stored)
    return {rel: f"{asset.storage_prefix}/{rel}" for rel in stored}


def download_urls(asset: Asset) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = _asset_files(asset)
    wanted = [("model.glb", "glb"), ("model.gltf", "gltf"), ("model.obj", "obj"),
              ("model.ply", "ply"), ("model.stl", "stl"), ("basecolor.png", "png"),
              (("turntable.mp4", "mp4") if "turntable.mp4" in keys else ("turntable.gif", "gif"))]
    for filename, fmt in wanted:
        key = keys.get(filename)
        if not key:
            continue
        bundled = fmt in {"obj", "gltf"}
        companions = ("model.mtl" if fmt == "obj" else "model.bin", "basecolor.png", "normal.png", "orm.png")
        size = storage.size_of_prefix(key)
        if bundled:
            size += sum(storage.size_of_prefix(keys[name]) for name in companions if name in keys)
        entry = {
            "format": fmt,
            "label": DOWNLOAD_LABELS.get(fmt, fmt.upper()),
            "size_bytes": size,
            "url": f"/api/v1/assets/{asset.id}/download?format={fmt}" + ("&bundle=true" if bundled else ""),
        }
        out.append(entry)
    return out


def upload_public(upload: Upload) -> dict[str, Any]:
    return {
        "id": upload.id,
        "filename": upload.filename,
        "mime": upload.mime,
        "size_bytes": upload.size_bytes,
        "width": upload.width,
        "height": upload.height,
        "sha256": upload.sha256,
        "created_at": upload.created_at,
        "image_url": f"/api/v1/uploads/{upload.id}/image",
    }


def generation_public(gen: Generation, asset_id: str | None = None, reference: Upload | None = None) -> dict[str, Any]:
    asset = gen.asset
    manifest = (asset.manifest or {}) if asset else {}
    return {
        "id": gen.id,
        "status": gen.status.value if hasattr(gen.status, "value") else str(gen.status),
        "stage": gen.stage,
        "progress": gen.progress,
        "quality": gen.quality,
        "preset": gen.preset,
        "prompt": gen.prompt,
        "credits_reserved": gen.credits_reserved,
        "credits_charged": gen.credits_charged,
        "privacy_mode": gen.privacy_mode,
        "queue_position": gen.queue_position,
        "error_code": gen.error_code,
        "error_message": gen.error_message,
        "retryable": is_retryable(gen.error_code),
        "support_code": gen.support_code,
        "provider": gen.provider_slug,
        "model": gen.model,
        "created_at": gen.created_at,
        "started_at": gen.started_at,
        "completed_at": gen.completed_at,
        "duration_ms": gen.duration_ms,
        "asset_id": asset_id or (asset.id if asset and not asset.deleted_at else None),
        "upload_id": gen.upload_id,
        "reference": {"name": reference.filename, "digest": reference.sha256} if reference else None,
        "measurement": {
            "triangles": asset.polygons,
            "watertight": asset.watertight,
            "bytes": asset.size_bytes,
            "hardware": manifest.get("hardware"),
            "config_hash": manifest.get("config_hash"),
        } if asset and not asset.deleted_at else None,
        "source_image_url": f"/api/v1/uploads/{gen.upload_id}/image" if gen.upload_id else None,
        "options": gen.options or {},
    }


def asset_public(asset: Asset, owner: User | None = None, include_files: bool = True) -> dict[str, Any]:
    keys = _asset_files(asset)

    def url(key: str | None) -> str | None:
        if not key:
            return None
        return f"/api/v1/assets/{asset.id}/file?name={Path(key).name}"

    return {
        "id": asset.id,
        "name": asset.name,
        "description": asset.description,
        "tags": asset.tags or [],
        "visibility": asset.visibility.value if hasattr(asset.visibility, "value") else str(asset.visibility),
        "owner_id": asset.user_id,
        "owner_name": (owner.display_name or owner.email.split("@")[0]) if owner else "unknown",
        "polygons": asset.polygons,
        "vertices": asset.vertices,
        "watertight": asset.watertight,
        "size_bytes": asset.size_bytes,
        "like_count": asset.like_count,
        "view_count": asset.view_count,
        "featured": asset.featured,
        "version": asset.version,
        "share_slug": asset.share_slug,
        "preview_url": url(asset.preview_key),
        "cover_url": url(asset.cover_key),
        "turntable_url": url(keys.get("turntable.gif") or asset.turntable_key),
        "model_url": url(keys.get("model.glb")) if keys.get("model.glb") else None,
        "downloads": download_urls(asset) if include_files else [],
        "metrics": flatten_metrics(asset.metrics, asset.manifest if isinstance(asset.manifest, dict) else None),
        "manifest": {
            "engine": (asset.manifest or {}).get("engine", {}),
            "timings": (asset.manifest or {}).get("timings_seconds", {}),
            "warnings": (asset.manifest or {}).get("warnings", []),
            "hardware": (asset.manifest or {}).get("hardware"),
            "limitations": (asset.manifest or {}).get("limitations", []),
            "config_hash": (asset.manifest or {}).get("config_hash"),
            "resolution_calibration": (asset.manifest or {}).get("resolution_calibration", {}),
        },
        "created_at": asset.created_at,
        "generation_id": asset.generation_id,
    }


def transaction_public(tx: CreditTransaction) -> dict[str, Any]:
    return {
        "id": tx.id,
        "reason": tx.reason.value if hasattr(tx.reason, "value") else str(tx.reason),
        "amount": tx.amount,
        "balance_after": tx.balance_after,
        "generation_id": tx.generation_id,
        "created_at": tx.created_at,
        "metadata": tx.metadata_json or {},
    }


def user_public(user: User, credits: int = 0) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "email_verified": user.email_verified,
        "mfa_enabled": user.mfa_enabled,
        "created_at": user.created_at,
        "credits": credits,
    }


# Fixed-width keys the UI reads straight out of the (nested) engine manifest so
# the studio never has to know the manifest layout.
_TOKEN_KEYS = {"aspect_ratio_mean", "aspect_ratio_p95", "aspect_ratio_p99",
               "boundary_edges", "nonmanifold_edges", "degenerate_faces",
               "surface_area", "euler_characteristic", "watertight",
               "edge_length_median", "edge_length_max", "volume",
               "signed_volume", "textured", "uv_atlas", "uv_corner_count"}
_TIME_KEYS = {"total", "render", "geometry", "texture", "depth", "export",
              "preprocessing", "volume", "uv"}


def flatten_metrics(raw: dict[str, Any] | None, manifest_raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Split the engine manifest into flat UI numbers and a bounded raw block."""
    metrics = dict(raw or {})
    flat: dict[str, Any] = {}
    for key, value in list(metrics.items()):
        if isinstance(value, (int, float, bool)) or value is None:
            flat[key] = value
    for key, value in list(metrics.items()):
        if isinstance(value, dict):
            for sub_key in _TOKEN_KEYS:
                if sub_key in value and isinstance(value[sub_key], (int, float, bool)):
                    flat[sub_key] = value[sub_key]
            if key == "timings" or set(value) & _TIME_KEYS:
                for time_key in _TIME_KEYS:
                    if time_key in value and isinstance(value[time_key], (int, float)):
                        flat[f"timings.{time_key}"] = round(float(value[time_key]), 3)
    manifest = (manifest_raw if isinstance(manifest_raw, dict)
                else metrics.get("manifest") if isinstance(metrics.get("manifest"), dict) else {})
    timings = (manifest.get("timings_seconds") or manifest.get("timings")
               or metrics.get("timings") or {})
    if isinstance(timings, dict):
        total = timings.get("total")
        if isinstance(total, (int, float)):
            flat["total_seconds"] = round(float(total), 3)
        for stage in ("depth", "volume", "geometry", "texture", "export", "render"):
            if isinstance(timings.get(stage), (int, float)):
                flat[f"timings.{stage}"] = round(float(timings[stage]), 3)
    config_hash = manifest.get("config_hash") or metrics.get("config_hash")
    if config_hash:
        flat["config_hash"] = str(config_hash)[:16]
    calibration = manifest.get("resolution_calibration") or {}
    if isinstance(calibration, dict) and calibration.get("chosen_res"):
        flat["grid_res"] = int(calibration["chosen_res"])
    if isinstance(flat.get("texture"), dict):
        flat["texture_size"] = flat["texture"].get("texture_size")
        flat["uv_coverage"] = flat["texture"].get("uv_coverage")
    if not isinstance(flat.get("uv_coverage"), (int, float)):
        tex = metrics.get("texture")
        if isinstance(tex, dict):
            flat["texture_size"] = tex.get("texture_size")
            flat["uv_coverage"] = tex.get("uv_coverage")
    flat.pop("texture", None)
    return flat
