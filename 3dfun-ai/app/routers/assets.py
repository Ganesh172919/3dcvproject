"""Asset endpoints: list, detail, update, downloads, public share, embed."""
from __future__ import annotations

import datetime as dt
import io
from html import escape
from zipfile import ZipFile

from fastapi import APIRouter, Depends, Header, Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..auth import get_current_user, get_optional_user
from ..config import settings
from ..db import Asset, AssetVersion, User, Visibility, get_db, utcnow
from ..schemas import ApiError, AssetUpdateRequest, ok
from ..serializers import asset_public
from ..storage import storage

router = APIRouter(prefix="/assets", tags=["assets"])

DELETION_GRACE_DAYS = 7


def _owned_asset(db: Session, asset_id: str, user: User) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    if asset.user_id != user.id and user.role != "admin":
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)   # IDOR-safe
    return asset


def _public_asset(db: Session, asset_id: str, user: User | None) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    restricted = asset.visibility == Visibility.PRIVATE or asset.moderation_state != "APPROVED"
    if restricted and (user is None or (asset.user_id != user.id and user.role != "admin")):
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    return asset


def _attachment_header(name: str, filename: str) -> str:
    """RFC 6266 / 5987 attachment header.

    Non-ASCII asset names (common: em dashes, CJK, accents) must not be emitted
    raw in an HTTP header — that is a latin-1 encoding error at the protocol
    level. We send an ASCII fallback plus a UTF-8 ``filename*`` parameter.
    """
    import urllib.parse

    stem = "".join(ch for ch in (name or "model") if ch.isascii() and ch.isprintable())
    stem = stem.replace('"', "").replace("\\", "").strip().replace(" ", "_")[:60] or "model"
    ascii_name = f"{stem}_{filename}"
    utf8_name = urllib.parse.quote(f"{name.replace(' ', '_')[:60]}_{filename}")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"


@router.get("")
def list_assets(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                page: int = 1, page_size: int = 24, q: str | None = None,
                visibility: str | None = None, sort: str = "newest"):
    page = max(1, page)
    page_size = min(max(page_size, 1), 100)
    base = select(Asset).where(Asset.user_id == user.id, Asset.deleted_at.is_(None))
    count = select(func.count()).select_from(Asset).where(Asset.user_id == user.id,
                                                          Asset.deleted_at.is_(None))
    if q:
        like = f"%{q.lower()}%"
        crit = or_(func.lower(Asset.name).like(like), func.lower(Asset.description).like(like))
        base, count = base.where(crit), count.where(crit)
    if visibility:
        try:
            vis = Visibility(visibility.upper())
        except ValueError:
            raise ApiError("INVALID_FILTER", "Unknown visibility.", status=422)
        base, count = base.where(Asset.visibility == vis), count.where(Asset.visibility == vis)
    total = db.execute(count).scalar_one()
    order = {"newest": Asset.created_at.desc(), "oldest": Asset.created_at.asc(),
             "polygons": Asset.polygons.desc(), "likes": Asset.like_count.desc()}.get(
        sort, Asset.created_at.desc())
    rows = db.execute(base.order_by(order).offset((page - 1) * page_size)
                      .limit(page_size)).scalars().all()
    return ok({"items": [asset_public(a, user) for a in rows], "total": total,
               "page": page, "page_size": page_size})


@router.get("/{asset_id}")
def get_asset(asset_id: str, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    asset = _public_asset(db, asset_id, user)
    asset.view_count += 1
    db.commit()
    return ok(asset_public(asset, user))


@router.patch("/{asset_id}")
def update_asset(asset_id: str, payload: AssetUpdateRequest,
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    asset = _owned_asset(db, asset_id, user)
    before = {"name": asset.name, "visibility": asset.visibility.value}
    if payload.name is not None:
        asset.name = payload.name.strip() or asset.name
    if payload.description is not None:
        asset.description = payload.description
    if payload.tags is not None:
        asset.tags = payload.tags
    if payload.visibility is not None:
        asset.visibility = Visibility(payload.visibility)
    asset.version += 1
    db.add(AssetVersion(asset_id=asset.id, version=asset.version, files=asset.files,
                        metrics=asset.metrics, note="metadata update"))
    db.commit()
    db.refresh(asset)
    return ok({**asset_public(asset, user), "previous": before})


@router.delete("/{asset_id}")
def delete_asset(asset_id: str, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    asset = _owned_asset(db, asset_id, user)
    asset.deleted_at = utcnow()
    db.commit()
    return ok({"deleted": asset.id,
               "restorable_until": (utcnow() + dt.timedelta(days=DELETION_GRACE_DAYS)).isoformat()})


@router.post("/{asset_id}/restore")
def restore_asset(asset_id: str, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    asset = db.get(Asset, asset_id)
    if asset is None or asset.user_id != user.id:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    if asset.deleted_at is None:
        return ok(asset_public(asset, user))
    if (utcnow() - asset.deleted_at.replace(tzinfo=dt.timezone.utc)
            if asset.deleted_at.tzinfo is None else utcnow() - asset.deleted_at
            ).days > DELETION_GRACE_DAYS:
        raise ApiError("RESTORE_WINDOW_CLOSED", "The restore window has expired.", status=409)
    asset.deleted_at = None
    db.commit()
    return ok(asset_public(asset, user))


@router.post("/{asset_id}/duplicate")
def duplicate_asset(asset_id: str, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    source = _owned_asset(db, asset_id, user)
    import shutil
    from ..db import new_id
    from ..storage import asset_prefix
    copy = Asset(
        user_id=user.id, generation_id=None,
        name=f"{source.name} (copy)", description=source.description,
        tags=list(source.tags or []), visibility=Visibility.PRIVATE,
        storage_prefix=asset_prefix(user.id, "copies", new_id("ast")),
        files=dict(source.files or {}), manifest=source.manifest, metrics=source.metrics,
        polygons=source.polygons, vertices=source.vertices, watertight=source.watertight,
        size_bytes=source.size_bytes, version=1,
    )
    db.add(copy)
    db.commit()
    db.refresh(copy)
    # share the byte payloads through the storage layer (copy, not move)
    for rel in (source.files or {}):
        src_key = f"{source.storage_prefix}/{rel}"
        if storage.exists(src_key):
            storage.put_bytes(f"{copy.storage_prefix}/{rel}", storage.read_bytes(src_key))
    copy.preview_key = source.preview_key
    copy.cover_key = source.cover_key
    copy.turntable_key = source.turntable_key
    db.commit()
    _ = shutil
    return ok(asset_public(copy, user))


def _resolve_file(db: Session, asset: Asset, name: str) -> str:
    key = f"{asset.storage_prefix}/{name}"
    if not storage.exists(key):
        raise ApiError("FILE_NOT_FOUND", "That file is not part of this asset.", status=404)
    return key


@router.get("/{asset_id}/download")
def download_asset(asset_id: str, format: str = "glb", bundle: bool = False,
                   user: User | None = Depends(get_optional_user), db: Session = Depends(get_db)):
    asset = _public_asset(db, asset_id, user)
    mapping = {
        "glb": ["model.glb"], "gltf": ["model.gltf"], "obj": ["model.obj"],
        "ply": ["model.ply"], "stl": ["model.stl"], "mtl": ["model.mtl"],
        "textures": ["basecolor.png", "normal.png", "orm.png"], "png": ["basecolor.png"],
        "preview": ["preview.png"], "turntable": ["turntable.mp4", "turntable.gif"],
        "mp4": ["turntable.mp4"], "gif": ["turntable.gif"],
        "manifest": ["manifest.json"],
    }
    if format not in mapping:
        raise ApiError("INVALID_FORMAT", f"Unsupported format '{format}'.", status=422)
    filename = next((name for name in mapping[format] if storage.exists(f"{asset.storage_prefix}/{name}")),
                    mapping[format][0])
    key = _resolve_file(db, asset, filename)
    if bundle and format in {"obj", "gltf"}:
        companion = "model.mtl" if format == "obj" else "model.bin"
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr(filename, storage.read_bytes(key))
            for name in (companion, "basecolor.png", "normal.png", "orm.png"):
                companion_key = f"{asset.storage_prefix}/{name}"
                if storage.exists(companion_key):
                    archive.writestr(name, storage.read_bytes(companion_key))
        return Response(content=buffer.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": _attachment_header(asset.name, f"{format}.zip"),
            "Cache-Control": "private, max-age=300",
        })
    data = storage.read_bytes(key)
    if isinstance(data, str):
        data = data.encode()
    content_type = ("model/gltf-binary" if filename.endswith(".glb") else
                    "application/json" if filename.endswith(".json") else
                    "text/plain" if filename.endswith((".obj", ".mtl", ".gltf")) else
                    "image/png" if filename.endswith(".png") else
                    "image/gif" if filename.endswith(".gif") else
                    "video/mp4" if filename.endswith(".mp4") else
                    "application/octet-stream")
    return Response(content=data, media_type=content_type, headers={
        "Content-Disposition": _attachment_header(asset.name, filename),
        "Cache-Control": "private, max-age=300",
        "X-Content-Type-Options": "nosniff",
    })


@router.get("/{asset_id}/preview")
def asset_preview(asset_id: str, user: User | None = Depends(get_optional_user),
                  db: Session = Depends(get_db)):
    asset = _public_asset(db, asset_id, user)
    if not asset.preview_key or not storage.exists(asset.preview_key):
        raise ApiError("NOT_FOUND", "No preview image.", status=404)
    return Response(content=storage.read_bytes(asset.preview_key), media_type="image/png",
                    headers={"Cache-Control": "private, max-age=600"})


@router.get("/{asset_id}/file")
def asset_file(asset_id: str, name: str, user: User | None = Depends(get_optional_user),
               db: Session = Depends(get_db)):
    """Serve any file inside the asset prefix (used by the 3D viewer)."""
    asset = _public_asset(db, asset_id, user)
    if "/" in name or ".." in name:
        raise ApiError("INVALID_NAME", "Invalid file name.", status=422)
    key = _resolve_file(db, asset, name)
    content_type = ("model/gltf-binary" if name.endswith(".glb") else
                    "model/gltf+json" if name.endswith(".gltf") else
                    "application/json" if name.endswith(".json") else
                    "image/png" if name.endswith(".png") else
                    "video/mp4" if name.endswith(".mp4") else
                    "image/gif" if name.endswith(".gif") else
                    "application/octet-stream")
    return Response(content=storage.read_bytes(key), media_type=content_type,
                    headers={"Cache-Control": "private, max-age=600"})


@router.get("/{asset_id}/embed")
def asset_embed(asset_id: str, db: Session = Depends(get_db)):
    asset = db.get(Asset, asset_id)
    if asset is None or asset.visibility != Visibility.PUBLIC or asset.deleted_at is not None:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    snippet = (f'<iframe src="{settings.app_url}/embed/{asset.share_slug}" '
               f'width="640" height="420" style="border:0;border-radius:12px" '
               f'title="{escape(asset.name, quote=True)}" loading="lazy"></iframe>')
    return ok({"embed_html": snippet,
               "public_url": f"{settings.app_url}/m/{asset.share_slug}",
               "license_note": "Embedding is allowed for public assets; attribution retained."})
