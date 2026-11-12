"""Image upload endpoints (presigned intent + completion + secure reads)."""
from __future__ import annotations

import hashlib
import io

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import settings
from ..db import Upload, User, get_db, new_id
from ..ratelimit import enforce, upload_limiter
from ..schemas import ApiError, PresignRequest, ok
from ..security import sha256_hex
from ..serializers import upload_public
from ..storage import storage, upload_key
from ..recon_bridge import validate_image

router = APIRouter(prefix="/uploads", tags=["uploads"])


def _image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return 0, 0
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def _store_thumbnail(data: bytes, upload_id: str, user_id: str) -> str | None:
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = 320.0 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        okp, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not okp:
            return None
        key = f"users/{user_id}/uploads/{upload_id}/thumb.jpg"
        storage.put_bytes(key, buf.tobytes(), "image/jpeg")
        return key
    except Exception:
        return None


@router.post("/presign")
def presign(payload: PresignRequest, request: Request,
            user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Register upload intent and receive a direct upload URL.

    With the local driver the URL points back at ``PUT /uploads/{id}/content``;
    with S3 it is a presigned PUT so bytes never transit the API.
    """
    enforce(upload_limiter, user.id, "upload")
    if payload.size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise ApiError("FILE_TOO_LARGE",
                       f"Images must be smaller than {settings.max_upload_mb} MB.", status=413)
    upload_id = new_id("upl")
    if settings.storage_driver == "s3":
        url = storage.signed_url(upload_key(user.id, upload_id, payload.filename), ttl=900)
    else:
        url = f"/api/v1/uploads/{upload_id}/content"
    return ok({"upload_id": upload_id, "upload_url": url,
               "max_bytes": settings.max_upload_mb * 1024 * 1024,
               "filename": payload.filename})


@router.post("/")
def upload_direct(request: Request, file: UploadFile = File(...),
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Single-request upload (used by the web client and the test-suite)."""
    enforce(upload_limiter, user.id, "upload")
    # Stream-capped read: never buffer more than the configured limit, so a
    # hostile client cannot exhaust memory before validation runs.
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise ApiError("FILE_TOO_LARGE",
                       f"Image exceeds the {settings.max_upload_mb} MB limit.",
                       status=413,
                       details=[{"limitBytes": limit}])
    try:
        mime, width, height = validate_image(data, settings.max_upload_mb)
    except ValueError as exc:
        raise ApiError("INVALID_IMAGE", str(exc), status=415,
                       details=[{"field": "file"}]) from exc
    upload_id = new_id("upl")
    key = upload_key(user.id, upload_id, file.filename or "image.png")
    stored = storage.put_bytes(key, data, mime)
    thumb_key = _store_thumbnail(data, upload_id, user.id)
    upload = Upload(
        id=upload_id, user_id=user.id,
        filename=(file.filename or "image.png")[:255], mime=mime,
        size_bytes=stored.size, width=width, height=height,
        sha256=sha256_hex(data), storage_key=key, thumb_key=thumb_key,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    duplicate = db.execute(select(Upload).where(Upload.user_id == user.id,
                                                Upload.sha256 == upload.sha256,
                                                Upload.id != upload.id)).scalars().first()
    payload = upload_public(upload)
    payload["duplicate_of"] = duplicate.id if duplicate else None
    return ok(payload)


@router.put("/{upload_id}/content")
async def upload_content(upload_id: str, request: Request,
                         user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    """Body upload target for the presign flow (local storage driver)."""
    data = await request.body()
    mime, width, height = validate_image(data, settings.max_upload_mb)
    key = upload_key(user.id, upload_id, "image.png")
    stored = storage.put_bytes(key, data, mime)
    existing = db.get(Upload, upload_id)
    if existing is not None and existing.user_id == user.id:
        raise ApiError("UPLOAD_EXISTS", "This upload already has content.", status=409)
    upload = Upload(id=upload_id, user_id=user.id, filename="image.png", mime=mime,
                    size_bytes=stored.size, width=width, height=height,
                    sha256=sha256_hex(data), storage_key=key,
                    thumb_key=_store_thumbnail(data, upload_id, user.id))
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return ok(upload_public(upload))


@router.get("/")
def list_uploads(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                 limit: int = 30):
    rows = db.execute(select(Upload).where(Upload.user_id == user.id)
                      .order_by(Upload.created_at.desc()).limit(min(limit, 100))).scalars().all()
    return ok([upload_public(u) for u in rows])


@router.get("/{upload_id}")
def get_upload(upload_id: str, user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    upload = db.get(Upload, upload_id)
    if upload is None or upload.user_id != user.id:
        raise ApiError("NOT_FOUND", "Upload not found.", status=404)
    return ok(upload_public(upload))


@router.get("/{upload_id}/image")
def get_upload_image(upload_id: str, user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    upload = db.get(Upload, upload_id)
    if upload is None or upload.user_id != user.id:
        raise ApiError("NOT_FOUND", "Upload not found.", status=404)
    data = storage.read_bytes(upload.storage_key)
    return Response(content=data, media_type=upload.mime,
                    headers={"Cache-Control": "private, max-age=3600",
                             "Content-Disposition": f'inline; filename="{upload.filename}"'})
