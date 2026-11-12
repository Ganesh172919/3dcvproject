"""Gallery, social actions and moderation reporting."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..auth import get_current_user, get_optional_user
from ..db import (Asset, GalleryLike, Report, User, Visibility, get_db)
from ..schemas import ApiError, ok
from ..serializers import asset_public

router = APIRouter(prefix="/gallery", tags=["gallery"])


@router.get("")
def gallery(db: Session = Depends(get_db), user: User | None = Depends(get_optional_user),
            page: int = 1, page_size: int = 24, q: str | None = None,
            tag: str | None = None, sort: str = "newest", featured: bool | None = None):
    page = max(1, page)
    page_size = min(max(page_size, 1), 60)
    base = select(Asset).where(Asset.visibility == Visibility.PUBLIC,
                               Asset.deleted_at.is_(None),
                               Asset.moderation_state == "APPROVED")
    count = select(func.count()).select_from(Asset).where(
        Asset.visibility == Visibility.PUBLIC, Asset.deleted_at.is_(None),
        Asset.moderation_state == "APPROVED")
    if q:
        like = f"%{q.lower()}%"
        crit = or_(func.lower(Asset.name).like(like), func.lower(Asset.description).like(like))
        base, count = base.where(crit), count.where(crit)
    if featured:
        base, count = base.where(Asset.featured.is_(True)), count.where(Asset.featured.is_(True))
    total = db.execute(count).scalar_one()
    order = {"newest": Asset.created_at.desc(), "popular": Asset.like_count.desc(),
             "trending": (Asset.like_count * 2 + Asset.view_count).desc()}.get(
        sort, Asset.created_at.desc())
    rows = db.execute(base.order_by(order).offset((page - 1) * page_size)
                      .limit(page_size)).scalars().all()
    items = []
    for asset in rows:
        owner = db.get(User, asset.user_id)
        payload = asset_public(asset, owner)
        if tag and tag not in (asset.tags or []):
            continue
        items.append(payload)
    return ok({"items": items, "total": total, "page": page, "page_size": page_size})


@router.get("/tags")
def gallery_tags(db: Session = Depends(get_db)):
    rows = db.execute(select(Asset.tags).where(Asset.visibility == Visibility.PUBLIC,
                                               Asset.deleted_at.is_(None))).scalars().all()
    counter: dict[str, int] = {}
    for tags in rows:
        for tag in tags or []:
            counter[tag] = counter.get(tag, 0) + 1
    return ok(sorted(({"tag": k, "count": v} for k, v in counter.items()),
                     key=lambda x: -x["count"])[:40])


@router.get("/{asset_id}")
def gallery_item(asset_id: str, db: Session = Depends(get_db),
                 user: User | None = Depends(get_optional_user)):
    asset = db.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None or asset.moderation_state != "APPROVED":
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    if asset.visibility == Visibility.PRIVATE and (user is None or asset.user_id != user.id):
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    asset.view_count += 1
    db.commit()
    liked = False
    if user:
        liked = db.execute(select(GalleryLike).where(GalleryLike.asset_id == asset.id,
                                                     GalleryLike.user_id == user.id)
                           ).scalar_one_or_none() is not None
    owner = db.get(User, asset.user_id)
    return ok({**asset_public(asset, owner), "liked": liked})


@router.get("/share/{slug}")
def shared_model(slug: str, db: Session = Depends(get_db)):
    asset = db.scalar(select(Asset).where(
        Asset.share_slug == slug, Asset.visibility.in_((Visibility.PUBLIC, Visibility.UNLISTED)),
        Asset.deleted_at.is_(None), Asset.moderation_state == "APPROVED",
    ))
    if asset is None:
        raise ApiError("NOT_FOUND", "Model not available.", status=404)
    return ok(asset_public(asset, db.get(User, asset.user_id)))


@router.post("/{asset_id}/like")
def like(asset_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    asset = db.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    existing = db.execute(select(GalleryLike).where(GalleryLike.asset_id == asset.id,
                                                    GalleryLike.user_id == user.id)
                          ).scalar_one_or_none()
    if existing:
        db.delete(existing)
        asset.like_count = max(0, asset.like_count - 1)
        liked = False
    else:
        db.add(GalleryLike(asset_id=asset.id, user_id=user.id))
        asset.like_count += 1
        liked = True
    db.commit()
    return ok({"liked": liked, "like_count": asset.like_count})


@router.post("/{asset_id}/report")
def report(asset_id: str, payload: dict, user: User = Depends(get_current_user),
           db: Session = Depends(get_db)):
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise ApiError("NOT_FOUND", "Asset not found.", status=404)
    reason = str(payload.get("reason", "other"))[:60]
    details = str(payload.get("details", ""))[:2000]
    row = Report(asset_id=asset.id, reporter_id=user.id, reason=reason, details=details)
    db.add(row)
    asset.moderation_state = "REVIEW"
    db.commit()
    return ok({"report_id": row.id, "state": "OPEN"})
