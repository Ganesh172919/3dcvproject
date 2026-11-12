"""Idempotent development seed.

Creates providers, feature flags, an admin account and a demo account, and
optionally one *real* completed generation (engine runs, files stored, manifest
written) so the dashboard, gallery and viewer have genuine content rather than
fabricated rows.

Everything here is explicitly non-production: passwords are fixed dev values,
no external credentials are created, and re-running is safe — rows are matched
on natural keys.

    python -m app.seed                # providers + flags + users
    python -m app.seed --with-sample  # ... plus a real demo generation (~20 s)
    python -m app.seed --reset-users  # drop the seeded accounts first
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from . import credits as credits_mod
from .config import ROOT, settings
from .db import (Asset, CreditReason, FeatureFlag, Generation, GenerationStatus,
                 Provider, SessionLocal, Upload, User, init_db, utcnow)

log = logging.getLogger("3dfun.seed")

ADMIN_EMAIL = "admin@3dfun.dev"
DEMO_EMAIL = "demo@3dfun.dev"
DEMO_PASSWORD = "DemoPassword123!"

PROVIDERS = [
    {"slug": "local", "display_name": "Local CPU engine", "kind": "local",
     "enabled": True, "weight": 100, "concurrency": 2,
     "notes": "Bundled ONNX depth model + volumetric mesher. No network egress."},
    {"slug": "http-generic", "display_name": "Generic HTTP provider", "kind": "http",
     "enabled": False, "weight": 50, "concurrency": 4,
     "notes": "Disabled until GENERATION_API_URL / GENERATION_API_KEY are set."},
    {"slug": "mock", "display_name": "Mock provider (tests)", "kind": "mock",
     "enabled": True, "weight": 10, "concurrency": 8,
     "notes": "Deterministic fixture provider used by the automated test-suite."},
]

FEATURE_FLAGS = [
    ("studio_upload", "Image upload and generation studio", True, 100),
    ("public_gallery", "Public gallery and share pages", True, 100),
    ("turntable_render", "MP4/GIF turntable rendering", True, 100),
    ("stl_export", "STL geometry export", True, 100),
    ("billing_stripe", "Stripe checkout (off = development ledger)", False, 0),
    ("registration_open", "Allow new account registrations", True, 100),
    ("api_keys", "Personal API keys for the REST API", True, 100),
]


def _ensure_user(db, email: str, role: str, credits: int) -> User:
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is not None:
        return user
    from .security import hash_password

    user = User(email=email, password_hash=hash_password(DEMO_PASSWORD),
                display_name="Administrator" if role == "admin" else "Demo User",
                role=role, email_verified=True,
                terms_accepted_at=utcnow(), privacy_accepted_at=utcnow())
    db.add(user)
    db.commit()
    credits_mod.grant(db, user.id, credits, reason=CreditReason.SIGNUP_GRANT,
                      idempotency_key=f"seed:{email}",
                      meta={"source": "app.seed"})
    return user


def seed_providers(db) -> None:
    for spec in PROVIDERS:
        row = db.execute(select(Provider).where(Provider.slug == spec["slug"])).scalar_one_or_none()
        if row is None:
            db.add(Provider(**spec))
        else:
            for key, value in spec.items():
                setattr(row, key, value)
    db.commit()


def seed_flags(db) -> None:
    for key, description, enabled, rollout in FEATURE_FLAGS:
        row = db.execute(select(FeatureFlag).where(FeatureFlag.key == key)).scalar_one_or_none()
        if row is None:
            db.add(FeatureFlag(key=key, description=description, enabled=enabled,
                               rollout_pct=rollout))
    db.commit()


def seed_sample(db, user: User, source: "object"):
    """Run the real engine once so seeded content is genuine output."""
    from pathlib import Path

    from . import jobs, storage
    from .config import settings as cfg

    existing = db.execute(
        select(Asset).where(Asset.user_id == user.id, Asset.name.like("Demo sneaker%"))
    ).scalars().first()
    if existing is not None:
        log.info("seed: demo asset already present (%s)", existing.id)
        return existing

    source = Path(source)
    if not source.exists():
        log.warning("seed: %s missing — skipping demo generation", source)
        return None

    data = source.read_bytes()
    from .recon_bridge import validate_image

    mime, width, height = validate_image(data, cfg.max_upload_mb)
    upload = Upload(user_id=user.id, filename=source.name, mime=mime,
                    size_bytes=len(data), width=width, height=height,
                    sha256=__import__("hashlib").sha256(data).hexdigest(),
                    storage_key=f"users/{user.id}/uploads/seed-{source.name}")
    storage.storage.put_bytes(upload.storage_key, data, mime)
    db.add(upload)
    db.commit()

    quality = "draft"
    gen = Generation(user_id=user.id, upload_id=upload.id, quality=quality,
                     preset="product", prompt="teal running shoe, product photo",
                     status=GenerationStatus.QUEUED, stage="QUEUED",
                     provider_slug="local",
                     credits_charged=cfg.credit_costs.get(quality, 4))
    db.add(gen)
    db.commit()

    log.info("seed: running the engine for generation %s (this takes ~20 s)", gen.id)
    jobs.runner._run(gen.id)

    db.expire_all()
    gen = db.get(Generation, gen.id)
    if gen is None or gen.status != GenerationStatus.COMPLETED:
        log.warning("seed: demo generation did not complete (%s)", gen and gen.status)
        return None
    asset = db.execute(
        select(Asset).where(Asset.generation_id == gen.id)).scalars().first()
    if asset is not None:
        asset.name = "Demo sneaker (seeded)"
        asset.description = ("Produced at seed time by the local reconstruction "
                             "engine from engine/testdata/sneaker.png.")
        asset.tags = ["demo", "sneaker", "seeded"]
        asset.visibility = "PUBLIC"
        db.commit()
        log.info("seed: demo asset %s (%s faces)", asset.id, asset.polygons)
    return asset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed 3DFUN AI development data")
    parser.add_argument("--with-sample", action="store_true",
                        help="also run the engine to create one real demo asset")
    parser.add_argument("--reset-users", action="store_true",
                        help="delete the seeded accounts before re-creating them")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    init_db()

    with SessionLocal() as db:
        if args.reset_users:
            for email in (ADMIN_EMAIL, DEMO_EMAIL):
                row = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
                if row is not None:
                    db.delete(row)
            db.commit()
            log.info("seed: seeded accounts removed")

        seed_providers(db)
        seed_flags(db)
        _ensure_user(db, ADMIN_EMAIL, "admin", 1000)
        demo = _ensure_user(db, DEMO_EMAIL, "user", 500)
        if args.with_sample:
            seed_sample(db, demo, ROOT / "engine" / "testdata" / "sneaker.png")

    print("\nSeed complete — development data only.\n")
    print(f"  admin  {ADMIN_EMAIL:<22} password: {DEMO_PASSWORD}")
    print(f"  user   {DEMO_EMAIL:<22} password: {DEMO_PASSWORD}")
    print(f"\n  database: {settings.database_url}")
    print(f"  storage : {settings.storage_root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
