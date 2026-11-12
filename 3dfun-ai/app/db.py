"""Database models and session management (SQLAlchemy 2.x).

The schema covers: identity (users, sessions, verification tokens, roles,
api keys), the generation pipeline (uploads, generations, attempts, events),
assets and versions, credits (immutable ledger), billing, gallery/social,
moderation, feature flags and audit logs.

SQLite is the default for local development; PostgreSQL is supported by
changing DATABASE_URL. Row-level safety for credit spend is enforced with
`SELECT ... FOR UPDATE` when the backend supports it and with an explicit
optimistic version column otherwise.
"""
from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Iterator

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, create_engine, event, select,
)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker, Session)

from .config import settings


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------
class GenerationStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    PROCESSING = "PROCESSING"
    GENERATING_GEOMETRY = "GENERATING_GEOMETRY"
    GENERATING_TEXTURE = "GENERATING_TEXTURE"
    OPTIMIZING = "OPTIMIZING"
    RENDERING_PREVIEW = "RENDERING_PREVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


class Visibility(str, enum.Enum):
    PRIVATE = "PRIVATE"
    UNLISTED = "UNLISTED"
    PUBLIC = "PUBLIC"


class CreditReason(str, enum.Enum):
    SIGNUP_GRANT = "SIGNUP_GRANT"
    RESERVATION = "RESERVATION"
    SPEND = "SPEND"
    REFUND = "REFUND"
    PURCHASE = "PURCHASE"
    ADMIN_ADJUSTMENT = "ADMIN_ADJUSTMENT"
    EXPIRY = "EXPIRY"


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(80), default="")
    role: Mapped[str] = mapped_column(String(20), default="user")          # user | admin
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    terms_accepted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    privacy_accepted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user",
                                                         cascade="all, delete-orphan")
    wallet: Mapped["CreditWallet | None"] = relationship(back_populates="user", uselist=False)


class UserSession(Base):
    __tablename__ = "user_sessions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ses"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    refresh_token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    user_agent: Mapped[str] = mapped_column(String(255), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="sessions")


class VerificationToken(Base):
    __tablename__ = "verification_tokens"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("vtk"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30))          # email_verify | password_reset
    token_hash: Mapped[str] = mapped_column(String(128), unique=True)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("key"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(128), unique=True)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# uploads / generations / assets
# ---------------------------------------------------------------------------
class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("upl"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    mime: Mapped[str] = mapped_column(String(80))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(400))
    thumb_key: Mapped[str | None] = mapped_column(String(400), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Provider(Base):
    __tablename__ = "providers"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("prv"))
    slug: Mapped[str] = mapped_column(String(40), unique=True)
    display_name: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(30), default="local")       # local | http | mock
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    weight: Mapped[int] = mapped_column(Integer, default=100)
    concurrency: Mapped[int] = mapped_column(Integer, default=1)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(Text, default="")


class Generation(Base):
    __tablename__ = "generations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("gen"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    upload_id: Mapped[str | None] = mapped_column(ForeignKey("uploads.id"), nullable=True, index=True)
    provider_slug: Mapped[str] = mapped_column(String(40), default="local")
    model: Mapped[str] = mapped_column(String(60), default="3dfun-recon-v1")
    status: Mapped[GenerationStatus] = mapped_column(Enum(GenerationStatus),
                                                     default=GenerationStatus.DRAFT, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(40), default="DRAFT")
    quality: Mapped[str] = mapped_column(String(20), default="standard")
    prompt: Mapped[str] = mapped_column(Text, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    preset: Mapped[str] = mapped_column(String(40), default="product")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    credits_reserved: Mapped[int] = mapped_column(Integer, default=0)
    credits_charged: Mapped[int] = mapped_column(Integer, default=0)
    privacy_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    queue_position: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    support_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    provider_job_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                    onupdate=utcnow)

    attempts: Mapped[list["GenerationAttempt"]] = relationship(back_populates="generation",
                                                               cascade="all, delete-orphan")
    events: Mapped[list["GenerationEvent"]] = relationship(back_populates="generation",
                                                           cascade="all, delete-orphan")
    asset: Mapped["Asset | None"] = relationship(back_populates="generation", uselist=False)

    __table_args__ = (
        Index("ix_generations_user_status", "user_id", "status"),
        Index("ix_generations_user_created", "user_id", "created_at"),
    )


class GenerationAttempt(Base):
    __tablename__ = "generation_attempts"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("att"))
    generation_id: Mapped[str] = mapped_column(ForeignKey("generations.id", ondelete="CASCADE"),
                                               index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    provider_slug: Mapped[str] = mapped_column(String(40), default="local")
    status: Mapped[str] = mapped_column(String(30), default="running")
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    generation: Mapped[Generation] = relationship(back_populates="attempts")


class GenerationEvent(Base):
    __tablename__ = "generation_events"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("evt"))
    generation_id: Mapped[str] = mapped_column(ForeignKey("generations.id", ondelete="CASCADE"),
                                               index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    type: Mapped[str] = mapped_column(String(60))
    stage: Mapped[str] = mapped_column(String(40), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    generation: Mapped[Generation] = relationship(back_populates="events")

    __table_args__ = (Index("ix_generation_events_gen_seq", "generation_id", "sequence"),)


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ast"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    generation_id: Mapped[str | None] = mapped_column(ForeignKey("generations.id"), nullable=True,
                                                      unique=True)
    name: Mapped[str] = mapped_column(String(140), default="Untitled model")
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    visibility: Mapped[Visibility] = mapped_column(Enum(Visibility), default=Visibility.PRIVATE,
                                                   index=True)
    storage_prefix: Mapped[str] = mapped_column(String(400))
    files: Mapped[dict] = mapped_column(JSON, default=dict)          # format -> storage key
    manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    preview_key: Mapped[str | None] = mapped_column(String(400), nullable=True)
    cover_key: Mapped[str | None] = mapped_column(String(400), nullable=True)
    turntable_key: Mapped[str | None] = mapped_column(String(400), nullable=True)
    polygons: Mapped[int] = mapped_column(Integer, default=0)
    vertices: Mapped[int] = mapped_column(Integer, default=0)
    watertight: Mapped[bool] = mapped_column(Boolean, default=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    featured: Mapped[bool] = mapped_column(Boolean, default=False)
    moderation_state: Mapped[str] = mapped_column(String(20), default="APPROVED")
    share_slug: Mapped[str] = mapped_column(String(40), unique=True,
                                            default=lambda: uuid.uuid4().hex[:12])
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                    onupdate=utcnow)

    generation: Mapped[Generation | None] = relationship(back_populates="asset")
    versions: Mapped[list["AssetVersion"]] = relationship(back_populates="asset",
                                                          cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_assets_visibility_created", "visibility", "created_at"),
        CheckConstraint("polygons >= 0", name="ck_assets_polygons"),
    )


class AssetVersion(Base):
    __tablename__ = "asset_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ver"))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    files: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    asset: Mapped[Asset] = relationship(back_populates="versions")


class GalleryLike(Base):
    __tablename__ = "gallery_likes"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("lik"))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("asset_id", "user_id", name="uq_like_asset_user"),)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("rpt"))
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    reporter_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    reason: Mapped[str] = mapped_column(String(60))
    details: Mapped[str] = mapped_column(Text, default="")
    state: Mapped[str] = mapped_column(String(20), default="OPEN")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("col"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CollectionItem(Base):
    __tablename__ = "collection_items"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("cit"))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"),
                                               index=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)


# ---------------------------------------------------------------------------
# credits & billing
# ---------------------------------------------------------------------------
class CreditWallet(Base):
    __tablename__ = "credit_wallets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("wal"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    balance: Mapped[int] = mapped_column(Integer, default=0)
    reserved: Mapped[int] = mapped_column(Integer, default=0)
    lifetime_granted: Mapped[int] = mapped_column(Integer, default=0)
    lifetime_spent: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                    onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="wallet")


class CreditTransaction(Base):
    """Append-only ledger. Rows are never updated or deleted."""
    __tablename__ = "credit_transactions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ctx"))
    wallet_id: Mapped[str] = mapped_column(ForeignKey("credit_wallets.id", ondelete="CASCADE"),
                                           index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    generation_id: Mapped[str | None] = mapped_column(ForeignKey("generations.id"), nullable=True,
                                                      index=True)
    reason: Mapped[CreditReason] = mapped_column(Enum(CreditReason))
    amount: Mapped[int] = mapped_column(Integer)          # negative = debit
    balance_after: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    idempotency_key: Mapped[str | None] = mapped_column(String(120), unique=True, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Invoice(Base):
    __tablename__ = "invoices"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("inv"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="usd")
    credits: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="paid")
    external_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WebhookEvent(Base):
    __tablename__ = "webhook_events"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("whk"))
    provider: Mapped[str] = mapped_column(String(30), default="stripe")
    external_id: Mapped[str] = mapped_column(String(140), unique=True)
    event_type: Mapped[str] = mapped_column(String(80))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    received_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
class FeatureFlag(Base):
    __tablename__ = "feature_flags"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ffg"))
    key: Mapped[str] = mapped_column(String(60), unique=True)
    description: Mapped[str] = mapped_column(String(255), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    rollout_pct: Mapped[int] = mapped_column(Integer, default=0)
    allowlist: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                    onupdate=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("aud"))
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    target_type: Mapped[str] = mapped_column(String(40), default="")
    target_id: Mapped[str] = mapped_column(String(60), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ntf"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(140))
    body: Mapped[str] = mapped_column(Text, default="")
    read_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UsageRecord(Base):
    __tablename__ = "usage_records"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usg"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    generation_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    provider_slug: Mapped[str] = mapped_column(String(40), default="local")
    stage: Mapped[str] = mapped_column(String(40), default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    credits: Mapped[int] = mapped_column(Integer, default=0)
    storage_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


# ---------------------------------------------------------------------------
# engine / session factory
# ---------------------------------------------------------------------------
def _build_engine(url: str):
    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):       # pragma: no cover
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return engine


engine = _build_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(engine)


def reset_db() -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
