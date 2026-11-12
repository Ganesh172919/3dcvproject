"""Shared contracts: response envelope, errors, and request/response schemas."""
from __future__ import annotations

import datetime as dt
import re
import time
import uuid
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

T = TypeVar("T")


# ---------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------
def request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


class Meta(BaseModel):
    requestId: str
    timestamp: str
    durationMs: float | None = None


class Envelope(BaseModel, Generic[T]):
    success: bool = True
    data: T | None = None
    meta: Meta
    error: None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)
    retryable: bool = False


class ErrorEnvelope(BaseModel):
    success: bool = False
    data: None = None
    meta: Meta
    error: ErrorBody


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ok(data: Any = None, started: float | None = None) -> dict[str, Any]:
    meta = {"requestId": request_id(), "timestamp": now_iso()}
    if started is not None:
        meta["durationMs"] = round((time.perf_counter() - started) * 1000, 2)
    return {"success": True, "data": data, "meta": meta, "error": None}


class ApiError(Exception):
    """Domain error carrying an HTTP status and a stable machine code."""

    def __init__(self, code: str, message: str, status: int = 400,
                 details: list[dict[str, Any]] | None = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or []
        self.retryable = retryable

    def envelope(self) -> dict[str, Any]:
        return {
            "success": False,
            "data": None,
            "meta": {"requestId": request_id(), "timestamp": now_iso()},
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "retryable": self.retryable,
            },
        }


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    display_name: str = Field(default="", max_length=80)
    accept_terms: bool = True


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)
    mfa_code: str | None = Field(default=None, max_length=10)


class MfaVerifyRequest(BaseModel):
    code: str = Field(min_length=6, max_length=10)


class RefreshResponse(BaseModel):
    access_token: str
    expires_in: int
    csrf_token: str


class UserPublic(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    email_verified: bool
    mfa_enabled: bool
    created_at: dt.datetime
    credits: int = 0


class UpdateProfileRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    password: str | None = Field(default=None, min_length=10, max_length=200)


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------
class PresignRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="image/png", max_length=80)
    size_bytes: int = Field(gt=0)


class PresignResponse(BaseModel):
    upload_id: str
    upload_url: str
    max_bytes: int


class UploadPublic(BaseModel):
    id: str
    filename: str
    mime: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    created_at: dt.datetime
    image_url: str


# ---------------------------------------------------------------------------
# generations
# ---------------------------------------------------------------------------
QUALITY = Literal["draft", "standard", "high"]
PRESET = Literal["product", "character", "sculpt", "game_asset", "low_poly", "realistic", "stylized"]


class AdvancedOptions(BaseModel):
    engine: Literal["local", "triposr", "midas"] | None = None
    geometry_detail: int = Field(default=50, ge=0, le=100)
    texture_quality: Literal["512", "1024", "2048"] = "1024"
    symmetry: bool = True
    background_removal: bool = True
    target_faces: int | None = Field(default=None, ge=1000, le=200000)
    formats: list[Literal["glb", "gltf", "obj", "ply", "stl"]] = Field(default=["glb", "obj", "ply"])
    render_preview: bool = True
    turntable: bool = True
    uv_optimize: bool = True
    delighting: bool = True


class EstimateRequest(BaseModel):
    quality: QUALITY = "standard"
    options: AdvancedOptions = Field(default_factory=AdvancedOptions)


class EstimateResponse(BaseModel):
    credits: int
    balance: int
    estimated_seconds: int
    provider: str
    model: str
    queue_position: int


class CreateGenerationRequest(BaseModel):
    upload_id: str
    quality: QUALITY = "standard"
    preset: PRESET = "product"
    prompt: str = Field(default="", max_length=600)
    negative_prompt: str = Field(default="", max_length=400)
    privacy_mode: bool = False
    options: AdvancedOptions = Field(default_factory=AdvancedOptions)
    idempotency_key: str | None = Field(default=None, max_length=80)
    accept_terms: bool = True


class GenerationEventPublic(BaseModel):
    id: str
    type: str
    stage: str
    progress: int
    sequence: int
    occurredAt: dt.datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class GenerationPublic(BaseModel):
    id: str
    status: str
    stage: str
    progress: int
    quality: str
    preset: str
    prompt: str
    credits_reserved: int
    credits_charged: int
    privacy_mode: bool
    queue_position: int
    error_code: str | None = None
    error_message: str | None = None
    support_code: str | None = None
    provider: str
    model: str
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    completed_at: dt.datetime | None = None
    duration_ms: int | None = None
    asset_id: str | None = None
    source_image_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class GenerationListResponse(BaseModel):
    items: list[GenerationPublic]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------
class AssetUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=140)
    description: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = None
    visibility: Literal["PRIVATE", "UNLISTED", "PUBLIC"] | None = None

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        out = []
        for tag in v[:12]:
            tag = re.sub(r"[^a-zA-Z0-9 _-]", "", tag).strip()[:24]
            if tag:
                out.append(tag)
        return out


class DownloadOption(BaseModel):
    format: str
    label: str
    size_bytes: int
    url: str


class AssetPublic(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    visibility: str
    owner_id: str
    owner_name: str
    polygons: int
    vertices: int
    watertight: bool
    size_bytes: int
    like_count: int
    view_count: int
    featured: bool
    version: int
    share_slug: str
    preview_url: str | None = None
    cover_url: str | None = None
    turntable_url: str | None = None
    model_url: str | None = None
    downloads: list[DownloadOption] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)
    created_at: dt.datetime
    generation_id: str | None = None


class AssetListResponse(BaseModel):
    items: list[AssetPublic]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# credits / billing
# ---------------------------------------------------------------------------
class CreditTransactionPublic(BaseModel):
    id: str
    reason: str
    amount: int
    balance_after: int
    generation_id: str | None
    created_at: dt.datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class BalanceResponse(BaseModel):
    balance: int
    reserved: int
    available: int
    lifetime_granted: int
    lifetime_spent: int


class CheckoutRequest(BaseModel):
    pack: Literal["starter", "studio", "scale"]


class CheckoutResponse(BaseModel):
    checkout_url: str
    pack: str
    credits: int
    amount_cents: int
    mode: Literal["stripe", "development"]


# ---------------------------------------------------------------------------
# admin / ops
# ---------------------------------------------------------------------------
class AdminUserPatch(BaseModel):
    role: Literal["user", "admin"] | None = None
    is_active: bool | None = None
    credits_delta: int | None = Field(default=None, ge=-100000, le=100000)
    reason: str = Field(default="", max_length=200)


class FeatureFlagPatch(BaseModel):
    enabled: bool | None = None
    rollout_pct: int | None = Field(default=None, ge=0, le=100)
    allowlist: list[str] | None = None


class AdminJobPublic(BaseModel):
    id: str
    user_email: str
    status: str
    stage: str
    progress: int
    quality: str
    provider: str
    created_at: dt.datetime
    duration_ms: int | None
    error_code: str | None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "fail"]
    version: str
    time: str
    checks: dict[str, Any]


class MetricsResponse(BaseModel):
    counters: dict[str, int]
    gauges: dict[str, float]
    histograms: dict[str, dict[str, float]]
