"""Application configuration (12-factor, environment driven)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
_GEMINI_ENV = {name: os.environ[name] for name in ("GEMINI_API_KEY", "GEMINI_MODEL") if name in os.environ}
load_dotenv(ENV_FILE, encoding="utf-8-sig")


def gemini_settings() -> tuple[str, str]:
    """Read the local chat settings so edits take effect without a restart."""
    values = dotenv_values(ENV_FILE, encoding="utf-8-sig", interpolate=False)
    values.update(_GEMINI_ENV)
    return (values.get("GEMINI_API_KEY") or "").strip(), (values.get("GEMINI_MODEL") or "gemini-3.5-flash-lite").strip()


def _env(name: str, default: str | None = None) -> str:
    val = os.getenv(name)
    if val is None:
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return val


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    app_name: str = "3DFUN AI"
    app_url: str = field(default_factory=lambda: _env("APP_URL", "http://localhost:8000"))
    environment: str = field(default_factory=lambda: _env("NODE_ENV", "development"))
    version: str = "1.0.0"

    # ---- database / queue ------------------------------------------
    database_url: str = field(default_factory=lambda: _env(
        "DATABASE_URL", f"sqlite:///{ROOT / 'data' / '3dfun.db'}"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))

    # ---- auth --------------------------------------------------------
    jwt_access_secret: str = field(default_factory=lambda: _env(
        "JWT_ACCESS_SECRET", "dev-only-access-secret-change-me"))
    jwt_refresh_secret: str = field(default_factory=lambda: _env(
        "JWT_REFRESH_SECRET", "dev-only-refresh-secret-change-me"))
    access_token_ttl: int = field(default_factory=lambda: _env_int("ACCESS_TOKEN_TTL", 900))
    refresh_token_ttl: int = field(default_factory=lambda: _env_int("REFRESH_TOKEN_TTL", 60 * 60 * 24 * 30))
    cookie_secure: bool = field(default_factory=lambda: _env_bool("COOKIE_SECURE", False))
    allow_registration: bool = field(default_factory=lambda: _env_bool("ALLOW_REGISTRATION", True))

    # ---- storage -----------------------------------------------------
    storage_driver: str = field(default_factory=lambda: _env("STORAGE_DRIVER", "local"))
    storage_root: Path = field(default_factory=lambda: Path(_env("STORAGE_ROOT", str(ROOT / "data" / "storage"))))
    storage_endpoint: str = field(default_factory=lambda: _env("STORAGE_ENDPOINT", ""))
    storage_bucket: str = field(default_factory=lambda: _env("STORAGE_BUCKET", "3dfun-assets"))
    storage_access_key: str = field(default_factory=lambda: _env("STORAGE_ACCESS_KEY", ""))
    storage_secret_key: str = field(default_factory=lambda: _env("STORAGE_SECRET_KEY", ""))
    signed_url_ttl: int = field(default_factory=lambda: _env_int("SIGNED_URL_TTL", 900))

    # ---- limits ------------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 20))
    rate_limit_max: int = field(default_factory=lambda: _env_int("RATE_LIMIT_MAX", 120))
    rate_limit_window: int = field(default_factory=lambda: _env_int("RATE_LIMIT_WINDOW", 60))
    login_max_attempts: int = field(default_factory=lambda: _env_int("LOGIN_MAX_ATTEMPTS", 8))
    job_timeout_seconds: int = field(default_factory=lambda: _env_int("JOB_TIMEOUT_SECONDS", 480))
    job_concurrency: int = field(default_factory=lambda: _env_int("JOB_CONCURRENCY", 1))

    # ---- credits -----------------------------------------------------
    signup_credits: int = field(default_factory=lambda: _env_int("SIGNUP_CREDITS", 60))
    credit_costs: dict[str, int] = field(default_factory=lambda: {
        "draft": _env_int("CREDITS_DRAFT", 4),
        "standard": _env_int("CREDITS_STANDARD", 8),
        "high": _env_int("CREDITS_HIGH", 16),
    })

    # ---- providers ---------------------------------------------------
    generation_provider: str = field(default_factory=lambda: _env("GENERATION_PROVIDER", "local"))
    generation_api_url: str = field(default_factory=lambda: _env("GENERATION_API_URL", ""))
    generation_api_key: str = field(default_factory=lambda: _env("GENERATION_API_KEY", ""))
    generation_model: str = field(default_factory=lambda: _env("GENERATION_MODEL", "3dfun-recon-v1"))

    # ---- billing / email --------------------------------------------
    stripe_secret_key: str = field(default_factory=lambda: _env("STRIPE_SECRET_KEY", ""))
    stripe_webhook_secret: str = field(default_factory=lambda: _env("STRIPE_WEBHOOK_SECRET", "whsec_dev"))
    email_provider: str = field(default_factory=lambda: _env("EMAIL_PROVIDER", "console"))
    resend_api_key: str = field(default_factory=lambda: _env("RESEND_API_KEY", ""))

    # ---- observability ----------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    sentry_dsn: str = field(default_factory=lambda: _env("SENTRY_DSN", ""))

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    def validate_production(self) -> list[str]:
        problems = []
        if self.is_production:
            if "dev-only" in self.jwt_access_secret:
                problems.append("JWT_ACCESS_SECRET must be set in production")
            if "dev-only" in self.jwt_refresh_secret:
                problems.append("JWT_REFRESH_SECRET must be set in production")
            if not self.cookie_secure:
                problems.append("COOKIE_SECURE must be true in production")
            if self.storage_driver == "local" and self.environment == "production":
                problems.append("local storage driver in production (S3-compatible expected)")
        return problems


settings = Settings()
settings.storage_root.mkdir(parents=True, exist_ok=True)
(ROOT / "data").mkdir(parents=True, exist_ok=True)


def generate_secret(nbytes: int = 32) -> str:
    """Helper used by scripts/rotate_secrets.py and the docs."""
    return secrets.token_urlsafe(nbytes)
