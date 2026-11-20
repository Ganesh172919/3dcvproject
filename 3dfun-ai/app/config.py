"""Application configuration (12-factor, environment driven)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
_GEMINI_ENV = {name: os.environ[name] for name in ("GEMINI_API_KEY", "GEMINI_MODEL") if name in os.environ}
_OPENROUTER_ENV = {name: os.environ[name] for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_BASE_URL") if name in os.environ}
_PROVIDER_KEYS = {name: os.environ[name] for name in ("GROQ_API_KEY", "GROQ_MODEL", "XAI_API_KEY", "TOGETHER_API_KEY", "FIREWORKS_API_KEY") if name in os.environ}
load_dotenv(ENV_FILE, encoding="utf-8-sig")


def gemini_settings() -> tuple[str, str]:
    """Read the local chat settings so edits take effect without a restart."""
    values = dotenv_values(ENV_FILE, encoding="utf-8-sig", interpolate=False)
    values.update(_GEMINI_ENV)
    return (values.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip(), (values.get("GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()


def openrouter_settings() -> tuple[str, str, str]:
    """Read the OpenRouter chat settings dynamically from env and .env."""
    values = dotenv_values(ENV_FILE, encoding="utf-8-sig", interpolate=False)
    values.update(_OPENROUTER_ENV)
    key = (values.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip()
    model = (values.get("OPENROUTER_MODEL") or os.getenv("OPENROUTER_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free").strip()
    base_url = (values.get("OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
    return key, model, base_url


def provider_api_keys() -> dict[str, str]:
    """Return dictionary of configured LLM provider keys."""
    values = dotenv_values(ENV_FILE, encoding="utf-8-sig", interpolate=False)
    values.update(_PROVIDER_KEYS)
    return {
        "groq": (values.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip(),
        "xai": (values.get("XAI_API_KEY") or os.getenv("XAI_API_KEY") or "").strip(),
        "together": (values.get("TOGETHER_API_KEY") or os.getenv("TOGETHER_API_KEY") or "").strip(),
        "fireworks": (values.get("FIREWORKS_API_KEY") or os.getenv("FIREWORKS_API_KEY") or "").strip(),
    }


OPENROUTER_FREE_MODELS: list[dict[str, Any]] = [
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "name": "NVIDIA: Nemotron 3 Ultra (free)",
        "provider": "NVIDIA",
        "tokens": "3.3T tokens",
        "ranking": "Programming (#39)",
        "context_length": "1M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Reasoning & Orchestration",
        "description": "Frontier-reasoning and orchestration MoE model (55B active / 550B total) on hybrid Transformer-Mamba architecture. Suited for multi-step agent pipelines.",
    },
    {
        "id": "nvidia/nemotron-3.5-super-vl:free",
        "name": "NVIDIA: Nemotron 3.5 Super VL (free)",
        "provider": "NVIDIA",
        "tokens": "3.04T tokens",
        "ranking": "Programming (#38)",
        "context_length": "1M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text", "image"],
        "category": "Vision-Language Multimodal",
        "description": "Multimodal mixture-of-experts model (12B active / 120B total) with 1M context. Supports text, image, GUI screenshots, and video reasoning.",
    },
    {
        "id": "baidu/ernie-5.5-flash-preview:free",
        "name": "Baidu: ERNIE 5.5 Flash Preview (free)",
        "provider": "Baidu",
        "tokens": "2.57T tokens",
        "ranking": "Programming (#41)",
        "context_length": "1.02M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "General & Reasoning",
        "description": "Baidu ERNIE 5.5 Flash Preview (280B total / 14B active parameters) with 1.02M context, delivering >200 tokens/sec.",
    },
    {
        "id": "thinkingmachines/inkling-2-small:free",
        "name": "Thinking Machines: Inkling 2 Small (free)",
        "provider": "Thinking Machines",
        "tokens": "1.32T tokens",
        "ranking": "Programming (#47) · Trivia (#25)",
        "context_length": "1.05M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text", "image"],
        "category": "Lightweight Reasoning",
        "description": "Multimodal mixture-of-experts reasoning model (17B active / 594B total) with 1.05M context for agentic coding and research.",
    },
    {
        "id": "nvidia/nemotron-3.5-lightning:free",
        "name": "NVIDIA: Nemotron 3.5 Lightning (free)",
        "provider": "NVIDIA",
        "tokens": "1.18T tokens",
        "ranking": "Programming (#46)",
        "context_length": "1M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Fast Reasoning",
        "description": "High-throughput mixture-of-experts model (3B active / 30B total) with 1M context for accelerated agentic workflows.",
    },
    {
        "id": "dots-studio/dots-3-note-preview:free",
        "name": "Dots Studio: Dots 3 Note Preview (free)",
        "provider": "Dots Studio",
        "tokens": "755B tokens",
        "ranking": "Synthesis",
        "context_length": "512K context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Note Synthesis",
        "description": "Open-weight mixture-of-experts model (16B active / 280B total) with 512K context for reasoning, coding, and multi-step agents.",
    },
    {
        "id": "particle-ai/particle-1:free",
        "name": "Particle AI: Particle 1 (free)",
        "provider": "Particle AI",
        "tokens": "494B tokens",
        "ranking": "Programming (#49) · Science (#32)",
        "context_length": "1.05M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Agent Planning",
        "description": "Token-efficient agentic coding model post-trained on DeepSeek V4.1 Flash with 1.05M context.",
    },
    {
        "id": "cohere/north-mini-code:free",
        "name": "Cohere: North Mini Code (free)",
        "provider": "Cohere",
        "tokens": "193B tokens",
        "ranking": "Programming",
        "context_length": "256K context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Programming & Code",
        "description": "Cohere sparse MoE agentic coding model (3B active / 30B total) with 256K context and up to 64K output tokens.",
    },
    {
        "id": "inclusionai/ling-3.2-flash:free",
        "name": "inclusionAI: Ling 3.2 Flash (free)",
        "provider": "inclusionAI",
        "tokens": "181B tokens",
        "ranking": "Vision & Text",
        "context_length": "1.05M context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text", "image"],
        "category": "Vision-Language Multimodal",
        "description": "Hybrid reasoning model accepting text and images with 1.05M context and up to 200 images per request.",
    },
    {
        "id": "recursive/flint-v1.0:free",
        "name": "Recursive: Flint v1.0 (free)",
        "provider": "Recursive",
        "tokens": "139B tokens",
        "ranking": "Deep Reasoning",
        "context_length": "262K context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Deep Reasoning",
        "description": "Eureka Machine MoE coding model (2.5B active / 40B total) with 262K context for codebase changes and bug fixes.",
    },
    {
        "id": "perplexity/pplx-decider-v1.3:free",
        "name": "Perplexity: Decider V1.3 (free)",
        "provider": "Perplexity",
        "tokens": "131B tokens",
        "ranking": "Academia (#14)",
        "context_length": "Decision State",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Decider & Router",
        "description": "Perplexity probabilistic decision model for classification, routing, and text state extraction.",
    },
    {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "name": "NVIDIA: Nemotron 3 Nano Omni (free)",
        "provider": "NVIDIA",
        "tokens": "101B tokens",
        "ranking": "Omni Reasoning",
        "context_length": "256K context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Omni Reasoning",
        "description": "Multimodal perception & context sub-agent (30B total / 3B active MoE) with 256K context and extended reasoning.",
    },
    {
        "id": "liquid/lfm-3-1.7b:free",
        "name": "LiquidAI: LFM 3 1.7B (free)",
        "provider": "LiquidAI",
        "tokens": "44.3B tokens",
        "ranking": "Liquid Architecture",
        "context_length": "256K context",
        "pricing": "$0/M input · $0/M output",
        "modalities": ["text"],
        "category": "Dynamic Neural Architecture",
        "description": "Compact 1.7B text model with 256K context post-trained for agents, tool calling, and structured outputs.",
    },
    {
        "id": "3dfun/academic-autonomous:local",
        "name": "Course Academic Core (Local Engine)",
        "provider": "IIIT Dharwad CV Lab",
        "tokens": "Local In-Memory",
        "ranking": "Academic Core",
        "context_length": "Unlimited",
        "pricing": "100% Free · Zero API Keys Needed",
        "modalities": ["text"],
        "category": "Autonomous Academic Engine",
        "description": "Local deterministic CV academic reasoning and autonomous tool execution engine for Group 17 3D Computer Vision course project.",
    },
]


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
    app_name: str = "3D Computer Vision Lab"
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
    signup_credits: int = field(default_factory=lambda: _env_int("CREDITS_SIGNUP_GRANT", _env_int("SIGNUP_CREDITS", 50)))
    daily_streak_credits: int = field(default_factory=lambda: _env_int("DAILY_STREAK_CREDITS", 10))
    lab_run_credits: int = field(default_factory=lambda: _env_int("LAB_RUN_CREDITS", 10))
    credit_costs: dict[str, int] = field(default_factory=lambda: {
        "draft": _env_int("CREDIT_COST_DRAFT", _env_int("CREDITS_DRAFT", 1)),
        "standard": _env_int("CREDIT_COST_STANDARD", _env_int("CREDITS_STANDARD", 3)),
        "high": _env_int("CREDIT_COST_HIGH", _env_int("CREDITS_HIGH", 5)),
    })

    # ---- providers ---------------------------------------------------
    generation_provider: str = field(default_factory=lambda: _env("GENERATION_PROVIDER", "local"))
    generation_api_url: str = field(default_factory=lambda: _env("GENERATION_API_URL", ""))
    generation_api_key: str = field(default_factory=lambda: _env("GENERATION_API_KEY", ""))
    generation_model: str = field(default_factory=lambda: _env("GENERATION_MODEL", "3dfun-recon-v1"))
    recon_warmup: bool = field(default_factory=lambda: _env_bool("RECON_WARMUP", False))

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

    @property
    def demo_login_enabled(self) -> bool:
        return self.environment.lower() in {"development", "test"}

    def validate_production(self) -> list[str]:
        if not self.is_production:
            return []
        problems = []
        for name, secret in (("JWT_ACCESS_SECRET", self.jwt_access_secret),
                             ("JWT_REFRESH_SECRET", self.jwt_refresh_secret)):
            if len(secret.encode()) < 32 or "dev-only" in secret:
                problems.append(f"{name} must be at least 32 bytes and replace the development default")
        if self.jwt_access_secret == self.jwt_refresh_secret:
            problems.append("JWT access and refresh secrets must be different")
        if not self.cookie_secure:
            problems.append("COOKIE_SECURE must be true in production")
        try:
            origin = urlsplit(self.app_url)
            https_origin = (origin.scheme == "https" and origin.hostname and not origin.username
                            and not origin.password and origin.path in {"", "/"}
                            and not origin.query and not origin.fragment)
        except ValueError:
            https_origin = False
        if not https_origin:
            problems.append("APP_URL must be the HTTPS origin of the deployment")
        return problems


settings = Settings()
settings.storage_root.mkdir(parents=True, exist_ok=True)
(ROOT / "data").mkdir(parents=True, exist_ok=True)


def generate_secret(nbytes: int = 32) -> str:
    """Helper used by scripts/rotate_secrets.py and the docs."""
    return secrets.token_urlsafe(nbytes)
