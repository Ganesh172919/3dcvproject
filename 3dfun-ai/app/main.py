"""3DFUN AI - FastAPI application factory, middleware and API wiring.

Course: 3D Computer Vision Course Project · IIIT Dharwad
Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
         Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import metrics as metrics_mod
from .config import ROOT, settings
from .db import init_db
from .events import backend_name as event_backend
from .jobs import backend as queue_backend
from .jobs import runner
from .schemas import ApiError, now_iso, ok, request_id
from .routers import admin, assets, auth, billing, chat, gallery, generations, system, uploads

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("3dfun.api")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
CSP = ("default-src 'self'; "
       "script-src 'self' 'unsafe-inline'; "
       "style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; "
       "media-src 'self' blob: data:; "
       "worker-src 'self' blob:; "
       "connect-src 'self' blob:; "
       "frame-ancestors 'self'; "
       "base-uri 'self'; "
       "form-action 'self'")


def create_app() -> FastAPI:
    app = FastAPI(
        title="3DFUN AI API",
        version=settings.version,
        description=("Single-image to 3D reconstruction platform. "
                     "Course project, 3D Computer Vision, IIIT Dharwad."),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
    )
    init_db()
    problems = settings.validate_production()
    if problems:
        for p in problems:
            log.warning("production configuration problem: %s", p)

    @app.middleware("http")
    async def observability_and_security(request: Request, call_next):
        started = time.perf_counter()
        rid = request.headers.get("x-request-id") or request_id()
        request.state.request_id = rid
        origin = request.headers.get("origin")
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)
        metrics_mod.inc("http_requests_total", method=request.method, path=path_label,
                        status=str(response.status_code))
        metrics_mod.record_timing("http_request_duration_ms", elapsed, method=request.method)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        # static assets are imported as ES modules by the SPA, which requires
        # an explicit CORS grant when the app runs inside a sandboxed iframe
        if request.url.path.startswith(("/static/", "/assets/", "/figures/", "/presentations/", "/paper-files/")) or request.url.path in {"/paper/main.pdf", "/report.pdf", "/docs/report.pdf"}:
            response.headers.setdefault("Access-Control-Allow-Origin", "*")
            response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        policy = CSP
        if request.url.path.startswith("/embed/"):
            response.headers.pop("X-Frame-Options", None)
            policy = policy.replace("frame-ancestors 'self'", "frame-ancestors *")
        if request.url.path in {"/docs", "/redoc"}:
            policy = policy.replace("script-src 'self'", "script-src 'self' https://cdn.jsdelivr.net")
            policy = policy.replace("style-src 'self'", "style-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com")
            policy = policy.replace("img-src 'self'", "img-src 'self' https://fastapi.tiangolo.com")
            policy += "; font-src 'self' https://fonts.gstatic.com"
        response.headers.setdefault("Content-Security-Policy", policy)
        response.headers["X-Request-Id"] = rid
        response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.1f}"
        if settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        # The SPA authenticates with a bearer token, so reflected CORS without
        # credentials is safe; cookie flows remain protected by the CSRF token.
        if origin:
            trusted = {settings.app_url.rstrip("/"), "http://localhost:8000",
                       "http://127.0.0.1:8000"}
            response.headers.setdefault("Access-Control-Allow-Origin", origin)
            response.headers.setdefault("Vary", "Origin")
            if origin.rstrip("/") in trusted:
                response.headers.setdefault("Access-Control-Allow-Credentials", "true")
            response.headers.setdefault(
                "Access-Control-Allow-Headers",
                "content-type,x-csrf-token,idempotency-key,authorization")
            response.headers.setdefault(
                "Access-Control-Allow-Methods", "GET,POST,PATCH,DELETE,PUT,OPTIONS")
            response.headers.setdefault("Access-Control-Max-Age", "600")
        if request.method == "OPTIONS":
            return JSONResponse(status_code=204, content=None, headers=dict(response.headers))
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        body = exc.envelope()
        body["meta"]["requestId"] = getattr(request.state, "request_id", body["meta"]["requestId"])
        metrics_mod.inc("api_errors_total", code=exc.code)
        return JSONResponse(status_code=exc.status, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(x) for x in err["loc"][1:]), "issue": err["msg"]}
                   for err in exc.errors()]
        body = ApiError("VALIDATION_ERROR", "The request could not be processed.",
                        details=details).envelope()
        metrics_mod.inc("api_errors_total", code="VALIDATION_ERROR")
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        metrics_mod.inc("api_errors_total", code="INTERNAL")
        body = ApiError("INTERNAL_ERROR",
                        "Something went wrong on our side. Quote the request id when reporting.",
                        status=500, retryable=True).envelope()
        return JSONResponse(status_code=500, content=body)

    api = FastAPI(openapi_url=None)
    for module in (system, auth, uploads, generations, assets, gallery, billing, admin, chat):
        app.include_router(module.router, prefix="/api/v1")
    _ = api

    web_dir = ROOT / "web"
    if web_dir.exists():
        app.mount("/assets", StaticFiles(directory=web_dir / "assets"), name="assets")
        app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

    results_figures = ROOT / "results" / "figures"
    if results_figures.exists():
        app.mount("/figures", StaticFiles(directory=results_figures), name="figures")
    presentations_dir = ROOT / "presentations"
    if presentations_dir.exists():
        app.mount("/presentations", StaticFiles(directory=presentations_dir), name="presentations")

    paper_dir = ROOT / "paper"
    docs_dir = ROOT / "docs"

    @app.get("/paper/main.pdf", include_in_schema=False)
    def paper_pdf():
        pdf_path = paper_dir / "main.pdf"
        if pdf_path.exists():
            return FileResponse(pdf_path, media_type="application/pdf", filename="3DFUN_AI_IEEE_Paper.pdf")
        return JSONResponse(status_code=404, content={"error": "IEEE Paper PDF not found"})

    @app.get("/report.pdf", include_in_schema=False)
    @app.get("/docs/report.pdf", include_in_schema=False)
    def report_pdf():
        pdf_path = docs_dir / "report.pdf"
        if pdf_path.exists():
            return FileResponse(pdf_path, media_type="application/pdf", filename="3DFUN_AI_Project_Report.pdf")
        return JSONResponse(status_code=404, content={"error": "Project Report PDF not found"})

    @app.get("/", include_in_schema=False)
    def landing():
        index = web_dir / "public" / "index.html"
        if index.exists():
            return FileResponse(index)
        return RedirectResponse(url="/app/")

    spa_index = web_dir / "app" / "index.html"

    def _spa() -> FileResponse:
        # The single-page app owns every client route; all of its real assets
        # live under /static, so anything else under these prefixes is a route.
        return FileResponse(spa_index, headers={"Cache-Control": "no-cache"})

    app.get("/m/{slug}", include_in_schema=False)(_spa)
    app.get("/embed/{slug}", include_in_schema=False)(_spa)

    def legacy_route(request: Request) -> RedirectResponse:
        target = f"/app{request.url.path}"
        return RedirectResponse(f"{target}?{request.url.query}" if request.url.query else target)

    for _route in ("/studio", "/dashboard", "/gallery", "/admin", "/login", "/register",
                   "/settings", "/benchmark", "/forgot", "/reset-password", "/verify-email",
                   "/team", "/work"):
        app.get(_route, include_in_schema=False)(legacy_route)

    @app.get("/app", include_in_schema=False)
    @app.get("/app/", include_in_schema=False)
    @app.get("/app/{path:path}", include_in_schema=False)
    def spa(path: str = ""):
        return _spa()

    app.mount("/site", StaticFiles(directory=web_dir / "public", html=True), name="site")

    @app.on_event("startup")
    def _startup() -> None:
        log.info("3DFUN AI %s starting (env=%s, provider=%s, queue=%s, events=%s)",
                 settings.version, settings.environment, settings.generation_provider,
                 queue_backend()["kind"], event_backend())
        log.info("health endpoint: %s/api/v1/health", settings.app_url)

        def _warm_models():
            try:
                from recon.depth import _load_session
                _load_session("depth_anything_v2_small_quantized.onnx")
                from recon.preprocessing import _get_u2net_session, MODELS_DIR
                _get_u2net_session(MODELS_DIR / "u2netp.onnx")
                from recon.triposr import _session, available as triposr_avail
                if triposr_avail():
                    _session("triposr_encoder_int8.onnx")
                    _session("triposr_decoder.onnx")
                log.info("Local ONNX sessions warmed up successfully in memory.")
            except Exception as e:
                log.info("Model warm-up completed or deferred: %s", e)

        if settings.recon_warmup and settings.generation_provider == "local":
            from recon.runtime import CPU, runtime_status

            if runtime_status()["preferred"] == CPU:
                import threading

                threading.Thread(target=_warm_models, daemon=True, name="model-warmup").start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        log.info("graceful shutdown: draining generation runner")
        runner.shutdown(wait=True)
        log.info("runner stopped")

    return app


app = create_app()
