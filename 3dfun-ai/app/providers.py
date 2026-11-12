"""Generation provider abstraction and adapters.

The platform never talks to a vendor directly: it depends on this contract.
Three adapters ship in the box.

``local``  the in-process reconstruction engine (real image-to-3D, CPU).
``http``   a generic HTTP adapter (env-configured) for a commercial vendor.
``mock``   deterministic fake job used by the test-suite for fast CI runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import settings
from .schemas import ApiError

log = logging.getLogger(__name__)

ProviderState = str  # queued | running | succeeded | failed | cancelled


@dataclass
class GenerationInput:
    generation_id: str
    quality: str
    options: dict[str, Any]
    source_image: bytes
    workdir: Path
    progress: Callable[[str, int, str], None] = lambda stage, pct, msg: None
    cancelled: Callable[[], bool] = lambda: False


@dataclass
class ProviderJob:
    provider_job_id: str
    state: ProviderState
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderStatus:
    state: ProviderState
    progress: int = 0
    stage: str = ""
    artifact_dir: str | None = None
    manifest: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False


@dataclass
class CostEstimate:
    credits: int
    estimated_seconds: int
    provider: str


class GenerationProvider(Protocol):
    slug: str

    def create_generation(self, payload: GenerationInput) -> ProviderJob: ...
    def get_generation(self, provider_job_id: str) -> ProviderStatus: ...
    def cancel_generation(self, provider_job_id: str) -> None: ...
    def estimate_cost(self, quality: str, options: dict[str, Any]) -> CostEstimate: ...


# ---------------------------------------------------------------------------
# local engine adapter
# ---------------------------------------------------------------------------
class LocalReconstructionProvider:
    """Runs the in-repo reconstruction engine (ONNX depth + marching cubes)."""

    slug = "local"
    depth_model: str | None = None

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def estimate_cost(self, quality: str, options: dict[str, Any]) -> CostEstimate:
        base = {"draft": 90, "standard": 150, "high": 240}.get(quality, 150)
        if not options.get("turntable", True):
            base -= 15
        return CostEstimate(credits=settings.credit_costs.get(quality, 8),
                            estimated_seconds=base, provider=self.slug)

    def create_generation(self, payload: GenerationInput) -> ProviderJob:
        job_id = f"local_{payload.generation_id}"
        self._jobs[job_id] = {"status": ProviderStatus(state="running")}
        return ProviderJob(provider_job_id=job_id, state="running")

    def run_blocking(self, payload: GenerationInput, override_config: dict[str, Any] | None = None) -> ProviderStatus:
        """Synchronous execution (the worker calls this inside its thread)."""
        import sys
        engine_root = Path(__file__).resolve().parent.parent / "engine"
        if str(engine_root) not in sys.path:
            sys.path.insert(0, str(engine_root))
        try:
            from recon import preset_config, reconstruct  # type: ignore
        except Exception as exc:  # pragma: no cover - engine not importable
            return ProviderStatus(state="failed", error_code="ENGINE_UNAVAILABLE",
                                  error_message=f"Reconstruction engine unavailable: {exc}",
                                  retryable=False)
        quality = payload.quality if payload.quality in {"draft", "standard", "high"} else "standard"
        overrides: dict[str, Any] = {}
        if self.depth_model:
            overrides["depth_model"] = self.depth_model
        opts = payload.options or {}
        if opts.get("target_faces"):
            overrides["target_faces"] = int(opts["target_faces"])
        if opts.get("texture_quality"):
            overrides["texture_size"] = int(opts["texture_quality"])
        if opts.get("formats"):
            overrides["formats"] = tuple(dict.fromkeys(["glb", *opts["formats"]]))
        # A turntable costs one full rasterisation per frame; the option is
        # honoured, but the frame count stays bounded so a draft run on CPU
        # never spends more time encoding video than reconstructing geometry.
        overrides["render_turntable"] = bool(opts.get("turntable", True))
        if overrides["render_turntable"]:
            overrides["turntable_frames"] = {"draft": 10, "standard": 12, "high": 16}.get(quality, 12)
        overrides["render_preview"] = bool(opts.get("render_preview", True))
        overrides["bake_texture"] = bool(opts.get("delighting", True))
        overrides["height_symmetrize"] = bool(opts.get("symmetry", True))
        overrides["matte_model"] = "u2netp" if opts.get("background_removal", True) else "threshold"
        if override_config:
            overrides.update(override_config)
        cfg = preset_config(quality, **overrides)

        def progress(stage: str, pct: int, msg: str) -> None:
            if payload.cancelled():
                raise InterruptedError("Generation cancelled")
            payload.progress(stage, pct, msg)

        try:
            result = reconstruct(payload.source_image, payload.workdir, cfg,
                                 progress=progress, asset_name=f"3dfun-{payload.generation_id[-6:]}")
        except InterruptedError:
            return ProviderStatus(state="cancelled")
        except ValueError as exc:
            return ProviderStatus(state="failed", error_code="INVALID_INPUT",
                                  error_message=str(exc), retryable=False)
        except MemoryError:
            return ProviderStatus(state="failed", error_code="OUT_OF_MEMORY",
                                  error_message="The worker ran out of memory. Retry with draft quality.",
                                  retryable=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("reconstruction failed")
            return ProviderStatus(state="failed", error_code="ENGINE_ERROR",
                                  error_message=str(exc), retryable=True)
        return ProviderStatus(state="succeeded", progress=100, stage="COMPLETED",
                              artifact_dir=str(result.out_dir), manifest=result.manifest)

    def get_generation(self, provider_job_id: str) -> ProviderStatus:
        job = self._jobs.get(provider_job_id)
        return job["status"] if job else ProviderStatus(state="failed",
                                                       error_code="UNKNOWN_JOB",
                                                       error_message="Unknown job")

    def cancel_generation(self, provider_job_id: str) -> None:
        job = self._jobs.get(provider_job_id)
        if job:
            job["status"] = ProviderStatus(state="cancelled")


class MiDaSProvider(LocalReconstructionProvider):
    slug = "midas"
    model_name = "MiDaS v2.1 Small"
    depth_model = "midas-v2.1-small"

    def estimate_cost(self, quality: str, options: dict[str, Any]) -> CostEstimate:
        from recon.depth import midas_available

        if not midas_available():
            raise ApiError("MODEL_NOT_INSTALLED", "Install MiDaS with scripts/fetch_models.py --midas.", status=503)
        return super().estimate_cost(quality, options)


class TripoSRProvider(LocalReconstructionProvider):
    slug = "triposr"
    model_name = "TripoSR"

    def estimate_cost(self, quality: str, options: dict[str, Any]) -> CostEstimate:
        from recon.triposr import available

        if not available():
            raise ApiError("MODEL_NOT_INSTALLED", "Install the TripoSR weights with scripts/fetch_models.py --triposr.", status=503)
        estimate = super().estimate_cost(quality, options)
        estimate.estimated_seconds = {"draft": 75, "standard": 100, "high": 150}[quality]
        return estimate

    def run_blocking(self, payload: GenerationInput) -> ProviderStatus:
        from recon.triposr import ReconstructionCancelled, reconstruct_triposr

        try:
            result = reconstruct_triposr(payload.source_image, payload.workdir, payload.quality,
                                         payload.options, payload.progress, payload.cancelled)
        except ReconstructionCancelled:
            return ProviderStatus(state="cancelled")
        except ValueError as exc:
            return ProviderStatus(state="failed", error_code="INVALID_INPUT", error_message=str(exc))
        except Exception:
            log.exception("TripoSR reconstruction failed")
            return ProviderStatus(state="failed", error_code="ENGINE_ERROR", retryable=True,
                                  error_message="TripoSR failed. Try Draft or switch to Depth + silhouette.")
        return ProviderStatus(state="succeeded", progress=100, stage="COMPLETED",
                              artifact_dir=str(result.out_dir), manifest=result.manifest)


class HttpGenerationProvider:
    """Adapter for a REST vendor: POST /generations, GET /generations/{id}."""

    slug = "http"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: int = 30, max_retries: int = 3):
        self.base_url = (base_url or settings.generation_api_url).rstrip("/")
        self.api_key = api_key or settings.generation_api_key
        self.model = model or settings.generation_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if not self.base_url:
            raise ApiError("PROVIDER_MISCONFIGURED",
                           "GENERATION_API_URL is not configured", status=503)
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "X-Idempotency-Key": payload.get("idempotency_key", "") if payload else ""})
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt * 0.5, 8))
        raise ApiError("PROVIDER_UNAVAILABLE", f"Provider request failed: {last_error}",
                       status=503, retryable=True)

    def estimate_cost(self, quality: str, options: dict[str, Any]) -> CostEstimate:
        return CostEstimate(credits=settings.credit_costs.get(quality, 8),
                            estimated_seconds={"draft": 60, "standard": 120, "high": 200}.get(quality, 120),
                            provider=self.slug)

    def create_generation(self, payload: GenerationInput) -> ProviderJob:
        image_sha = hashlib.sha256(payload.source_image).hexdigest()
        body = {"model": self.model, "quality": payload.quality,
                "options": payload.options, "image_sha256": image_sha,
                "idempotency_key": payload.generation_id}
        resp = self._request("POST", "/generations", body)
        return ProviderJob(provider_job_id=resp["id"], state=resp.get("state", "queued"),
                           meta={"request": body})

    def get_generation(self, provider_job_id: str) -> ProviderStatus:
        resp = self._request("GET", f"/generations/{provider_job_id}")
        return ProviderStatus(state=resp.get("state", "running"),
                              progress=int(resp.get("progress", 0)),
                              stage=resp.get("stage", ""),
                              artifact_dir=resp.get("artifact_url"),
                              error_code=resp.get("error_code"),
                              error_message=resp.get("error_message"),
                              retryable=bool(resp.get("retryable", False)))

    def cancel_generation(self, provider_job_id: str) -> None:
        try:
            self._request("POST", f"/generations/{provider_job_id}/cancel", {})
        except ApiError:
            log.warning("provider cancel failed for %s", provider_job_id)


# ---------------------------------------------------------------------------
# mock adapter (tests / CI)
# ---------------------------------------------------------------------------
class MockProvider:
    """Deterministic adapter used by the automated test-suite."""

    slug = "mock"

    def __init__(self, fail: bool = False, delay: float = 0.0):
        self.fail = fail
        self.delay = delay

    def estimate_cost(self, quality: str, options: dict[str, Any]) -> CostEstimate:
        return CostEstimate(credits=settings.credit_costs.get(quality, 8),
                            estimated_seconds=5, provider=self.slug)

    def create_generation(self, payload: GenerationInput) -> ProviderJob:
        return ProviderJob(provider_job_id=f"mock_{payload.generation_id}", state="queued")

    def run_blocking(self, payload: GenerationInput, override_config: dict | None = None) -> ProviderStatus:
        if os.environ.get("THREEFUN_FORCE_MOCK_FAILURE") == "1":
            self.fail = True
        for pct, stage in ((10, "PREPROCESSING"), (40, "GEOMETRY"), (70, "TEXTURE"), (95, "EXPORT")):
            if self.delay:
                time.sleep(self.delay)
            payload.progress(stage, pct, f"mock {stage}")
        if self.fail:
            return ProviderStatus(state="failed", error_code="MOCK_FAILURE",
                                  error_message="Mock provider failure (test)", retryable=True)
        cube = _write_unit_cube(payload.workdir)
        written = _mock_artifacts(payload.workdir, cube)
        return ProviderStatus(state="succeeded", progress=100, stage="COMPLETED",
                              artifact_dir=str(payload.workdir),
                              manifest={"mesh": {"faces": 12, "vertices": 8,
                                                 "watertight_trimesh": True,
                                                 "boundary_edges": 0,
                                                 "nonmanifold_edges": 0,
                                                 "euler_characteristic": 2,
                                                 "watertight": True,
                                                 "bounding_box": [2.0, 2.0, 2.0]},
                                        "files": written,
                                        "provider": "mock",
                                        "engine": {"version": "mock-1.0.0",
                                                   "depth_model": "unit-cube"},
                                        "timings": {stage: 0.05 for stage in
                                                    ("preprocess", "depth", "volume",
                                                     "geometry", "uv", "texture",
                                                     "export", "render")},
                                        "texture": None, "warnings": []})

    def get_generation(self, provider_job_id: str) -> ProviderStatus:
        return ProviderStatus(state="succeeded", progress=100, stage="COMPLETED")

    def cancel_generation(self, provider_job_id: str) -> None:
        return None


def _write_unit_cube(workdir: Path) -> Path:
    """Emit a valid GLB cube so mock runs still produce a viewable asset."""
    import struct

    positions = [
        -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
        -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
    ]
    indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
               3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]
    bin_blob = struct.pack(f"<{len(positions)}f", *positions)
    bin_blob += struct.pack(f"<{len(indices)}H", *indices)
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    gltf = {
        "asset": {"version": "2.0", "generator": "3DFUN mock provider"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
             "min": [-1, -1, -1], "max": [1, 1, 1]},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions) * 4, "target": 34962},
            {"buffer": 0, "byteOffset": len(positions) * 4,
             "byteLength": len(indices) * 2, "target": 34963},
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(bin_blob)
    out = workdir / "model.glb"
    with open(out, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        fh.write(bin_blob)
    return out


def _mock_artifacts(workdir: Path, glb: Path) -> dict[str, str]:
    """Write sibling OBJ/PLY/STL/PNG previews so every download path is exercised."""
    import numpy as np

    verts = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5),
             (0, 5, 4), (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3),
             (1, 2, 6), (1, 6, 5)]
    written = {"glb": glb.name}
    obj = workdir / "model.obj"
    with open(obj, "w", encoding="utf-8") as fh:
        fh.write("# 3DFUN AI mock fixture\nmtllib model.mtl\no cube\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        fh.write("usemtl mock\n")
        for f in faces:
            fh.write("f {} {} {}\n".format(*(i + 1 for i in f)))
    (workdir / "model.mtl").write_text(
        "newmtl mock\nKd 0.60 0.75 0.80\nKa 1.0 1.0 1.0\nKs 0.10 0.10 0.10\nNs 40\n",
        encoding="utf-8")
    written["obj"] = obj.name
    written["mtl"] = "model.mtl"

    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
        mesh.export(str(workdir / "model.ply"))
        mesh.export(str(workdir / "model.stl"))
        written["ply"] = "model.ply"
        written["stl"] = "model.stl"
    except Exception as exc:                              # pragma: no cover
        log.warning("mock: trimesh export skipped (%s)", exc)

    try:
        from PIL import Image

        Image.new("RGB", (256, 256), (36, 44, 54)).save(workdir / "preview.png")
        written["png"] = "preview.png"
    except Exception as exc:                              # pragma: no cover
        log.warning("mock: preview render skipped (%s)", exc)
    return written


_REGISTRY: dict[str, Callable[[], GenerationProvider]] = {
    "local": LocalReconstructionProvider,
    "triposr": TripoSRProvider,
    "midas": MiDaSProvider,
    "http": HttpGenerationProvider,
    "mock": MockProvider,
}


def get_provider(slug: str | None = None) -> GenerationProvider:
    key = (slug or settings.generation_provider or "local").lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ApiError("PROVIDER_NOT_FOUND", f"Unknown provider '{key}'", status=400)
    return factory()


def provider_health(slug: str) -> dict[str, Any]:
    """Cheap liveness probe shown on the admin provider page."""
    try:
        provider = get_provider(slug)
        est = provider.estimate_cost("draft", {})
        return {"slug": slug, "healthy": True, "estimated_seconds": est.estimated_seconds}
    except Exception as exc:
        return {"slug": slug, "healthy": False, "error": str(exc)}
