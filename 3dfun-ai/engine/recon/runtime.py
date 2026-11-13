"""Select a local ONNX provider and retry failed GPU inference on the CPU."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)
CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DIRECTML = "DmlExecutionProvider"


def requested_device() -> str:
    return os.environ.get("RECON_DEVICE", "cpu").strip().lower()


@lru_cache(maxsize=1)
def _available() -> tuple[str, ...]:
    import onnxruntime as ort

    return tuple(ort.get_available_providers())


def runtime_status() -> dict:
    requested = requested_device()
    available = _available()
    candidates = {"cuda": (CUDA,), "directml": (DIRECTML,), "auto": (CUDA, DIRECTML)}
    selected = next((name for name in candidates.get(requested, ()) if name in available), CPU)
    return {"requested": requested, "preferred": selected, "available": list(available)}


@lru_cache(maxsize=1)
def _preload_cuda() -> None:
    import onnxruntime as ort

    if os.name == "nt" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


class InferenceSession:
    def __init__(self, path: Path, options, provider: str | None = None):
        self.path = path
        self.options = options
        self._run_lock = Lock()
        desired = provider or runtime_status()["preferred"]
        try:
            self.session = self._open(desired)
        except Exception as error:
            if desired == CPU:
                raise
            log.warning("%s could not start %s. Using CPU (%s).", path.name, desired, type(error).__name__)
            self.session = self._open(CPU)

    def _open(self, provider: str):
        import onnxruntime as ort

        providers = [CPU]
        if provider == CUDA:
            _preload_cuda()
            # Bound the CUDA arena on small GPUs. CUDA libraries can also allocate separately.
            budget = max(256, int(os.environ.get("RECON_CUDA_MEMORY_MB", "2048")))
            providers.insert(0, (CUDA, {"gpu_mem_limit": budget * 1024 ** 2,
                                       "arena_extend_strategy": "kSameAsRequested",
                                       "cudnn_conv_algo_search": "HEURISTIC",
                                       "cudnn_conv_use_max_workspace": "0"}))
        elif provider == DIRECTML:
            self.options.enable_mem_pattern = False
            self.options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # The Iris Xe can spend minutes compiling fused graphs at session startup.
            # Keep DirectML kernels but skip whole-graph fusion for interactive jobs.
            self.options.add_session_config_entry("ep.dml.disable_graph_fusion", "1")
            providers.insert(0, DIRECTML)
        session = ort.InferenceSession(str(self.path), self.options, providers=providers)
        session.disable_fallback()
        return session

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, outputs, feeds):
        with self._run_lock:
            try:
                return self.session.run(outputs, feeds)
            except Exception as error:
                if self.provider == CPU:
                    raise
                log.warning("%s failed on %s. Retrying on CPU (%s).", self.path.name, self.provider, type(error).__name__)
                self.session = self._open(CPU)
                return self.session.run(outputs, feeds)


def inference_hardware(**stages: str | None) -> dict:
    providers = {stage: provider for stage, provider in stages.items() if provider}
    return {"device": "gpu" if any(provider != CPU for provider in providers.values()) else "cpu",
            "requested": requested_device(), "providers": providers, "inference_threads": 2}
