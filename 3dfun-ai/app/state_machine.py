"""Generation state machine.

Transitions are validated centrally so that no endpoint or worker can move a
job into an illegal state. The table below is the single source of truth and is
unit-tested.
"""
from __future__ import annotations

from .db import GenerationStatus as S

TRANSITIONS: dict[S, set[S]] = {
    S.DRAFT: {S.UPLOADING, S.QUEUED, S.CANCELLED, S.FAILED},
    S.UPLOADING: {S.QUEUED, S.VALIDATING, S.FAILED, S.CANCELLED},
    S.QUEUED: {S.VALIDATING, S.PROCESSING, S.CANCEL_REQUESTED, S.FAILED, S.CANCELLED},
    S.VALIDATING: {S.PROCESSING, S.QUEUED, S.FAILED, S.CANCEL_REQUESTED, S.CANCELLED},
    S.PROCESSING: {S.GENERATING_GEOMETRY, S.GENERATING_TEXTURE, S.FAILED,
                   S.CANCEL_REQUESTED, S.CANCELLED},
    S.GENERATING_GEOMETRY: {S.GENERATING_TEXTURE, S.PROCESSING, S.FAILED,
                            S.CANCEL_REQUESTED, S.CANCELLED},
    S.GENERATING_TEXTURE: {S.OPTIMIZING, S.RENDERING_PREVIEW, S.FAILED,
                           S.CANCEL_REQUESTED, S.CANCELLED},
    S.OPTIMIZING: {S.RENDERING_PREVIEW, S.COMPLETED, S.FAILED, S.CANCELLED},
    S.RENDERING_PREVIEW: {S.COMPLETED, S.FAILED, S.CANCELLED},
    S.COMPLETED: set(),
    S.FAILED: {S.QUEUED},              # retry
    S.CANCEL_REQUESTED: {S.CANCELLED, S.COMPLETED, S.FAILED},
    S.CANCELLED: {S.QUEUED},           # re-queue
}

TERMINAL = {S.COMPLETED, S.FAILED, S.CANCELLED}

# stage name emitted by the engine -> generation status
STAGE_TO_STATUS: dict[str, S] = {
    "VALIDATING": S.VALIDATING,
    "PREPROCESSING": S.PROCESSING,
    "DEPTH": S.PROCESSING,
    "VOLUME": S.GENERATING_GEOMETRY,
    "GEOMETRY": S.GENERATING_GEOMETRY,
    "UV": S.GENERATING_GEOMETRY,
    "TEXTURE": S.GENERATING_TEXTURE,
    "EXPORT": S.OPTIMIZING,
    "RENDER": S.RENDERING_PREVIEW,
    "COMPLETED": S.COMPLETED,
}

RETRYABLE_CODES = {"ENGINE_ERROR", "OUT_OF_MEMORY", "PROVIDER_UNAVAILABLE", "TIMEOUT",
                   "MOCK_FAILURE", "RENDER_FAILED"}


class InvalidTransition(Exception):
    def __init__(self, current: S, target: S):
        super().__init__(f"Illegal generation transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


def can_transition(current: S, target: S) -> bool:
    if current == target:
        return True
    return target in TRANSITIONS.get(current, set())


def assert_transition(current: S, target: S) -> None:
    if not can_transition(current, target):
        raise InvalidTransition(current, target)


def status_for_stage(stage: str) -> S:
    return STAGE_TO_STATUS.get(stage.upper(), S.PROCESSING)


def is_retryable(code: str | None) -> bool:
    return (code or "") in RETRYABLE_CODES


def mermaid() -> str:
    """Diagram source embedded in the documentation."""
    lines = ["stateDiagram-v2", "  [*] --> DRAFT"]
    for src, dests in TRANSITIONS.items():
        for dst in sorted(dests, key=lambda s: s.value):
            lines.append(f"  {src.value} --> {dst.value}")
    lines.append("  COMPLETED --> [*]")
    return "\n".join(lines)
