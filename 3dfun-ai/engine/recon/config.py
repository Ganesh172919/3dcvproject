"""Pipeline configuration for the 3DFUN AI reconstruction engine.

Every knob that influences output geometry/texture lives here so that a run is
fully reproducible from a serialised config (stored on each Generation row).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ENGINE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ENGINE_ROOT / "models"

Quality = Literal["draft", "standard", "high"]
OutputFormat = Literal["glb", "gltf", "obj", "ply", "stl"]


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single image-to-3D reconstruction."""

    # ---- preprocessing -------------------------------------------------
    canvas_size: int = 640            # working resolution (square canvas)
    depth_input_size: int = 518       # Depth Anything V2 native resolution
    matte_model: str = "u2netp"       # "u2netp" | "grabcut" | "none"
    alpha_erode_px: int = 1           # shrink matte to kill halo pixels
    crop_padding_pct: float = 0.04    # padding around the object bbox
    background: Literal["white", "gray", "midgray"] = "white"

    # ---- depth / height field -----------------------------------------
    depth_percentile_low: float = 2.0
    depth_percentile_high: float = 98.0
    depth_smooth_sigma: float = 2.0   # px, gaussian on the closeness field
    height_symmetrize: bool = True    # left/right symmetry prior toggle

    # ---- volume / silhouette inflation --------------------------------
    voxel_res: int = 128              # voxel grid resolution along X (width)
    inflation_power: float = 0.85     # t = tmax * (f + (1-f) * d ** power)
    # t_max as a fraction of the object width. Measured sweep (results/summary/
    # thickness_sweep.csv): 0.25 -> CD 0.1026, 0.35 -> 0.0910, 0.42 -> 0.0842,
    # 0.55 -> 0.0749, 0.70 -> 0.0639 (best), 0.85 -> 0.0691, 1.00 -> 0.0625.
    thickness_scale: float = 0.70
    thickness_floor_ratio: float = 0.55   # f: constant part of the slab thickness
    thickness_depth_gain: float = 0.45    # g: weight of the depth term (f + g = 1 by default)
    min_thickness: float = 0.02       # floor so the rim stays closed
    field_smooth_sigma: float = 0.8   # voxels, smoothing of the implicit field
    # (the old `depth_bias` knob was never read by the volume builder and has been
    #  replaced by the explicit `thickness_depth_gain` above.)

    # ---- mesh extraction ----------------------------------------------
    quality: Quality = "standard"
    target_faces: int = 24000
    taubin_iters: int = 12
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    remove_components_below: float = 0.02   # fraction of largest component volume

    # ---- texturing -----------------------------------------------------
    bake_texture: bool = True
    texture_size: int = 2048
    back_mirror_texture: bool = True
    generate_pbr_maps: bool = True
    ao_strength: float = 0.55

    # ---- export --------------------------------------------------------
    formats: tuple[str, ...] = ("glb", "obj", "ply")
    render_preview: bool = True
    render_turntable: bool = True
    turntable_frames: int = 24
    render_size: int = 512

    # ---- resource guards -----------------------------------------------
    memory_budget_mb: int = 1400      # hard ceiling used to size the atlas
    time_budget_seconds: int = 300    # worker-level watchdog

    # ---- provenance ----------------------------------------------------
    depth_model: str = "depth-anything-v2-small-int8"
    pipeline_version: str = "1.1.0"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["formats"] = list(self.formats)
        return d

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_overrides(self, **kwargs: Any) -> "PipelineConfig":
        data = asdict(self)
        known = {k: v for k, v in kwargs.items() if k in data}
        if "formats" in kwargs:
            known["formats"] = tuple(kwargs["formats"])
        return PipelineConfig(**{**data, **known})


QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "draft": dict(canvas_size=384, voxel_res=80, target_faces=8000,
                  texture_size=512, render_size=384, turntable_frames=10,
                  quality="draft"),
    "standard": dict(canvas_size=640, voxel_res=128, target_faces=24000,
                     texture_size=1024, render_size=512, turntable_frames=16,
                     quality="standard"),
    "high": dict(canvas_size=768, voxel_res=160, target_faces=48000,
                 texture_size=1536, render_size=640, turntable_frames=24,
                 quality="high"),
}


def preset_config(quality: Quality, **overrides: Any) -> PipelineConfig:
    base = PipelineConfig().with_overrides(**QUALITY_PRESETS[quality])
    return base.with_overrides(**overrides) if overrides else base
