from __future__ import annotations

import os


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_int_list(name: str) -> list[int]:
    raw = os.getenv(name)
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of ints (got {raw!r})") from exc
    return out


class Settings:
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    strict_lossless: bool = _get_bool("STRICT_LOSSLESS", True)
    allow_lossy_jpeg: bool = _get_bool("ALLOW_LOSSY_JPEG", True)
    mask_max_side: int = int(os.getenv("MASK_MAX_SIDE", "640"))
    min_component_area: int = int(os.getenv("MIN_COMPONENT_AREA", "256"))

    # SCHP (human parsing) knobs (used only when SKIN_BACKEND=onnx_schp).
    schp_model_path: str = os.getenv("SCHP_MODEL_PATH", "models/schp.onnx").strip()
    # SCHP (LIP) commonly uses 473x473 in the original codebase.
    schp_input_size: int = int(os.getenv("SCHP_INPUT_SIZE", "473"))
    # Default to LIP label ids: face=13, leftArm=14, rightArm=15, leftLeg=16, rightLeg=17.
    schp_skin_class_ids: list[int] = _get_int_list("SCHP_SKIN_CLASS_IDS") or [13, 14, 15, 16, 17]
    schp_min_confidence: float = float(os.getenv("SCHP_MIN_CONFIDENCE", "0.0"))

    # Skin detection backends:
    # - onnx_smp: ONNX Skin/Clothes/Hair segmentation (best quality offline)
    # - onnx_schp: ONNX human parsing (best separation of skin vs clothing, requires SCHP model)
    # - cv: classic CV heuristics (fallback)
    _default_backend: str = "onnx_schp" if os.path.exists(schp_model_path) else "onnx_smp"
    skin_backend: str = os.getenv("SKIN_BACKEND", _default_backend).strip().lower()
    skin_model_path: str = os.getenv("SKIN_MODEL_PATH", "models/skin_smp.onnx").strip()
    skin_score_threshold: float = float(os.getenv("SKIN_SCORE_THRESHOLD", "0.5"))
    skin_channel_index: int = int(os.getenv("SKIN_CHANNEL_INDEX", "0"))
    skin_require_max: bool = _get_bool("SKIN_REQUIRE_MAX", True)
    skin_margin: float = float(os.getenv("SKIN_MARGIN", "0.05"))

    # Optional: also censor hair (only supported with the onnx_smp backend).
    censor_hair: bool = _get_bool("CENSOR_HAIR", False)
    hair_score_threshold: float = float(os.getenv("HAIR_SCORE_THRESHOLD", "0.5"))
    hair_channel_index: int = int(os.getenv("HAIR_CHANNEL_INDEX", "2"))
    hair_require_max: bool = _get_bool("HAIR_REQUIRE_MAX", True)
    hair_margin: float = float(os.getenv("HAIR_MARGIN", "0.05"))
    hair_min_component_area: int = int(os.getenv("HAIR_MIN_COMPONENT_AREA", "128"))

    # CV fallback knobs (used only when SKIN_BACKEND=cv)
    skin_mode: str = os.getenv("SKIN_MODE", "auto").strip().lower()  # "auto" | "adaptive" | "heuristic"
    skin_maha_threshold: float = float(os.getenv("SKIN_MAHA_THRESHOLD", "3.5"))
    skin_min_seed_pixels: int = int(os.getenv("SKIN_MIN_SEED_PIXELS", "350"))


settings = Settings()
