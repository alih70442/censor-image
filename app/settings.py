from __future__ import annotations

import os


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings:
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    strict_lossless: bool = _get_bool("STRICT_LOSSLESS", True)
    allow_lossy_jpeg: bool = _get_bool("ALLOW_LOSSY_JPEG", True)
    mask_max_side: int = int(os.getenv("MASK_MAX_SIDE", "640"))
    min_component_area: int = int(os.getenv("MIN_COMPONENT_AREA", "256"))

    # Skin detection: CV-only.
    skin_mode: str = os.getenv("SKIN_MODE", "auto").strip().lower()  # "auto" | "adaptive" | "heuristic"
    skin_maha_threshold: float = float(os.getenv("SKIN_MAHA_THRESHOLD", "3.5"))
    skin_min_seed_pixels: int = int(os.getenv("SKIN_MIN_SEED_PIXELS", "350"))


settings = Settings()
