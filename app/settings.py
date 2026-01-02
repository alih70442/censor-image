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
    censor_hair: bool = _get_bool("CENSOR_HAIR", True)

    # Skin detection.
    skin_backend: str = os.getenv("SKIN_BACKEND", "schp").strip().lower()  # "auto" | "schp" | "cv"

    # CV-only skin detection (used when SKIN_BACKEND=cv or as fallback).
    skin_mode: str = os.getenv("SKIN_MODE", "auto").strip().lower()  # "auto" | "adaptive" | "heuristic"
    skin_maha_threshold: float = float(os.getenv("SKIN_MAHA_THRESHOLD", "3.5"))
    skin_min_seed_pixels: int = int(os.getenv("SKIN_MIN_SEED_PIXELS", "350"))

    # SCHP (ONNX) human parsing.
    schp_model_path: str = os.getenv("SCHP_MODEL_PATH", "models/schp_lip.onnx")
    schp_dataset: str = os.getenv("SCHP_DATASET", "lip").strip().lower()
    schp_expand_skin: bool = _get_bool("SCHP_EXPAND_SKIN", True)
    schp_max_expand_px: int = int(os.getenv("SCHP_MAX_EXPAND_PX", "45"))
    schp_intra_threads: int = int(os.getenv("SCHP_INTRA_THREADS", "0"))
    schp_inter_threads: int = int(os.getenv("SCHP_INTER_THREADS", "0"))

    # MHP (Detectron2 person instances + SCHP on crops). Optional.
    mhp_d2_config: str = os.getenv("MHP_D2_CONFIG", "").strip()
    mhp_d2_weights: str = os.getenv("MHP_D2_WEIGHTS", "").strip()
    mhp_d2_score_thresh: float = float(os.getenv("MHP_D2_SCORE_THRESH", "0.5"))
    mhp_d2_max_people: int = int(os.getenv("MHP_D2_MAX_PEOPLE", "25"))
    mhp_d2_person_class_id: int = int(os.getenv("MHP_D2_PERSON_CLASS_ID", "0"))
    mhp_bbox_expand_ratio: float = float(os.getenv("MHP_BBOX_EXPAND_RATIO", "1.2"))


settings = Settings()
