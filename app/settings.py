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


def _get_str_list(name: str) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
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
    # Default to LIP label id: hair=2.
    schp_hair_class_ids: list[int] = _get_int_list("SCHP_HAIR_CLASS_IDS") or [2]
    schp_min_confidence: float = float(os.getenv("SCHP_MIN_CONFIDENCE", "0.0"))

    # SegFormer Human Parse 24 (used only when SKIN_BACKEND=onnx_segformer24).
    segformer24_model_path: str = os.getenv("SEGFORMER24_MODEL_PATH", "models/model.onnx").strip()
    segformer24_input_size: int = int(os.getenv("SEGFORMER24_INPUT_SIZE", "512"))
    # Defaults match `yolo12138/segformer-b2-human-parse-24` id2label:
    # skin_around_neck_region=11, face=14, left_arm=15, right_arm=16, left_leg=17, right_leg=18, hair=2
    segformer24_skin_class_ids: list[int] = _get_int_list("SEGFORMER24_SKIN_CLASS_IDS") or [11, 14, 15, 16, 17, 18]
    segformer24_hair_class_ids: list[int] = _get_int_list("SEGFORMER24_HAIR_CLASS_IDS") or [2]
    segformer24_face_class_ids: list[int] = _get_int_list("SEGFORMER24_FACE_CLASS_IDS") or [14]
    segformer24_neck_from_face: bool = _get_bool("SEGFORMER24_NECK_FROM_FACE", True)
    segformer24_neck_extend_frac: float = float(os.getenv("SEGFORMER24_NECK_EXTEND_FRAC", "0.35"))
    segformer24_neck_expand_x_frac: float = float(os.getenv("SEGFORMER24_NECK_EXPAND_X_FRAC", "0.10"))
    segformer24_min_confidence: float = float(os.getenv("SEGFORMER24_MIN_CONFIDENCE", "0.0"))

    # LV-MHP v2 human parsing backend (used only when SKIN_BACKEND=onnx_lvmhpv2).
    # Provide an ONNX model trained on the LV-MHP v2 label set and configure the class IDs to treat as "skin parts"
    # (and optionally hair) via env vars.
    lvmhp_model_path: str = os.getenv("LVMHP_MODEL_PATH", "models/lvmhp_v2.onnx").strip()
    lvmhp_input_size: int = int(os.getenv("LVMHP_INPUT_SIZE", "512"))
    # Some exports (including detection-style models) may have multiple outputs; by default we auto-pick the one
    # that looks like a segmentation map. Set to a non-negative index to force-select an output.
    lvmhp_output_index: int = int(os.getenv("LVMHP_OUTPUT_INDEX", "-1"))
    # Some models output 58 classes with NO background class (0..57), while dataset IDs are often 1..58 with 0=background.
    # This offset is applied to configured class IDs before matching them against the model output.
    # Example: set to -1 if your config uses background-included IDs but the model outputs 58 classes without background.
    lvmhp_class_id_offset: int = int(os.getenv("LVMHP_CLASS_ID_OFFSET", "0"))
    # Preprocessing mode for the ONNX export:
    # - rgb_imagenet: RGB, ImageNet mean/std (common for SegFormer/DeepLab exports)
    # - schp_bgr_imagenet: BGR, ImageNet mean/std in SCHP order (common for SCHP/AugmentCE2P exports)
    lvmhp_preprocess: str = os.getenv("LVMHP_PREPROCESS", "rgb_imagenet").strip().lower()
    # MHParsNet-style decoding knobs (used when the ONNX output is kernel+mask features, not per-pixel logits).
    lvmhp_cate_threshold: float = float(os.getenv("LVMHP_CATE_THRESHOLD", "0.35"))
    lvmhp_mask_threshold: float = float(os.getenv("LVMHP_MASK_THRESHOLD", "0.5"))
    lvmhp_max_instances: int = int(os.getenv("LVMHP_MAX_INSTANCES", "200"))
    lvmhp_skin_class_ids: list[int] = _get_int_list("LVMHP_SKIN_CLASS_IDS")
    lvmhp_hair_class_ids: list[int] = _get_int_list("LVMHP_HAIR_CLASS_IDS")
    lvmhp_face_class_ids: list[int] = _get_int_list("LVMHP_FACE_CLASS_IDS")
    lvmhp_neck_from_face: bool = _get_bool("LVMHP_NECK_FROM_FACE", True)
    lvmhp_neck_extend_frac: float = float(os.getenv("LVMHP_NECK_EXTEND_FRAC", "0.35"))
    lvmhp_neck_expand_x_frac: float = float(os.getenv("LVMHP_NECK_EXPAND_X_FRAC", "0.10"))
    lvmhp_min_confidence: float = float(os.getenv("LVMHP_MIN_CONFIDENCE", "0.0"))

    # Skin detection backends:
    # - onnx_smp: ONNX Skin/Clothes/Hair segmentation (best quality offline)
    # - onnx_schp: ONNX human parsing (best separation of skin vs clothing, requires SCHP model)
    # - onnx_segformer24: ONNX human parsing (SegFormer Human Parse 24)
    # - onnx_lvmhpv2: ONNX human parsing trained on LV-MHP v2
    # - cv: classic CV heuristics (fallback)
    _default_backend: str = "onnx_schp" if os.path.exists(schp_model_path) else "onnx_smp"
    skin_backend: str = os.getenv("SKIN_BACKEND", _default_backend).strip().lower()
    skin_model_path: str = os.getenv("SKIN_MODEL_PATH", "models/skin_smp.onnx").strip()
    skin_score_threshold: float = float(os.getenv("SKIN_SCORE_THRESHOLD", "0.5"))
    skin_channel_index: int = int(os.getenv("SKIN_CHANNEL_INDEX", "0"))
    skin_require_max: bool = _get_bool("SKIN_REQUIRE_MAX", True)
    skin_margin: float = float(os.getenv("SKIN_MARGIN", "0.05"))

    # Optional: also censor hair (supported with onnx_smp, onnx_schp, onnx_segformer24 backends).
    censor_hair: bool = _get_bool("CENSOR_HAIR", True)
    hair_score_threshold: float = float(os.getenv("HAIR_SCORE_THRESHOLD", "0.5"))
    hair_channel_index: int = int(os.getenv("HAIR_CHANNEL_INDEX", "2"))
    hair_require_max: bool = _get_bool("HAIR_REQUIRE_MAX", True)
    hair_margin: float = float(os.getenv("HAIR_MARGIN", "0.05"))
    hair_min_component_area: int = int(os.getenv("HAIR_MIN_COMPONENT_AREA", "128"))

    # CV fallback knobs (used only when SKIN_BACKEND=cv)
    skin_mode: str = os.getenv("SKIN_MODE", "auto").strip().lower()  # "auto" | "adaptive" | "heuristic"
    skin_maha_threshold: float = float(os.getenv("SKIN_MAHA_THRESHOLD", "3.5"))
    skin_min_seed_pixels: int = int(os.getenv("SKIN_MIN_SEED_PIXELS", "350"))

    # Multi-backend fusion (optional): combine masks from multiple backends.
    # Modes:
    # - none: use a single backend (SKIN_BACKEND)
    # - overlap_filter: union the two masks, but keep only union-components that overlap their intersection
    mask_fusion_mode: str = os.getenv("MASK_FUSION_MODE", "none").strip().lower()
    mask_fusion_backends: list[str] = _get_str_list("MASK_FUSION_BACKENDS")
    mask_fusion_binarize_threshold: float = float(os.getenv("MASK_FUSION_BINARIZE_THRESHOLD", "0.5"))
    mask_fusion_overlap_dilate_px: int = int(os.getenv("MASK_FUSION_OVERLAP_DILATE_PX", "3"))
    mask_fusion_min_overlap_pixels: int = int(os.getenv("MASK_FUSION_MIN_OVERLAP_PIXELS", "25"))
    mask_fusion_min_overlap_frac: float = float(os.getenv("MASK_FUSION_MIN_OVERLAP_FRAC", "0.0"))
    mask_fusion_min_component_area: int = int(os.getenv("MASK_FUSION_MIN_COMPONENT_AREA", str(min_component_area)))


settings = Settings()
