from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from app.detectron2_people import detect_people_instances
from app.schp_onnx import parse_schp_onnx


def _resize_max_side(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _postprocess_mask(mask01: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = (mask01 * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = 255
    return (keep > 0).astype(np.float32)


def _skin_mask_rules(rgb: np.ndarray) -> np.ndarray:
    """
    Conservative skin rules to reduce false positives on clothing.
    Returns float32 mask in [0, 1].
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    # Narrower chroma bounds than typical, to avoid beige/wood-like clothing.
    ycrcb_mask = (cr >= 140) & (cr <= 175) & (cb >= 85) & (cb <= 135) & (y >= 30)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # Skin hue tends to be in warm range; limit saturation/value to avoid white/gray clothes and vivid colors.
    hsv_mask = (((h <= 25) | (h >= 160)) & (s >= 25) & (s <= 200) & (v >= 40))

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    rgb_mask = (r > 60) & (g > 35) & (b > 20) & (r > g) & (g >= b) & (np.abs(r - g) > 10) & ((r - b) > 10)

    return (ycrcb_mask & hsv_mask & rgb_mask).astype(np.float32)


def skin_mask_heuristic(rgb: np.ndarray, *, max_side: int, min_area: int) -> np.ndarray:
    """
    Returns HxW float32 mask in [0, 1] where 1 indicates likely skin.
    Heuristic (no ML) to keep the service fully offline by default.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    small, scale = _resize_max_side(rgb, max_side=max_side)
    mask = _skin_mask_rules(small)
    mask = _postprocess_mask(mask, min_area=min_area)

    if scale != 1.0:
        height, width = rgb.shape[:2]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask


def _detect_largest_face(rgb_small: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
    except Exception:
        return None

    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if faces is None or len(faces) == 0:
        return None
    # Pick largest face by area.
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def skin_mask_adaptive(
    rgb: np.ndarray,
    *,
    max_side: int,
    min_area: int,
    threshold: float = 3.5,
    min_seed_pixels: int = 350,
) -> np.ndarray:
    """
    Face-adaptive skin mask: estimate the person's skin chroma from the face and segment pixels with similar chroma.
    This significantly reduces censoring clothing compared to pure color thresholds.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    small, scale = _resize_max_side(rgb, max_side=max_side)
    face = _detect_largest_face(small)
    if face is None:
        return skin_mask_seeded(
            rgb,
            max_side=max_side,
            min_area=min_area,
            threshold=threshold,
            min_seed_pixels=min_seed_pixels,
        )

    x, y, w, h = face
    # Use a central face patch to avoid hair/background; shrink box.
    pad_x = int(round(w * 0.18))
    pad_y = int(round(h * 0.22))
    x0 = max(0, x + pad_x)
    y0 = max(0, y + pad_y)
    x1 = min(small.shape[1], x + w - pad_x)
    y1 = min(small.shape[0], y + h - pad_y)
    if x1 <= x0 or y1 <= y0:
        return skin_mask_seeded(
            rgb,
            max_side=max_side,
            min_area=min_area,
            threshold=threshold,
            min_seed_pixels=min_seed_pixels,
        )

    face_roi = small[y0:y1, x0:x1]
    roi_rule = _skin_mask_rules(face_roi) > 0

    ycrcb_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2YCrCb)
    cr_roi = ycrcb_roi[:, :, 1].astype(np.float32)
    cb_roi = ycrcb_roi[:, :, 2].astype(np.float32)

    if int(roi_rule.sum()) < 200:
        # Not enough confident pixels to estimate; fallback.
        return skin_mask_seeded(
            rgb,
            max_side=max_side,
            min_area=min_area,
            threshold=threshold,
            min_seed_pixels=min_seed_pixels,
        )

    cr_samples = cr_roi[roi_rule].reshape(-1, 1)
    cb_samples = cb_roi[roi_rule].reshape(-1, 1)
    samples = np.concatenate([cr_samples, cb_samples], axis=1)

    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)
    cov = cov + np.eye(2, dtype=np.float32) * 1.0  # regularization
    try:
        cov_inv = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        return skin_mask_seeded(
            rgb,
            max_side=max_side,
            min_area=min_area,
            threshold=threshold,
            min_seed_pixels=min_seed_pixels,
        )

    ycrcb = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)
    y_chan = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    X = np.stack([cr.reshape(-1), cb.reshape(-1)], axis=1)
    d = X - mean.reshape(1, 2)
    dist2 = np.einsum("ni,ij,nj->n", d, cov_inv, d)
    thr2 = float(threshold) ** 2
    maha_mask = (dist2 <= thr2).reshape(small.shape[0], small.shape[1])

    # Extra constraints to reduce clothing false positives:
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.uint8)
    v = hsv[:, :, 2].astype(np.uint8)
    constraints = (y_chan >= 30.0) & (s >= 20) & (s <= 210) & (v >= 35)

    mask = (maha_mask & constraints).astype(np.float32)
    mask = _postprocess_mask(mask, min_area=min_area)

    if scale != 1.0:
        height, width = rgb.shape[:2]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask


def skin_mask_seeded(
    rgb: np.ndarray,
    *,
    max_side: int,
    min_area: int,
    threshold: float = 3.2,
    min_seed_pixels: int = 350,
) -> np.ndarray:
    """
    Seeded adaptive skin mask (no face required):
    1) Build a very conservative "seed" mask using strict rules.
    2) Fit a Cr/Cb Gaussian to those seed pixels.
    3) Expand skin region by Mahalanobis distance + extra constraints.

    This is usually less likely to censor clothing than pure thresholds, but may miss skin if seeds are too sparse.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    small, scale = _resize_max_side(rgb, max_side=max_side)

    seed = _skin_mask_rules(small) > 0
    # Additional conservative constraints for seeds.
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    s_seed = hsv[:, :, 1].astype(np.uint8)
    v_seed = hsv[:, :, 2].astype(np.uint8)
    seed &= (s_seed >= 25) & (s_seed <= 180) & (v_seed >= 50)

    if int(seed.sum()) < int(min_seed_pixels):
        return skin_mask_heuristic(rgb, max_side=max_side, min_area=min_area)

    ycrcb = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)
    y_chan = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    samples = np.stack([cr[seed], cb[seed]], axis=1)
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)
    cov = cov + np.eye(2, dtype=np.float32) * 1.0
    try:
        cov_inv = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        return skin_mask_heuristic(rgb, max_side=max_side, min_area=min_area)

    X = np.stack([cr.reshape(-1), cb.reshape(-1)], axis=1)
    d = X - mean.reshape(1, 2)
    dist2 = np.einsum("ni,ij,nj->n", d, cov_inv, d)
    thr2 = float(threshold) ** 2
    maha_mask = (dist2 <= thr2).reshape(small.shape[0], small.shape[1])

    # Constraints (avoid very dark/white areas that are often clothing/background).
    s = hsv[:, :, 1].astype(np.uint8)
    v = hsv[:, :, 2].astype(np.uint8)
    constraints = (y_chan >= 25.0) & (s >= 15) & (s <= 220) & (v >= 35)

    mask = (maha_mask & constraints).astype(np.float32)
    mask = _postprocess_mask(mask, min_area=min_area)

    if scale != 1.0:
        height, width = rgb.shape[:2]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask


@dataclass(frozen=True)
class SkinMaskResult:
    mask01: np.ndarray
    backend_requested: str
    backend_used: str
    backend_error: str | None
    schp_dataset: str | None


def skin_mask_with_meta(
    rgb: np.ndarray,
    *,
    max_side: int,
    min_area: int,
    backend: str = "cv",  # "auto" | "schp" | "cv"
    mode: str = "auto",
    threshold: float = 3.5,
    min_seed_pixels: int = 350,
    include_hair: bool = False,
    schp_model_path: str | None = None,
    schp_dataset: str = "lip",
    schp_expand_skin: bool = True,
    schp_max_expand_px: int = 0,
    schp_intra_threads: int = 0,
    schp_inter_threads: int = 0,
    mhp_d2_config: str | None = None,
    mhp_d2_weights: str | None = None,
    mhp_d2_score_thresh: float = 0.5,
    mhp_d2_max_people: int = 25,
    mhp_d2_person_class_id: int = 0,
    mhp_bbox_expand_ratio: float = 1.2,
) -> SkinMaskResult:
    backend_requested = (backend or "cv").strip().lower()
    backend = backend_requested

    if backend == "auto":
        try:
            if mhp_d2_config and mhp_d2_weights:
                inner = skin_mask_with_meta(
                    rgb,
                    max_side=max_side,
                    min_area=min_area,
                    backend="mhp",
                    mode=mode,
                    threshold=threshold,
                    min_seed_pixels=min_seed_pixels,
                    include_hair=include_hair,
                    schp_model_path=schp_model_path,
                    schp_dataset=schp_dataset,
                    schp_expand_skin=schp_expand_skin,
                    schp_max_expand_px=schp_max_expand_px,
                    schp_intra_threads=schp_intra_threads,
                    schp_inter_threads=schp_inter_threads,
                    mhp_d2_config=mhp_d2_config,
                    mhp_d2_weights=mhp_d2_weights,
                    mhp_d2_score_thresh=mhp_d2_score_thresh,
                    mhp_d2_max_people=mhp_d2_max_people,
                    mhp_d2_person_class_id=mhp_d2_person_class_id,
                    mhp_bbox_expand_ratio=mhp_bbox_expand_ratio,
                )
                return SkinMaskResult(
                    mask01=inner.mask01,
                    backend_requested=backend_requested,
                    backend_used=inner.backend_used,
                    backend_error=inner.backend_error,
                    schp_dataset=inner.schp_dataset,
                )
            inner = skin_mask_with_meta(
                rgb,
                max_side=max_side,
                min_area=min_area,
                backend="schp",
                mode=mode,
                threshold=threshold,
                min_seed_pixels=min_seed_pixels,
                include_hair=include_hair,
                schp_model_path=schp_model_path,
                schp_dataset=schp_dataset,
                schp_expand_skin=schp_expand_skin,
                schp_max_expand_px=schp_max_expand_px,
                schp_intra_threads=schp_intra_threads,
                schp_inter_threads=schp_inter_threads,
            )
            return SkinMaskResult(
                mask01=inner.mask01,
                backend_requested=backend_requested,
                backend_used=inner.backend_used,
                backend_error=inner.backend_error,
                schp_dataset=inner.schp_dataset,
            )
        except Exception as exc:
            fallback = skin_mask_with_meta(
                rgb,
                max_side=max_side,
                min_area=min_area,
                backend="cv",
                mode=mode,
                threshold=threshold,
                min_seed_pixels=min_seed_pixels,
                include_hair=False,
            )
            return SkinMaskResult(
                mask01=fallback.mask01,
                backend_requested=backend_requested,
                backend_used=fallback.backend_used,
                backend_error=f"AUTO backend failed; using CV: {exc}",
                schp_dataset=None,
            )

    if backend in {"mhp", "detectron2"}:
        if not schp_model_path:
            schp_model_path = "models/schp_lip.onnx"
        try:
            if not mhp_d2_config or not mhp_d2_weights:
                raise ValueError("MHP_D2_CONFIG and MHP_D2_WEIGHTS must be set for SKIN_BACKEND=mhp")
            h, w = rgb.shape[:2]
            bgr = rgb[:, :, ::-1]
            people = detect_people_instances(
                bgr,
                config_file=mhp_d2_config,
                weights_path=mhp_d2_weights,
                score_thresh=float(mhp_d2_score_thresh),
                max_people=int(mhp_d2_max_people),
                person_class_id=int(mhp_d2_person_class_id),
            )
            if not people:
                schp_fallback = skin_mask_with_meta(
                    rgb,
                    max_side=max_side,
                    min_area=min_area,
                    backend="schp",
                    mode=mode,
                    threshold=threshold,
                    min_seed_pixels=min_seed_pixels,
                    include_hair=include_hair,
                    schp_model_path=schp_model_path,
                    schp_dataset=schp_dataset,
                    schp_expand_skin=schp_expand_skin,
                    schp_max_expand_px=schp_max_expand_px,
                    schp_intra_threads=schp_intra_threads,
                    schp_inter_threads=schp_inter_threads,
                )
                return SkinMaskResult(
                    mask01=schp_fallback.mask01,
                    backend_requested=backend_requested,
                    backend_used=schp_fallback.backend_used,
                    backend_error="MHP found no people; using SCHP full-image parsing",
                    schp_dataset=schp_fallback.schp_dataset,
                )

            out = np.zeros((h, w), dtype=np.float32)
            used_spec_name: str | None = None
            for person in people:
                x1, y1, x2, y2 = _expand_bbox_xyxy(
                    person.bbox_xyxy, ratio=float(mhp_bbox_expand_ratio), img_w=w, img_h=h
                )
                crop_rgb = rgb[y1:y2, x1:x2]
                if crop_rgb.size == 0:
                    continue
                parsing, spec = parse_schp_onnx(
                    crop_rgb,
                    model_path=schp_model_path,
                    dataset=schp_dataset,
                    intra_threads=int(schp_intra_threads),
                    inter_threads=int(schp_inter_threads),
                )
                used_spec_name = spec.name
                crop_mask = _schp_skin_mask_from_parsing(
                    crop_rgb,
                    parsing=parsing,
                    spec_name=spec.name,
                    label_to_id=spec.label_to_id,
                    min_area=min_area,
                    threshold=threshold,
                    min_seed_pixels=min_seed_pixels,
                    expand_skin=schp_expand_skin,
                    max_expand_px=schp_max_expand_px,
                    include_hair=include_hair,
                )
                inst_mask_crop = person.mask[y1:y2, x1:x2].astype(np.float32)
                crop_mask = crop_mask * inst_mask_crop
                out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], crop_mask)

            out = _postprocess_mask(out, min_area=min_area)
            return SkinMaskResult(
                mask01=out,
                backend_requested=backend_requested,
                backend_used="mhp-detectron2",
                backend_error=None,
                schp_dataset=used_spec_name,
            )
        except Exception as exc:
            fallback = skin_mask_with_meta(
                rgb,
                max_side=max_side,
                min_area=min_area,
                backend="schp",
                mode=mode,
                threshold=threshold,
                min_seed_pixels=min_seed_pixels,
                include_hair=include_hair,
                schp_model_path=schp_model_path,
                schp_dataset=schp_dataset,
                schp_expand_skin=schp_expand_skin,
                schp_max_expand_px=schp_max_expand_px,
                schp_intra_threads=schp_intra_threads,
                schp_inter_threads=schp_inter_threads,
            )
            return SkinMaskResult(
                mask01=fallback.mask01,
                backend_requested=backend_requested,
                backend_used=fallback.backend_used,
                backend_error=f"MHP failed; using SCHP: {exc}",
                schp_dataset=fallback.schp_dataset,
            )

    if backend in {"schp", "schp-onnx", "onnx"}:
        try:
            if not schp_model_path:
                schp_model_path = "models/schp_lip.onnx"
            parsing, spec = parse_schp_onnx(
                rgb,
                model_path=schp_model_path,
                dataset=schp_dataset,
                intra_threads=int(schp_intra_threads),
                inter_threads=int(schp_inter_threads),
            )

            mask01 = _schp_skin_mask_from_parsing(
                rgb,
                parsing=parsing,
                spec_name=spec.name,
                label_to_id=spec.label_to_id,
                min_area=min_area,
                threshold=threshold,
                min_seed_pixels=min_seed_pixels,
                expand_skin=schp_expand_skin,
                max_expand_px=schp_max_expand_px,
                include_hair=include_hair,
            )
            return SkinMaskResult(
                mask01=mask01,
                backend_requested=backend_requested,
                backend_used="schp-onnx",
                backend_error=None,
                schp_dataset=spec.name,
            )
        except Exception as exc:
            fallback = skin_mask_with_meta(
                rgb,
                max_side=max_side,
                min_area=min_area,
                backend="cv",
                mode=mode,
                threshold=threshold,
                min_seed_pixels=min_seed_pixels,
                include_hair=False,
            )
            return SkinMaskResult(
                mask01=fallback.mask01,
                backend_requested=backend_requested,
                backend_used=fallback.backend_used,
                backend_error=f"SCHP failed; using CV: {exc}",
                schp_dataset=None,
            )

    if backend in {"cv", "opencv"}:
        mode = (mode or "auto").strip().lower()
        if mode == "heuristic":
            return SkinMaskResult(
                mask01=skin_mask_heuristic(rgb, max_side=max_side, min_area=min_area),
                backend_requested=backend_requested,
                backend_used="cv",
                backend_error=None,
                schp_dataset=None,
            )
        if mode in {"adaptive", "auto"}:
            return SkinMaskResult(
                mask01=skin_mask_adaptive(
                    rgb,
                    max_side=max_side,
                    min_area=min_area,
                    threshold=threshold,
                    min_seed_pixels=min_seed_pixels,
                ),
                backend_requested=backend_requested,
                backend_used="cv",
                backend_error=None,
                schp_dataset=None,
            )
        raise ValueError(f"Unknown SKIN_MODE: {mode}")

    raise ValueError(f"Unknown SKIN_BACKEND: {backend}")


def skin_mask(
    rgb: np.ndarray,
    *,
    max_side: int,
    min_area: int,
    backend: str = "cv",
    mode: str = "auto",
    threshold: float = 3.5,
    min_seed_pixels: int = 350,
    include_hair: bool = False,
    schp_model_path: str | None = None,
    schp_dataset: str = "lip",
    schp_expand_skin: bool = True,
    schp_max_expand_px: int = 0,
    schp_intra_threads: int = 0,
    schp_inter_threads: int = 0,
    mhp_d2_config: str | None = None,
    mhp_d2_weights: str | None = None,
    mhp_d2_score_thresh: float = 0.5,
    mhp_d2_max_people: int = 25,
    mhp_d2_person_class_id: int = 0,
    mhp_bbox_expand_ratio: float = 1.2,
) -> np.ndarray:
    return skin_mask_with_meta(
        rgb,
        max_side=max_side,
        min_area=min_area,
        backend=backend,
        mode=mode,
        threshold=threshold,
        min_seed_pixels=min_seed_pixels,
        include_hair=include_hair,
        schp_model_path=schp_model_path,
        schp_dataset=schp_dataset,
        schp_expand_skin=schp_expand_skin,
        schp_max_expand_px=schp_max_expand_px,
        schp_intra_threads=schp_intra_threads,
        schp_inter_threads=schp_inter_threads,
        mhp_d2_config=mhp_d2_config,
        mhp_d2_weights=mhp_d2_weights,
        mhp_d2_score_thresh=mhp_d2_score_thresh,
        mhp_d2_max_people=mhp_d2_max_people,
        mhp_d2_person_class_id=mhp_d2_person_class_id,
        mhp_bbox_expand_ratio=mhp_bbox_expand_ratio,
    ).mask01


def _expand_bbox_xyxy(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    ratio: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    ratio = float(max(1.0, ratio))
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    bw = (float(x2 - x1)) * ratio
    bh = (float(y2 - y1)) * ratio
    nx1 = int(max(0, np.floor(cx - bw * 0.5)))
    ny1 = int(max(0, np.floor(cy - bh * 0.5)))
    nx2 = int(min(int(img_w), np.ceil(cx + bw * 0.5)))
    ny2 = int(min(int(img_h), np.ceil(cy + bh * 0.5)))
    if nx2 <= nx1:
        nx2 = min(int(img_w), nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(int(img_h), ny1 + 1)
    return nx1, ny1, nx2, ny2


def _schp_skin_mask_from_parsing(
    rgb: np.ndarray,
    *,
    parsing: np.ndarray,
    spec_name: str,
    label_to_id: dict[str, int],
    min_area: int,
    threshold: float,
    min_seed_pixels: int,
    expand_skin: bool,
    max_expand_px: int,
    include_hair: bool,
) -> np.ndarray:
    spec_name = (spec_name or "lip").strip().lower()
    if spec_name == "lip":
        skin_label_ids = [
            label_to_id.get("Face"),
            label_to_id.get("Left-arm"),
            label_to_id.get("Right-arm"),
            label_to_id.get("Left-leg"),
            label_to_id.get("Right-leg"),
        ]
    elif spec_name == "atr":
        skin_label_ids = [
            label_to_id.get("Face"),
            label_to_id.get("Left-arm"),
            label_to_id.get("Right-arm"),
            label_to_id.get("Left-leg"),
            label_to_id.get("Right-leg"),
        ]
    elif spec_name == "pascal":
        # Pascal-Person-Part does not separate clothes vs skin; this is best-effort.
        skin_label_ids = [
            label_to_id.get("Head"),
            label_to_id.get("Torso"),
            label_to_id.get("Upper Arms"),
            label_to_id.get("Lower Arms"),
            label_to_id.get("Upper Legs"),
            label_to_id.get("Lower Legs"),
        ]
    else:
        skin_label_ids = []

    skin_label_ids = [int(x) for x in skin_label_ids if x is not None]
    if not skin_label_ids:
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    seed_mask = np.isin(parsing, np.array(skin_label_ids, dtype=np.uint8)).astype(np.uint8)
    if int(seed_mask.sum()) == 0:
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    mask01 = seed_mask.astype(np.float32)
    if expand_skin:
        person_mask = (parsing != 0).astype(np.uint8)
        fit_mask = seed_mask

        # Prefer fitting the Cr/Cb model from the face label when available.
        # This avoids sleeves/pants (rare SCHP mistakes) from contaminating the color model.
        face_id = label_to_id.get("Face")
        if spec_name in {"lip", "atr"} and face_id is not None:
            face_mask = (parsing == int(face_id)).astype(np.uint8)
            min_fit_pixels = max(60, int(min_seed_pixels) // 5)
            if int(face_mask.sum()) >= min_fit_pixels:
                fit_mask = face_mask
            else:
                # Fallback: use only seed pixels that pass conservative skin rules.
                rule_u8 = (_skin_mask_rules(rgb) > 0).astype(np.uint8)
                seed_rule = seed_mask & rule_u8
                if int(seed_rule.sum()) >= min_fit_pixels:
                    fit_mask = seed_rule

        mask01 = _expand_skin_from_seed(
            rgb,
            seed_mask=seed_mask,
            fit_mask=fit_mask,
            person_mask=person_mask,
            threshold=float(threshold),
            min_seed_pixels=int(min_seed_pixels),
            max_expand_px=int(max_expand_px),
        )

    if include_hair and spec_name in {"lip", "atr"}:
        hair_id = label_to_id.get("Hair")
        if hair_id is not None:
            hair_u8 = (parsing == int(hair_id)).astype(np.uint8)
            if int(hair_u8.sum()) > 0:
                mask01 = np.maximum(mask01, hair_u8.astype(np.float32))

    mask01 = _postprocess_mask(mask01, min_area=min_area)
    return mask01


def _expand_skin_from_seed(
    rgb: np.ndarray,
    *,
    seed_mask: np.ndarray,
    fit_mask: np.ndarray | None,
    person_mask: np.ndarray | None,
    threshold: float,
    min_seed_pixels: int,
    max_expand_px: int,
) -> np.ndarray:
    """
    Expand SCHP part-based skin mask using a Cr/Cb Gaussian estimated from seed pixels.

    This helps cover additional exposed skin (e.g. neck / torso) while remaining offline/CPU.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")
    if seed_mask.dtype != np.uint8:
        seed_u8 = seed_mask.astype(np.uint8)
    else:
        seed_u8 = seed_mask

    if fit_mask is None:
        fit_u8 = seed_u8
    elif fit_mask.dtype != np.uint8:
        fit_u8 = fit_mask.astype(np.uint8)
    else:
        fit_u8 = fit_mask

    if int(fit_u8.sum()) < int(min_seed_pixels):
        return seed_u8.astype(np.float32)

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y_chan = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    fit_bool = fit_u8 > 0
    samples = np.stack([cr[fit_bool], cb[fit_bool]], axis=1)
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)
    cov = cov + np.eye(2, dtype=np.float32) * 1.0
    try:
        cov_inv = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        return seed_u8.astype(np.float32)

    X = np.stack([cr.reshape(-1), cb.reshape(-1)], axis=1)
    d = X - mean.reshape(1, 2)
    dist2 = np.einsum("ni,ij,nj->n", d, cov_inv, d)
    thr2 = float(threshold) ** 2
    maha = (dist2 <= thr2).reshape(rgb.shape[0], rgb.shape[1])

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.uint8)
    v = hsv[:, :, 2].astype(np.uint8)
    constraints = (y_chan >= 25.0) & (s >= 15) & (s <= 220) & (v >= 35)

    cand = (maha & constraints).astype(np.uint8)
    if person_mask is not None:
        cand &= (person_mask > 0).astype(np.uint8)

    if int(max_expand_px) > 0:
        r = int(max_expand_px)
        k = 2 * r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        near_seed = cv2.dilate(seed_u8, kernel, iterations=1) > 0
        cand &= near_seed.astype(np.uint8)

    # Only keep expanded regions that touch the seed (reduces false positives on background / clothing).
    seed_bool = seed_u8 > 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((cand > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(cand, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        region = labels == label
        if not np.any(seed_bool & region):
            continue
        keep[region] = 1

    expanded = (seed_u8 > 0) | (keep > 0)
    return expanded.astype(np.float32)
