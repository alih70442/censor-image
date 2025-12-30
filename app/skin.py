from __future__ import annotations

from functools import lru_cache

import numpy as np
import cv2


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

def _letterbox_to_size(rgb: np.ndarray, *, target_h: int, target_w: int) -> tuple[np.ndarray, float, int, int]:
    """
    Resize to fit within (target_h x target_w) preserving aspect ratio, then pad.
    Returns (boxed_rgb, scale, pad_x, pad_y) where pad is top-left offset.
    """
    target_h = int(target_h)
    target_w = int(target_w)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target size must be > 0")

    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("invalid image")

    scale = min(target_w / float(width), target_h / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return out, float(scale), int(pad_x), int(pad_y)


def _unletterbox_mask01(
    mask01_square: np.ndarray,
    *,
    orig_h: int,
    orig_w: int,
    target_h: int,
    target_w: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    target_h = int(target_h)
    target_w = int(target_w)
    if mask01_square.shape[:2] != (target_h, target_w):
        raise ValueError("mask01_square must match the letterbox size")
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    cropped = mask01_square[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)

def _letterbox_to_square(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    size = int(size)
    return _letterbox_to_size(rgb, target_h=size, target_w=size)


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


def skin_mask(
    rgb: np.ndarray,
    *,
    max_side: int,
    min_area: int,
    mode: str = "auto",
    threshold: float = 3.5,
    min_seed_pixels: int = 350,
) -> np.ndarray:
    mode = (mode or "auto").strip().lower()
    if mode == "heuristic":
        return skin_mask_heuristic(rgb, max_side=max_side, min_area=min_area)
    if mode == "adaptive":
        return skin_mask_adaptive(
            rgb,
            max_side=max_side,
            min_area=min_area,
            threshold=threshold,
            min_seed_pixels=min_seed_pixels,
        )
    if mode == "auto":
        return skin_mask_adaptive(
            rgb,
            max_side=max_side,
            min_area=min_area,
            threshold=threshold,
            min_seed_pixels=min_seed_pixels,
        )
    raise ValueError(f"Unknown SKIN_MODE: {mode}")


def _softmax_channel_first(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=0, keepdims=True)


def _smp_infer_logits(rgb: np.ndarray, *, model_path: str) -> tuple[np.ndarray, int, int]:
    session = _get_onnx_session(model_path)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape  # e.g. [1, 3, 512, 512]

    target_h = 512
    target_w = 512
    if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
        if isinstance(input_shape[2], int):
            target_h = int(input_shape[2])
        if isinstance(input_shape[3], int):
            target_w = int(input_shape[3])

    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    output_name = session.get_outputs()[0].name
    out = session.run([output_name], {input_name: x})[0]
    out = np.array(out)

    if out.ndim == 4:
        out = out[0]
    if out.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {out.shape}")

    return out, target_h, target_w

def _schp_expected_hw(model_path: str) -> tuple[int | None, int | None]:
    session = _get_onnx_session(model_path)
    shape = session.get_inputs()[0].shape  # e.g. [1, 3, 512, 512]
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None, None
    h = shape[2] if isinstance(shape[2], int) else None
    w = shape[3] if isinstance(shape[3], int) else None
    return h, w


def _schp_infer_logits(rgb_boxed: np.ndarray, *, model_path: str) -> np.ndarray:
    """
    Run SCHP human-parsing model and return logits (or scores) as a numpy array.
    Expected outputs typically look like:
      - [1, C, H, W] or [C, H, W]
    """
    session = _get_onnx_session(model_path)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    # SCHP (AugmentCE2P) expects BGR input normalized with ImageNet stats in BGR order:
    # mean=[0.406,0.456,0.485], std=[0.225,0.224,0.229]
    x = rgb_boxed[:, :, ::-1].astype(np.float32) / 255.0  # RGB -> BGR
    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    output_name = session.get_outputs()[0].name
    out = session.run([output_name], {input_name: x})[0]
    return np.array(out)


def skin_mask_onnx_schp(
    rgb: np.ndarray,
    *,
    model_path: str,
    input_size: int,
    skin_class_ids: tuple[int, ...],
    min_confidence: float,
    min_area: int,
) -> np.ndarray:
    """
    Human parsing (SCHP-style) backend.
    Builds a binary "skin" mask by selecting semantic classes that represent exposed skin
    (e.g. face/arms/legs), which generally avoids skin-colored clothing false positives.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    # If the ONNX model has a fixed input size, prefer it (avoids dimension mismatch).
    expected_h, expected_w = _schp_expected_hw(model_path)
    if expected_h is not None and expected_w is not None:
        target_h, target_w = int(expected_h), int(expected_w)
    else:
        input_size = int(input_size)
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        target_h, target_w = input_size, input_size

    skin_class_ids = tuple(int(i) for i in skin_class_ids)
    if not skin_class_ids:
        raise ValueError("skin_class_ids must not be empty")

    height, width = rgb.shape[:2]
    boxed, scale, pad_x, pad_y = _letterbox_to_size(rgb, target_h=target_h, target_w=target_w)
    out = _schp_infer_logits(boxed, model_path=model_path)

    if out.ndim == 4:
        out = out[0]
    if out.ndim == 3:
        logits = out  # [C, H, W]
        if logits.shape[1:] != (target_h, target_w):
            # Be permissive if the model returns a different spatial size.
            logits = cv2.resize(
                logits.transpose(1, 2, 0),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)
        pred = np.argmax(logits, axis=0).astype(np.int32)
        mask = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))

        min_confidence = float(min_confidence)
        if min_confidence > 0.0:
            # Probability of predicted class: exp(l_i - max) / sum exp(l_j - max)
            shifted = logits.astype(np.float32) - np.max(logits.astype(np.float32), axis=0, keepdims=True)
            exp = np.exp(shifted)
            denom = np.sum(exp, axis=0)
            maxexp = np.max(exp, axis=0)
            conf = (maxexp / np.maximum(denom, 1e-8)).astype(np.float32)
            mask &= conf >= min_confidence

        mask01_square = _postprocess_mask(mask.astype(np.float32), min_area=min_area)
    elif out.ndim == 2:
        # Some exports return class IDs directly as [H, W].
        pred = out.astype(np.int32)
        if pred.shape != (target_h, target_w):
            pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        mask01_square = _postprocess_mask(mask.astype(np.float32), min_area=min_area)
    else:
        raise RuntimeError(f"Unexpected SCHP model output shape: {out.shape}")

    return _unletterbox_mask01(
        mask01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def _smp_class_mask_from_probs(
    probs: np.ndarray,
    *,
    channel_index: int,
    score_threshold: float,
    require_max: bool,
    margin: float,
    min_area: int,
) -> np.ndarray:
    channels = probs.shape[0]
    if not (0 <= int(channel_index) < int(channels)):
        raise ValueError(f"channel_index out of range: {channel_index} for channels={channels}")

    channel_index = int(channel_index)
    score_threshold = float(score_threshold)
    margin = float(margin)

    cls = probs[channel_index]
    other = np.max(np.delete(probs, channel_index, axis=0), axis=0) if channels > 1 else np.zeros_like(cls)

    mask = cls > score_threshold
    if margin > 0:
        mask &= (cls - other) > margin
    if require_max and channels > 1:
        mask &= cls == np.max(probs, axis=0)

    mask01 = mask.astype(np.float32)
    return _postprocess_mask(mask01, min_area=min_area)


def class_mask_onnx_smp(
    rgb: np.ndarray,
    *,
    model_path: str,
    score_threshold: float,
    channel_index: int,
    require_max: bool = True,
    margin: float = 0.05,
    min_area: int,
) -> np.ndarray:
    """
    Generic ONNX mask extractor for the SMP skin/clothes/hair model.
    Returns a float32 mask in [0, 1] for the requested output channel.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    logits, target_h, target_w = _smp_infer_logits(rgb, model_path=model_path)
    probs = _softmax_channel_first(logits)
    mask01 = _smp_class_mask_from_probs(
        probs,
        channel_index=channel_index,
        score_threshold=score_threshold,
        require_max=require_max,
        margin=margin,
        min_area=min_area,
    )

    height, width = rgb.shape[:2]
    if (target_h, target_w) != (height, width):
        mask01 = cv2.resize(mask01, (width, height), interpolation=cv2.INTER_LINEAR)
        mask01 = np.clip(mask01, 0.0, 1.0).astype(np.float32)

    return mask01


def skin_hair_masks_onnx_smp(
    rgb: np.ndarray,
    *,
    model_path: str,
    skin_score_threshold: float,
    skin_channel_index: int,
    skin_require_max: bool,
    skin_margin: float,
    skin_min_area: int,
    hair_score_threshold: float,
    hair_channel_index: int,
    hair_require_max: bool,
    hair_margin: float,
    hair_min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-pass ONNX inference to get both skin + hair masks from the SMP model.
    Returns two float32 masks in [0, 1] (skin_mask01, hair_mask01).
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    logits, target_h, target_w = _smp_infer_logits(rgb, model_path=model_path)
    probs = _softmax_channel_first(logits)

    skin01 = _smp_class_mask_from_probs(
        probs,
        channel_index=skin_channel_index,
        score_threshold=skin_score_threshold,
        require_max=skin_require_max,
        margin=skin_margin,
        min_area=skin_min_area,
    )
    hair01 = _smp_class_mask_from_probs(
        probs,
        channel_index=hair_channel_index,
        score_threshold=hair_score_threshold,
        require_max=hair_require_max,
        margin=hair_margin,
        min_area=hair_min_area,
    )

    height, width = rgb.shape[:2]
    if (target_h, target_w) != (height, width):
        skin01 = cv2.resize(skin01, (width, height), interpolation=cv2.INTER_LINEAR)
        hair01 = cv2.resize(hair01, (width, height), interpolation=cv2.INTER_LINEAR)
        skin01 = np.clip(skin01, 0.0, 1.0).astype(np.float32)
        hair01 = np.clip(hair01, 0.0, 1.0).astype(np.float32)

    return skin01, hair01


@lru_cache(maxsize=4)
def _get_onnx_session(model_path: str):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is not installed") from exc

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])


def skin_mask_onnx_smp(
    rgb: np.ndarray,
    *,
    model_path: str,
    score_threshold: float,
    skin_channel_index: int,
    require_max: bool = True,
    margin: float = 0.05,
    min_area: int,
) -> np.ndarray:
    """
    ONNX skin/clothes/hair segmentation from:
      Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP

    Output is a float32 mask in [0, 1] (1 indicates skin).
    """
    return class_mask_onnx_smp(
        rgb,
        model_path=model_path,
        score_threshold=score_threshold,
        channel_index=skin_channel_index,
        require_max=require_max,
        margin=margin,
        min_area=min_area,
    )


def skin_mask_dispatch(
    rgb: np.ndarray,
    *,
    backend: str,
    model_path: str,
    score_threshold: float,
    skin_channel_index: int,
    require_max: bool,
    margin: float,
    min_area: int,
    schp_model_path: str = "models/schp.onnx",
    schp_input_size: int = 473,
    schp_skin_class_ids: tuple[int, ...] = (13, 14, 15, 16, 17),
    schp_min_confidence: float = 0.0,
    cv_max_side: int,
    cv_mode: str,
    cv_maha_threshold: float,
    cv_min_seed_pixels: int,
) -> np.ndarray:
    mask01, _used, _err = skin_mask_dispatch_info(
        rgb,
        backend=backend,
        model_path=model_path,
        score_threshold=score_threshold,
        skin_channel_index=skin_channel_index,
        require_max=require_max,
        margin=margin,
        min_area=min_area,
        schp_model_path=schp_model_path,
        schp_input_size=schp_input_size,
        schp_skin_class_ids=schp_skin_class_ids,
        schp_min_confidence=schp_min_confidence,
        cv_max_side=cv_max_side,
        cv_mode=cv_mode,
        cv_maha_threshold=cv_maha_threshold,
        cv_min_seed_pixels=cv_min_seed_pixels,
    )
    return mask01


def skin_mask_dispatch_info(
    rgb: np.ndarray,
    *,
    backend: str,
    model_path: str,
    score_threshold: float,
    skin_channel_index: int,
    require_max: bool,
    margin: float,
    min_area: int,
    schp_model_path: str = "models/schp.onnx",
    schp_input_size: int = 473,
    schp_skin_class_ids: tuple[int, ...] = (13, 14, 15, 16, 17),
    schp_min_confidence: float = 0.0,
    cv_max_side: int = 640,
    cv_mode: str = "auto",
    cv_maha_threshold: float = 3.5,
    cv_min_seed_pixels: int = 350,
) -> tuple[np.ndarray, str, str | None]:
    fallback_error: str | None = None
    backend = (backend or "").strip().lower()
    if backend == "onnx_schp":
        try:
            return (
                skin_mask_onnx_schp(
                    rgb,
                    model_path=schp_model_path,
                    input_size=schp_input_size,
                    skin_class_ids=schp_skin_class_ids,
                    min_confidence=schp_min_confidence,
                    min_area=min_area,
                ),
                "onnx_schp",
                None,
            )
        except Exception as exc:
            fallback_error = f"onnx_schp failed: {type(exc).__name__}: {exc}"
            # Fallback to SMP or CV if the model is missing or ONNX fails.
            backend = "onnx_smp"

    if backend == "onnx_smp":
        try:
            return (
                skin_mask_onnx_smp(
                    rgb,
                    model_path=model_path,
                    score_threshold=score_threshold,
                    skin_channel_index=skin_channel_index,
                    require_max=require_max,
                    margin=margin,
                    min_area=min_area,
                ),
                "onnx_smp",
                fallback_error,
            )
        except Exception as exc:
            msg = f"onnx_smp failed: {type(exc).__name__}: {exc}"
            fallback_error = msg if fallback_error is None else f"{fallback_error}; {msg}"
            # Fallback to CV if the model is missing or ONNX fails.
            return (
                skin_mask(
                    rgb,
                    max_side=cv_max_side,
                    min_area=min_area,
                    mode=cv_mode,
                    threshold=cv_maha_threshold,
                    min_seed_pixels=cv_min_seed_pixels,
                ),
                "cv",
                fallback_error,
            )

    if backend == "cv":
        return (
            skin_mask(
                rgb,
                max_side=cv_max_side,
                min_area=min_area,
                mode=cv_mode,
                threshold=cv_maha_threshold,
                min_seed_pixels=cv_min_seed_pixels,
            ),
            "cv",
            None,
        )

    raise ValueError(f"Unknown SKIN_BACKEND: {backend}")
