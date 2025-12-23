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
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

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

    channels = out.shape[0]
    if not (0 <= skin_channel_index < channels):
        raise ValueError(f"skin_channel_index out of range: {skin_channel_index} for channels={channels}")

    # Prefer softmax for mutually-exclusive classes; if model already outputs probabilities,
    # softmax still yields reasonable ordering.
    probs = _softmax_channel_first(out)
    skin = probs[skin_channel_index]
    other = np.max(np.delete(probs, skin_channel_index, axis=0), axis=0) if probs.shape[0] > 1 else np.zeros_like(skin)

    mask = skin > float(score_threshold)
    if float(margin) > 0:
        mask &= (skin - other) > float(margin)
    if require_max and probs.shape[0] > 1:
        mask &= skin == np.max(probs, axis=0)

    mask01 = mask.astype(np.float32)
    mask01 = _postprocess_mask(mask01, min_area=min_area)

    height, width = rgb.shape[:2]
    if (target_h, target_w) != (height, width):
        mask01 = cv2.resize(mask01, (width, height), interpolation=cv2.INTER_LINEAR)
        mask01 = np.clip(mask01, 0.0, 1.0).astype(np.float32)

    return mask01


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
    cv_max_side: int,
    cv_mode: str,
    cv_maha_threshold: float,
    cv_min_seed_pixels: int,
) -> np.ndarray:
    backend = (backend or "").strip().lower()
    if backend == "onnx_smp":
        try:
            return skin_mask_onnx_smp(
                rgb,
                model_path=model_path,
                score_threshold=score_threshold,
                skin_channel_index=skin_channel_index,
                require_max=require_max,
                margin=margin,
                min_area=min_area,
            )
        except Exception:
            # Fallback to CV if the model is missing or ONNX fails.
            return skin_mask(
                rgb,
                max_side=cv_max_side,
                min_area=min_area,
                mode=cv_mode,
                threshold=cv_maha_threshold,
                min_seed_pixels=cv_min_seed_pixels,
            )

    if backend == "cv":
        return skin_mask(
            rgb,
            max_side=cv_max_side,
            min_area=min_area,
            mode=cv_mode,
            threshold=cv_maha_threshold,
            min_seed_pixels=cv_min_seed_pixels,
        )

    raise ValueError(f"Unknown SKIN_BACKEND: {backend}")
