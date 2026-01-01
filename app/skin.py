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


def _add_neck_band_from_anchor(
    base_mask: np.ndarray,
    *,
    pred: np.ndarray,
    primary_anchor_class_ids: tuple[int, ...],
    fallback_anchor_class_ids: tuple[tuple[int, ...], ...] = (),
    extend_frac: float,
    expand_x_frac: float,
) -> np.ndarray:
    """
    LIP-style parsing often predicts neck pixels as "upper clothes" (no dedicated neck class).
    This adds a small band below the detected face region to cover the neck area.
    """
    if base_mask.dtype != np.bool_:
        base_mask = base_mask.astype(bool)

    def _bbox_for_ids(class_ids: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        class_ids = tuple(int(i) for i in class_ids)
        if not class_ids:
            return None
        mask = np.isin(pred, np.array(class_ids, dtype=np.int32))
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return None
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        return y0, y1, x0, x1

    bbox = _bbox_for_ids(primary_anchor_class_ids)
    if bbox is None:
        for ids in fallback_anchor_class_ids:
            bbox = _bbox_for_ids(ids)
            if bbox is not None:
                break
    if bbox is None:
        return base_mask

    y0, y1, x0, x1 = bbox
    face_h = max(1, y1 - y0)
    face_w = max(1, x1 - x0)

    extend_frac = float(extend_frac)
    expand_x_frac = float(expand_x_frac)
    if extend_frac <= 0:
        return base_mask

    y_neck0 = y1
    y_neck1 = min(pred.shape[0], y1 + int(round(face_h * extend_frac)))
    if y_neck1 <= y_neck0:
        return base_mask

    x_pad = int(round(face_w * max(0.0, expand_x_frac)))
    x_neck0 = max(0, x0 - x_pad)
    x_neck1 = min(pred.shape[1], x1 + x_pad)
    if x_neck1 <= x_neck0:
        return base_mask

    out = base_mask.copy()
    out[y_neck0:y_neck1, x_neck0:x_neck1] = True
    return out


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


def _segformer24_expected_hw(model_path: str) -> tuple[int | None, int | None]:
    session = _get_onnx_session(model_path)
    shape = session.get_inputs()[0].shape  # e.g. [1, 3, 512, 512]
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None, None
    h = shape[2] if isinstance(shape[2], int) else None
    w = shape[3] if isinstance(shape[3], int) else None
    return h, w


def _segformer24_infer_logits(rgb_boxed: np.ndarray, *, model_path: str) -> np.ndarray:
    """
    Run SegFormer human parsing model and return logits/scores (or class IDs) as numpy.
    Expected outputs typically look like:
      - [1, C, H, W] or [C, H, W] or sometimes [1, H, W]
    """
    session = _get_onnx_session(model_path)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    # Most SegFormer exports use RGB input normalized with ImageNet stats.
    x = rgb_boxed.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    output_name = session.get_outputs()[0].name
    out = session.run([output_name], {input_name: x})[0]
    return np.array(out)


def _onnx_expected_hw(model_path: str) -> tuple[int | None, int | None]:
    session = _get_onnx_session(model_path)
    shape = session.get_inputs()[0].shape  # e.g. [1, 3, 512, 512]
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None, None
    h = shape[2] if isinstance(shape[2], int) else None
    w = shape[3] if isinstance(shape[3], int) else None
    return h, w


def _parsing_infer_outputs(rgb_boxed: np.ndarray, *, model_path: str, preprocess: str) -> tuple[list[str], list[np.ndarray]]:
    """
    Run an ONNX human-parsing model and return all raw outputs as numpy arrays.

    `preprocess` controls channel order + normalization:
      - rgb_imagenet: RGB in [0,1], ImageNet mean/std (common for SegFormer/DeepLab exports)
      - schp_bgr_imagenet: BGR in [0,1], ImageNet mean/std in SCHP order (common for SCHP/AugmentCE2P exports)
    """
    session = _get_onnx_session(model_path)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    preprocess = (preprocess or "rgb_imagenet").strip().lower()

    if preprocess in {"rgb_imagenet", "imagenet_rgb", "segformer"}:
        x = rgb_boxed.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    elif preprocess in {"schp_bgr_imagenet", "bgr_imagenet", "schp"}:
        x = rgb_boxed[:, :, ::-1].astype(np.float32) / 255.0  # RGB -> BGR
        mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    else:
        raise ValueError(f"Unknown preprocess mode: {preprocess!r}")

    outs = session.run(None, {input_name: x})
    names = [o.name for o in session.get_outputs()]
    return names, [np.array(o) for o in outs]


def _select_parsing_output(
    output_names: list[str],
    outputs: list[np.ndarray],
    *,
    target_h: int,
    target_w: int,
    output_index: int,
) -> np.ndarray:
    if not outputs:
        raise RuntimeError("Model returned no outputs")
    if len(outputs) != len(output_names):
        # Be defensive; onnxruntime should keep these aligned.
        output_names = [f"output_{i}" for i in range(len(outputs))]

    output_index = int(output_index)
    if output_index >= 0:
        if output_index >= len(outputs):
            raise RuntimeError(f"LVMHP_OUTPUT_INDEX={output_index} out of range for outputs={len(outputs)}")
        return np.array(outputs[output_index])

    # Auto-pick: find the output that most resembles a parsing map at (target_h,target_w).
    best_i: int | None = None
    best_score: int = -10_000

    for i, out in enumerate(outputs):
        arr = np.array(out)

        # Drop batch dim if it's exactly 1.
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr_s = arr[0]
        else:
            arr_s = arr

        score = 0
        if arr_s.ndim == 2 and arr_s.shape == (target_h, target_w):
            score = 1000
        elif arr_s.ndim == 3:
            if arr_s.shape[1:] == (target_h, target_w):
                score = 900  # [C,H,W]
            elif arr_s.shape[:2] == (target_h, target_w):
                score = 850  # [H,W,C]
        elif arr_s.ndim == 4:
            # Common: [N,C,H,W] but batch isn't 1 (rare), or [N,H,W,C]
            if arr_s.shape[-2:] == (target_h, target_w):
                score = 700
            elif arr_s.shape[1:3] == (target_h, target_w):
                score = 650

        # Prefer float logits over unrelated integer tensors (but allow int32 maps).
        if arr_s.dtype.kind == "f":
            score += 10
        elif arr_s.dtype.kind in {"i", "u"}:
            score += 5

        # Prefer outputs with a plausible channel count.
        if arr_s.ndim == 3:
            c = arr_s.shape[0] if arr_s.shape[1:] == (target_h, target_w) else arr_s.shape[2]
            if 2 <= int(c) <= 256:
                score += 10

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None or best_score < 600:
        shapes = ", ".join(
            f"{name}:{tuple(np.array(o).shape)}:{np.array(o).dtype}" for name, o in zip(output_names, outputs)
        )
        raise RuntimeError(
            "Could not auto-select a parsing output; set LVMHP_OUTPUT_INDEX. "
            f"Expected something like [1,C,H,W]/[C,H,W]/[H,W] at ({target_h},{target_w}). Outputs: {shapes}"
        )

    return np.array(outputs[best_i])


def _parsing_logits_or_pred(
    out: np.ndarray,
    *,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Normalize common parsing-model outputs to either:
      - (logits_chw, pred_hw) for multi-class logits, or
      - (None, pred_hw) for pre-argmax class IDs.
    """
    out = np.array(out)

    if out.ndim == 4:
        # [1, C, H, W] or [1, H, W, C]
        if out.shape[0] == 1:
            out = out[0]
        else:
            raise RuntimeError(f"Unexpected batch dimension in output: {out.shape}")

    if out.ndim == 3 and out.shape[0] == 1:
        # Some exports return class IDs as [1, H, W].
        out = out[0]

    if out.ndim == 3 and out.shape[-1] == 1 and out.shape[0] == target_h and out.shape[1] == target_w:
        # [H, W, 1] class IDs.
        out = out[:, :, 0]

    if out.ndim == 3:
        # [C, H, W] or [H, W, C]
        if out.shape[0] == target_h and out.shape[1] == target_w:
            logits = out.transpose(2, 0, 1)
        else:
            logits = out

        if logits.ndim != 3:
            raise RuntimeError(f"Unexpected logits shape: {out.shape}")

        if logits.shape[1:] != (target_h, target_w):
            logits = cv2.resize(
                logits.transpose(1, 2, 0),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)

        pred = np.argmax(logits, axis=0).astype(np.int32)
        return logits.astype(np.float32), pred

    if out.ndim == 2:
        pred = out.astype(np.int32)
        if pred.shape != (target_h, target_w):
            pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return None, pred

    raise RuntimeError(f"Unexpected model output shape: {out.shape}")


def _parsing_confidence(logits_chw: np.ndarray) -> np.ndarray:
    shifted = logits_chw.astype(np.float32) - np.max(logits_chw.astype(np.float32), axis=0, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=0)
    maxexp = np.max(exp, axis=0)
    return (maxexp / np.maximum(denom, 1e-8)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    # Numerically stable sigmoid.
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def _normalize_class_ids_for_model(
    class_ids: tuple[int, ...],
    *,
    num_classes: int,
    offset: int,
) -> tuple[int, ...]:
    ids = [int(i) + int(offset) for i in class_ids]
    ids = [i for i in ids if 0 <= i < int(num_classes)]
    # If user passed 1-based IDs for a 0-based model (and offset wasn't set), try to help.
    if not ids and class_ids:
        maybe = [int(i) - 1 for i in class_ids]
        maybe = [i for i in maybe if 0 <= i < int(num_classes)]
        ids = maybe
    return tuple(sorted(set(ids)))


def _bbox_from_bool_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return y0, y1, x0, x1


def _add_neck_band_from_anchor_mask(
    base_mask: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    extend_frac: float,
    expand_x_frac: float,
) -> np.ndarray:
    if base_mask.dtype != np.bool_:
        base_mask = base_mask.astype(bool)
    if anchor_mask.dtype != np.bool_:
        anchor_mask = anchor_mask.astype(bool)
    bbox = _bbox_from_bool_mask(anchor_mask)
    if bbox is None:
        return base_mask

    y0, y1, x0, x1 = bbox
    face_h = max(1, y1 - y0)
    face_w = max(1, x1 - x0)

    extend_frac = float(extend_frac)
    expand_x_frac = float(expand_x_frac)
    if extend_frac <= 0:
        return base_mask

    y_neck0 = y1
    y_neck1 = min(base_mask.shape[0], y1 + int(round(face_h * extend_frac)))
    if y_neck1 <= y_neck0:
        return base_mask

    x_pad = int(round(face_w * max(0.0, expand_x_frac)))
    x_neck0 = max(0, x0 - x_pad)
    x_neck1 = min(base_mask.shape[1], x1 + x_pad)
    if x_neck1 <= x_neck0:
        return base_mask

    out = base_mask.copy()
    out[y_neck0:y_neck1, x_neck0:x_neck1] = True
    return out


def _mhparsnet_part_union_masks(
    output_names: list[str],
    outputs: list[np.ndarray],
    *,
    target_h: int,
    target_w: int,
    class_id_offset: int,
    skin_class_ids: tuple[int, ...],
    hair_class_ids: tuple[int, ...],
    face_class_ids: tuple[int, ...],
    cate_threshold: float,
    mask_threshold: float,
    max_instances: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode MHParsNet-style outputs (category logits + dynamic kernels + mask feature map).

    Returns (skin01_square, hair01_square, face_mask_bool_square) at (target_h,target_w).
    """
    name_to_idx = {str(n): i for i, n in enumerate(output_names)}
    required = ("parts_cate", "parts_kernel", "parts_mask_feat")
    if not all(k in name_to_idx for k in required):
        raise RuntimeError("MHParsNet outputs not found (expected parts_cate/parts_kernel/parts_mask_feat)")

    cate = np.array(outputs[name_to_idx["parts_cate"]], dtype=np.float32)
    kernel = np.array(outputs[name_to_idx["parts_kernel"]], dtype=np.float32)
    feat = np.array(outputs[name_to_idx["parts_mask_feat"]], dtype=np.float32)

    if cate.ndim != 3 or kernel.ndim != 3 or feat.ndim != 4:
        raise RuntimeError(f"Unexpected MHParsNet output shapes: cate={cate.shape}, kernel={kernel.shape}, feat={feat.shape}")
    if cate.shape[0] != 1 or kernel.shape[0] != 1 or feat.shape[0] != 1:
        raise RuntimeError("Only batch=1 is supported")
    if cate.shape[1] != kernel.shape[1]:
        raise RuntimeError("parts_cate and parts_kernel K dimension mismatch")

    num_classes = int(cate.shape[2])
    kernel_dim = int(kernel.shape[2])

    # feat could be NCHW or NHWC depending on export; normalize to (C,H,W).
    feat0 = feat[0]
    if feat0.shape[0] == kernel_dim:
        feat_chw = feat0  # [C,H,W]
    elif feat0.shape[-1] == kernel_dim:
        feat_chw = feat0.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
    else:
        raise RuntimeError(f"parts_mask_feat channels do not match kernel_dim={kernel_dim}: {feat.shape}")

    # Category logits are typically sigmoid-activated (SOLO/CondInst-style).
    cate_prob = _sigmoid(cate[0])  # [K, C]
    scores = np.max(cate_prob, axis=1)  # [K]
    cls = np.argmax(cate_prob, axis=1).astype(np.int32)  # [K]

    cate_threshold = float(cate_threshold)
    keep = np.where(scores >= cate_threshold)[0]
    if keep.size == 0:
        return (
            np.zeros((target_h, target_w), dtype=np.float32),
            np.zeros((target_h, target_w), dtype=np.float32),
            np.zeros((target_h, target_w), dtype=bool),
        )

    max_instances = int(max_instances)
    if max_instances > 0 and keep.size > max_instances:
        top = np.argsort(scores[keep])[::-1][:max_instances]
        keep = keep[top]

    keep_scores = scores[keep].astype(np.float32)
    keep_cls = cls[keep].astype(np.int32)
    keep_kernels = kernel[0, keep, :].astype(np.float32)  # [M, Kdim]

    c, h, w = feat_chw.shape
    feat_flat = feat_chw.reshape(int(c), int(h) * int(w)).astype(np.float32)  # [Kdim, HW]

    # masks_logits: [M, HW]
    masks_logits = keep_kernels @ feat_flat
    masks_prob = _sigmoid(masks_logits).reshape(-1, int(h), int(w))  # [M, h, w]

    skin_ids = _normalize_class_ids_for_model(skin_class_ids, num_classes=num_classes, offset=class_id_offset)
    hair_ids = _normalize_class_ids_for_model(hair_class_ids, num_classes=num_classes, offset=class_id_offset)
    face_ids = _normalize_class_ids_for_model(face_class_ids, num_classes=num_classes, offset=class_id_offset)

    skin01 = np.zeros((target_h, target_w), dtype=np.float32)
    hair01 = np.zeros((target_h, target_w), dtype=np.float32)
    face_mask = np.zeros((target_h, target_w), dtype=bool)

    mask_threshold = float(mask_threshold)

    for i in range(masks_prob.shape[0]):
        cls_id = int(keep_cls[i])
        m = masks_prob[i]
        if (h, w) != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
        weighted = m * float(keep_scores[i])

        if cls_id in skin_ids:
            skin01 = np.maximum(skin01, weighted)
        if cls_id in hair_ids:
            hair01 = np.maximum(hair01, weighted)
        if cls_id in face_ids:
            face_mask |= (m >= mask_threshold)

    return skin01, hair01, face_mask

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


def skin_hair_masks_onnx_schp(
    rgb: np.ndarray,
    *,
    model_path: str,
    input_size: int,
    skin_class_ids: tuple[int, ...],
    hair_class_ids: tuple[int, ...],
    face_class_ids: tuple[int, ...] = (13,),
    neck_from_face: bool = True,
    neck_extend_frac: float = 0.35,
    neck_expand_x_frac: float = 0.10,
    min_confidence: float,
    skin_min_area: int,
    hair_min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-pass SCHP inference to get both skin + hair masks from semantic classes.

    This is useful when you want to censor "all visible human skin parts" (face/arms/legs, etc.)
    plus hair, while avoiding most false positives from skin-colored clothing.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    expected_h, expected_w = _schp_expected_hw(model_path)
    if expected_h is not None and expected_w is not None:
        target_h, target_w = int(expected_h), int(expected_w)
    else:
        input_size = int(input_size)
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        target_h, target_w = input_size, input_size

    skin_class_ids = tuple(int(i) for i in skin_class_ids)
    hair_class_ids = tuple(int(i) for i in hair_class_ids)
    if not skin_class_ids:
        raise ValueError("skin_class_ids must not be empty")
    if not hair_class_ids:
        raise ValueError("hair_class_ids must not be empty")

    height, width = rgb.shape[:2]
    boxed, scale, pad_x, pad_y = _letterbox_to_size(rgb, target_h=target_h, target_w=target_w)
    out = _schp_infer_logits(boxed, model_path=model_path)

    min_confidence = float(min_confidence)

    if out.ndim == 4:
        out = out[0]

    if out.ndim == 3:
        logits = out  # [C, H, W]
        if logits.shape[1:] != (target_h, target_w):
            logits = cv2.resize(
                logits.transpose(1, 2, 0),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)
        pred = np.argmax(logits, axis=0).astype(np.int32)

        conf: np.ndarray | None
        if min_confidence > 0.0:
            shifted = logits.astype(np.float32) - np.max(logits.astype(np.float32), axis=0, keepdims=True)
            exp = np.exp(shifted)
            denom = np.sum(exp, axis=0)
            maxexp = np.max(exp, axis=0)
            conf = (maxexp / np.maximum(denom, 1e-8)).astype(np.float32)
        else:
            conf = None

        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        hair = np.isin(pred, np.array(hair_class_ids, dtype=np.int32))
        if neck_from_face:
            skin = _add_neck_band_from_anchor(
                skin,
                pred=pred,
                primary_anchor_class_ids=face_class_ids,
                fallback_anchor_class_ids=(hair_class_ids, skin_class_ids),
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
        if conf is not None:
            skin &= conf >= min_confidence
            hair &= conf >= min_confidence

        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=skin_min_area)
        hair01_square = _postprocess_mask(hair.astype(np.float32), min_area=hair_min_area)
    elif out.ndim == 2:
        pred = out.astype(np.int32)
        if pred.shape != (target_h, target_w):
            pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        hair = np.isin(pred, np.array(hair_class_ids, dtype=np.int32))
        if neck_from_face:
            skin = _add_neck_band_from_anchor(
                skin,
                pred=pred,
                primary_anchor_class_ids=face_class_ids,
                fallback_anchor_class_ids=(hair_class_ids, skin_class_ids),
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=skin_min_area)
        hair01_square = _postprocess_mask(hair.astype(np.float32), min_area=hair_min_area)
    else:
        raise RuntimeError(f"Unexpected SCHP model output shape: {out.shape}")

    skin01 = _unletterbox_mask01(
        skin01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    hair01 = _unletterbox_mask01(
        hair01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return skin01, hair01


def skin_hair_masks_onnx_segformer24(
    rgb: np.ndarray,
    *,
    model_path: str,
    input_size: int,
    skin_class_ids: tuple[int, ...],
    hair_class_ids: tuple[int, ...],
    face_class_ids: tuple[int, ...] = (13,),
    neck_from_face: bool = True,
    neck_extend_frac: float = 0.35,
    neck_expand_x_frac: float = 0.10,
    min_confidence: float = 0.0,
    skin_min_area: int,
    hair_min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-pass SegFormer Human Parse 24 inference to get skin + hair masks.

    Note: Class IDs are model/dataset-specific; configure via env vars.
    Defaults mirror common LIP IDs (face=13, arms=14-15, legs=16-17, hair=2).
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    expected_h, expected_w = _segformer24_expected_hw(model_path)
    if expected_h is not None and expected_w is not None:
        target_h, target_w = int(expected_h), int(expected_w)
    else:
        input_size = int(input_size)
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        target_h, target_w = input_size, input_size

    skin_class_ids = tuple(int(i) for i in skin_class_ids)
    hair_class_ids = tuple(int(i) for i in hair_class_ids)
    if not skin_class_ids:
        raise ValueError("skin_class_ids must not be empty")
    if not hair_class_ids:
        raise ValueError("hair_class_ids must not be empty")

    height, width = rgb.shape[:2]
    boxed, scale, pad_x, pad_y = _letterbox_to_size(rgb, target_h=target_h, target_w=target_w)
    out = _segformer24_infer_logits(boxed, model_path=model_path)

    min_confidence = float(min_confidence)

    if out.ndim == 4:
        out = out[0]

    if out.ndim == 3 and out.shape[0] == 1:
        # Some exports return class IDs as [1, H, W].
        out = out[0]

    if out.ndim == 3:
        logits = out  # [C, H, W]
        if logits.shape[1:] != (target_h, target_w):
            logits = cv2.resize(
                logits.transpose(1, 2, 0),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)

        pred = np.argmax(logits, axis=0).astype(np.int32)
        conf: np.ndarray | None
        if min_confidence > 0.0:
            shifted = logits.astype(np.float32) - np.max(logits.astype(np.float32), axis=0, keepdims=True)
            exp = np.exp(shifted)
            denom = np.sum(exp, axis=0)
            maxexp = np.max(exp, axis=0)
            conf = (maxexp / np.maximum(denom, 1e-8)).astype(np.float32)
        else:
            conf = None

        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        hair = np.isin(pred, np.array(hair_class_ids, dtype=np.int32))
        if neck_from_face:
            skin = _add_neck_band_from_anchor(
                skin,
                pred=pred,
                primary_anchor_class_ids=face_class_ids,
                fallback_anchor_class_ids=(hair_class_ids, skin_class_ids),
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
        if conf is not None:
            skin &= conf >= min_confidence
            hair &= conf >= min_confidence

        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=skin_min_area)
        hair01_square = _postprocess_mask(hair.astype(np.float32), min_area=hair_min_area)
    elif out.ndim == 2:
        pred = out.astype(np.int32)
        if pred.shape != (target_h, target_w):
            pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        hair = np.isin(pred, np.array(hair_class_ids, dtype=np.int32))
        if neck_from_face:
            skin = _add_neck_band_from_anchor(
                skin,
                pred=pred,
                primary_anchor_class_ids=face_class_ids,
                fallback_anchor_class_ids=(hair_class_ids, skin_class_ids),
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=skin_min_area)
        hair01_square = _postprocess_mask(hair.astype(np.float32), min_area=hair_min_area)
    else:
        raise RuntimeError(f"Unexpected SegFormer model output shape: {out.shape}")

    skin01 = _unletterbox_mask01(
        skin01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    hair01 = _unletterbox_mask01(
        hair01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return skin01, hair01


def skin_mask_onnx_segformer24(
    rgb: np.ndarray,
    *,
    model_path: str,
    input_size: int,
    skin_class_ids: tuple[int, ...],
    face_class_ids: tuple[int, ...] = (13,),
    neck_from_face: bool = True,
    neck_extend_frac: float = 0.35,
    neck_expand_x_frac: float = 0.10,
    min_confidence: float = 0.0,
    min_area: int,
) -> np.ndarray:
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    expected_h, expected_w = _segformer24_expected_hw(model_path)
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
    out = _segformer24_infer_logits(boxed, model_path=model_path)

    min_confidence = float(min_confidence)

    if out.ndim == 4:
        out = out[0]

    if out.ndim == 3 and out.shape[0] == 1:
        # Some exports return class IDs as [1, H, W].
        out = out[0]

    if out.ndim == 3:
        logits = out  # [C, H, W]
        if logits.shape[1:] != (target_h, target_w):
            logits = cv2.resize(
                logits.transpose(1, 2, 0),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)

        pred = np.argmax(logits, axis=0).astype(np.int32)
        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        if neck_from_face:
            skin = _add_neck_band_from_face(
                skin,
                pred=pred,
                face_class_ids=face_class_ids,
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )

        if min_confidence > 0.0:
            shifted = logits.astype(np.float32) - np.max(logits.astype(np.float32), axis=0, keepdims=True)
            exp = np.exp(shifted)
            denom = np.sum(exp, axis=0)
            maxexp = np.max(exp, axis=0)
            conf = (maxexp / np.maximum(denom, 1e-8)).astype(np.float32)
            skin &= conf >= min_confidence

        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=min_area)
    elif out.ndim == 2:
        pred = out.astype(np.int32)
        if pred.shape != (target_h, target_w):
            pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        if neck_from_face:
            skin = _add_neck_band_from_face(
                skin,
                pred=pred,
                face_class_ids=face_class_ids,
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=min_area)
    else:
        raise RuntimeError(f"Unexpected SegFormer model output shape: {out.shape}")

    return _unletterbox_mask01(
        skin01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def skin_hair_masks_onnx_lvmhpv2(
    rgb: np.ndarray,
    *,
    model_path: str,
    preprocess: str,
    input_size: int,
    output_index: int,
    class_id_offset: int,
    skin_class_ids: tuple[int, ...],
    hair_class_ids: tuple[int, ...],
    face_class_ids: tuple[int, ...] = (),
    neck_from_face: bool = True,
    neck_extend_frac: float = 0.35,
    neck_expand_x_frac: float = 0.10,
    min_confidence: float = 0.0,
    cate_threshold: float = 0.35,
    mask_threshold: float = 0.5,
    max_instances: int = 200,
    skin_min_area: int,
    hair_min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    LV-MHP v2 human parsing backend.

    This is intentionally dataset/model-agnostic: you provide the ONNX model + label IDs to treat
    as "skin parts" and (optionally) hair.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    expected_h, expected_w = _onnx_expected_hw(model_path)
    if expected_h is not None and expected_w is not None:
        target_h, target_w = int(expected_h), int(expected_w)
    else:
        input_size = int(input_size)
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        target_h, target_w = input_size, input_size

    skin_class_ids = tuple(int(i) for i in skin_class_ids)
    hair_class_ids = tuple(int(i) for i in hair_class_ids)
    face_class_ids = tuple(int(i) for i in face_class_ids)
    if not skin_class_ids:
        raise ValueError("skin_class_ids must not be empty")
    if not hair_class_ids:
        raise ValueError("hair_class_ids must not be empty")

    height, width = rgb.shape[:2]
    boxed, scale, pad_x, pad_y = _letterbox_to_size(rgb, target_h=target_h, target_w=target_w)
    output_names, outputs = _parsing_infer_outputs(boxed, model_path=model_path, preprocess=preprocess)

    # MHParsNet-style outputs (kernel + mask features) need custom decoding.
    if output_index < 0 and {"parts_cate", "parts_kernel", "parts_mask_feat"}.issubset(set(output_names)):
        skin01_square, hair01_square, face_mask = _mhparsnet_part_union_masks(
            output_names,
            outputs,
            target_h=target_h,
            target_w=target_w,
            class_id_offset=class_id_offset,
            skin_class_ids=skin_class_ids,
            hair_class_ids=hair_class_ids,
            face_class_ids=face_class_ids,
            cate_threshold=cate_threshold,
            mask_threshold=mask_threshold,
            max_instances=max_instances,
        )
        if neck_from_face and face_class_ids:
            skin_bool = skin01_square > float(mask_threshold)
            skin_bool = _add_neck_band_from_anchor_mask(
                skin_bool,
                anchor_mask=face_mask,
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
            skin01_square = np.maximum(skin01_square, skin_bool.astype(np.float32))

        skin01_square = _postprocess_mask(skin01_square.astype(np.float32), min_area=skin_min_area)
        hair01_square = _postprocess_mask(hair01_square.astype(np.float32), min_area=hair_min_area)
    else:
        out = _select_parsing_output(output_names, outputs, target_h=target_h, target_w=target_w, output_index=output_index)
        logits, pred = _parsing_logits_or_pred(out, target_h=target_h, target_w=target_w)

        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))
        hair = np.isin(pred, np.array(hair_class_ids, dtype=np.int32))

        min_confidence = float(min_confidence)
        if min_confidence > 0.0 and logits is not None:
            conf = _parsing_confidence(logits)
            skin &= conf >= min_confidence
            hair &= conf >= min_confidence

        if neck_from_face and face_class_ids:
            skin = _add_neck_band_from_anchor(
                skin,
                pred=pred,
                primary_anchor_class_ids=face_class_ids,
                fallback_anchor_class_ids=(hair_class_ids, skin_class_ids),
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )

        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=skin_min_area)
        hair01_square = _postprocess_mask(hair.astype(np.float32), min_area=hair_min_area)

    skin01 = _unletterbox_mask01(
        skin01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    hair01 = _unletterbox_mask01(
        hair01_square,
        orig_h=height,
        orig_w=width,
        target_h=target_h,
        target_w=target_w,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return skin01, hair01


def skin_mask_onnx_lvmhpv2(
    rgb: np.ndarray,
    *,
    model_path: str,
    preprocess: str,
    input_size: int,
    output_index: int,
    class_id_offset: int,
    skin_class_ids: tuple[int, ...],
    face_class_ids: tuple[int, ...] = (),
    neck_from_face: bool = True,
    neck_extend_frac: float = 0.35,
    neck_expand_x_frac: float = 0.10,
    min_confidence: float = 0.0,
    cate_threshold: float = 0.35,
    mask_threshold: float = 0.5,
    max_instances: int = 200,
    min_area: int,
) -> np.ndarray:
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    expected_h, expected_w = _onnx_expected_hw(model_path)
    if expected_h is not None and expected_w is not None:
        target_h, target_w = int(expected_h), int(expected_w)
    else:
        input_size = int(input_size)
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        target_h, target_w = input_size, input_size

    skin_class_ids = tuple(int(i) for i in skin_class_ids)
    face_class_ids = tuple(int(i) for i in face_class_ids)
    if not skin_class_ids:
        raise ValueError("skin_class_ids must not be empty")

    height, width = rgb.shape[:2]
    boxed, scale, pad_x, pad_y = _letterbox_to_size(rgb, target_h=target_h, target_w=target_w)
    output_names, outputs = _parsing_infer_outputs(boxed, model_path=model_path, preprocess=preprocess)

    if output_index < 0 and {"parts_cate", "parts_kernel", "parts_mask_feat"}.issubset(set(output_names)):
        skin01_square, _hair01_square, face_mask = _mhparsnet_part_union_masks(
            output_names,
            outputs,
            target_h=target_h,
            target_w=target_w,
            class_id_offset=class_id_offset,
            skin_class_ids=skin_class_ids,
            hair_class_ids=(),
            face_class_ids=face_class_ids,
            cate_threshold=cate_threshold,
            mask_threshold=mask_threshold,
            max_instances=max_instances,
        )
        if neck_from_face and face_class_ids:
            skin_bool = skin01_square > float(mask_threshold)
            skin_bool = _add_neck_band_from_anchor_mask(
                skin_bool,
                anchor_mask=face_mask,
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )
            skin01_square = np.maximum(skin01_square, skin_bool.astype(np.float32))
        skin01_square = _postprocess_mask(skin01_square.astype(np.float32), min_area=min_area)
    else:
        out = _select_parsing_output(output_names, outputs, target_h=target_h, target_w=target_w, output_index=output_index)
        logits, pred = _parsing_logits_or_pred(out, target_h=target_h, target_w=target_w)
        skin = np.isin(pred, np.array(skin_class_ids, dtype=np.int32))

        min_confidence = float(min_confidence)
        if min_confidence > 0.0 and logits is not None:
            conf = _parsing_confidence(logits)
            skin &= conf >= min_confidence

        if neck_from_face and face_class_ids:
            skin = _add_neck_band_from_anchor(
                skin,
                pred=pred,
                primary_anchor_class_ids=face_class_ids,
                fallback_anchor_class_ids=(skin_class_ids,),
                extend_frac=neck_extend_frac,
                expand_x_frac=neck_expand_x_frac,
            )

        skin01_square = _postprocess_mask(skin.astype(np.float32), min_area=min_area)
    return _unletterbox_mask01(
        skin01_square,
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
    segformer24_model_path: str = "models/model.onnx",
    segformer24_input_size: int = 512,
    segformer24_skin_class_ids: tuple[int, ...] = (13, 14, 15, 16, 17),
    segformer24_face_class_ids: tuple[int, ...] = (13,),
    segformer24_neck_from_face: bool = True,
    segformer24_neck_extend_frac: float = 0.35,
    segformer24_neck_expand_x_frac: float = 0.10,
    segformer24_min_confidence: float = 0.0,
    lvmhp_model_path: str = "models/lvmhp_v2.onnx",
    lvmhp_preprocess: str = "rgb_imagenet",
    lvmhp_input_size: int = 512,
    lvmhp_output_index: int = -1,
    lvmhp_class_id_offset: int = 0,
    lvmhp_cate_threshold: float = 0.35,
    lvmhp_mask_threshold: float = 0.5,
    lvmhp_max_instances: int = 200,
    lvmhp_skin_class_ids: tuple[int, ...] = (),
    lvmhp_face_class_ids: tuple[int, ...] = (),
    lvmhp_neck_from_face: bool = True,
    lvmhp_neck_extend_frac: float = 0.35,
    lvmhp_neck_expand_x_frac: float = 0.10,
    lvmhp_min_confidence: float = 0.0,
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
        segformer24_model_path=segformer24_model_path,
        segformer24_input_size=segformer24_input_size,
        segformer24_skin_class_ids=segformer24_skin_class_ids,
        segformer24_face_class_ids=segformer24_face_class_ids,
        segformer24_neck_from_face=segformer24_neck_from_face,
        segformer24_neck_extend_frac=segformer24_neck_extend_frac,
        segformer24_neck_expand_x_frac=segformer24_neck_expand_x_frac,
        segformer24_min_confidence=segformer24_min_confidence,
        lvmhp_model_path=lvmhp_model_path,
        lvmhp_preprocess=lvmhp_preprocess,
        lvmhp_input_size=lvmhp_input_size,
        lvmhp_output_index=lvmhp_output_index,
        lvmhp_class_id_offset=lvmhp_class_id_offset,
        lvmhp_cate_threshold=lvmhp_cate_threshold,
        lvmhp_mask_threshold=lvmhp_mask_threshold,
        lvmhp_max_instances=lvmhp_max_instances,
        lvmhp_skin_class_ids=lvmhp_skin_class_ids,
        lvmhp_face_class_ids=lvmhp_face_class_ids,
        lvmhp_neck_from_face=lvmhp_neck_from_face,
        lvmhp_neck_extend_frac=lvmhp_neck_extend_frac,
        lvmhp_neck_expand_x_frac=lvmhp_neck_expand_x_frac,
        lvmhp_min_confidence=lvmhp_min_confidence,
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
    segformer24_model_path: str = "models/model.onnx",
    segformer24_input_size: int = 512,
    segformer24_skin_class_ids: tuple[int, ...] = (13, 14, 15, 16, 17),
    segformer24_face_class_ids: tuple[int, ...] = (13,),
    segformer24_neck_from_face: bool = True,
    segformer24_neck_extend_frac: float = 0.35,
    segformer24_neck_expand_x_frac: float = 0.10,
    segformer24_min_confidence: float = 0.0,
    lvmhp_model_path: str = "models/lvmhp_v2.onnx",
    lvmhp_preprocess: str = "rgb_imagenet",
    lvmhp_input_size: int = 512,
    lvmhp_output_index: int = -1,
    lvmhp_class_id_offset: int = 0,
    lvmhp_cate_threshold: float = 0.35,
    lvmhp_mask_threshold: float = 0.5,
    lvmhp_max_instances: int = 200,
    lvmhp_skin_class_ids: tuple[int, ...] = (),
    lvmhp_face_class_ids: tuple[int, ...] = (),
    lvmhp_neck_from_face: bool = True,
    lvmhp_neck_extend_frac: float = 0.35,
    lvmhp_neck_expand_x_frac: float = 0.10,
    lvmhp_min_confidence: float = 0.0,
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

    if backend == "onnx_segformer24":
        try:
            return (
                skin_mask_onnx_segformer24(
                    rgb,
                    model_path=segformer24_model_path,
                    input_size=segformer24_input_size,
                    skin_class_ids=segformer24_skin_class_ids,
                    face_class_ids=segformer24_face_class_ids,
                    neck_from_face=segformer24_neck_from_face,
                    neck_extend_frac=segformer24_neck_extend_frac,
                    neck_expand_x_frac=segformer24_neck_expand_x_frac,
                    min_confidence=segformer24_min_confidence,
                    min_area=min_area,
                ),
                "onnx_segformer24",
                None,
            )
        except Exception as exc:
            fallback_error = f"onnx_segformer24 failed: {type(exc).__name__}: {exc}"
            backend = "onnx_smp"

    if backend == "onnx_lvmhpv2":
        try:
            return (
                skin_mask_onnx_lvmhpv2(
                    rgb,
                    model_path=lvmhp_model_path,
                    preprocess=lvmhp_preprocess,
                    input_size=lvmhp_input_size,
                    output_index=lvmhp_output_index,
                    class_id_offset=lvmhp_class_id_offset,
                    skin_class_ids=lvmhp_skin_class_ids,
                    face_class_ids=lvmhp_face_class_ids,
                    neck_from_face=lvmhp_neck_from_face,
                    neck_extend_frac=lvmhp_neck_extend_frac,
                    neck_expand_x_frac=lvmhp_neck_expand_x_frac,
                    min_confidence=lvmhp_min_confidence,
                    cate_threshold=lvmhp_cate_threshold,
                    mask_threshold=lvmhp_mask_threshold,
                    max_instances=lvmhp_max_instances,
                    min_area=min_area,
                ),
                "onnx_lvmhpv2",
                None,
            )
        except Exception as exc:
            fallback_error = f"onnx_lvmhpv2 failed: {type(exc).__name__}: {exc}"
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


def _mask01_to_bool(mask01: np.ndarray, *, threshold: float) -> np.ndarray:
    threshold = float(threshold)
    if mask01.dtype != np.float32 and mask01.dtype != np.float64:
        mask01 = mask01.astype(np.float32)
    return (mask01 > threshold).astype(bool)


def mask_fuse_overlap_filter(
    mask_a01: np.ndarray,
    mask_b01: np.ndarray,
    *,
    binarize_threshold: float = 0.5,
    overlap_dilate_px: int = 3,
    min_overlap_pixels: int = 25,
    min_overlap_frac: float = 0.0,
    min_area: int = 256,
) -> np.ndarray:
    """
    Fuse two masks by:
      1) union = A ∪ B
      2) intersection = A ∩ B
      3) Keep each connected component of the union only if it overlaps the intersection.

    This removes regions detected by only one backend (no overlap), while allowing each
    backend to contribute within regions where they agree.
    Returns float32 mask in [0, 1].
    """
    if mask_a01.shape != mask_b01.shape:
        raise ValueError("mask shapes must match")

    a = _mask01_to_bool(mask_a01, threshold=binarize_threshold)
    b = _mask01_to_bool(mask_b01, threshold=binarize_threshold)
    union = (a | b).astype(np.uint8)
    if int(union.sum()) == 0:
        return np.zeros_like(mask_a01, dtype=np.float32)

    inter = (a & b)
    if overlap_dilate_px > 0:
        k = 2 * int(overlap_dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inter_u8 = cv2.dilate(inter.astype(np.uint8) * 255, kernel, iterations=1)
        inter = inter_u8 > 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    keep = np.zeros_like(union, dtype=np.uint8)

    min_overlap_pixels = int(max(0, min_overlap_pixels))
    min_overlap_frac = float(max(0.0, min_overlap_frac))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == label
        overlap = int(np.count_nonzero(comp & inter))
        if overlap < min_overlap_pixels:
            continue
        if min_overlap_frac > 0.0 and (overlap / float(area)) < min_overlap_frac:
            continue
        keep[comp] = 255

    return _postprocess_mask((keep > 0).astype(np.float32), min_area=int(min_area))


def _single_backend_mask01(
    rgb: np.ndarray,
    *,
    backend: str,
    use_hair: bool,
    # onnx_smp
    model_path: str,
    score_threshold: float,
    skin_channel_index: int,
    require_max: bool,
    margin: float,
    min_area: int,
    hair_score_threshold: float,
    hair_channel_index: int,
    hair_require_max: bool,
    hair_margin: float,
    hair_min_area: int,
    # schp
    schp_model_path: str,
    schp_input_size: int,
    schp_skin_class_ids: tuple[int, ...],
    schp_hair_class_ids: tuple[int, ...],
    schp_min_confidence: float,
    # segformer24
    segformer24_model_path: str,
    segformer24_input_size: int,
    segformer24_skin_class_ids: tuple[int, ...],
    segformer24_hair_class_ids: tuple[int, ...],
    segformer24_face_class_ids: tuple[int, ...],
    segformer24_neck_from_face: bool,
    segformer24_neck_extend_frac: float,
    segformer24_neck_expand_x_frac: float,
    segformer24_min_confidence: float,
    # lvmhpv2
    lvmhp_model_path: str,
    lvmhp_preprocess: str,
    lvmhp_input_size: int,
    lvmhp_output_index: int,
    lvmhp_class_id_offset: int,
    lvmhp_cate_threshold: float,
    lvmhp_mask_threshold: float,
    lvmhp_max_instances: int,
    lvmhp_skin_class_ids: tuple[int, ...],
    lvmhp_hair_class_ids: tuple[int, ...],
    lvmhp_face_class_ids: tuple[int, ...],
    lvmhp_neck_from_face: bool,
    lvmhp_neck_extend_frac: float,
    lvmhp_neck_expand_x_frac: float,
    lvmhp_min_confidence: float,
) -> np.ndarray:
    backend = (backend or "").strip().lower()

    if backend == "onnx_smp":
        if use_hair:
            skin01, hair01 = skin_hair_masks_onnx_smp(
                rgb,
                model_path=model_path,
                skin_score_threshold=score_threshold,
                skin_channel_index=skin_channel_index,
                skin_require_max=require_max,
                skin_margin=margin,
                skin_min_area=min_area,
                hair_score_threshold=hair_score_threshold,
                hair_channel_index=hair_channel_index,
                hair_require_max=hair_require_max,
                hair_margin=hair_margin,
                hair_min_area=hair_min_area,
            )
            return np.maximum(skin01, hair01)
        return skin_mask_onnx_smp(
            rgb,
            model_path=model_path,
            score_threshold=score_threshold,
            skin_channel_index=skin_channel_index,
            require_max=require_max,
            margin=margin,
            min_area=min_area,
        )

    if backend == "onnx_schp":
        if use_hair:
            skin01, hair01 = skin_hair_masks_onnx_schp(
                rgb,
                model_path=schp_model_path,
                input_size=schp_input_size,
                skin_class_ids=schp_skin_class_ids,
                hair_class_ids=schp_hair_class_ids,
                min_confidence=schp_min_confidence,
                skin_min_area=min_area,
                hair_min_area=hair_min_area,
            )
            return np.maximum(skin01, hair01)
        return skin_mask_onnx_schp(
            rgb,
            model_path=schp_model_path,
            input_size=schp_input_size,
            skin_class_ids=schp_skin_class_ids,
            min_confidence=schp_min_confidence,
            min_area=min_area,
        )

    if backend == "onnx_segformer24":
        if use_hair:
            skin01, hair01 = skin_hair_masks_onnx_segformer24(
                rgb,
                model_path=segformer24_model_path,
                input_size=segformer24_input_size,
                skin_class_ids=segformer24_skin_class_ids,
                hair_class_ids=segformer24_hair_class_ids,
                face_class_ids=segformer24_face_class_ids,
                neck_from_face=segformer24_neck_from_face,
                neck_extend_frac=segformer24_neck_extend_frac,
                neck_expand_x_frac=segformer24_neck_expand_x_frac,
                min_confidence=segformer24_min_confidence,
                skin_min_area=min_area,
                hair_min_area=hair_min_area,
            )
            return np.maximum(skin01, hair01)
        return skin_mask_onnx_segformer24(
            rgb,
            model_path=segformer24_model_path,
            input_size=segformer24_input_size,
            skin_class_ids=segformer24_skin_class_ids,
            face_class_ids=segformer24_face_class_ids,
            neck_from_face=segformer24_neck_from_face,
            neck_extend_frac=segformer24_neck_extend_frac,
            neck_expand_x_frac=segformer24_neck_expand_x_frac,
            min_confidence=segformer24_min_confidence,
            min_area=min_area,
        )

    if backend == "onnx_lvmhpv2":
        if use_hair and lvmhp_hair_class_ids:
            skin01, hair01 = skin_hair_masks_onnx_lvmhpv2(
                rgb,
                model_path=lvmhp_model_path,
                preprocess=lvmhp_preprocess,
                input_size=lvmhp_input_size,
                output_index=lvmhp_output_index,
                class_id_offset=lvmhp_class_id_offset,
                skin_class_ids=lvmhp_skin_class_ids,
                hair_class_ids=lvmhp_hair_class_ids,
                face_class_ids=lvmhp_face_class_ids,
                neck_from_face=lvmhp_neck_from_face,
                neck_extend_frac=lvmhp_neck_extend_frac,
                neck_expand_x_frac=lvmhp_neck_expand_x_frac,
                min_confidence=lvmhp_min_confidence,
                cate_threshold=lvmhp_cate_threshold,
                mask_threshold=lvmhp_mask_threshold,
                max_instances=lvmhp_max_instances,
                skin_min_area=min_area,
                hair_min_area=hair_min_area,
            )
            return np.maximum(skin01, hair01)
        return skin_mask_onnx_lvmhpv2(
            rgb,
            model_path=lvmhp_model_path,
            preprocess=lvmhp_preprocess,
            input_size=lvmhp_input_size,
            output_index=lvmhp_output_index,
            class_id_offset=lvmhp_class_id_offset,
            skin_class_ids=lvmhp_skin_class_ids,
            face_class_ids=lvmhp_face_class_ids,
            neck_from_face=lvmhp_neck_from_face,
            neck_extend_frac=lvmhp_neck_extend_frac,
            neck_expand_x_frac=lvmhp_neck_expand_x_frac,
            min_confidence=lvmhp_min_confidence,
            cate_threshold=lvmhp_cate_threshold,
            mask_threshold=lvmhp_mask_threshold,
            max_instances=lvmhp_max_instances,
            min_area=min_area,
        )

    if backend == "cv":
        return skin_mask(
            rgb,
            max_side=640,
            min_area=min_area,
            mode="auto",
            threshold=3.5,
            min_seed_pixels=350,
        )

    raise ValueError(f"Unknown backend: {backend}")


def fused_mask_dispatch_info(
    rgb: np.ndarray,
    *,
    mode: str,
    backends: tuple[str, ...],
    use_hair: bool,
    # common sizes/filters
    min_area: int,
    hair_min_area: int,
    # fusion knobs
    binarize_threshold: float,
    overlap_dilate_px: int,
    min_overlap_pixels: int,
    min_overlap_frac: float,
    fusion_min_area: int,
    # onnx_smp
    model_path: str,
    score_threshold: float,
    skin_channel_index: int,
    require_max: bool,
    margin: float,
    hair_score_threshold: float,
    hair_channel_index: int,
    hair_require_max: bool,
    hair_margin: float,
    # schp
    schp_model_path: str,
    schp_input_size: int,
    schp_skin_class_ids: tuple[int, ...],
    schp_hair_class_ids: tuple[int, ...],
    schp_min_confidence: float,
    # segformer24
    segformer24_model_path: str,
    segformer24_input_size: int,
    segformer24_skin_class_ids: tuple[int, ...],
    segformer24_hair_class_ids: tuple[int, ...],
    segformer24_face_class_ids: tuple[int, ...],
    segformer24_neck_from_face: bool,
    segformer24_neck_extend_frac: float,
    segformer24_neck_expand_x_frac: float,
    segformer24_min_confidence: float,
    # lvmhpv2
    lvmhp_model_path: str,
    lvmhp_preprocess: str,
    lvmhp_input_size: int,
    lvmhp_output_index: int,
    lvmhp_class_id_offset: int,
    lvmhp_cate_threshold: float,
    lvmhp_mask_threshold: float,
    lvmhp_max_instances: int,
    lvmhp_skin_class_ids: tuple[int, ...],
    lvmhp_hair_class_ids: tuple[int, ...],
    lvmhp_face_class_ids: tuple[int, ...],
    lvmhp_neck_from_face: bool,
    lvmhp_neck_extend_frac: float,
    lvmhp_neck_expand_x_frac: float,
    lvmhp_min_confidence: float,
) -> tuple[np.ndarray, tuple[str, ...], str | None]:
    """
    Compute a fused mask from multiple backends.
    Currently supports `mode=overlap_filter` for exactly two backends.
    Returns (mask01, used_backends, error_message_or_none).
    """
    mode = (mode or "").strip().lower()
    backends = tuple((b or "").strip().lower() for b in backends if (b or "").strip())
    if mode not in {"overlap_filter"}:
        raise ValueError(f"Unknown fusion mode: {mode}")
    if len(backends) != 2:
        raise ValueError("MASK_FUSION_BACKENDS must have exactly 2 items for overlap_filter mode")

    masks: list[np.ndarray] = []
    used: list[str] = []
    errors: list[str] = []

    for backend in backends:
        try:
            m = _single_backend_mask01(
                rgb,
                backend=backend,
                use_hair=use_hair,
                model_path=model_path,
                score_threshold=score_threshold,
                skin_channel_index=skin_channel_index,
                require_max=require_max,
                margin=margin,
                min_area=min_area,
                hair_score_threshold=hair_score_threshold,
                hair_channel_index=hair_channel_index,
                hair_require_max=hair_require_max,
                hair_margin=hair_margin,
                hair_min_area=hair_min_area,
                schp_model_path=schp_model_path,
                schp_input_size=schp_input_size,
                schp_skin_class_ids=schp_skin_class_ids,
                schp_hair_class_ids=schp_hair_class_ids,
                schp_min_confidence=schp_min_confidence,
                segformer24_model_path=segformer24_model_path,
                segformer24_input_size=segformer24_input_size,
                segformer24_skin_class_ids=segformer24_skin_class_ids,
                segformer24_hair_class_ids=segformer24_hair_class_ids,
                segformer24_face_class_ids=segformer24_face_class_ids,
                segformer24_neck_from_face=segformer24_neck_from_face,
                segformer24_neck_extend_frac=segformer24_neck_extend_frac,
                segformer24_neck_expand_x_frac=segformer24_neck_expand_x_frac,
                segformer24_min_confidence=segformer24_min_confidence,
                lvmhp_model_path=lvmhp_model_path,
                lvmhp_preprocess=lvmhp_preprocess,
                lvmhp_input_size=lvmhp_input_size,
                lvmhp_output_index=lvmhp_output_index,
                lvmhp_class_id_offset=lvmhp_class_id_offset,
                lvmhp_cate_threshold=lvmhp_cate_threshold,
                lvmhp_mask_threshold=lvmhp_mask_threshold,
                lvmhp_max_instances=lvmhp_max_instances,
                lvmhp_skin_class_ids=lvmhp_skin_class_ids,
                lvmhp_hair_class_ids=lvmhp_hair_class_ids,
                lvmhp_face_class_ids=lvmhp_face_class_ids,
                lvmhp_neck_from_face=lvmhp_neck_from_face,
                lvmhp_neck_extend_frac=lvmhp_neck_extend_frac,
                lvmhp_neck_expand_x_frac=lvmhp_neck_expand_x_frac,
                lvmhp_min_confidence=lvmhp_min_confidence,
            )
            masks.append(m)
            used.append(backend)
        except Exception as exc:
            errors.append(f"{backend} failed: {type(exc).__name__}: {exc}")

    if len(masks) == 2:
        fused01 = mask_fuse_overlap_filter(
            masks[0],
            masks[1],
            binarize_threshold=binarize_threshold,
            overlap_dilate_px=overlap_dilate_px,
            min_overlap_pixels=min_overlap_pixels,
            min_overlap_frac=min_overlap_frac,
            min_area=fusion_min_area,
        )
    elif len(masks) == 1:
        fused01 = masks[0]
    else:
        fused01 = np.zeros(rgb.shape[:2], dtype=np.float32)

    err = "; ".join(errors) if errors else None
    return fused01.astype(np.float32), tuple(used), err
