from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass(frozen=True)
class CensorParams:
    method: str  # "blur" | "pixelate" | "solid"
    sigma: float
    pixel_size: int
    color_rgb: tuple[int, int, int]
    feather: float


def _gaussian_blur_rgb(rgb: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.1, float(sigma))
    return cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def _pixelate_rgb(rgb: np.ndarray, pixel_size: int) -> np.ndarray:
    pixel_size = int(max(2, pixel_size))
    height, width = rgb.shape[:2]
    small_w = max(1, width // pixel_size)
    small_h = max(1, height // pixel_size)
    small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _solid_rgb_like(rgb: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    solid = np.zeros_like(rgb, dtype=np.uint8)
    solid[:, :, 0] = color_rgb[0]
    solid[:, :, 1] = color_rgb[1]
    solid[:, :, 2] = color_rgb[2]
    return solid


def _feather_mask(mask01: np.ndarray, feather: float) -> np.ndarray:
    feather = float(feather)
    if feather <= 0:
        return np.clip(mask01, 0.0, 1.0).astype(np.float32)
    # Convert "pixels" feather to sigma; sigma≈feather/2 is a reasonable UX mapping.
    sigma = max(0.1, feather / 2.0)
    blurred = cv2.GaussianBlur(mask01.astype(np.float32), ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def apply_censor(rgb: np.ndarray, mask01: np.ndarray, params: CensorParams) -> np.ndarray:
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")
    if mask01.dtype not in (np.float32, np.float64):
        mask01 = mask01.astype(np.float32)
    if mask01.shape[:2] != rgb.shape[:2]:
        raise ValueError("mask and image size mismatch")

    method = params.method.lower().strip()
    if method == "blur":
        censored = _gaussian_blur_rgb(rgb, sigma=params.sigma)
    elif method == "pixelate":
        censored = _pixelate_rgb(rgb, pixel_size=params.pixel_size)
    elif method == "solid":
        censored = _solid_rgb_like(rgb, color_rgb=params.color_rgb)
    else:
        raise ValueError(f"unknown method: {params.method}")

    mask = _feather_mask(mask01, feather=params.feather)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    out = (rgb.astype(np.float32) * (1.0 - mask3) + censored.astype(np.float32) * mask3).round()
    return np.clip(out, 0, 255).astype(np.uint8)

