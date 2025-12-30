from __future__ import annotations

import io
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageColor, ImageOps

from app.censor import CensorParams, apply_censor
from app.settings import settings
from app.skin import skin_hair_masks_onnx_smp, skin_mask_dispatch_info

app = FastAPI(title="Censor Image", version="0.1.0")

_FORMAT_ALIASES: dict[str, str] = {"JPG": "JPEG"}
_CONTENT_TYPE_TO_FMT: dict[str, str] = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_EXT_TO_FMT: dict[str, str] = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}


def _content_type_for(fmt: str) -> str:
    fmt = _FORMAT_ALIASES.get(fmt.upper(), fmt.upper())
    if fmt == "PNG":
        return "image/png"
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "WEBP":
        return "image/webp"
    return "application/octet-stream"


def _ensure_supported(fmt: str) -> None:
    fmt = _FORMAT_ALIASES.get(fmt.upper(), fmt.upper())
    if fmt not in {"PNG", "JPEG", "WEBP"}:
        raise HTTPException(status_code=415, detail=f"Unsupported format: {fmt}")


def _read_upload_bytes(upload: UploadFile) -> bytes:
    data = upload.file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb}MB)")
    return data


def _open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img.load()
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _detect_input_format(upload: UploadFile, img: Image.Image) -> str:
    pil_fmt = (img.format or "").upper().strip()
    pil_fmt = _FORMAT_ALIASES.get(pil_fmt, pil_fmt)
    if pil_fmt:
        return pil_fmt

    ct = (upload.content_type or "").lower().strip()
    if ct in _CONTENT_TYPE_TO_FMT:
        return _CONTENT_TYPE_TO_FMT[ct]

    _, ext = os.path.splitext(upload.filename or "")
    ext = ext.lower().strip()
    if ext in _EXT_TO_FMT:
        return _EXT_TO_FMT[ext]

    raise HTTPException(status_code=415, detail="Unsupported or unknown image format (expected PNG/JPG/WebP).")


def _pil_to_rgb_array(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    if img.mode in {"RGBA", "LA"}:
        rgba = img.convert("RGBA")
        rgba_arr = np.array(rgba, dtype=np.uint8)
        alpha = rgba_arr[:, :, 3]
        rgb = rgba_arr[:, :, :3]
        return rgb, alpha
    rgb = img.convert("RGB")
    return np.array(rgb, dtype=np.uint8), None


def _compose_with_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> Image.Image:
    if alpha is None:
        return Image.fromarray(rgb, mode="RGB")
    rgba = np.dstack([rgb, alpha.astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def _encode_image(img: Image.Image, fmt: str) -> bytes:
    fmt = _FORMAT_ALIASES.get(fmt.upper(), fmt.upper())
    out = io.BytesIO()

    if fmt == "PNG":
        img.save(out, format="PNG", optimize=False)
        return out.getvalue()

    if fmt == "WEBP":
        img.save(out, format="WEBP", lossless=True, quality=100, method=6)
        return out.getvalue()

    if fmt == "JPEG":
        if settings.strict_lossless and not settings.allow_lossy_jpeg:
            raise HTTPException(
                status_code=422,
                detail="JPEG cannot be lossless after pixel edits; set ALLOW_LOSSY_JPEG=true to enable best-effort JPEG output.",
            )
        rgb = img.convert("RGB")
        rgb.save(out, format="JPEG", quality=95, optimize=True, subsampling=0)
        return out.getvalue()

    raise HTTPException(status_code=415, detail=f"Unsupported format: {fmt}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mask")
def mask(
    image: UploadFile = File(...),
) -> Response:
    data = _read_upload_bytes(image)
    img = _open_image(data)
    fmt = _detect_input_format(image, img)
    _ensure_supported(fmt)

    rgb, _alpha = _pil_to_rgb_array(img)
    mask01, used_backend, backend_error = skin_mask_dispatch_info(
        rgb,
        backend=settings.skin_backend,
        model_path=settings.skin_model_path,
        score_threshold=settings.skin_score_threshold,
        skin_channel_index=settings.skin_channel_index,
        require_max=settings.skin_require_max,
        margin=settings.skin_margin,
        min_area=settings.min_component_area,
        schp_model_path=settings.schp_model_path,
        schp_input_size=settings.schp_input_size,
        schp_skin_class_ids=tuple(settings.schp_skin_class_ids),
        schp_min_confidence=settings.schp_min_confidence,
        cv_max_side=settings.mask_max_side,
        cv_mode=settings.skin_mode,
        cv_maha_threshold=settings.skin_maha_threshold,
        cv_min_seed_pixels=settings.skin_min_seed_pixels,
    )
    mask_u8 = (np.clip(mask01, 0.0, 1.0) * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_u8, mode="L")
    out = io.BytesIO()
    mask_img.save(out, format="PNG", optimize=False)
    headers = {
        "X-Skin-Backend-Requested": settings.skin_backend,
        "X-Skin-Backend-Used": used_backend,
    }
    if backend_error:
        safe = " ".join(str(backend_error).splitlines()).strip()
        safe = safe.encode("ascii", "replace").decode("ascii")
        headers["X-Skin-Backend-Error"] = safe[:800]
    return Response(content=out.getvalue(), media_type="image/png", headers=headers)


@app.post("/censor")
def censor(
    method: Literal["blur", "pixelate", "solid"] = "blur",
    sigma: float = 12.0,
    pixel_size: int = 18,
    color: str = "#000000",
    feather: float = 6.0,
    hair: bool | None = None,
    image: UploadFile = File(...),
) -> Response:
    data = _read_upload_bytes(image)
    img = _open_image(data)
    fmt = _detect_input_format(image, img)
    _ensure_supported(fmt)

    try:
        rgb_color = ImageColor.getrgb(color)
        color_rgb = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid color: {color}") from exc

    rgb, alpha = _pil_to_rgb_array(img)
    use_hair = settings.censor_hair if hair is None else bool(hair)
    used_backend: str

    if use_hair and settings.skin_backend == "onnx_smp":
        try:
            skin01, hair01 = skin_hair_masks_onnx_smp(
                rgb,
                model_path=settings.skin_model_path,
                skin_score_threshold=settings.skin_score_threshold,
                skin_channel_index=settings.skin_channel_index,
                skin_require_max=settings.skin_require_max,
                skin_margin=settings.skin_margin,
                skin_min_area=settings.min_component_area,
                hair_score_threshold=settings.hair_score_threshold,
                hair_channel_index=settings.hair_channel_index,
                hair_require_max=settings.hair_require_max,
                hair_margin=settings.hair_margin,
                hair_min_area=settings.hair_min_component_area,
            )
            mask01 = np.maximum(skin01, hair01)
            used_backend = "onnx_smp"
            backend_error = None
        except Exception:
            skin01, used_backend, backend_error = skin_mask_dispatch_info(
                rgb,
                backend=settings.skin_backend,
                model_path=settings.skin_model_path,
                score_threshold=settings.skin_score_threshold,
                skin_channel_index=settings.skin_channel_index,
                require_max=settings.skin_require_max,
                margin=settings.skin_margin,
                min_area=settings.min_component_area,
                schp_model_path=settings.schp_model_path,
                schp_input_size=settings.schp_input_size,
                schp_skin_class_ids=tuple(settings.schp_skin_class_ids),
                schp_min_confidence=settings.schp_min_confidence,
                cv_max_side=settings.mask_max_side,
                cv_mode=settings.skin_mode,
                cv_maha_threshold=settings.skin_maha_threshold,
                cv_min_seed_pixels=settings.skin_min_seed_pixels,
            )
            mask01 = skin01
    else:
        mask01, used_backend, backend_error = skin_mask_dispatch_info(
            rgb,
            backend=settings.skin_backend,
            model_path=settings.skin_model_path,
            score_threshold=settings.skin_score_threshold,
            skin_channel_index=settings.skin_channel_index,
            require_max=settings.skin_require_max,
            margin=settings.skin_margin,
            min_area=settings.min_component_area,
            schp_model_path=settings.schp_model_path,
            schp_input_size=settings.schp_input_size,
            schp_skin_class_ids=tuple(settings.schp_skin_class_ids),
            schp_min_confidence=settings.schp_min_confidence,
            cv_max_side=settings.mask_max_side,
            cv_mode=settings.skin_mode,
            cv_maha_threshold=settings.skin_maha_threshold,
            cv_min_seed_pixels=settings.skin_min_seed_pixels,
        )

    params = CensorParams(
        method=method,
        sigma=float(sigma),
        pixel_size=int(pixel_size),
        color_rgb=color_rgb,
        feather=float(feather),
    )
    out_rgb = apply_censor(rgb, mask01, params)
    out_img = _compose_with_alpha(out_rgb, alpha)
    out_bytes = _encode_image(out_img, fmt)

    headers = {
        "X-Skin-Backend-Requested": settings.skin_backend,
        "X-Skin-Backend-Used": used_backend,
    }
    if backend_error:
        safe = " ".join(str(backend_error).splitlines()).strip()
        safe = safe.encode("ascii", "replace").decode("ascii")
        headers["X-Skin-Backend-Error"] = safe[:800]
    return Response(content=out_bytes, media_type=_content_type_for(fmt), headers=headers)
