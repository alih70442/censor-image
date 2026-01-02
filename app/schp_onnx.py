from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import cv2
import numpy as np


@dataclass(frozen=True)
class SchpDatasetSpec:
    name: str
    input_size: tuple[int, int]
    num_classes: int
    labels: tuple[str, ...]

    @property
    def label_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.labels)}


_DATASETS: Final[dict[str, SchpDatasetSpec]] = {
    "lip": SchpDatasetSpec(
        name="lip",
        input_size=(473, 473),
        num_classes=20,
        labels=(
            "Background",
            "Hat",
            "Hair",
            "Glove",
            "Sunglasses",
            "Upper-clothes",
            "Dress",
            "Coat",
            "Socks",
            "Pants",
            "Jumpsuits",
            "Scarf",
            "Skirt",
            "Face",
            "Left-arm",
            "Right-arm",
            "Left-leg",
            "Right-leg",
            "Left-shoe",
            "Right-shoe",
        ),
    ),
    "atr": SchpDatasetSpec(
        name="atr",
        input_size=(512, 512),
        num_classes=18,
        labels=(
            "Background",
            "Hat",
            "Hair",
            "Sunglasses",
            "Upper-clothes",
            "Skirt",
            "Pants",
            "Dress",
            "Belt",
            "Left-shoe",
            "Right-shoe",
            "Face",
            "Left-leg",
            "Right-leg",
            "Left-arm",
            "Right-arm",
            "Bag",
            "Scarf",
        ),
    ),
    "pascal": SchpDatasetSpec(
        name="pascal",
        input_size=(512, 512),
        num_classes=7,
        labels=(
            "Background",
            "Head",
            "Torso",
            "Upper Arms",
            "Lower Arms",
            "Upper Legs",
            "Lower Legs",
        ),
    ),
}


def get_dataset_spec(name: str) -> SchpDatasetSpec:
    key = (name or "lip").strip().lower()
    if key not in _DATASETS:
        raise ValueError(f"Unknown SCHP dataset: {name}")
    return _DATASETS[key]


def get_palette(num_cls: int) -> list[int]:
    """
    Standard segmentation palette used by SCHP scripts.
    Returns a flat list of length num_cls*3.
    """
    n = int(num_cls)
    palette = [0] * (n * 3)
    for j in range(n):
        lab = j
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def _import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore

        return ort
    except Exception as exc:  # pragma: no cover - runtime-dependent
        raise RuntimeError(
            "onnxruntime is required for SCHP parsing; install onnxruntime or set SKIN_BACKEND=cv"
        ) from exc


def _xywh2cs(x: float, y: float, w: float, h: float, aspect_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.zeros((2,), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def _get_dir(src_point: np.ndarray, rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([src_point[0] * cs - src_point[1] * sn, src_point[0] * sn + src_point[1] * cs], dtype=np.float32)


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple[int, int],
    shift: np.ndarray | None = None,
    inv: bool = False,
) -> np.ndarray:
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)

    scale_tmp = scale.astype(np.float32)
    src_w = float(scale_tmp[0])
    dst_w = float(output_size[0])
    dst_h = float(output_size[1])

    rot_rad = np.pi * float(rot) / 180.0
    src_dir = _get_dir(np.array([0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5], dtype=np.float32)
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5], dtype=np.float32) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def _prepare_bgr_tensor(bgr: np.ndarray) -> np.ndarray:
    # SCHP default preprocessing (BGR input, ToTensor()+Normalize with ImageNet stats in BGR order).
    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    x = bgr.astype(np.float32) / 255.0
    x = (x - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    x = np.transpose(x, (2, 0, 1))[None, :, :, :]  # NCHW
    return np.ascontiguousarray(x, dtype=np.float32)


@lru_cache(maxsize=4)
def _get_ort_session(model_path: str, intra_threads: int, inter_threads: int):
    ort = _import_onnxruntime()
    opts = ort.SessionOptions()
    if int(intra_threads) > 0:
        opts.intra_op_num_threads = int(intra_threads)
    if int(inter_threads) > 0:
        opts.inter_op_num_threads = int(inter_threads)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def parse_schp_onnx(
    rgb: np.ndarray,
    *,
    model_path: str,
    dataset: str,
    input_size: tuple[int, int] | None = None,
    intra_threads: int = 0,
    inter_threads: int = 0,
) -> tuple[np.ndarray, SchpDatasetSpec]:
    """
    Returns (parsing_map, dataset_spec).

    parsing_map is a uint8 HxW image with label ids.
    """
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")

    spec = get_dataset_spec(dataset)
    in_w, in_h = (input_size or spec.input_size)
    # enforce square-ish shapes, but keep API generic
    in_w = int(in_w)
    in_h = int(in_h)
    if in_w <= 0 or in_h <= 0:
        raise ValueError("input_size must be positive")

    model_path_abs = str(Path(model_path).expanduser().resolve())
    if not Path(model_path_abs).exists():
        raise FileNotFoundError(model_path_abs)

    height, width = rgb.shape[:2]
    # Full-image center/scale (same logic as SCHP simple extractor for a single box covering the image).
    aspect = float(in_w) / float(in_h)
    center, scale = _xywh2cs(0.0, 0.0, float(width - 1), float(height - 1), aspect_ratio=aspect)

    trans = _get_affine_transform(center, scale, rot=0.0, output_size=(in_w, in_h), inv=False)
    bgr = rgb[:, :, ::-1]
    warped = cv2.warpAffine(
        bgr,
        trans,
        (in_w, in_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    inp = _prepare_bgr_tensor(warped)

    sess = _get_ort_session(model_path_abs, intra_threads=int(intra_threads), inter_threads=int(inter_threads))
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: inp})
    if not outputs:
        raise RuntimeError("SCHP ONNX session returned no outputs")
    out0 = outputs[0]

    # Common layouts: NCHW logits or NHWC logits. Convert to HxW labels.
    if out0.ndim == 4:
        if out0.shape[1] == spec.num_classes:
            logits = out0[0].astype(np.float32)  # C,H,W
            if logits.shape[1] != in_h or logits.shape[2] != in_w:
                resized = []
                for c in range(int(logits.shape[0])):
                    resized.append(cv2.resize(logits[c], (in_w, in_h), interpolation=cv2.INTER_LINEAR))
                logits = np.stack(resized, axis=0)
            parsing_small = np.argmax(logits, axis=0).astype(np.uint8)
        elif out0.shape[-1] == spec.num_classes:
            logits = out0[0].astype(np.float32)  # H,W,C
            if logits.shape[0] != in_h or logits.shape[1] != in_w:
                logits = cv2.resize(logits, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            parsing_small = np.argmax(logits, axis=-1).astype(np.uint8)
        else:
            raise RuntimeError(f"Unexpected SCHP output shape: {tuple(out0.shape)}")
    elif out0.ndim == 3:
        # Could be [N,H,W] already.
        parsing_small = out0[0].astype(np.uint8)
        if parsing_small.shape[0] != in_h or parsing_small.shape[1] != in_w:
            parsing_small = cv2.resize(parsing_small, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
    else:
        raise RuntimeError(f"Unexpected SCHP output shape: {tuple(out0.shape)}")

    inv_trans = _get_affine_transform(center, scale, rot=0.0, output_size=(in_w, in_h), inv=True)
    parsing = cv2.warpAffine(
        parsing_small,
        inv_trans,
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )
    parsing = np.ascontiguousarray(parsing, dtype=np.uint8)
    return parsing, spec
