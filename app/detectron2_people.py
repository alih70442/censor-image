from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PersonInstance:
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray  # HxW uint8 {0,1}


def _import_detectron2():
    try:
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.engine import DefaultPredictor  # type: ignore
        from detectron2 import model_zoo  # type: ignore

        return get_cfg, DefaultPredictor, model_zoo
    except Exception as exc:  # pragma: no cover - runtime-dependent
        raise RuntimeError(
            "detectron2 is required for SKIN_BACKEND=mhp; install detectron2 + torch or use SKIN_BACKEND=schp"
        ) from exc


@lru_cache(maxsize=2)
def _get_predictor(config_file: str, weights_path: str, score_thresh: float):
    get_cfg, DefaultPredictor, model_zoo = _import_detectron2()

    cfg = get_cfg()
    if Path(config_file).exists():
        cfg.merge_from_file(config_file)
    else:
        cfg.merge_from_file(model_zoo.get_config_file(config_file))

    if not weights_path:
        raise ValueError("MHP_D2_WEIGHTS must be set to a local .pth file for offline use")
    weights_abs = str(Path(weights_path).expanduser().resolve())
    if not Path(weights_abs).exists():
        raise FileNotFoundError(weights_abs)

    cfg.MODEL.WEIGHTS = weights_abs
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


def detect_people_instances(
    bgr: np.ndarray,
    *,
    config_file: str,
    weights_path: str,
    score_thresh: float = 0.5,
    max_people: int = 25,
    person_class_id: int = 0,
) -> list[PersonInstance]:
    """
    Detect people using Detectron2 instance segmentation models (e.g. Mask R-CNN).
    Returns masks in input image coordinates.
    """
    if bgr.dtype != np.uint8:
        raise ValueError("bgr must be uint8")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be HxWx3")

    predictor = _get_predictor(config_file, weights_path, float(score_thresh))
    outputs = predictor(bgr)
    if "instances" not in outputs:
        return []

    instances = outputs["instances"].to("cpu")
    if not hasattr(instances, "pred_masks") or not hasattr(instances, "pred_boxes") or not hasattr(instances, "scores"):
        return []

    classes = instances.pred_classes.numpy()
    keep = classes == int(person_class_id)
    if not np.any(keep):
        return []

    instances = instances[keep]
    scores = instances.scores.numpy().astype(np.float32)
    order = np.argsort(-scores)
    order = order[: int(max_people)]

    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)  # Nx4 xyxy
    masks = instances.pred_masks.numpy().astype(np.uint8)  # NxHxW {0,1}

    h, w = bgr.shape[:2]
    out: list[PersonInstance] = []
    for idx in order:
        x1, y1, x2, y2 = boxes[int(idx)].tolist()
        xi1 = int(max(0, min(w - 1, np.floor(x1))))
        yi1 = int(max(0, min(h - 1, np.floor(y1))))
        xi2 = int(max(0, min(w, np.ceil(x2))))
        yi2 = int(max(0, min(h, np.ceil(y2))))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        out.append(
            PersonInstance(
                score=float(scores[int(idx)]),
                bbox_xyxy=(xi1, yi1, xi2, yi2),
                mask=masks[int(idx)],
            )
        )
    return out

