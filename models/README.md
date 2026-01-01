# Models

This service can run fully offline once the ONNX model files are present in `models/`.

## Supported backends

### `onnx_schp` (human parsing)

Place an SCHP human-parsing ONNX model at:

- `models/schp.onnx`

This backend is designed to avoid censoring skin-colored clothing by selecting only body-part classes
(e.g. face/arms/legs) as "skin".

To generate `models/schp.onnx` from `GoGoDuck912/Self-Correction-Human-Parsing`, export the PyTorch model to ONNX
in that repo (once) and then copy the resulting `.onnx` here. The ONNX model should output per-pixel class scores
(typical shape: `[1, C, H, W]` or `[C, H, W]`).

### `onnx_smp` (skin/clothes/hair segmentation)

The Docker image downloads `skin_smp.onnx` during `docker build` by default, unless you provide it yourself at:

- `models/skin_smp.onnx`

### `onnx_segformer24` (human parsing)

Place a SegFormer human-parsing ONNX model at:

- `models/model.onnx` (or set `SEGFORMER24_MODEL_PATH`)

This backend is meant for models like `yolo12138/segformer-b2-human-parse-24` and builds a binary mask by selecting
configured semantic class IDs for "skin parts" plus (optionally) hair.

### `onnx_lvmhpv2` (human parsing)

If you have an ONNX human-parsing model trained on the LV-MHP v2 label set, you can use it via:

- `SKIN_BACKEND=onnx_lvmhpv2`
- `LVMHP_MODEL_PATH=models/lvmhp_v2.onnx` (or any path)
- `LVMHP_PREPROCESS=rgb_imagenet` (or `schp_bgr_imagenet`, depending on your export)
- `LVMHP_SKIN_CLASS_IDS=...` and optionally `LVMHP_HAIR_CLASS_IDS=...`

This backend is label-set agnostic: you must configure the label IDs that correspond to exposed skin/hair for your model.
