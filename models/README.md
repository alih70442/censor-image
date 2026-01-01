# Models

This service can run fully offline once the ONNX model files are present in `models/`.

## Supported backends

### `onnx_mhp` (LV-MHP v2 / MHParsNet human parsing)

Place your LV-MHP v2 ONNX export at:

- `models/MHParsNet.onnx`
- `models/MHParsNet_logits.onnx` (recommended)

This backend builds a skin mask from selected body-part classes (e.g. face/arms/legs).
Configure which classes count as “skin” via `MHP_SKIN_CLASS_IDS`.
If your ONNX export has fixed I/O names, set `MHP_INPUT_NAME`/`MHP_OUTPUT_NAME` accordingly (commonly `image`/`logits`).

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
