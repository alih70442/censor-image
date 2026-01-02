# Models

The service can load SCHP (Self-Correction Human Parsing) ONNX models from this folder when `SKIN_BACKEND=schp` (default).

Default:

- `SCHP_MODEL_PATH=models/schp_lip.onnx`

Notes:

- `schp_lip.onnx` uses external weights in `schp_lip.onnx.data`; keep both files together.
- You can replace the ONNX file(s) and point `SCHP_MODEL_PATH` to any local ONNX model compatible with `SCHP_DATASET`.
