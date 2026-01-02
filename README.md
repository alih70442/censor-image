# Censor Image (CPU, offline)

HTTP service that detects human body parts (SCHP human parsing) and censors visible skin (blur / pixelate / solid color). Designed for Debian servers without GPU and to run fully offline at runtime.

## Links

- GitHub (source): https://github.com/alih70442/censor-image
- GitHub (SCHP upstream): https://github.com/PeikeLi/Self-Correction-Human-Parsing
- HuggingFace (model weights): TODOhttps://huggingface.co/aravindhv10/Self-Correction-Human-Parsing/tree/main

## Run (Docker)

```bash
docker build -t censor-image .
docker run --rm -p 8002:8000 \
  -e MAX_UPLOAD_MB=20 \
  censor-image
```

If you prefer a `.env` file, copy `.env.example` → `.env` and run with either:

- `docker compose up --build` (loads `.env` via `env_file` in `docker-compose.yml`), or
- `uvicorn app.main:app --env-file .env` (when running locally outside Docker)

Health check:

```bash
curl -sS http://localhost:8002/healthz
```

## Tuning

Hair censoring is supported when SCHP is used (`CENSOR_HAIR=true` or `hair=true` on `/censor` and `/mask`).

### Backends

- `SKIN_BACKEND=schp` (default): SCHP human parsing (ONNX, CPU) → skin parts mask (+ optional expansion)
- `SKIN_BACKEND=mhp`: Detectron2 person instances + SCHP-on-crops (best multi-person quality; requires Detectron2 + weights)
- `SKIN_BACKEND=cv`: legacy color-based skin detection (fastest, lower quality)

### SCHP (recommended)

- `SCHP_MODEL_PATH` (default `models/schp_lip.onnx`)
- `SCHP_DATASET` (default `lip`)
- `SCHP_EXPAND_SKIN=true` (default): expands from SCHP skin parts using a per-image Cr/Cb model
- `SCHP_MAX_EXPAND_PX=45` (default): if >0, expansion only grows near SCHP skin parts (reduces clothing false positives)

The expansion uses:

- `SKIN_MAHA_THRESHOLD` (default `3.5`): lower values are more conservative (e.g. `2.6`–`3.0`)
- `SKIN_MIN_SEED_PIXELS` (default `350`): higher values require stronger evidence before expanding

### CV fallback (legacy)

- `SKIN_MODE=auto` (default)
- `SKIN_MODE=adaptive`
- `SKIN_MODE=heuristic`

### MHP (Detectron2, optional)

Follow Detectron2 installation requirements, then set:

- `SKIN_BACKEND=mhp`
- `MHP_D2_CONFIG` (e.g. `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`)
- `MHP_D2_WEIGHTS` (local `.pth` path; required for offline runtime)

## Model weights (best quality)

- SCHP single-person weights are linked in the upstream SCHP repo README (LIP / ATR / Pascal checkpoints).
- SCHP multi-person extension weights (CIHP demo) are linked in `mhp_extension/demo.ipynb` in the upstream repo (Detectron2 instance model + SCHP global/local parsing weights).

### Export `lip.pth` → ONNX (recommended)

If you have a SCHP checkpoint (e.g. `lip.pth`), export a compatible ONNX model with:

```bash
python scripts/export_schp_onnx.py --checkpoint /path/to/lip.pth --dataset lip --output models/schp_lip.onnx --external-data
```

This requires a separate env with `torch` and `onnx` installed (the service itself only needs `onnxruntime`).

If you see an `onnxruntime` error like `Unsupported model IR version`, re-export with the default `--ir-version 9` (or upgrade `onnxruntime`).

## Troubleshooting

- If you get `Missing boundary in multipart`, don’t manually set `Content-Type`; let your HTTP client set it for `multipart/form-data`.
- If SCHP fails to load (missing `onnxruntime`, bad model path), set `SKIN_BACKEND=cv` to keep the service running.
- If skin detection is too aggressive, lower `SKIN_MAHA_THRESHOLD` or set `SCHP_MAX_EXPAND_PX` (more conservative expansion).

## Endpoints

- `GET /health`
- `GET /healthz`
- `POST /censor` → returns the censored image (same dimensions; same format when possible)
- `POST /mask` → returns a PNG mask (debug)
- `POST /parts` → returns a PNG indexed parsing map (debug)
- `GET /parts/labels` → returns label names for a dataset

Both `POST /mask` and `POST /censor` include:

- `X-Skin-Backend-Requested`: `auto` | `schp` | `mhp` | `cv`
- `X-Skin-Backend-Used`: `schp-onnx` | `mhp-detectron2` | `cv`
- `X-SCHP-Dataset`: when SCHP is used
- `X-Skin-Backend-Error`: when a fallback was used

## Examples

Blur (PNG/JPEG/WebP input):

```bash
curl -sS -X POST "http://localhost:8002/censor?method=blur&sigma=12&feather=6" \
  -F "image=@input.png" \
  -o out.png
```

Pixelate:

```bash
curl -sS -X POST "http://localhost:8002/censor?method=pixelate&pixel_size=18&feather=4" \
  -F "image=@input.webp" \
  -o out.webp
```

Solid color:

```bash
curl -sS -X POST "http://localhost:8002/censor?method=solid&color=%23000000" \
  -F "image=@input.png" \
  -o out.png
```

## Notes about “lossless”

- PNG and lossless WebP are supported.
- JPEG cannot be lossless once pixels are modified; this app outputs best-effort JPEG by default. Set `ALLOW_LOSSY_JPEG=false` to reject JPEG outputs.
