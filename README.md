# Censor Image (CPU, offline)

HTTP service that detects visible skin and censors it (blur / pixelate / solid color). Designed for Debian servers without GPU.

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

If you place a human-parsing ONNX model at `models/schp.onnx`, the app will prefer it automatically (best separation of skin vs skin-colored clothing):

- `SKIN_BACKEND=onnx_schp`
- `SCHP_MODEL_PATH=models/schp.onnx` (default)
- `SCHP_INPUT_SIZE=473`
- `SCHP_SKIN_CLASS_IDS=13,14,15,16,17` (default assumes LIP labels: face/arms/legs)
- `SCHP_MIN_CONFIDENCE=0.0`

Otherwise, the Docker image bundles an offline ONNX model for skin/clothes/hair segmentation at build time and uses it at runtime:

- `SKIN_BACKEND=onnx_smp` (default)
- `SKIN_MODEL_PATH=models/skin_smp.onnx` (default)
- `SKIN_SCORE_THRESHOLD=0.5`
- `SKIN_CHANNEL_INDEX=0` (model output channel for “skin”)
- `SKIN_REQUIRE_MAX=true` (only keep pixels where “skin” is the top class)
- `SKIN_MARGIN=0.05` (require `skin_prob - max(other_probs)` to exceed this)

Optional: also censor hair (ONNX backend only):

- `CENSOR_HAIR=true` to enable by default (or pass `hair=true` to `POST /censor`)
- `HAIR_CHANNEL_INDEX=2` (model output channel for “hair”)
- `HAIR_SCORE_THRESHOLD=0.5`
- `HAIR_REQUIRE_MAX=true`
- `HAIR_MARGIN=0.05`

If you need a fully local build without downloading during `docker build`, set `SKIN_BACKEND=cv` and rebuild (lower quality).

CV fallback (used when `SKIN_BACKEND=cv` or ONNX fails):

- `SKIN_MODE=auto` (default)
- `SKIN_MODE=adaptive`
- `SKIN_MODE=heuristic`

If skin detection is too aggressive (censors clothing), reduce expansion:

- `SKIN_MAHA_THRESHOLD` (default `3.5`): lower values are more conservative (e.g. `2.6`–`3.0`)
- `SKIN_MIN_SEED_PIXELS` (default `350`): higher values require stronger evidence before adapting

## Troubleshooting

- If you get `Missing boundary in multipart`, don’t manually set `Content-Type`; let your HTTP client set it for `multipart/form-data`.
- If `POST /mask` looks wrong, try adjusting `SKIN_SCORE_THRESHOLD` (lower = more skin) or `SKIN_MARGIN` (higher = less clothing false positives).
- If skin-colored clothing is being censored, use the `onnx_schp` backend with a human-parsing model (it builds the mask from body-part classes instead of color/texture).

## Endpoints

- `GET /health`
- `GET /healthz`
- `POST /censor` → returns the censored image (same dimensions; same format when possible)
- `POST /mask` → returns a PNG mask (debug)

Both `POST /mask` and `POST /censor` include:

- `X-Skin-Backend-Requested`: configured backend (e.g. `onnx_schp`)
- `X-Skin-Backend-Used`: backend actually used after fallback (e.g. `onnx_smp`, `cv`)
- `X-Skin-Backend-Error`: reason for fallback (only present when a fallback happened)

For `onnx_schp`, the server auto-detects the model’s fixed input resolution from the ONNX graph (e.g. 512×512) and will resize accordingly.

## Examples

Blur (PNG/JPEG/WebP input):

```bash
curl -sS -X POST "http://localhost:8002/censor?method=blur&sigma=12&feather=6&hair=true" \
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
