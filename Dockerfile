FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl libgl1 libglib2.0-0 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/models
COPY models /app/models

ARG SCHP_MODEL_URL=""
RUN if [ -n "${SCHP_MODEL_URL}" ] && [ ! -f /app/models/schp.onnx ]; then \
      curl -L --fail -o /app/models/schp.onnx "${SCHP_MODEL_URL}"; \
    fi

ARG SKIN_SMP_MODEL_URL="https://raw.githubusercontent.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP/main/02.model/DeepLabV3Plus(timm-mobilenetv3_large_100)_1366_4.71M_0.8606/best_model_simplifier.onnx"
RUN if [ ! -f /app/models/skin_smp.onnx ]; then \
      curl -L --fail -o /app/models/skin_smp.onnx "${SKIN_SMP_MODEL_URL}"; \
    fi

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
