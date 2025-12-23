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

ARG SKIN_SMP_MODEL_URL="https://raw.githubusercontent.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP/main/02.model/DeepLabV3Plus(timm-mobilenetv3_large_100)_1366_4.71M_0.8606/best_model_simplifier.onnx"
RUN mkdir -p /app/models \
  && curl -L --fail -o /app/models/skin_smp.onnx "${SKIN_SMP_MODEL_URL}"

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
