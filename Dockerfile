FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE requirements.txt /workspace/
COPY src /workspace/src
COPY configs /workspace/configs
COPY docs /workspace/docs

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e ".[dev]" && \
    pip install -r requirements.txt

CMD ["python", "-m", "supreme_modeltx.foundation.training", "--config", "configs/foundation/training-smoke.yaml"]
