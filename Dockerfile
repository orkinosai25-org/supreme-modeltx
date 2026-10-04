FROM python:3.11-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

COPY pyproject.toml README.md LICENSE /workspace/
COPY src /workspace/src
COPY configs /workspace/configs

FROM base AS foundation

RUN pip install "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e ".[foundation,dev]"

CMD ["python", "-m", "supreme_modeltx.foundation.training", "--config", "configs/foundation/training-smoke.yaml"]

FROM base AS api

RUN pip install ".[api]" && \
    useradd --create-home --uid 10001 smtx

USER smtx
EXPOSE 9000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health/', timeout=2)" || exit 1

CMD ["smtx-serve"]
