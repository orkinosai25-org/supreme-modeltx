FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

LABEL maintainer="SMTX Project" \
      version="0.2.0" \
      description="SMTX T-101 training container (DeepSpeed ZeRO-3 + multi-node)"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    openssh-client \
    pdsh \
    && rm -rf /var/lib/apt/lists/*

# Install Python training dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    datasets>=2.19.0 \
    deepspeed>=0.14.0 \
    accelerate>=0.29.0 \
    sentencepiece>=0.2.0 \
    tokenizers>=0.19.0 \
    pydantic>=2.0.0 \
    azure-storage-blob>=12.20.0 \
    azure-identity>=1.16.0

# Copy project source
COPY . /workspace

# Create checkpoint mount point
RUN mkdir -p /mnt/checkpoints /mnt/data/raw /mnt/data/processed

ENV PYTHONPATH=/workspace \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_DISABLE=1

# Multi-node training is driven by scripts/run_training.sh.
# Override via environment variables:
#   CHECKPOINT_DIR, DATA_INPUT_DIR, DATA_OUTPUT_DIR, NUM_NODES,
#   NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, SKIP_DATASET, CONFIG
ENTRYPOINT ["bash", "scripts/run_training.sh"]
