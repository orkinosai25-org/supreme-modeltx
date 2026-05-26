#!/usr/bin/env bash
# run_training.sh — SMTX T-101 training launcher
#
# Supports three compute profiles (default: cpu-single-node):
#
#   cpu-single-node  — single process, no GPU required (default)
#     bash scripts/run_training.sh
#
#   cpu-distributed  — multi-node CPU via torchrun + Gloo backend
#     TRAINING_PROFILE=cpu-distributed \
#     NUM_NODES=2 NUM_PROCS_PER_NODE=1 MASTER_ADDR=<ip> \
#     bash scripts/run_training.sh
#
#   gpu-accelerated  — optional; only when CUDA quota is available
#     TRAINING_PROFILE=gpu-accelerated \
#     NUM_NODES=2 NUM_GPUS_PER_NODE=4 MASTER_ADDR=<ip> \
#     bash scripts/run_training.sh
#
# Environment variables (all optional, sane defaults shown):
#   TRAINING_PROFILE    — compute profile (default: auto-detected)
#   CHECKPOINT_DIR      — where to save checkpoints (default: /mnt/checkpoints/t101-cpu)
#   DATA_INPUT_DIR      — raw data directory         (default: /mnt/data/raw)
#   DATA_OUTPUT_DIR     — processed data output dir  (default: /mnt/data/processed)
#   TOKENIZER_PATH      — tokenizer directory         (default: tmodels/t101)
#   CONFIG              — training config JSON path   (default: auto-selected per profile)
#   NUM_NODES           — number of training nodes    (default: 1)
#   NUM_PROCS_PER_NODE  — CPU workers per node        (default: 1)
#   NUM_GPUS_PER_NODE   — GPUs per node (gpu profile) (default: auto-detected)
#   MASTER_ADDR         — rendezvous hostname/IP      (default: localhost)
#   MASTER_PORT         — rendezvous port             (default: 29500)
#   SKIP_DATASET        — set to "1" to skip dataset pipeline (default: unset)

set -euo pipefail

# ── Help / dry-run flag ───────────────────────────────────────────────────────
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'HELPEOF'
SMTX T-101 Training Launcher

Usage:
  bash scripts/run_training.sh [--help]

  Or via Docker:
    docker run --rm \
      -e TRAINING_PROFILE=cpu-single-node \
      <acr>.azurecr.io/smtx-train-cpu:latest \
      --help

Compute profiles (set via TRAINING_PROFILE env var):
  cpu-single-node   Single-process CPU training (default, no GPU required)
  cpu-distributed   Multi-node CPU via torchrun + Gloo backend
  gpu-accelerated   Multi-node GPU via DeepSpeed (requires CUDA quota)

Key environment variables:
  TRAINING_PROFILE      Compute profile (auto-detected if unset)
  CHECKPOINT_DIR        Checkpoint output directory
  DATA_INPUT_DIR        Raw training data directory   (default: /mnt/data/raw)
  DATA_OUTPUT_DIR       Processed data directory      (default: /mnt/data/processed)
  TOKENIZER_PATH        Tokenizer directory            (default: tmodels/t101)
  CONFIG                Training config JSON path      (auto-selected per profile)
  NUM_NODES             Number of training nodes       (default: 1)
  NUM_PROCS_PER_NODE    CPU workers per node           (default: 1)
  NUM_GPUS_PER_NODE     GPUs per node (gpu profile)   (default: auto-detected)
  MASTER_ADDR           Rendezvous hostname/IP         (default: localhost)
  MASTER_PORT           Rendezvous port                (default: 29500)
  OMP_NUM_THREADS       OpenMP thread count            (default: nproc)
  SKIP_DATASET          Set to "1" to skip dataset pipeline

Examples:
  # CPU single-node (default)
  TRAINING_PROFILE=cpu-single-node bash scripts/run_training.sh

  # CPU distributed (2 nodes)
  TRAINING_PROFILE=cpu-distributed NUM_NODES=2 MASTER_ADDR=<ip> bash scripts/run_training.sh

  # GPU accelerated
  TRAINING_PROFILE=gpu-accelerated NUM_NODES=2 NUM_GPUS_PER_NODE=4 bash scripts/run_training.sh
HELPEOF
  exit 0
fi

# ── Resolve script root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Detect compute profile ────────────────────────────────────────────────────
# Auto-detect if TRAINING_PROFILE is not set:
#   GPU present → gpu-accelerated
#   WORLD_SIZE > 1 in env → cpu-distributed
#   Otherwise → cpu-single-node
if [ -z "${TRAINING_PROFILE:-}" ]; then
  # Check for CUDA GPUs: nvidia-smi must succeed AND report at least one device.
  GPU_COUNT=0
  if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader >/tmp/smtx_gpu_list 2>/dev/null; then
    GPU_COUNT=$(wc -l < /tmp/smtx_gpu_list)
    rm -f /tmp/smtx_gpu_list
  fi

  if [ "${GPU_COUNT}" -gt 0 ]; then
    TRAINING_PROFILE="gpu-accelerated"
    echo "[INFO] ${GPU_COUNT} CUDA GPU(s) detected — using gpu-accelerated profile."
  elif [ "${WORLD_SIZE:-1}" -gt 1 ]; then
    TRAINING_PROFILE="cpu-distributed"
    echo "[INFO] WORLD_SIZE=${WORLD_SIZE} — using cpu-distributed profile."
  else
    TRAINING_PROFILE="cpu-single-node"
    echo "[INFO] No GPU found, single process — using cpu-single-node profile (default)."
  fi
fi
export TRAINING_PROFILE

# ── Profile-specific config defaults ─────────────────────────────────────────
case "${TRAINING_PROFILE}" in
  gpu-accelerated)
    DEFAULT_CONFIG="training/config_gpu_accelerated.json"
    DEFAULT_CHECKPOINT_DIR="/mnt/checkpoints/t101-gpu"
    ;;
  cpu-distributed)
    DEFAULT_CONFIG="training/config_cpu_distributed.json"
    DEFAULT_CHECKPOINT_DIR="/mnt/checkpoints/t101-cpu-dist"
    ;;
  cpu-single-node|*)
    DEFAULT_CONFIG="training/config_cpu_single.json"
    DEFAULT_CHECKPOINT_DIR="/mnt/checkpoints/t101-cpu"
    ;;
esac

# ── Defaults ──────────────────────────────────────────────────────────────────
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
export DATA_INPUT_DIR="${DATA_INPUT_DIR:-/mnt/data/raw}"
export DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-/mnt/data/processed}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-tmodels/t101}"
export NUM_NODES="${NUM_NODES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"

echo "============================================================"
echo " SMTX T-101 Training"
echo "============================================================"
echo "  Profile         : ${TRAINING_PROFILE}"
echo "  Config          : ${CONFIG}"
echo "  Checkpoint dir  : ${CHECKPOINT_DIR}"
echo "  Data input      : ${DATA_INPUT_DIR}"
echo "  Data output     : ${DATA_OUTPUT_DIR}"
echo "  Nodes           : ${NUM_NODES}"
echo "  Master addr     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================================"

# ── Step 1: Dataset pipeline ──────────────────────────────────────────────────
if [ "${SKIP_DATASET:-0}" != "1" ]; then
  echo "[$(date -u +%H:%M:%S)] Running dataset pipeline …"
  python training/dataset_pipeline.py \
    --input_dir  "${DATA_INPUT_DIR}" \
    --output_dir "${DATA_OUTPUT_DIR}" \
    --tokenizer_path "${TOKENIZER_PATH}"
  echo "[$(date -u +%H:%M:%S)] Dataset pipeline complete."
else
  echo "[$(date -u +%H:%M:%S)] Skipping dataset pipeline (SKIP_DATASET=1)."
fi

# ── Step 2: Create checkpoint directory ───────────────────────────────────────
mkdir -p "${CHECKPOINT_DIR}"

# ── Step 3: Launch training ────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Launching training (profile: ${TRAINING_PROFILE}) …"

case "${TRAINING_PROFILE}" in

  # ── GPU-accelerated: use DeepSpeed ──────────────────────────────────────────
  gpu-accelerated)
    # Auto-detect GPU count if not provided
    if [ -z "${NUM_GPUS_PER_NODE:-}" ]; then
      NUM_GPUS_PER_NODE=0
      if nvidia-smi --query-gpu=name --format=csv,noheader >/tmp/smtx_gpu_list 2>/dev/null; then
        NUM_GPUS_PER_NODE=$(wc -l < /tmp/smtx_gpu_list)
        rm -f /tmp/smtx_gpu_list
      fi
      # Default to 1 if detection fails to avoid passing --num_gpus=0 to deepspeed
      [ "${NUM_GPUS_PER_NODE}" -gt 0 ] || NUM_GPUS_PER_NODE=1
    fi
    export NUM_GPUS_PER_NODE

    echo "  GPUs / node : ${NUM_GPUS_PER_NODE}"

    deepspeed \
      --num_nodes="${NUM_NODES}" \
      --num_gpus="${NUM_GPUS_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      training/train_t101.py \
      --config "${CONFIG}" \
      --profile gpu-accelerated
    ;;

  # ── CPU distributed: use torchrun with Gloo backend ─────────────────────────
  cpu-distributed)
    NUM_PROCS_PER_NODE="${NUM_PROCS_PER_NODE:-1}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    echo "  Procs / node : ${NUM_PROCS_PER_NODE}"
    echo "  Gloo iface   : ${GLOO_SOCKET_IFNAME}"

    torchrun \
      --nproc_per_node="${NUM_PROCS_PER_NODE}" \
      --nnodes="${NUM_NODES}" \
      --node_rank="${NODE_RANK:-0}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      --rdzv_backend=c10d \
      training/train_t101.py \
      --config "${CONFIG}" \
      --profile cpu-distributed
    ;;

  # ── CPU single-node: plain python, no distribution ──────────────────────────
  cpu-single-node|*)
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 1)}"

    echo "  OMP threads  : ${OMP_NUM_THREADS}"

    python training/train_t101.py \
      --config "${CONFIG}" \
      --profile cpu-single-node
    ;;

esac

echo "[$(date -u +%H:%M:%S)] Training complete. Checkpoints at: ${CHECKPOINT_DIR}"

