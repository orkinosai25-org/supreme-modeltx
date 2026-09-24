# GPU runbook

## Scope

This runbook covers the first controlled single-GPU pilot for the private Supreme Model T-X PyTorch development foundation. It preserves the CPU-safe local workflow and adds a documented CUDA host path for a single approved GPU instance.

## Operating rules

- Keep the repository private.
- Do not commit datasets, checkpoints, logs, model weights, or generated run artifacts.
- Do not place credentials, API keys, or cloud secrets in configs, shell history, or source control.
- Do not claim the project is a completed or production-trained LLM.

## CPU versus GPU environment setup

### CPU-safe local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.6"
pip install -e ".[foundation,dev]"
```

CPU validation commands:

```bash
python -m pytest tests/unit/test_foundation_config.py tests/unit/test_foundation_training.py tests/unit/test_foundation_gpu_diagnostics.py -v
python -m pytest tests/smoke/test_foundation_cpu_smoke.py -v
```

### CUDA host setup

Use a clean virtual environment on the GPU host instead of replacing the CPU image in-place.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PYTORCH_WHL_INDEX_URL="${PYTORCH_WHL_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
pip install --index-url "${PYTORCH_WHL_INDEX_URL}" "torch>=2.4,<2.6"
pip install -e ".[foundation,dev]"
```

## Cloud-host preparation

1. Provision a single GPU host with current NVIDIA drivers.
2. Confirm the host has enough local disk for temporary checkpoints and logs.
3. Mount or provision approved private storage for dataset input and checkpoint backup.
4. Copy private datasets outside the repository working tree.
5. Set a budget cap and shutdown time before starting the instance.

## CUDA/PyTorch compatibility guidance

- Keep Python within the repository constraint: `>=3.10`.
- Keep PyTorch within the repository constraint: `torch>=2.4,<2.6`.
- Choose `PYTORCH_WHL_INDEX_URL` to match the host driver/runtime combination.
- If CUDA diagnostics fail, fix the host driver or wheel selection before starting training.

## Diagnostics and strict preflight

Basic diagnostics:

```bash
python scripts/gpu_diagnostics.py
```

Strict single-GPU preflight:

```bash
python scripts/gpu_diagnostics.py --config configs/foundation/training-single-gpu.yaml --require-cuda
```

The strict command must report:

- CUDA availability
- visible device count
- GPU name and VRAM
- PyTorch version
- compiled CUDA version
- runtime details when available through `nvidia-smi`
- BF16/FP16 support status
- config/device/precision preflight status

If the command exits non-zero, do not start training. Fix the reported issue first.

## BF16 and FP16 selection guidance

- Prefer `bf16` when the GPU and installed PyTorch runtime report BF16 support.
- Use `fp16` only when CUDA is available but BF16 is not supported or not desired.
- Use `off` for CPU smoke runs.
- CPU smoke runs should not request `fp16`.
- If BF16 is requested on unsupported hardware, the preflight fails before training starts.

## First-run sequence

1. Activate the approved environment.
2. Validate private JSONL data and scans:
   ```bash
   python -m supreme_modeltx.foundation.dataset_tools validate /private/path/train.jsonl --scan-pii --scan-secrets
   ```
3. Review and update dataset paths in `configs/foundation/training-single-gpu.yaml`.
4. Run diagnostics and strict preflight:
   ```bash
   python scripts/gpu_diagnostics.py --config configs/foundation/training-single-gpu.yaml --require-cuda
   ```
5. Start a CPU smoke validation if the host image is new:
   ```bash
   python -m pytest tests/smoke/test_foundation_cpu_smoke.py -v
   ```
6. Launch the first single-GPU pilot:
   ```bash
   python -m supreme_modeltx.foundation.training --config configs/foundation/training-single-gpu.yaml
   ```
7. Record the config used, diagnostics output, and operator notes in `docs/experiment-report-template.md`.

## Determinism note

The training scaffold seeds Python and PyTorch, enables deterministic algorithms with `warn_only=True`, and disables cuDNN benchmarking. CUDA kernels can still have practical determinism limits on some operators, so repeatability should be treated as best-effort rather than bit-for-bit guaranteed across every host/runtime combination.

## Checkpoint backup and retention

- Keep live checkpoints under the configured output directory, not in Git.
- Back up retained checkpoints to approved private storage after each checkpoint interval.
- Retain at minimum:
  - the latest checkpoint
  - the checkpoint used for any recorded evaluation result
  - `training_summary.json` and `metrics.jsonl`
- Verify that backup copies complete before shutting the host down.

## Cost controls and shutdown steps

- Use one GPU only for this pilot.
- Stop the run if preflight, dataset validation, or checkpoint backup is incomplete.
- Set manual reminders for checkpoint review and host shutdown.
- After the run:
  1. verify checkpoint backup
  2. archive evaluation output and run notes
  3. stop training
  4. shut down the GPU instance
  5. delete the instance if it is no longer needed

## No-public-data / no-secrets rules

- Do not add proprietary dataset rows or samples to docs, tests, or logs.
- Do not print dataset contents in diagnostics.
- Do not commit real checkpoints, tokenizer models trained on private data, or weights.
- Do not commit credentials, tokens, `.env` values, or cloud-provider secrets.
