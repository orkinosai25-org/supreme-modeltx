# Lambda A10 first run

This runbook uses a single Lambda A10 for private model experiments. It does not
provision or change Azure resources. Keep Azure as the application and
governance layer; the Lambda instance is a separate, manually operated GPU
execution environment until an authenticated integration is deliberately built.

## Connect and protect the instance

1. In Lambda, confirm the instance is **Running** and copy its SSH command.
2. Connect from your computer using the **private key kept on that computer**:

   ```bash
   chmod 600 /path/to/lambda-private-key
   ssh -i /path/to/lambda-private-key ubuntu@<instance-ip>
   ```

   Add only the corresponding public key to Lambda. Never copy or commit the
   private key, API tokens, or model credentials to the instance repository.
3. Inspect available storage with `df -hT`. Put the repository, model cache,
   and experiment outputs on the attached persistent filesystem if available;
   use its actual mount path rather than assuming a default.
4. Keep the instance private. For the examples below, services listen on
   `127.0.0.1` and are reached through an SSH tunnel, not an open public port.

## First chat: run a pretrained model

The repository's optional vLLM backend can download a Hugging Face model by ID
and serve it on the A10. This is the fastest way to verify a GPU-backed chat
path; it is separate from training or serving the repository's T-Dev-6L model.

From the repository root on the instance, set the Hugging Face cache to
persistent storage (replace the example path):

```bash
export HF_HOME=/path/to/persistent-filesystem/huggingface
mkdir -p "$HF_HOME"
nvidia-smi
```

Install vLLM into its own environment so its PyTorch requirements do not
conflict with the repository's pinned training environment:

```bash
python3 -m venv "$HOME/.venvs/smtx-vllm"
source "$HOME/.venvs/smtx-vllm/bin/activate"
python -m pip install --upgrade pip
python -m pip install vllm
```

Start the existing server. The model is downloaded to `HF_HOME` on first load:

```bash
python inference/vllm_server.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --host 127.0.0.1 \
  --port 8000 \
  --max_model_len 4096 \
  --gpu_memory_utilization 0.90
```

From a second terminal on your computer, forward the port:

```bash
ssh -i /path/to/lambda-private-key -L 8000:127.0.0.1:8000 ubuntu@<instance-ip>
```

Then test from your computer:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Give a one-sentence greeting in Turkish.","max_tokens":80}'
```

If the model is gated, accept its terms with the model provider and supply
`HF_TOKEN` through a secure shell environment; never put it in source control.
If model loading runs out of VRAM, lower `--max_model_len` before changing GPU
settings.

## Run the repository's T-Dev-6L GPU experiment

The canonical first training run is
[`t_dev_6l_first_gpu_run.md`](t_dev_6l_first_gpu_run.md) with config
`configs/real_training/t_dev_6l_first_gpu_run.json`. It trains the repository's
small T-Dev-6L model from the configured corpus; it does not fine-tune Mistral
or produce a general-purpose instruction-following assistant.

Use a separate environment from vLLM:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh --gpu --api
nvidia-smi
python scripts/gpu_diagnostics.py
bash scripts/validate_first_gpu_environment.sh
```

The validation script checks CUDA, prepares the versioned tokenizer if absent,
and runs the trainer preflight against the configured manifest and output
paths. Review the config and preflight results before starting the run:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json
```

The canonical config enables BF16 and targets a 24-GB-or-larger GPU. If
diagnostics show BF16 is unsupported, adjust the config to FP16 before running.
Keep generated checkpoints, tokenizer files, and logs on persistent storage
and outside Git; copy retained artifacts off the instance before stopping it.

To load a trained T-Dev-6L checkpoint in the repository's platform API, set
these variables to the actual paths and start `smtx-serve` from the repository
root:

```bash
export SMTX_CHECKPOINT_PATH=/path/to/checkpoints/step_<step>.pt
export SMTX_TOKENIZER_PATH=/path/to/tokenizer.model
export SMTX_MODEL_CONFIG_PATH="$PWD/configs/real_training/t_dev_6l_first_gpu_run.json"
export SMTX_INFERENCE_DTYPE=bfloat16
smtx-serve
```

The platform API uses port `9000`; it is not the vLLM server on port `8000`.
Keep it bound to loopback or behind an authenticated private network. The
existing `.env.example` documents additional optional API variables, and `.env`
is not loaded automatically.

## Azure relationship and shutdown

Azure remains the app/control-plane deployment. Neither the Lambda server nor
its SSH tunnel is automatically wired into the Azure app. Do not expose an
unauthenticated inference port or add Azure settings/resources as part of this
experiment; decide on an authenticated private connection separately.

Before ending a session, retain any required artifacts on persistent storage,
stop the model process, and stop the GPU instance when it is not being used.
