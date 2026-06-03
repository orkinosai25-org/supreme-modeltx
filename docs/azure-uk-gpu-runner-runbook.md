# Azure UK GPU Runner Runbook for the First T-Dev-6L Run

This runbook provides a repeatable, startup-credit-conscious path to:

- provision a Linux GPU VM in Azure
- prefer `uksouth`, then fall back to `ukwest`
- validate GPU readiness
- prepare the repo runtime
- optionally register the VM as a GitHub self-hosted runner
- run the existing first GPU workflow or equivalent commands
- stop/deallocate the VM when finished

## 1. Provision the VM

Use the plain Ubuntu 22.04 image path, not a Marketplace GPU image:

```bash
cd /path/to/supreme-modeltx

bash scripts/provision_azure_gpu_vm.sh \
  --resource-group smtx-gpu-rg \
  --vm-name smtx-gpu-runner \
  --ssh-public-key ~/.ssh/id_ed25519.pub
```

Default behavior:

- region preference: `uksouth`, then `ukwest`
- image: `Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest`
- GPU size order:
  1. `Standard_NC24ads_A100_v4`
  2. `Standard_NC8as_T4_v3`
  3. `Standard_NC4as_T4_v3`

Notes:

- The first size is the best fit for the canonical config.
- The T4 sizes are cheaper fallback/bootstrap options; they are useful for runner bring-up and preflight, but may require reduced batch settings for full training because the canonical run plan expects `24 GB+` VRAM.
- If quota or capacity blocks `uksouth`, the script automatically retries in `ukwest`.

If you want a different SKU order, repeat `--vm-size` flags explicitly.

## 2. Connect and bootstrap

SSH into the VM using the public IP printed by the provisioning script:

```bash
ssh smtxadmin@<public-ip>
```

Then clone the repo on the VM and bootstrap it:

```bash
git clone https://github.com/orkinosai25-org/supreme-modeltx.git
cd supreme-modeltx
sudo bash scripts/bootstrap_azure_gpu_vm.sh
```

Bootstrap covers:

- system package install
- `nvidia-smi` verification (and a clear failure if the driver is not ready yet)
- repo checkout/update
- Python virtualenv creation
- `pip install -e ".[dev]"`
- PyTorch CUDA verification

If the Azure NVIDIA extension was skipped or failed, install/retry it from your workstation:

```bash
az vm extension set \
  --resource-group smtx-gpu-rg \
  --vm-name smtx-gpu-runner \
  --publisher Microsoft.HpcCompute \
  --name NvidiaGpuDriverLinux \
  --enable-auto-upgrade true
```

Then reconnect and rerun:

```bash
sudo bash scripts/bootstrap_azure_gpu_vm.sh
```

## 3. Optional: register as a GitHub self-hosted runner

The existing workflow expects labels:

- `self-hosted`
- `linux`
- `x64`
- `gpu`

Safe default: generate an ephemeral registration token in GitHub, then pass it only at bootstrap time.

Example:

```bash
export GITHUB_RUNNER_URL="https://github.com/orkinosai25-org/supreme-modeltx"
export GITHUB_RUNNER_TOKEN="<ephemeral-runner-token>"
sudo -E bash scripts/bootstrap_azure_gpu_vm.sh
```

If you prefer manual registration instead of passing the token into the script:

1. Open **GitHub → Settings → Actions → Runners → New self-hosted runner**
2. Choose **Linux / x64**
3. Use labels: `self-hosted,linux,x64,gpu`
4. Start the runner service on the VM

## 4. Run first-pass validation before consuming GPU time

From the VM:

```bash
cd ~/supreme-modeltx
bash scripts/validate_first_gpu_environment.sh
```

Validation covers:

- `nvidia-smi`
- PyTorch CUDA availability
- manifest path verification
- tokenizer path verification (auto-prepares the canonical tokenizer if missing)
- output directory readiness
- training preflight:

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json \
  --preflight
```

Expected failure conditions before a full run:

- `nvidia-smi` unavailable
- `torch.cuda.is_available()` is false
- manifest/config/tokenizer inputs missing
- artifact directories not writable

## 5. Execute the first GPU path

### Option A — use the existing GitHub workflow

Once the runner is online, trigger:

- `.github/workflows/first-gpu-experiment.yml`

This runs:

- package install
- CUDA verification
- tokenizer preparation
- trainer preflight
- canonical first GPU training run
- benchmark scoring
- CPU-vs-GPU comparison artifact generation

### Option B — run equivalent commands manually on the VM

```bash
cd ~/supreme-modeltx
. .venv/bin/activate

python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json \
  --preflight

python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_gpu_run.json
```

## 6. Cost controls and shutdown

When idle, deallocate instead of leaving the VM running:

```bash
az vm deallocate --resource-group smtx-gpu-rg --name smtx-gpu-runner
```

Restart later with:

```bash
az vm start --resource-group smtx-gpu-rg --name smtx-gpu-runner
```

Delete completely when the experiment is over:

```bash
az vm delete --yes --resource-group smtx-gpu-rg --name smtx-gpu-runner
```

Practical cost/quota notes:

- GPU quota or transient capacity is the most likely blocker in `uksouth`; the scripted fallback is `ukwest`.
- Prefer a single-GPU VM for the first run.
- Use the cheaper T4 SKUs for bootstrap/preflight if A100 capacity is unavailable, but treat them as a fallback rather than the canonical target.
