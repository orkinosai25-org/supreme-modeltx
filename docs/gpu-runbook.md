# GPU runbook

## Scope

This runbook is for the first private single-GPU pilot of Supreme Model T-X. It assumes the repository stays private and that no proprietary datasets, credentials, or checkpoints are committed back into source control.

## Before provision

- Confirm the approved config file: `configs/foundation/training-single-gpu.yaml`
- Confirm the training dataset has passed JSONL validation, duplicate checks, and optional PII/secret scans
- Confirm destination storage for checkpoints and copied logs
- Set a manual budget ceiling before powering on a GPU instance

## Initial machine setup

1. Provision a single GPU instance.
2. Install Python 3.11 and create a virtual environment.
3. Install the project with `pip install -e ".[foundation,dev]"`.
4. Run `python scripts/gpu_diagnostics.py`.
5. Save the diagnostics output with the experiment record.

## CUDA and PyTorch verification

The diagnostics output must confirm:

- CUDA is available
- the expected GPU name is visible
- reported VRAM is enough for the configured batch size and accumulation plan
- the installed PyTorch version matches the approved environment
- the reported CUDA runtime version matches the driver environment
- BF16 or FP16 support is present before enabling mixed precision

## Training launch

1. Copy the approved config file into the run workspace.
2. Set the private dataset paths in `configs/foundation/training-single-gpu.yaml`.
3. Launch the run with `python -m supreme_modeltx.foundation.training --config configs/foundation/training-single-gpu.yaml`.
4. Record the exact config, start time, diagnostics output, and operator name in the experiment report.

## Checkpoint handling

- Keep checkpoints outside the repository working tree if they contain real model state
- Copy each retained checkpoint to approved private storage after creation
- Retain at least the latest checkpoint and the checkpoint tied to any recorded evaluation result
- Never commit checkpoints into Git

## Cost controls

- Use a single GPU only for the initial pilot
- Prefer fixed run windows with manual stop times
- Review utilisation before extending the run
- Shut down the instance immediately after checkpoint backup and log capture

## Interrupted run recovery

If the process stops intentionally or via interruption:

1. Confirm the latest checkpoint under the configured `output_dir/checkpoints/`
2. Copy the checkpoint and `training_summary.json` to backup storage
3. Set `resume_from` or leave `auto_resume_latest: true`
4. Relaunch with the same config and document the resumed run in the experiment report

## Shutdown and deletion

- confirm checkpoint copy completion
- confirm evaluation outputs are archived
- stop the GPU instance
- delete the instance if it is no longer required
- rotate any temporary runtime credentials used outside source control
