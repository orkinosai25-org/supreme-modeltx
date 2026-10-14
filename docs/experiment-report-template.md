# Experiment report template

## Summary

- experiment name:
- date:
- operator:
- purpose:

## Configuration

- training config path:
- evaluation config path:
- dataset version:
- seed:
- device:
- mixed precision mode:
- configured precision enabled/dtype and executed effective dtype:
- model/config/settings SHA-256:
- dataset content and manifest SHA-256:
- tokenizer artifact SHA-256 (path/version alone is not identity):
- checkpoint/model artifact SHA-256:
- code revision:

## Environment verification

- PyTorch version:
- CUDA version:
- NVIDIA driver version:
- compute capability:
- cuDNN version:
- resolved wheel environment (`pip freeze` artifact):
- GPU name:
- VRAM:
- diagnostics artifact path:
- synchronized CUDA operation and FP16/BF16 numerical results:
- readiness status: candidate / GPU experiment validated (attach evidence):

## Training record

- start time:
- end time:
- max steps:
- batch size:
- gradient accumulation:
- interruption or resume details:

## Checkpoints

- checkpoint directory:
- retained checkpoints:
- backup location:

## Evaluation results

- evaluation suite:
- suite version:
- output JSON path:
- passed cases:
- failed cases:

## Notes

- hardware comparison eligibility (identical hashes, seed, precision, batches,
  accumulation, steps, eval cadence, sampling and benchmark settings):
- noncomparable reasons (if any; report no hardware-attributed deltas):
- missing/nonfinite metrics: null/N/A, never zero or NaN:
- observed issues:
- next actions:
- approval needed before the next run:
