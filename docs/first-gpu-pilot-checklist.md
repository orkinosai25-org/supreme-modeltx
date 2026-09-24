# First GPU pilot checklist

- [ ] Confirm the run stays inside the private Supreme Model T-X repository workflow.
- [ ] Install the approved CUDA-enabled PyTorch environment on the GPU host.
- [ ] Run `python scripts/gpu_diagnostics.py --config configs/foundation/training-single-gpu.yaml --require-cuda`.
- [ ] Validate the private dataset with `python -m supreme_modeltx.foundation.dataset_tools validate ... --scan-pii --scan-secrets`.
- [ ] Run `python -m pytest tests/smoke/test_foundation_cpu_smoke.py -v` if the environment image is new.
- [ ] Start the pilot with `python -m supreme_modeltx.foundation.training --config configs/foundation/training-single-gpu.yaml`.
- [ ] Verify that checkpoints, `training_summary.json`, and `metrics.jsonl` were written and copied to approved private storage.
- [ ] Run the planned evaluation command against the retained checkpoint.
- [ ] Capture diagnostics, config, checkpoint path, and operator notes in the experiment record.
- [ ] Stop training and shut the GPU host down when validation and backup are complete.
