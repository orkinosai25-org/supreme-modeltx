# GPU Readiness and Scaling Plan from T-Dev-6L Evidence

The two benchmarked T-Dev-6L runs now provide enough engineering evidence to define a concrete
GPU-readiness plan. The core conclusion is: the training/benchmark pipeline works end to end, the
expanded run shows real learning, and CPU-only execution is now the main constraint on moving from
engineering validation to application-ready model development.

## 1) What the benchmark evidence already proves

| Evidence area | `t_dev_6l_first_run` | `t_dev_6l_expanded_run` | What it means |
|---|---:|---:|---|
| Corpus | 20 synthetic sentences | 288 mixed code/reasoning examples | The pipeline scales beyond the tiny wiring-only corpus. |
| Training steps | 20 | 200 | A 10x longer schedule produces measurable quality gains. |
| Wall-clock | ~4.0s | ~15.8m | CPU runtime rises sharply as the experiment becomes more realistic. |
| Best validation loss | 5.9428 | 4.7820 | More data + longer schedule improved generalisation. |
| Best perplexity | 381.02 | 119.37 | Learning is no longer noise-only at the expanded scale. |
| Best internal benchmark score | 0.00 | 1.00 | The expanded run solved all 4 benchmark prompts at its best checkpoint. |
| Sample quality | empty/repetitive outputs | correct canonical answers by step 150 | The model can now learn the benchmarked task pattern, not just reduce loss. |

### Bottom line

- **Baseline run:** proved the artifact, checkpoint, sampling, and benchmark plumbing.
- **Expanded run:** proved that the same pipeline can produce real learning signal when given more
  data and a longer schedule.
- **Remaining gap:** both runs were still **CPU-only, float32, and materially smaller than the
  canonical GPU-oriented T-Dev-6L config**.

## 2) Current CPU bottlenecks, quantified

The current bottleneck is no longer “does training work?” but “can training scale beyond small CPU
experiments quickly enough to support model iteration?”

| Constraint | Current evidence |
|---|---|
| Device path | Both benchmarked runs used `device=cpu` and `precision.enabled=false`. |
| Effective end-to-end throughput | Expanded run took **948.6s for 200 steps** = **4.74s/step** (~0.21 steps/s). |
| Token throughput at current proxy settings | With batch size 2 and sequence length 128, the expanded run processed ~**54 packed tokens/s** end to end. |
| Time to longer schedules on CPU | At expanded-run throughput, **1,000 steps ≈ 79 minutes** and **5,000 steps ≈ 6.6 hours** before adding broader eval sweeps. |
| Model scale compromise | Benchmarked runs used a **256-hidden / 128-context proxy config**; the canonical dev config is **512-hidden / 512-context / BF16**. |
| Batch-size compromise | Benchmarked runs used batch size **2**; the canonical dev config targets batch size **8** with gradient accumulation **4**. |

This means CPU is constraining all four scaling axes at once:

1. **Schedule length** — practical runs stop at 200 steps instead of the 1k–10k range already
   described by the canonical config.
2. **Corpus size** — the expanded run is better than baseline, but still far below a full
   application-readiness training corpus.
3. **Model/context size** — the benchmarked proxy is smaller than the intended T-Dev-6L geometry.
4. **Experiment velocity** — once a single 200-step CPU run takes ~16 minutes, hyperparameter and
   benchmark iteration slows down materially.

## 3) Next target model and training scale

The immediate target should be the **canonical T-Dev-6L GPU validation run**, not a direct jump to
T-101.

### Immediate funded target

- **Model:** canonical T-Dev-6L (~58M params)
- **Shape:** 6 layers, hidden size 512, FFN 2048, context 512
- **Precision:** BF16
- **Training recipe target:** batch size 8, gradient accumulation 4, up to 10,000 steps
- **Data target:** move from the 288-example expanded set to the full SMTX-Baby/`data/raw` style
  corpus and follow-on larger corpora

### Why this is the right next target

It is the smallest GPU-backed experiment that closes the credibility gap between:

- a **benchmarked CPU proof** that the training stack learns, and
- an **application-ready training recipe** that uses the intended model shape, context window,
  precision mode, and a meaningfully larger corpus.

### Promotion target after that

Once canonical T-Dev-6L is stable on GPU, the next model-scale milestone is the repository roadmap
target:

- **T-101 (7B) first training run with GPU allocation**

That is the point where the project moves from dev-model validation into larger-scale sovereign model
development.

## 4) What GPU access would unlock

GPU access would not just make the current run faster; it would enable experiments that the current
CPU evidence cannot responsibly claim yet.

| Area | CPU-evidenced today | GPU-funded unlock |
|---|---|---|
| Schedule length | 20–200 steps | practical 1k–10k step training windows |
| Corpus size | 20 sentences → 288 examples | full seed corpus plus larger packed corpora |
| Precision | float32 only | BF16 / mixed-precision validation on the real training path |
| Model size | 256-hidden proxy | canonical ~58M T-Dev-6L, then T-101 preparation |
| Sequence length | 128 | 512-context canonical dev runs |
| Batch regime | batch size 2 | larger effective batch sizes with accumulation |
| Iteration speed | single longer CPU run per tuning loop | multiple benchmarked experiments in the same working day |

In practical terms, funded GPU access unlocks:

1. **Longer schedules** that can test whether the benchmark win at 200 steps survives at 1k–5k+.
2. **Larger corpora** that move beyond the current small benchmark-focused dataset toward
   application-relevant training evidence.
3. **Bigger batch / sequence / model settings** aligned with the intended T-Dev-6L configuration.
4. **Faster experiment iteration** so optimiser, schedule, tokenizer, and benchmark changes can be
   compared within days instead of serial CPU-bound loops.

## 5) Staged compute plan

### Minimum plan — prove canonical T-Dev-6L on GPU

**Goal:** validate the intended training path on GPU with the canonical dev model.

- Run canonical T-Dev-6L in BF16 on CUDA
- Use 512 context and the planned higher effective batch regime
- Train for **1,000 steps**
- Evaluate multiple checkpoints on the internal benchmark
- Record throughput, stability, and checkpoint-resume behaviour

**Success metrics**

- GPU/BF16 run completes end to end with no CUDA or mixed-precision failures
- Best validation loss beats the expanded CPU best of **4.7820**
- Internal benchmark remains at least as strong as the expanded run best checkpoint
- Throughput improvement over the CPU expanded run is measured and reported
- Checkpoint, sample, and benchmark artifacts remain reproducible

### Target plan — turn engineering proof into application-readiness evidence

**Goal:** show that the canonical dev model can train long enough and on enough data to support a
credible funded-development case.

- Run **3–5 benchmarked GPU experiments**
- Extend schedules into the **5,000-step** range
- Use the full seed corpus plus larger packed training mixtures
- Compare at least one schedule/optimizer variant
- Add broader benchmark coverage once the internal benchmark is stable

**Success metrics**

- Repeated runs show a stable validation-loss improvement trend, not a one-off win
- Benchmark gains persist beyond one checkpoint and one dataset slice
- At least one run demonstrates better sample quality on held prompts outside the initial 4-task set
- Per-run turnaround is fast enough to support multiple experiments per week
- Results are strong enough to justify promotion from dev-model validation to T-101 preparation

### Stretch plan — prepare the first T-101 GPU programme

**Goal:** use the validated T-Dev-6L recipe as the launchpad for larger-model training.

- Finalise the canonical T-Dev-6L recipe and benchmark contract
- Stand up the first **T-101 (7B)** GPU-backed training run
- Exercise distributed-training features needed for larger-scale experiments
- Publish benchmark and model-card evidence for the first T-101 milestone

**Success metrics**

- T-101 training starts and completes a first benchmarked run on allocated GPU compute
- The larger-model workflow preserves run-artifact, checkpoint, and benchmark reproducibility
- GPU utilisation evidence is clear enough for milestone-based funder reporting
- The project can point to a credible path from dev-model experiments to deployable sovereign model work

## 6) Why this is the bridge to application readiness

The current benchmark evidence already establishes three key facts:

1. **The engineering stack is real** — training, checkpointing, sampling, and benchmarking all work.
2. **Scaling the recipe improves outcomes** — the expanded run materially outperformed baseline.
3. **CPU is now the blocker** — the remaining limitations are schedule, corpus, precision, and model
   scale rather than core pipeline correctness.

That is exactly the point where funded GPU compute becomes application-relevant. The ask is no longer
for speculative infrastructure; it is for the compute required to turn an evidence-backed training
prototype into a repeatable, benchmarked, application-ready development programme.

## Evidence sources

- `artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.json`
- `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/training_summary.json`
- `artifacts/runs/t_dev_6l_expanded_run/benchmark_outputs/benchmark_results.md`
- `artifacts/runs/t_dev_6l_expanded_run/run_artifacts/samples.md`
- `configs/real_training/t_dev_6l_first_run.json`
- `configs/real_training/t_dev_6l_expanded_run.json`
- `configs/t_dev_6l.json`
- `docs/architecture/model-core.md`
- `docs/dataset-overview.md`
