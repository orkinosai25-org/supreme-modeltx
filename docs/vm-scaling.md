# SMTX VM Scaling Guide

SUMOTX is a **CPU-only LLM platform**. All training, inference, and serving layers run on standard Azure CPU VMs — no GPU quota required.

This document describes how to scale each SUMOTX architectural layer by adding more VMs, along with recommended CPU sizes, estimated costs, and the one-click GitHub Actions workflow.

---

## Overview

SMTX uses a pure VM-based architecture split into **7 layers**. Each layer starts with a conservative baseline (1–2 VMs) and can be scaled independently as traffic, training load, or budget allows.

All scaling is done through the **Scale VM Layer** GitHub Actions workflow, which re-applies the Bicep templates with an updated count. Because Bicep is idempotent, existing VMs are not recreated — only new VMs are added.

---

## Workflow: Scale VM Layer

Go to **Actions → Scale VM Layer** and trigger with:

| Input | Description |
|-------|-------------|
| `layer` | One of: `api`, `frontend`, `data`, `inference`, `training` |
| `new_vm_count` | New **total** count (not the number to add) |
| `training_vm_size` | CPU VM size override (only for `training` layer) |
| `inference_vm_size` | CPU VM size override (only for `inference` layer) |

### Example: add a second training VM

```
layer: training
new_vm_count: 2
training_vm_size: Standard_D16s_v5
```

### Example: upgrade training to larger CPU VM for faster throughput

```
layer: training
new_vm_count: 2
training_vm_size: Standard_E32s_v5   # 32 vCPU / 256 GiB
```

### Example: add a second CPU inference VM for higher concurrency

```
layer: inference
new_vm_count: 2
inference_vm_size: Standard_D16s_v5
```

### Example: scale data layer to 4 VMs for SharePoint-level load

```
layer: data
new_vm_count: 4
```

---

## Per-Layer Scaling Reference

### Layer 1 — Control / Orchestration

| | |
|-|-|
| **Purpose** | Job scheduler, deployment control, health monitor |
| **Default** | 1 VM |
| **Max** | 2 VMs (edit `parameters.json` directly) |
| **VM sizes** | `Standard_B2ms` (default) → `Standard_D2s_v5` |
| **Est. cost** | ~$60–120 / month |
| **Scale trigger** | Only if control plane becomes a bottleneck (rare) |

---

### Layer 2 — API / Backend

| | |
|-|-|
| **Purpose** | REST API, authentication, business logic |
| **Default** | 2 VMs |
| **Max** | 10 VMs |
| **VM sizes** | `Standard_D4s_v5` (default) → `Standard_D8s_v5` |
| **Est. cost** | ~$125/mo per VM (D4s_v5) |
| **Scale trigger** | API response latency > 500 ms under load |

**To scale:**
```
layer: api
new_vm_count: 4    # was 2
```

---

### Layer 3 — Frontend / Web / Chat

| | |
|-|-|
| **Purpose** | Public website, investor demo, chat interface |
| **Default** | 2 VMs (with public IPs) |
| **Max** | 10 VMs |
| **VM sizes** | `Standard_D4s_v5` (default) → `Standard_D8s_v5` |
| **Est. cost** | ~$125/mo per VM |
| **Scale trigger** | Concurrent visitor count or slow page load |

**To scale:**
```
layer: frontend
new_vm_count: 4
```

---

### Layer 4 — Model Lifecycle / Registry

| | |
|-|-|
| **Purpose** | Model versioning, artifact store, training job tracking |
| **Default** | 1 VM |
| **Max** | 2 VMs (edit `parameters.json` directly) |
| **VM sizes** | `Standard_D4s_v5` |
| **Est. cost** | ~$125/mo |
| **Scale trigger** | High model artifact upload/download throughput |

---

### Layer 5 — Data / Governance / SharePoint-like

| | |
|-|-|
| **Purpose** | Document ingestion, vector search, content governance |
| **Default** | 2 VMs |
| **Max** | 8 VMs |
| **VM sizes** | `Standard_D8s_v5` (default) → `Standard_E8s_v5` (memory optimised) |
| **Est. cost** | ~$250/mo per VM (D8s_v5) |
| **Scale trigger** | Large document corpus, high concurrent search queries |

**To scale:**
```
layer: data
new_vm_count: 6
```

---

### Layer 6 — CPU Inference

| | |
|-|-|
| **Purpose** | CPU-based LLM inference via llama-cpp-python (GGUF quantised models) |
| **Default** | 1 VM |
| **Max** | 5 VMs |
| **VM sizes** | `Standard_D8s_v5` (8 vCPU / 32 GiB, default — handles Q4 7B models) |
| | `Standard_D16s_v5` (16 vCPU / 64 GiB — 13B models) |
| | `Standard_E8s_v5` (8 vCPU / 64 GiB — memory-heavy models) |
| | `Standard_E16s_v5` (16 vCPU / 128 GiB — larger quantised models) |
| **Est. cost** | ~$250–500/mo per VM |
| **Scale trigger** | Inference latency under concurrent demo/public load |

**Model size guide (CPU GGUF):**

| Quantised model | Min RAM | Recommended VM |
|----------------|---------|----------------|
| 7B Q4_K_M | ~6 GiB | D8s_v5 |
| 13B Q4_K_M | ~10 GiB | D8s_v5 or D16s_v5 |
| 20B Q4_K_M | ~14 GiB | D16s_v5 or E8s_v5 |
| 34B Q4_K_M | ~22 GiB | E16s_v5 |

**To scale inference:**
```
layer: inference
new_vm_count: 2
inference_vm_size: Standard_D16s_v5
```

---

### Layer 7 — CPU LLM Training

| | |
|-|-|
| **Purpose** | CPU-based LoRA/QLoRA fine-tuning on Turkish culture, history, and Islamic AI datasets |
| **Default** | 1 VM |
| **Max** | 10 VMs |
| **VM sizes** | `Standard_D16s_v5` (16 vCPU / 64 GiB, default) |
| | `Standard_D32s_v5` (32 vCPU / 128 GiB — 2× faster multi-threaded training) |
| | `Standard_E16s_v5` (16 vCPU / 128 GiB — memory-optimised for larger models) |
| | `Standard_E32s_v5` (32 vCPU / 256 GiB — heavy fine-tuning workloads) |
| **Est. cost** | ~$500–1,600/mo per VM |
| **Scale trigger** | Training speed / dataset size |

**Practical CPU training path:**

1. Start with **1 × `Standard_D16s_v5`** (default)
2. Choose a small open base model: Phi-3 Mini 3.8B, Llama-3 8B, Qwen2 7B, or Mistral 7B
3. Use `peft` + LoRA with gradient checkpointing and a small batch size for CPU
4. Validate quality on your Turkish + Islamic AI dataset
5. If training speed is the bottleneck, scale to a larger CPU VM or add more nodes:

```
layer: training
new_vm_count: 2
training_vm_size: Standard_E32s_v5   # 32 vCPU / 256 GiB
```

---

## Budget Estimates (Monthly, CPU-Only)

> **Note:** All costs are approximate based on Azure East US pay-as-you-go pricing as of early 2026. Actual costs vary by region and commitment tier (reserved instances can cut costs 40–60%). Always verify at the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/).

| Layer | Baseline VMs | Est. cost / mo | Notes |
|-------|--------------|----------------|-------|
| Control | 1 × B2ms | ~$60 | |
| API | 2 × D4s_v5 | ~$250 | |
| Frontend | 2 × D4s_v5 | ~$250 | |
| Lifecycle | 1 × D4s_v5 | ~$125 | |
| Data | 2 × D8s_v5 | ~$500 | |
| Inference | 1 × D8s_v5 | ~$250 | CPU only |
| Training | 1 × D16s_v5 | ~$500 | CPU only |
| **Total baseline** | **10 CPU VMs** | **~$1,935 / mo** | No GPU needed |

With **$10k Azure credits** the baseline runs for approximately **4–5 months** before VM credits are exhausted. Additional costs for storage, egress, Key Vault operations, and ACR will reduce this further, so budget alerts are recommended from day one.

---

## Adding VMs Manually (CLI)

If you need to add VMs outside of GitHub Actions (e.g., for emergency capacity), use the Azure CLI:

```bash
# Scale training layer to 2 VMs with a larger CPU size
az deployment group create \
  --resource-group smtx-rg \
  --template-file infra/main.bicep \
  --parameters @infra/parameters.json \
               trainingVmCount=2 \
               trainingVmSize=Standard_E32s_v5 \
  --name smtx-scale-training-$(date +%s)

# Scale inference layer to 2 VMs
az deployment group create \
  --resource-group smtx-rg \
  --template-file infra/main.bicep \
  --parameters @infra/parameters.json \
               inferenceVmCount=2 \
               inferenceVmSize=Standard_D16s_v5 \
  --name smtx-scale-inference-$(date +%s)
```

Parameter reference per layer:

| Layer | Parameters |
|-------|-----------|
| API | `apiVmCount` |
| Frontend | `frontendVmCount` |
| Data | `dataVmCount` |
| Inference | `inferenceVmCount` + `inferenceVmSize` |
| Training | `trainingVmCount` + `trainingVmSize` |

---

## Pre-Deployment Checklist

1. ☐ Verify Azure VM quota for target region: `az vm list-usage --location eastus -o table`
2. ☐ Confirm remaining Azure credits are sufficient for the expected runtime
3. ☐ Ensure GGUF model weights are uploaded to the `models/gguf/` path in Azure Blob storage
4. ☐ Set a budget alert in the Azure Portal (Subscriptions → Cost Management → Budgets)
5. ☐ Add SSH public key as `SSH_PUBLIC_KEY` GitHub secret before running the infra workflow

---

## Related

- [`docs/azure.md`](azure.md) — full architecture overview and resource inventory
- [`.github/workflows/scale-vm-layer.yml`](../.github/workflows/scale-vm-layer.yml) — Scale VM Layer workflow
- [`.github/workflows/infra.yml`](../.github/workflows/infra.yml) — Full infrastructure provisioning
- [`.github/workflows/train.yml`](../.github/workflows/train.yml) — Training workflow (VM path is default)
