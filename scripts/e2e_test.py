#!/usr/bin/env python3
"""
e2e_test.py — SMTX End-to-End Integration Test

Tests the full pipeline: Prompt → T-X → T-101 → T-201 → T-301 → T-501 → Response.

Usage:
    python scripts/e2e_test.py [--orchestrator-url http://<tx-url>]

Environment variables:
    TX_ORCHESTRATOR_URL  — T-X orchestrator URL       (default: http://localhost:8080)
    CPU_VM_IP            — CPU VM private IP           (default: 10.0.2.4)
    GPU_VM_IP            — GPU VM private IP           (default: 10.0.1.4)
    VLLM_ENABLED         — Include vLLM health check   (default: false)

Results are written to docs/test-results.md.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.e2e_test")

# ── Configuration ─────────────────────────────────────────────────────────────

TX_URL = os.environ.get("TX_ORCHESTRATOR_URL", "http://localhost:8080")
CPU_VM_IP = os.environ.get("CPU_VM_IP", "10.0.2.4")
GPU_VM_IP = os.environ.get("GPU_VM_IP", "10.0.1.4")
# Set VLLM_ENABLED=true only when GPU quota is available and vLLM is deployed.
VLLM_ENABLED = os.environ.get("VLLM_ENABLED", "false").lower() == "true"

RESULTS_PATH = Path(__file__).parent.parent / "docs" / "test-results.md"

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    "What is the SMTX T-Series architecture?",
    "Explain the difference between T-101 and T-201 models.",
    "How does SMTX handle multi-model governance?",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def check_health(client: httpx.Client, name: str, url: str) -> Dict[str, Any]:
    try:
        r = client.get(f"{url}/health", timeout=15)
        status = r.json().get("status", "unknown") if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as exc:
        status = f"error: {exc}"
    passed = status in ("ok", "degraded")
    logger.info("[%s] health → %s", name, status)
    return {"name": name, "url": url, "status": status, "passed": passed}


def run_orchestrate(client: httpx.Client, prompt: str) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k_retrieval": 3,
        "verification_threshold": 0.4,
    }
    try:
        r = client.post(f"{TX_URL}/orchestrate", json=payload, timeout=180)
        if r.status_code == 200:
            result = r.json()
            passed = bool(result.get("final_response"))
            return {"prompt": prompt, "passed": passed, "response": result, "error": None}
        else:
            return {"prompt": prompt, "passed": False, "response": None, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as exc:
        return {"prompt": prompt, "passed": False, "response": None, "error": str(exc)}


# ── Result writer ─────────────────────────────────────────────────────────────


def write_results(
    timestamp: str,
    health_results: List[Dict],
    orchestrate_results: List[Dict],
) -> None:
    total = len(orchestrate_results)
    passed = sum(1 for r in orchestrate_results if r["passed"])
    health_pass = sum(1 for h in health_results if h["passed"])

    lines = [
        "# SMTX End-to-End Test Results",
        "",
        f"**Run timestamp:** {timestamp}",
        f"**Orchestrator URL:** {TX_URL}",
        "",
        "---",
        "",
        "## 1. Health Check Results",
        "",
        f"**{health_pass}/{len(health_results)} services healthy**",
        "",
        "| Service | URL | Status | Passed |",
        "|---|---|---|---|",
    ]
    for h in health_results:
        tick = "✅" if h["passed"] else "❌"
        lines.append(f"| {h['name']} | `{h['url']}` | `{h['status']}` | {tick} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Pipeline Test Results",
        "",
        f"**{passed}/{total} prompts succeeded**",
        "",
    ]

    for i, r in enumerate(orchestrate_results, 1):
        tick = "✅" if r["passed"] else "❌"
        lines.append(f"### Test {i} {tick}")
        lines.append(f"**Prompt:** `{r['prompt']}`")
        lines.append("")
        if r["error"]:
            lines.append(f"**Error:** {r['error']}")
        elif r["response"]:
            resp = r["response"]
            final = resp.get("final_response", "")
            lines.append(f"**Final response (truncated):** {final[:300]}{'…' if len(final) > 300 else ''}")
            lines.append("")
            lines.append("**Pipeline stages:**")
            for stage in resp.get("pipeline_stages", []):
                lines.append(f"- `{stage.get('stage')}` — {stage}")
            lines.append("")
            verif = resp.get("verification", {})
            lines.append(
                f"**Verification:** supported={verif.get('supported')} | confidence={verif.get('confidence', 0):.3f}"
            )
        lines.append("")

    lines += [
        "---",
        "",
        f"**Overall: {passed}/{total} pipeline tests passed, {health_pass}/{len(health_results)} services healthy.**",
        "",
    ]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Test results written to %s", RESULTS_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMTX end-to-end integration test.")
    parser.add_argument("--orchestrator-url", default=TX_URL)
    parser.add_argument("--cpu-ip", default=CPU_VM_IP)
    parser.add_argument("--gpu-ip", default=GPU_VM_IP)
    parser.add_argument(
        "--vllm-enabled",
        action="store_true",
        default=VLLM_ENABLED,
        help="Include vLLM GPU inference health check (only when GPU quota is available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global TX_URL, CPU_VM_IP, GPU_VM_IP, VLLM_ENABLED
    TX_URL = args.orchestrator_url
    CPU_VM_IP = args.cpu_ip
    GPU_VM_IP = args.gpu_ip
    VLLM_ENABLED = args.vllm_enabled

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logger.info("SMTX E2E test starting at %s", timestamp)
    logger.info(
        "Orchestrator: %s | CPU VM: %s | GPU VM: %s | vLLM enabled: %s",
        TX_URL, CPU_VM_IP, GPU_VM_IP, VLLM_ENABLED,
    )

    with httpx.Client() as client:
        # Health checks — CPU inference is the primary (default) backend.
        # vLLM is checked only when VLLM_ENABLED is set.
        health_targets = [
            ("T-X Orchestrator",     TX_URL),
            ("CPU Inference",        f"http://{CPU_VM_IP}:8003"),
            ("T-301 Retrieval",      f"http://{CPU_VM_IP}:8001"),
            ("T-501 Verification",   f"http://{CPU_VM_IP}:8002"),
        ]
        if VLLM_ENABLED:
            health_targets.append(("vLLM Inference (GPU)", f"http://{GPU_VM_IP}:8000"))

        health_results = [check_health(client, name, url) for name, url in health_targets]

        # Pipeline tests
        orchestrate_results = []
        for prompt in TEST_PROMPTS:
            logger.info("Testing prompt: %s", prompt)
            result = run_orchestrate(client, prompt)
            orchestrate_results.append(result)
            logger.info("  → passed=%s", result["passed"])

    # Write results
    write_results(timestamp, health_results, orchestrate_results)

    # Exit code
    all_passed = all(r["passed"] for r in orchestrate_results)
    if not all_passed:
        logger.error("Some pipeline tests FAILED. See docs/test-results.md for details.")
        sys.exit(1)
    logger.info("All tests PASSED.")


if __name__ == "__main__":
    main()
