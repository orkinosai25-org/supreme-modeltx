#!/usr/bin/env bash
# health_check.sh — SMTX Inference Layer Health Check
#
# Tests the /health endpoints on the GPU VM (vLLM) and CPU VM
# (retrieval + verification) and optionally sends a smoke-test generate request.
#
# Usage:
#   bash scripts/health_check.sh [--gpu-ip <ip>] [--cpu-ip <ip>] [--smoke-test]
#
# Defaults read from environment:
#   GPU_VM_IP   — GPU VM private IP  (default: 10.0.1.4)
#   CPU_VM_IP   — CPU VM private IP  (default: 10.0.2.4)

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
GPU_VM_IP="${GPU_VM_IP:-10.0.1.4}"
CPU_VM_IP="${CPU_VM_IP:-10.0.2.4}"
SMOKE_TEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ip)     GPU_VM_IP="$2"; shift 2 ;;
    --cpu-ip)     CPU_VM_IP="$2"; shift 2 ;;
    --smoke-test) SMOKE_TEST=1; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

PASS=0
FAIL=0

# ── Helper ────────────────────────────────────────────────────────────────────
check_health() {
  local name="$1"
  local url="$2"
  local http_code
  http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${url}" || echo "000")
  if [ "${http_code}" = "200" ]; then
    echo "[PASS] ${name} — ${url} → HTTP ${http_code}"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] ${name} — ${url} → HTTP ${http_code}"
    FAIL=$((FAIL + 1))
  fi
}

# ── Endpoint health checks ────────────────────────────────────────────────────
echo "=== SMTX Health Check ==="
echo "  GPU VM : ${GPU_VM_IP}"
echo "  CPU VM : ${CPU_VM_IP}"
echo ""

check_health "vLLM inference"  "http://${GPU_VM_IP}:8000/health"
check_health "T-301 retrieval" "http://${CPU_VM_IP}:8001/health"
check_health "T-501 verification" "http://${CPU_VM_IP}:8002/health"

# ── Smoke test ────────────────────────────────────────────────────────────────
if [ "${SMOKE_TEST}" -eq 1 ]; then
  echo ""
  echo "=== Smoke Test: POST /generate ==="
  RESPONSE=$(curl -s -X POST "http://${GPU_VM_IP}:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello, SMTX!","max_tokens":16,"temperature":0.0}' \
    --max-time 60 || echo '{"error":"request failed"}')
  echo "Response: ${RESPONSE}"
  if echo "${RESPONSE}" | grep -q '"text"'; then
    echo "[PASS] Smoke test generate — got text response"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] Smoke test generate — unexpected response"
    FAIL=$((FAIL + 1))
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
echo "  PASS: ${PASS}"
echo "  FAIL: ${FAIL}"

if [ "${FAIL}" -gt 0 ]; then
  exit 1
fi
