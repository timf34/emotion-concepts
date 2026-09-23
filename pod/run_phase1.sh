#!/usr/bin/env bash
# Phase 1 on the pod: vectors -> transcript probes -> self/other -> quantity sweep, one model at a time.
# Expects Phase 0 outputs (stories + transcripts + judgments) under $DPROBE_RESULTS (default /workspace/dprobe_results).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export DPROBE_RESULTS=${DPROBE_RESULTS:-/workspace/dprobe_results}
MODELS=${MODELS:-"gemma3_27b gemma3_27b_pt gemma4_31b"}
PY=${PY:-python}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for m in $MODELS; do
  echo "================ $m  $(date) ================"
  $PY -m dprobe.cli check_template "$m"
  $PY -m dprobe.cli pod_phase1 "$m"
  echo "---- $m done $(date) ----"
done
echo "ALL DONE $(date)"
