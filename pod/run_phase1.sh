#!/usr/bin/env bash
# Phase 1 on a pod. Pulls Phase 0 inputs from the private HF dataset, runs one or more models, pushes outputs back.
#   MODELS="gemma3_27b"            one model per pod when fanning out (pod/fanout.sh)
#   SMOKE=1                        ~10-minute validation of the whole GPU path on the real model (outputs under <model>_smoke)
set -euo pipefail
cd "$(dirname "$0")/.."
# Without a network volume, /workspace is a 50GB pod volume: too small for a 54-62GB model. Use the container disk.
if [ -z "${HF_HOME:-}" ]; then
  if [ "$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc 0-9)" -ge 100 ] 2>/dev/null; then HF_HOME=/workspace/hf; else HF_HOME=/hf_cache; fi
fi
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export DPROBE_RESULTS=${DPROBE_RESULTS:-/workspace/dprobe_results}
MODELS=${MODELS:-"gemma3_27b gemma3_27b_pt gemma4_31b"}
SMOKE=${SMOKE:-0}
SYNC=${SYNC:-1}
PY=${PY:-python}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ "$SYNC" = "1" ]; then
  $PY -m dprobe.cli sync_down --subsets stories,spiral
fi
for m in $MODELS; do
  echo "================ $m  smoke=$SMOKE  $(date) ================"
  $PY -m dprobe.cli check_template "$m"
  if [ "$SMOKE" = "1" ]; then
    $PY -m dprobe.cli pod_phase1 "$m" --smoke
  else
    $PY -m dprobe.cli pod_phase1 "$m"
  fi
  if [ "$SYNC" = "1" ]; then
    $PY -m dprobe.cli sync_up --subsets vectors,probe,selfother
  fi
  echo "---- $m done $(date) ----"
done
echo "ALL DONE $(date)"
