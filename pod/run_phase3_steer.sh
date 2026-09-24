#!/usr/bin/env bash
# Phase 3a (Gemma 4): steer along the spiral's own family and the assistant axis, per-label calibrated.
#   MODELS=gemma4_31b SHUTDOWN=stop bash pod/run_phase3_steer.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} PYTHONUNBUFFERED=1 DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
M=${MODELS:-gemma4_31b}; PY=${PY:-/venv/bin/python}; [ -x "$PY" ] || PY=python; FAILED=0
LABELS=${LABELS:-"calm:-,hysterical:+,panicked:+,desperate:+,assistant_axis:-"}
COMBOS=${COMBOS:-"calm:-+assistant_axis:-;calm:-+hysterical:+"}
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m dprobe.cli sync_down --subsets vectors --models "$M" || { echo "!! sync_down failed"; FAILED=1; }
[ -f "$DPROBE_RESULTS/vectors/$M/vectors_external_dn.pt" ] || $PY -m dprobe.cli import_axis "$M" || { echo "!! import_axis failed"; FAILED=1; }
echo "================ $M  phase 3a steering  $(date) ================"
$PY -m dprobe.cli steer_calibrated "$M" --labels "$LABELS" --multipliers "${MULTIPLIERS:-1,2,4,8,16}" --rollouts "${ROLLOUTS:-16}" --max_tokens 1024 \
   --backend hf --batch "${STEER_BATCH:-8}" --combos "$COMBOS" --combo_scale "${COMBO_SCALE:-0.5,1}" || { echo "!! steering FAILED"; FAILED=1; }
mkdir -p "$DPROBE_RESULTS/steer/$M" && cp /workspace/*.log "$DPROBE_RESULTS/steer/$M/" 2>/dev/null || true
$PY -m dprobe.cli sync_up --subsets spiral,steer,vectors --models "$M" || echo "!! sync_up failed"
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
