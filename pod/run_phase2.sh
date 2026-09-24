#!/usr/bin/env bash
# Phase 2 (steering) on a fresh pod for ONE model. Pulls Phase 1 vectors from HF, validates the HF-hooks backend on a
# tiny grid, tries EasySteer for throughput, runs the largest grid that works, judges, uploads, stops the pod.
#   MODELS=gemma3_27b SHUTDOWN=stop bash pod/run_phase2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
M=${MODELS:-gemma3_27b}
PY=${PY:-/venv/bin/python}; [ -x "$PY" ] || PY=python
FAILED=0

case "$M" in
  gemma3_27b)  LABELS="depressed,clinical_depression,worthless,sad,frustrated,calm";  LABELS_SMALL="depressed,clinical_depression,calm" ;;
  gemma4_31b)  LABELS="depressed,clinical_depression,worthless,frustrated";           LABELS_SMALL="depressed,clinical_depression" ;;
  *)           LABELS="depressed,calm"; LABELS_SMALL="depressed" ;;
esac

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m dprobe.cli sync_down --subsets vectors --models "$M" || { echo "!! sync_down failed"; FAILED=1; }

echo "================ $M  steering smoke (HF hooks)  $(date) ================"
$PY -m dprobe.cli steer "$M" --labels depressed --strengths=1 --rollouts 2 --max_tokens 200 --judge False --backend hf --batch "${STEER_BATCH:-8}" \
  || { echo "!! HF steering smoke FAILED"; FAILED=1; }

BACKEND=hf
if [ "$FAILED" = "0" ] && [ "${TRY_EASYSTEER:-1}" = "1" ]; then
  echo "================ EasySteer install  $(date) ================"
  if bash pod/easysteer_install.sh > /results/easysteer_install.log 2>&1; then
    echo "EasySteer installed; smoke test"
    if EASY_PY=/easysteer_venv/bin/python; DPROBE_RESULTS=$DPROBE_RESULTS $EASY_PY -m dprobe.cli steer "$M" --labels depressed --strengths=1 --rollouts 2 --max_tokens 200 --judge False --backend easysteer; then
      BACKEND=easysteer; PY=$EASY_PY
    else
      echo "!! EasySteer smoke failed; falling back to HF hooks"
    fi
  else
    echo "!! EasySteer install failed (see /results/easysteer_install.log); falling back to HF hooks"
  fi
fi

echo "================ $M  per-label calibration + grid  backend=$BACKEND  $(date) ================"
SIGNS=both; [ "$M" = "gemma4_31b" ] && SIGNS=pos
if [ "$BACKEND" = "easysteer" ]; then
  $PY -m dprobe.cli steer_calibrated "$M" --labels "$LABELS" --multipliers "${MULTIPLIERS:-1,2,4,8}" --rollouts "${ROLLOUTS:-40}" --max_tokens 2048 --backend easysteer --signs $SIGNS \
    || { echo "!! grid FAILED"; FAILED=1; }
else
  $PY -m dprobe.cli steer_calibrated "$M" --labels "$LABELS_SMALL" --multipliers "${MULTIPLIERS:-1,2,4,8}" --rollouts "${ROLLOUTS_SMALL:-16}" --max_tokens 1024 --backend hf --batch "${STEER_BATCH:-8}" --signs $SIGNS \
    || { echo "!! grid FAILED"; FAILED=1; }
fi

mkdir -p "$DPROBE_RESULTS/steer/$M" && cp /workspace/phase2.log "$DPROBE_RESULTS/steer/$M/phase2.log" 2>/dev/null || cp /workspace/*.log "$DPROBE_RESULTS/steer/$M/" 2>/dev/null || true   # keep the log with the results: stopped pods cannot be read
$PY -m dprobe.cli sync_up --subsets spiral,steer --models "$M" || echo "!! sync_up failed"
echo "ALL DONE $(date) failed=$FAILED backend=$BACKEND"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
