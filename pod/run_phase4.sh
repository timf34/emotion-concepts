#!/usr/bin/env bash
# Phase 4 follow-up steering cells, one pod per JOB (launched by pod/fanout_phase4.sh):
#   JOB=g4_2x2     Gemma 4: calm x assistant-axis factorial at 34-44 (-4 calm, -2 calm, -1 axis alone; -2 calm + -2 axis)
#   JOB=g3_family  Gemma 3: +-2 hysterical / panicked / assistant axis at 34-46, each label calibrated too
#   JOB=g3_early   Gemma 3: calibrated +-calm, +-assistant axis, +-(axis minus its calm component) at layers 20-26,
#                  the band where the axis and calm are entangled (cos 0.2-0.5; ~0 at 34-46)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} PYTHONUNBUFFERED=1 DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
PY=${PY:-/venv/bin/python}; [ -x "$PY" ] || PY=python; FAILED=0
JOB=${JOB:?set JOB=g4_2x2|g3_family|g3_early}
case "$JOB" in
  g4_2x2) M=gemma4_31b ;;
  g3_family|g3_early) M=gemma3_27b ;;
  *) echo "unknown JOB $JOB"; exit 1 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m dprobe.cli sync_down --subsets vectors --models "$M" || { echo "!! sync_down failed"; FAILED=1; }
[ -f "$DPROBE_RESULTS/vectors/$M/vectors_external_dn.pt" ] || $PY -m dprobe.cli import_axis "$M" || { echo "!! import_axis failed"; FAILED=1; }
echo "================ $M  phase 4 $JOB  $(date) ================"
R=${ROLLOUTS:-16}; B=${STEER_BATCH:-8}
case "$JOB" in
  g4_2x2)
    $PY -m dprobe.cli steer_cells "$M" --cells "calm:-4,calm:-2,assistant_axis:-1" --combos "calm:-2+assistant_axis:-2" \
       --rollouts "$R" --max_tokens 1024 --batch "$B" || { echo "!! steer_cells FAILED"; FAILED=1; } ;;
  g3_family)
    $PY -m dprobe.cli steer_cells "$M" --cells "hysterical:-2,hysterical:2,panicked:-2,panicked:2,assistant_axis:-2,assistant_axis:2" \
       --calibrate "hysterical:+,panicked:+,assistant_axis:-" --rollouts "$R" --max_tokens 1024 --batch "$B" || { echo "!! steer_cells FAILED"; FAILED=1; } ;;
  g3_early)
    $PY -m dprobe.cli steer_calibrated "$M" --labels "calm,assistant_axis,assistant_axis_minus_calm" --multipliers "1,2,4,8" --layers "20,22,24,26" \
       --rollouts "$R" --max_tokens 1024 --backend hf --batch "$B" --petri True || { echo "!! steer_calibrated FAILED"; FAILED=1; } ;;
esac
mkdir -p "$DPROBE_RESULTS/steer/$M" && cp /workspace/*.log "$DPROBE_RESULTS/steer/$M/" 2>/dev/null || true
$PY -m dprobe.cli sync_up --subsets spiral,steer --models "$M" || echo "!! sync_up failed"
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
