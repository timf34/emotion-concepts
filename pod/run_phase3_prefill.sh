#!/usr/bin/env bash
# Phase 3b: prefill-recovery trajectories. Gemma 4 (unsteered, -calm, -assistant_axis) and Gemma 3 (unsteered control)
# continue Gemma 3 spiral prefixes; continuation judged and probed token by token.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} PYTHONUNBUFFERED=1 DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
PY=${PY:-/venv/bin/python}; [ -x "$PY" ] || PY=python; FAILED=0
N=${N:-32}; TURN=${TURN:-6}
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m dprobe.cli sync_down --subsets vectors,spiral --models gemma3_27b,gemma4_31b || { echo "!! sync_down failed"; FAILED=1; }
for m in gemma4_31b gemma3_27b; do
  [ -f "$DPROBE_RESULTS/vectors/$m/vectors_external_dn.pt" ] || $PY -m dprobe.cli import_axis "$m" || echo "!! import_axis $m failed"
done
echo "================ gemma4_31b prefill  $(date) ================"
MODELS=google/gemma-4-31B-it bash pod/predownload.sh
$PY -m dprobe.cli prefill gemma4_31b --n $N --turn $TURN --batch "${PREFILL_BATCH:-4}" || { echo "!! gemma4 unsteered FAILED"; FAILED=1; }
# calibrate -calm and -assistant_axis on gemma4 (grid length), then steered continuations at the chosen multipliers
CAL_CALM=$($PY -m dprobe.cli calibrate gemma4_31b --label calm --multipliers=-1,-2,-4,-8 --max_tokens 1024 2>&1 | tee -a /results/cal_calm.log | sed -n 's/^CHOSEN_MULTIPLIER=//p' | tail -1)
CAL_AXIS=$($PY -m dprobe.cli calibrate gemma4_31b --label assistant_axis --multipliers=-1,-2,-4,-8,-16 --max_tokens 1024 2>&1 | tee -a /results/cal_axis.log | sed -n 's/^CHOSEN_MULTIPLIER=//p' | tail -1)
echo "calibrated: calm=$CAL_CALM assistant_axis=$CAL_AXIS"
[ -n "$CAL_CALM" ] && [ "$CAL_CALM" != "0.0" ] && { $PY -m dprobe.cli prefill gemma4_31b --n $N --turn $TURN --batch "${PREFILL_BATCH:-4}" --steer "calm:$CAL_CALM" || FAILED=1; }
[ -n "$CAL_AXIS" ] && [ "$CAL_AXIS" != "0.0" ] && { $PY -m dprobe.cli prefill gemma4_31b --n $N --turn $TURN --batch "${PREFILL_BATCH:-4}" --steer "assistant_axis:$CAL_AXIS" || FAILED=1; }
if [ "${SKIP_CONTROL:-0}" != "1" ]; then
  echo "================ gemma3_27b prefill (control)  $(date) ================"
  MODELS=google/gemma-3-27b-it bash pod/predownload.sh
  $PY -m dprobe.cli prefill gemma3_27b --n $N --turn $TURN --batch "${PREFILL_BATCH:-4}" || { echo "!! gemma3 control FAILED"; FAILED=1; }
fi
mkdir -p "$DPROBE_RESULTS/prefill" && cp /workspace/*.log "$DPROBE_RESULTS/prefill/" 2>/dev/null || true
$PY -m dprobe.cli sync_up --subsets prefill,steer,vectors || echo "!! sync_up failed"
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
