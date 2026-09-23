#!/usr/bin/env bash
# Steering grid on the spiral eval. Default backend = HF hooks (always works). Set BACKEND=easysteer after
# pod/easysteer_install.sh succeeded and `source /workspace/easysteer_venv/bin/activate`.
set -euo pipefail
cd "$(dirname "$0")/.."
export DPROBE_RESULTS=${DPROBE_RESULTS:-/workspace/dprobe_results} PYTHONUNBUFFERED=1
if [ -z "${HF_HOME:-}" ]; then
  if [ "$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc 0-9)" -ge 100 ] 2>/dev/null; then HF_HOME=/workspace/hf; else HF_HOME=/hf_cache; fi
fi
export HF_HOME
MODEL=${MODEL:-gemma3_27b}
LABELS=${LABELS:-"depressed,clinical_depression,worthless,sad,frustrated,calm"}
STRENGTHS=${STRENGTHS:-"-0.08,-0.04,0.04,0.08"}
BACKEND=${BACKEND:-hf}
ROLLOUTS=${ROLLOUTS:-40}
python -m dprobe.cli steer "$MODEL" --labels "$LABELS" --strengths="$STRENGTHS" --backend "$BACKEND" --rollouts "$ROLLOUTS"
