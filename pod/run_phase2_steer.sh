#!/usr/bin/env bash
# Steering grid on the spiral eval. Default backend = HF hooks (always works). Set BACKEND=easysteer after
# pod/easysteer_install.sh succeeded and `source /workspace/easysteer_venv/bin/activate`.
set -euo pipefail
cd "$(dirname "$0")/.."
export DPROBE_RESULTS=${DPROBE_RESULTS:-/results} PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/hf_cache}
MODEL=${MODEL:-gemma3_27b}
LABELS=${LABELS:-"depressed,clinical_depression,worthless,sad,frustrated,calm"}
STRENGTHS=${STRENGTHS:-"-0.08,-0.04,0.04,0.08"}
BACKEND=${BACKEND:-hf}
ROLLOUTS=${ROLLOUTS:-40}
python -m dprobe.cli steer "$MODEL" --labels "$LABELS" --strengths="$STRENGTHS" --backend "$BACKEND" --rollouts "$ROLLOUTS"
