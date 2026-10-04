#!/usr/bin/env bash
# Gemma Needs Help model organisms on a pod (see `dprobe.cli pod_organism`).
#   MODEL=gemma3_27b_dpo SMOKE=1 bash pod/run_organisms.sh     # ~15-minute validation of the whole path
#   MODEL=gemma3_27b_dpo bash pod/run_organisms.sh             # full: extract, axis, spiral, probe, cross
#   MODEL=gemma3_27b STAGES=axis,spiral,probe bash pod/run_organisms.sh   # plain-Gemma-3 control (vectors from HF)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache}
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
MODEL=${MODEL:?set MODEL}
SMOKE=${SMOKE:-0}
STAGES=${STAGES:-extract,axis,spiral,probe,cross}
PY=${PY:-/venv/bin/python}
[ -x "$PY" ] || PY=python
FAILED=0

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# inputs: Gemma 3's stories (every organism borrows them) and Gemma 3's own spiral transcripts (the cross stage)
$PY -m dprobe.cli sync_down --subsets stories --models gemma3_27b
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
from dprobe.config import HF_RESULTS_REPO, RESULTS_DIR
import os
snapshot_download(HF_RESULTS_REPO, repo_type="dataset", local_dir=str(RESULTS_DIR), token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["spiral/gemma3_27b/extended/*"])
PYEOF
# the control reuses Gemma 3's existing emotion vectors instead of re-extracting them
if [ "$MODEL" = "gemma3_27b" ]; then
  $PY -m dprobe.cli sync_down --subsets vectors --models gemma3_27b || true
fi
$PY -m dprobe.cli check_template "$MODEL"
if [ "$SMOKE" = "1" ]; then
  $PY -m dprobe.cli pod_organism "$MODEL" --stages "$STAGES" --smoke || FAILED=1
else
  $PY -m dprobe.cli pod_organism "$MODEL" --stages "$STAGES" || FAILED=1
fi
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
