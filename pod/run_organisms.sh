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
# long spiralling transcripts at batch 16 filled the H200; smaller batches + expandable segments avoid OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BATCH=${BATCH:-8}
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
# vectors already on HF (the control's Gemma 3 vectors, or stages finished by an earlier pod) are reused, not redone
$PY - "$MODEL" <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
from dprobe.config import HF_RESULTS_REPO, RESULTS_DIR
snapshot_download(HF_RESULTS_REPO, repo_type="dataset", local_dir=str(RESULTS_DIR), token=os.environ.get("HF_TOKEN"),
                  allow_patterns=[f"vectors/{sys.argv[1]}/*", f"vectors/{sys.argv[1]}/axis_reencoded/*"])
PYEOF
$PY -m dprobe.cli check_template "$MODEL"
if [ "$SMOKE" = "1" ]; then
  $PY -m dprobe.cli pod_organism "$MODEL" --stages "$STAGES" --batch "$BATCH" --smoke || FAILED=1
else
  $PY -m dprobe.cli pod_organism "$MODEL" --stages "$STAGES" --batch "$BATCH" || FAILED=1
fi
# on failure, push whatever finished (the eval resumes from saved conversations) before the pod stops
if [ "$FAILED" = "1" ]; then
  $PY -m dprobe.cli sync_up --subsets spiral,probe,vectors --models "$MODEL" || true
fi
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
