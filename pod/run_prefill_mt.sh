#!/usr/bin/env bash
# Multi-turn prefill on a pod (see `dprobe.cli pod_prefill_mt`): Gemma 3 high-frustration turns 1-4, model writes 5-8.
#   MODEL=gemma4_31b SMOKE=1 bash pod/run_prefill_mt.sh    # 2 conversations, 64 tokens: validates the path
#   MODEL=gemma3_27b_bct bash pod/run_prefill_mt.sh        # 48 conversations
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache}
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
# long spiralling transcripts at batch 16 filled the H200; smaller batches + expandable segments avoid OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
MODEL=${MODEL:?set MODEL}
# Batch sizes that fit an H200 (141GB) at 1024 tokens/reply: Gemma 3 variants OOM'd at 16 with ~7k-token contexts;
# Gemma 4 31B OOM'd at 8 with the ~11k-token prefill contexts. Override with BATCH=... only with a reason.
case "$MODEL" in gemma4_*) BATCH=${BATCH:-4} ;; *) BATCH=${BATCH:-8} ;; esac
SMOKE=${SMOKE:-0}
N=${N:-48}
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
  $PY -m dprobe.cli pod_prefill_mt "$MODEL" --batch "$BATCH" --smoke || FAILED=1
else
  $PY -m dprobe.cli pod_prefill_mt "$MODEL" --n "$N" --batch "$BATCH" || FAILED=1
fi
# on failure, push whatever finished (the eval resumes from saved conversations) before the pod stops
if [ "$FAILED" = "1" ]; then
  $PY -m dprobe.cli sync_up --subsets spiral,probe --models "$MODEL" || true
fi
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
