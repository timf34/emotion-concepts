#!/usr/bin/env bash
# Coordinate-2733 ablation: vectors + rejection-eval cells (see dprobe/ablate2733.py).
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMOKE=${SMOKE:-0}; STAGES=${STAGES:-vectors,cells}; CELLS=${CELLS:-ctrl,1.0,0.5,0.0}
PY=${PY:-/venv/bin/python}; [ -x "$PY" ] || PY=python
FAILED=0
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -m dprobe.cli sync_down --subsets stories --models gemma3_27b
$PY - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
from dprobe.config import HF_RESULTS_REPO, RESULTS_DIR
snapshot_download(HF_RESULTS_REPO, repo_type="dataset", local_dir=str(RESULTS_DIR), token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["spiral/gemma3_27b/extended/*", "vectors/gemma3_27b/vectors_external_*.pt", "vectors/gemma3_27b/neutral_pca.pt",
                                  "vectors/gemma3_27b/vectors_emotions_dn.pt", "vectors/gemma3_27b/external_meta.json"])
PYEOF
if [ "$SMOKE" = "1" ]; then
  $PY -m dprobe.cli pod_ablate2733 --stages "$STAGES" --cells "$CELLS" --smoke || FAILED=1
else
  $PY -m dprobe.cli pod_ablate2733 --stages "$STAGES" --cells "$CELLS" || FAILED=1
fi
[ "$FAILED" = "1" ] && { $PY -m dprobe.cli sync_up --subsets ablate2733,spiral,vectors --models gemma3_27b || true; }
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
