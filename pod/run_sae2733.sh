#!/usr/bin/env bash
# Gemma 3 coordinate-2733 job (see dprobe/sae2733.py).   SMOKE=1 bash pod/run_sae2733.sh  |  bash pod/run_sae2733.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMOKE=${SMOKE:-0}; STAGES=${STAGES:-collect,causal}
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
  $PY -m dprobe.cli pod_sae2733 --stages "$STAGES" --smoke || FAILED=1
else
  $PY -m dprobe.cli pod_sae2733 --stages "$STAGES" || FAILED=1
fi
[ "$FAILED" = "1" ] && { $PY -m dprobe.cli sync_up --subsets sae2733,spiral --models gemma3_27b || true; }
echo "ALL DONE $(date) failed=$FAILED"
SHUTDOWN=${SHUTDOWN:-} bash pod/self_stop.sh
