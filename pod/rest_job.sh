#!/usr/bin/env bash
# Unattended vectors job for a pod created by pod/launch_rest.py (no `rp` needed): build the venv, download the
# weights, run a smoke pass, and only if it passes the full run; then stop the pod. The log is pushed to the results
# dataset every few minutes (logs/<POD_NAME>/job.log) so progress can be followed without ssh.
#   env: MODEL (registry key), STAGES (default extract,axis), POD_NAME, HF_TOKEN, RUNPOD_API_KEY, SKIP_SMOKE=1 optional
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:?set MODEL}
STAGES=${STAGES:-extract,axis}
POD_NAME=${POD_NAME:-dprobe-$MODEL}
export DPROBE_RESULTS=${DPROBE_RESULTS:-/results}
mkdir -p "$DPROBE_RESULTS/logs"
LOG="$DPROBE_RESULTS/logs/job.log"
exec > >(tee -a "$LOG") 2>&1

push_log() {
  [ -x /venv/bin/python ] || return 0
  /venv/bin/python - "$LOG" "$POD_NAME" <<'PYEOF' >/dev/null 2>&1 || true
import os, sys
from huggingface_hub import HfApi
from dprobe.config import HF_RESULTS_REPO
HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=sys.argv[1], path_in_repo=f"logs/{sys.argv[2]}/job.log",
    repo_id=HF_RESULTS_REPO, repo_type="dataset", commit_message=f"log {sys.argv[2]}")
PYEOF
}
( while true; do sleep 240; push_log; done ) &
LOGGER=$!

echo "== $POD_NAME: MODEL=$MODEL STAGES=$STAGES branch=$(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
STATUS=failed
if bash pod/fast_venv.sh; then
  HF=$(/venv/bin/python -c "import sys; from dprobe.config import get_model; print(get_model(sys.argv[1]).hf_id)" "$MODEL")
  push_log
  if MODELS="$HF" bash pod/predownload.sh; then
    SMOKE_OK=1
    if [ "${SKIP_SMOKE:-0}" != "1" ]; then
      echo "================ smoke $(date) ================"
      SHUTDOWN= MODEL="$MODEL" STAGES="$STAGES" SMOKE=1 bash pod/run_organisms.sh 2>&1 | tee "$DPROBE_RESULTS/logs/smoke.log"
      grep -q "ALL DONE .*failed=0" "$DPROBE_RESULTS/logs/smoke.log" || SMOKE_OK=0
      push_log
    fi
    if [ "$SMOKE_OK" = "1" ]; then
      echo "================ full run $(date) ================"
      SHUTDOWN= MODEL="$MODEL" STAGES="$STAGES" bash pod/run_organisms.sh 2>&1 | tee "$DPROBE_RESULTS/logs/full.log"
      grep -q "ALL DONE .*failed=0" "$DPROBE_RESULTS/logs/full.log" && STATUS=ok
    else
      echo "!! smoke failed; skipping the full run"
    fi
  else
    echo "!! weight download failed"
  fi
else
  echo "!! venv build failed"
fi
echo "JOB FINISHED status=$STATUS $(date)"
kill $LOGGER 2>/dev/null; push_log
SHUTDOWN=${SHUTDOWN:-stop} bash pod/self_stop.sh
