#!/usr/bin/env bash
# Two pods: Gemma 4 steering (phase 3a) and the prefill-recovery experiment (phase 3b).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-https://github.com/timf34/emotion-concepts.git}
launch() {  # name job script predownload-models
  GPUS="h200" VOLUME=none bash pod/up.sh "$1" || { echo "skip $1: no GPU"; return; }
  rp bootstrap "$1" --repo "$REPO" --env .env
  rp run "$1" --job "$2" --dotenv --env SHUTDOWN=stop --env MODELS="${4:-gemma4_31b}" -- "bash pod/fast_venv.sh && MODELS='$3' bash pod/predownload.sh && bash pod/$2.sh"
  echo "launched $1 (rp logs $1 --job $2 -f)"
}
[ "${SKIP_STEER:-0}" = "1" ] || launch dprobe-p3-steer run_phase3_steer google/gemma-4-31B-it gemma4_31b
[ "${SKIP_PREFILL:-0}" = "1" ] || launch dprobe-p3-prefill run_phase3_prefill google/gemma-4-31B-it gemma4_31b
