#!/usr/bin/env bash
# Three pods for the Phase 4 follow-ups (see pod/run_phase4.sh). Each self-stops when done; results land on HF.
#   bash pod/fanout_phase4.sh            # all three
#   SKIP_G4=1 bash pod/fanout_phase4.sh  # skip one
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-https://github.com/timf34/emotion-concepts.git}
launch() {  # name job hf-model-to-predownload model-key
  GPUS="${GPUS:-h200 h100}" VOLUME=none bash pod/up.sh "$1" || { echo "skip $1: no GPU"; return; }
  rp bootstrap "$1" --repo "$REPO" --env .env
  rp run "$1" --job run_phase4 --dotenv --env SHUTDOWN=stop --env JOB="$2" --env MODELS="$4" -- "bash pod/fast_venv.sh && MODELS='$3' bash pod/predownload.sh && JOB=$2 bash pod/run_phase4.sh"
  echo "launched $1 (rp logs $1 --job run_phase4 -f)"
}
[ "${SKIP_G4:-0}" = "1" ] || launch dprobe-p4-g4 g4_2x2 google/gemma-4-31B-it gemma4_31b
[ "${SKIP_G3FAM:-0}" = "1" ] || launch dprobe-p4-g3fam g3_family google/gemma-3-27b-it gemma3_27b
[ "${SKIP_G3EARLY:-0}" = "1" ] || launch dprobe-p4-g3early g3_early google/gemma-3-27b-it gemma3_27b
