#!/usr/bin/env bash
# One steering pod per model. Requires Phase 1 vectors on HF (sync_up from the Phase 1 pods).
#   bash pod/fanout_phase2.sh                  # gemma3_27b gemma4_31b
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"gemma3_27b gemma4_31b"}
REPO=${REPO:-https://github.com/timf34/emotion-concepts.git}
for m in $MODELS; do
  name="dprobe-steer-$m"
  case "$m" in
    gemma3_27b) hf=google/gemma-3-27b-it ;; gemma3_27b_pt) hf=google/gemma-3-27b-pt ;; gemma4_31b) hf=google/gemma-4-31B-it ;; *) echo "unknown $m"; continue ;;
  esac
  # Gemma 4 31B (62 GB bf16) + batched 12k-token generation does not fit an 80 GB card: H200 only for it.
  gpus="h200 h100 a100"; [ "$m" = "gemma4_31b" ] && gpus="h200"
  GPUS="$gpus" VOLUME=none bash pod/up.sh "$name" || { echo "skip $m: no GPU"; continue; }
  rp bootstrap "$name" --repo "$REPO" --env .env
  rp run "$name" --job phase2 --dotenv --env MODELS="$m" --env SHUTDOWN=stop --env STEER_BATCH="${STEER_BATCH:-8}" -- "bash pod/fast_venv.sh && MODELS=$hf bash pod/predownload.sh && bash pod/run_phase2.sh"
  echo "launched $name  (rp logs $name --job phase2 -f)"
done
