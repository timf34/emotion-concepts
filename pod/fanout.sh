#!/usr/bin/env bash
# Launch one pod per model for Phase 1 (run only after pod/run_phase1.sh with SMOKE=1 has passed on one pod).
# Requires: rp (runpod-runner) on PATH, RUNPOD_API_KEY, a filled .env in this folder.
#   bash pod/fanout.sh                    # gemma3_27b gemma3_27b_pt gemma4_31b, one H200 each
#   GPU=h100 MODELS="gemma4_31b" bash pod/fanout.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"gemma3_27b gemma3_27b_pt gemma4_31b"}
GPU=${GPU:-h200}
REPO=${REPO:-https://github.com/timf34/emotion-concepts.git}
VOLUME=${VOLUME:-none}
for m in $MODELS; do
  name="dprobe-$m"
  GPUS="$GPU h100 a100" VOLUME="$VOLUME" bash pod/up.sh "$name" || { echo "skip $m: no GPU"; continue; }
  # bootstrap without --req: it would install into /workspace (often a slow FUSE mount). fast_venv.sh builds /venv on local disk.
  rp bootstrap "$name" --repo "$REPO" --env .env
  case "$m" in
    gemma3_27b)    hf=google/gemma-3-27b-it ;;
    gemma3_27b_pt) hf=google/gemma-3-27b-pt ;;
    gemma4_31b)    hf=google/gemma-4-31B-it ;;
    gemma3_12b)    hf=google/gemma-3-12b-it ;;
    *) echo "unknown model key $m"; continue ;;
  esac
  rp run "$name" --job phase1 --dotenv --env MODELS="$m" --env SHUTDOWN=stop -- "bash pod/fast_venv.sh && MODELS=$hf bash pod/predownload.sh && bash pod/run_phase1.sh"
  echo "launched $name  (rp logs $name --job phase1 -f)"
done
echo "When a pod prints ALL DONE: rp down <name>. Then locally: uv run python -m dprobe.cli sync_down --subsets vectors,probe,selfother && uv run python -m dprobe.cli analyze <model>"
