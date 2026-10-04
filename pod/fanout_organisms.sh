#!/usr/bin/env bash
# One pod per model organism (run only after a SMOKE=1 pass of pod/run_organisms.sh on one pod).
#   bash pod/fanout_organisms.sh                         # gemma3_27b_dpo gemma3_27b_sft gemma3_27b(control)
#   MODELS="gemma3_27b_sft" bash pod/fanout_organisms.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"gemma3_27b_dpo gemma3_27b_sft gemma3_27b"}
GPU=${GPU:-h200}
REPO=${REPO:-https://github.com/timf34/emotion-concepts.git}
BRANCH=${BRANCH:-organisms}
for m in $MODELS; do
  name="dprobe-org-${m#gemma3_27b_}"; [ "$m" = "gemma3_27b" ] && name="dprobe-org-control"
  GPUS="$GPU h100" VOLUME=none DISK=${DISK:-200} bash pod/up.sh "$name" || { echo "skip $m: no GPU"; continue; }
  rp bootstrap "$name" --repo "$REPO" --branch "$BRANCH" --env .env --no-req
  case "$m" in
    gemma3_27b_dpo) hf="google/gemma-3-27b-it annasoli/gemma3-27b-dpo-calm-full"; stages=extract,axis,spiral,probe,cross ;;
    gemma3_27b_sft) hf=annasoli/gemma3-27b-sft-diverse-calm-merged; stages=extract,axis,spiral,probe,cross ;;
    gemma3_27b)     hf=google/gemma-3-27b-it; stages=axis,spiral,probe ;;
    *) echo "unknown model key $m"; continue ;;
  esac
  rp run "$name" --job org --dotenv --env MODEL="$m" --env STAGES="$stages" --env SHUTDOWN=${SHUTDOWN:-stop} -- \
    "bash pod/fast_venv.sh && MODELS='$hf' bash pod/predownload.sh && bash pod/run_organisms.sh"
  echo "launched $name  (rp logs $name --job org -f)"
done
