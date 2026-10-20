#!/usr/bin/env bash
# Emotion + syndrome vectors only (extract stage) for extra models, one pod each. Every model reads Gemma 3's stories.
#   SMOKE=1 MODELS="qwen36_27b" bash pod/fanout_vectors.sh   # ~15-min validation, pod left running for inspection
#   MODELS="qwen36_27b qwen38_27b" bash pod/fanout_vectors.sh
# Full runs stop their pod when the job exits, success or not.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:?set MODELS}
GPU=${GPU:-h200}
REPO=${REPO:-https://github.com/timf34/emotion-concepts.git}
BRANCH=${BRANCH:-crossfamily}
SMOKE=${SMOKE:-0}
RP=${RP:-rp}
for m in $MODELS; do
  hf=$(uv run --quiet python -c "import sys; from dprobe.config import get_model; print(get_model(sys.argv[1]).hf_id)" "$m")
  name="dprobe-vec-${m//_/-}"
  # Qwen3.5-architecture models: Gated-DeltaNet layers fall back to slow torch kernels without flash-linear-attention
  extra=""; case "$m" in qwen36_*|qwen38_*) extra="(/venv/bin/pip install -q flash-linear-attention || true) && " ;; esac
  # reuse the pod if it already exists (e.g. the smoke pod), else create one
  $RP status "$name" >/dev/null 2>&1 || { GPUS="$GPU h100" VOLUME=none DISK=${DISK:-200} RP=$RP bash pod/up.sh "$name" || { echo "skip $m: no GPU"; continue; }; }
  $RP bootstrap "$name" --repo "$REPO" --branch "$BRANCH" --env .env --no-req
  if [ "$SMOKE" = "1" ]; then
    $RP run "$name" --job smoke --dotenv --env MODEL="$m" --env STAGES=extract --env SMOKE=1 -- \
      "bash pod/fast_venv.sh && ${extra}MODELS='$hf' bash pod/predownload.sh && bash pod/run_organisms.sh"
    echo "launched smoke on $name  ($RP logs $name --job smoke -f)"
  else
    # trailing self_stop runs even if the venv/download fails before run_organisms.sh starts
    $RP run "$name" --job vec --dotenv --env MODEL="$m" --env STAGES=extract -- \
      "bash pod/fast_venv.sh && ${extra}MODELS='$hf' bash pod/predownload.sh && bash pod/run_organisms.sh; SHUTDOWN=stop bash pod/self_stop.sh"
    echo "launched $name  ($RP logs $name --job vec -f)"
  fi
done
