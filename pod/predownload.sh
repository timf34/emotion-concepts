#!/usr/bin/env bash
# Pre-download model weights to the local HF cache so the first model load is fast. MODELS = HF ids.
set -euo pipefail
# rp run exports HF_HOME=/workspace/hf (a FUSE mount with a 50GB quota when no network volume): override unconditionally.
export HF_HOME=${DPROBE_HF_HOME:-/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=1
PY=${PY:-/venv/bin/python}
for m in ${MODELS:-google/gemma-3-27b-it}; do
  $PY - "$m" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
# gpt-oss repos also carry original/ and metal/ copies of the weights (another 2 x 65 GB for 120b): skip them
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
                      ignore_patterns=["original/*", "metal/*"])
print("weights ready:", sys.argv[1], "->", p)
PYEOF
done
