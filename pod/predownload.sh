#!/usr/bin/env bash
# Pre-download model weights to the local HF cache so the first model load is fast. MODELS = HF ids.
set -euo pipefail
export HF_HOME=${HF_HOME:-/hf_cache} HF_HUB_ENABLE_HF_TRANSFER=1
PY=${PY:-/venv/bin/python}
for m in ${MODELS:-google/gemma-3-27b-it}; do
  $PY - "$m" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print("weights ready:", sys.argv[1], "->", p)
PYEOF
done
