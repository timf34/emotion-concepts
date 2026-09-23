#!/usr/bin/env bash
# Build the Python env on the pod's LOCAL container disk (/venv). RunPod's /workspace is often a FUSE network
# filesystem (slow for thousands of small files), and rp bootstrap puts its venv there. Reuses the pip cache.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/venv}
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -r pod/requirements-pod.txt
"$VENV/bin/pip" install -e . --no-deps
"$VENV/bin/python" -c "import torch, transformers; print('venv ok: torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
