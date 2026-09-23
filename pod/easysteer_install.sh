#!/usr/bin/env bash
# EasySteer (ZJU-REAL) = official vLLM 0.29.0 wheel + overlay of the fork's files, in its own venv on LOCAL disk
# (never /workspace: often a slow FUSE mount). Exits non-zero if the import test fails.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${EASYSTEER_VENV:-/easysteer_venv}
SRC=${EASYSTEER_SRC:-/easysteer_src}
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install vllm==0.29.0
[ -d "$SRC/.git" ] || git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git "$SRC"
VLLM_DIR=$("$VENV/bin/python" -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a "$SRC/vllm-steer/vllm/" "$VLLM_DIR"/
"$VENV/bin/pip" install "$SRC"
"$VENV/bin/pip" install -q fire python-dotenv pyyaml pandas gguf openai scikit-learn scipy tqdm matplotlib seaborn huggingface_hub
"$VENV/bin/pip" install -q -e . --no-deps
"$VENV/bin/python" - <<'PY'
from vllm import LLM
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
import dprobe.steer
print("EasySteer import OK")
PY
