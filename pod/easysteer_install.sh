#!/usr/bin/env bash
# EasySteer (ZJU-REAL) = official vLLM 0.29.0 wheel + overlay of the fork's files. Separate venv so it
# cannot fight the extraction env's torch/transformers pins.
set -euo pipefail
VENV=${VENV:-/workspace/easysteer_venv}
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip
pip install vllm==0.29.0
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git /workspace/EasySteer || true
cd /workspace/EasySteer
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a vllm-steer/vllm/ "$VLLM_DIR"/
pip install .
pip install fire python-dotenv pyyaml pandas gguf openai
cd - >/dev/null
pip install -e . --no-deps
python - <<'PY'
from vllm import LLM
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
print("EasySteer import OK")
PY
echo "Use: source $VENV/bin/activate && python -m dprobe.cli steer gemma3_27b --backend easysteer ..."
