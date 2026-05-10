#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/streamlit ]; then
  echo "Missing .venv. Run scripts/setup_l4.sh first." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/home/.cache/huggingface}"
mkdir -p "$HF_HOME"

source .venv/bin/activate
exec streamlit run app/streamlit_app.py \
  --server.address "${STREAMLIT_ADDRESS:-0.0.0.0}" \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.headless true
