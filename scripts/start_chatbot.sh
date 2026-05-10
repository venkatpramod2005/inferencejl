#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Create the environment and install dependencies before starting."
  exit 1
fi

set -a
if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  source ".env"
fi
set +a

mkdir -p logs
if [ -f "streamlit.pid" ] && kill -0 "$(cat streamlit.pid)" 2>/dev/null; then
  echo "Streamlit is already running as PID $(cat streamlit.pid)."
  exit 0
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"
nohup streamlit run app/streamlit_app.py \
  --server.address "${STREAMLIT_ADDRESS:-0.0.0.0}" \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.headless true \
  > logs/streamlit.log 2>&1 < /dev/null &

echo "$!" > streamlit.pid
echo "Streamlit started as PID $(cat streamlit.pid)."
echo "Log: $REPO_DIR/logs/streamlit.log"

