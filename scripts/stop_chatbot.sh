#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f "streamlit.pid" ] && kill -0 "$(cat streamlit.pid)" 2>/dev/null; then
  kill "$(cat streamlit.pid)"
  echo "Stopped Streamlit PID $(cat streamlit.pid)."
  rm -f streamlit.pid
else
  pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null || true
  rm -f streamlit.pid
  echo "No running Streamlit PID file found."
fi

