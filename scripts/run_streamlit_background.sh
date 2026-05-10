#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f streamlit.pid ]; then
  kill "$(cat streamlit.pid)" 2>/dev/null || true
fi

nohup scripts/start_streamlit.sh > streamlit.log 2>&1 < /dev/null &
echo "$!" > streamlit.pid
echo "Streamlit started with PID $(cat streamlit.pid). Logs: $(pwd)/streamlit.log"
