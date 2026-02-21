#!/bin/bash

# dataDa unified launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"

echo "Launching dataDa..."
echo "Web UI:   http://localhost:8000"
echo "API docs: http://localhost:8000/docs"

exec uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 --reload
