#!/bin/bash

# dataDa launcher with startup checks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[warn] .venv not found. Using system Python."
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"

if [ ! -f "data/haikugraph.db" ] && [ -z "${HG_DB_PATH:-}" ]; then
  echo "[info] No default DB found at data/haikugraph.db."
  echo "[info] Ingest first:"
  echo "       PYTHONPATH=src python -m haikugraph.cli ingest --data-dir ./data --db-path ./data/haikugraph.db --force"
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -i :8000 >/dev/null 2>&1; then
    echo "[warn] Port 8000 already in use. Stop existing process or choose another port."
  fi
fi

RELOAD_FLAG="--reload"
if [ "${HG_DISABLE_RELOAD:-0}" = "1" ]; then
  RELOAD_FLAG=""
fi

echo "Launching dataDa..."
echo "Web UI:   http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo "Health:   http://localhost:8000/api/assistant/health"

exec uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 ${RELOAD_FLAG}
