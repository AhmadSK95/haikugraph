#!/bin/bash

# HaikuGraph Backend Server Startup Script

echo "🚀 Starting HaikuGraph Backend Server..."
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if database exists
if [ ! -f "./data/haikugraph.duckdb" ]; then
    echo "⚠️  Warning: Database not found at ./data/haikugraph.duckdb"
    echo "   Run 'haikugraph ingest' first to set up the database."
    echo ""
fi

# Start the server
echo "🌐 Server will be available at: http://localhost:8000"
echo "📊 API docs available at: http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"
uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 --reload
