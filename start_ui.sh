#!/bin/bash

# HaikuGraph Full Stack Startup Script

echo "🚀 Starting HaikuGraph POC..."
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

# Check if web dependencies are installed
if [ ! -d "./web/node_modules" ]; then
    echo "📦 Installing web dependencies..."
    cd web && npm install && cd ..
    echo ""
fi

# Start backend server in background with auto-reload
echo "🌐 Starting backend server at http://localhost:8000 (with auto-reload)..."
uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend dev server
echo "🎨 Starting frontend at http://localhost:5173..."
echo ""
echo "✨ HaikuGraph POC is ready!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

cd web && npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
