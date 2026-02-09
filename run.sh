#!/bin/bash

# HaikuGraph Full Application Startup Script
set -e

echo "🚀 Starting HaikuGraph Application..."
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

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

# Cleanup function to kill both processes
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    # Kill any remaining processes on the ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    echo "✅ Cleanup complete"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start backend server
echo "🌐 Starting backend server at http://localhost:8000..."
uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 > /tmp/haikugraph-backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check /tmp/haikugraph-backend.log for errors."
    exit 1
fi

# Start frontend dev server
echo "🎨 Starting frontend at http://localhost:5173..."
cd web
npm run dev > /tmp/haikugraph-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start. Check /tmp/haikugraph-frontend.log for errors."
    exit 1
fi

echo ""
echo "✨ HaikuGraph is ready!"
echo ""
echo "   🌐 Frontend: http://localhost:5173"
echo "   🔧 Backend:  http://localhost:8000"
echo "   📚 API Docs: http://localhost:8000/docs"
echo ""
echo "   📋 Logs:"
echo "      Backend:  /tmp/haikugraph-backend.log"
echo "      Frontend: /tmp/haikugraph-frontend.log"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for processes (this keeps the script running)
wait
