# HaikuGraph POC - Quick Start Guide

## Setup (One-Time)

### 1. Install Python Dependencies
```bash
source .venv/bin/activate
pip install -e .
```

### 2. Prepare Database (if not already done)
```bash
# Ingest your Excel files
haikugraph ingest

# Profile the data
haikugraph profile

# Generate data cards
haikugraph cards build

# Build relationship graph
haikugraph graph build
```

### 3. Install Web Dependencies (First Time Only)
```bash
cd web
npm install
cd ..
```

## Running the POC

### Option 1: Full Stack (Recommended)
```bash
./start_ui.sh
```
Then open http://localhost:5173 in your browser.

### Option 2: Backend Only
```bash
./start_server.sh
```
API docs at http://localhost:8000/docs

### Option 3: Manual Control
Terminal 1 (Backend):
```bash
source .venv/bin/activate
uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (Frontend):
```bash
cd web
npm run dev
```

## Demo Questions

Try these questions in the UI:

1. "What is total revenue?" - Single metric
2. "Show me revenue by customer" - Grouped metric with bar chart
3. "Compare revenue this month vs last month" - Comparison with delta
4. "List recent transactions" - Data table
5. "How many unique customers?" - Count metric
6. "Show payments by date" - Time series

## Features to Demo

✨ **Answer View**: Plain English explanation with metadata  
📊 **Visualizations**: Auto-generated charts, tables, number cards  
🔄 **Comparisons**: Side-by-side with delta and percentage  
🔍 **Transparency**: Intent, plan, SQL, and timing in collapsible tabs  
⚠️ **Error Handling**: User-friendly messages with retry hints  

## Architecture

```
User Question
    ↓
FastAPI Backend (/ask endpoint)
    ↓
Intent Classification (A8) [non-fatal if fails]
    ↓
Plan Generation (A6-A7) with Ollama
    ↓
SQL Execution (A4-A5) against DuckDB
    ↓
Narration (A9) with Ollama
    ↓
Visualization Hints Added
    ↓
JSON Response → React UI
    ↓
Rendered Answer + Charts + Explainability
```

## What Makes This POC Powerful

1. **Same Pipeline as CLI**: UI and CLI share identical execution path
2. **Zero Logic Changes**: No modifications to existing validators or pipeline
3. **Visualization Hints**: Backend suggests display types, UI decides rendering
4. **A11 Compliant**: Comparisons use normalized structure (no UI math)
5. **Full Transparency**: Users can inspect intent, plan, SQL, and results
6. **Demo Ready**: Clean UI with example questions built in

## Key Files

- `src/haikugraph/api/server.py` - Backend API
- `web/src/App.jsx` - Main UI component
- `web/src/components/` - Visualization components
- `start_ui.sh` - One-command startup
- `WEB_UI_README.md` - Full documentation

## Troubleshooting

**"Database not found"**  
→ Run `haikugraph ingest` first

**"Module not found: fastapi"**  
→ Run `pip install -e .` in activated venv

**"Cannot find module 'react'"**  
→ Run `npm install` in web/ directory

**Port already in use**  
→ Stop other services or change ports in config

## Next Steps

1. Try all demo questions
2. Ask your own questions
3. Check explainability tabs
4. Review API docs at /docs
5. Modify demo questions in App.jsx
6. Add new visualizations in VisualizationView.jsx

Enjoy the POC! 🚀
