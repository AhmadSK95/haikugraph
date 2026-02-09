# HaikuGraph Web UI - POC Documentation

A browser-based data assistant where users can ask natural language questions and receive:
- Clear final answers with explanations
- Rich visualizations (tables, charts, comparisons)
- Full transparency (intent, plan, SQL, results, metadata)

## Architecture

### Backend API
- **FastAPI** server wrapping the existing pipeline
- Endpoint: `POST /ask` with stable JSON contract
- Shares the same execution path as the CLI

### Frontend
- **React + Vite** for fast development
- **Recharts** for data visualization
- Minimal, clean, demo-ready UI

### Pipeline (Unchanged)
```
Question → Intent (A8) → Planner (A6-A7) → Executor (A4-A5) → Narrator (A9) → UI
```

## Quick Start

### Prerequisites
- Python 3.11+ with haikugraph installed
- Node.js 18+ and npm
- Database with data (run `haikugraph ingest` first)

### One-Command Startup

Start both backend and frontend:
```bash
./start_ui.sh
```

This will:
1. Activate Python virtual environment
2. Check for database
3. Install web dependencies (first run only)
4. Start backend at http://localhost:8000
5. Start frontend at http://localhost:5173

**Access the UI**: Open http://localhost:5173 in your browser

### Backend Only

To run just the API server:
```bash
./start_server.sh
```

API will be at http://localhost:8000  
API docs at http://localhost:8000/docs

### Manual Setup

If you prefer manual control:

**Backend**:
```bash
source .venv/bin/activate
uvicorn haikugraph.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend**:
```bash
cd web
npm install  # first time only
npm run dev
```

## Demo Questions

The UI includes 6 canonical demo questions that showcase different intent types:

1. **Metric**: "What is total revenue?"
   - Single aggregated value
   - Display: Large number card

2. **Grouped Metric**: "Show me revenue by customer"
   - Aggregation by dimension
   - Display: Bar chart or table

3. **Comparison**: "Compare revenue this month vs last month"
   - Temporal comparison with delta
   - Display: Comparison card with trend

4. **Lookup**: "List recent transactions"
   - Raw row listing
   - Display: Data table

5. **Metric (Count)**: "How many unique customers?"
   - Count aggregation
   - Display: Number card

6. **Time Series**: "Show payments by date"
   - Time-based grouping
   - Display: Line chart or table

## Features

### Answer View
- Final answer in plain English
- Execution metadata (time, row count, intent type)
- Auto-generated from narrator (A9)

### Comparison View
- Side-by-side period values
- Absolute delta and percentage change
- Visual direction indicator (↑ up, ↓ down, − flat)
- Only shown for comparison intent

### Visualizations
- **Number Display**: Single metric values with formatting
- **Bar Charts**: Grouped metrics (dimension + value)
- **Line Charts**: Time series data
- **Tables**: Multi-column or large result sets
- Automatic selection based on data shape

### Explainability (Collapsible)
- **Intent Tab**: Classification details (type, confidence, rationale)
- **Plan Tab**: Full plan JSON
- **SQL Tab**: Generated queries with syntax highlighting
- **Metadata Tab**: Execution timing, tables used, warnings

### Error Handling
- User-friendly error messages (no stack traces)
- Retry hints for common issues
- Graceful degradation (narration failures fall back to summary)

## API Contract

### Request
```json
POST /api/ask
{
  "question": "What is total revenue?"
}
```

### Response
```json
{
  "final_answer": "Total revenue is $45,678.",
  "intent": {
    "type": "metric",
    "confidence": 0.95,
    "rationale": "Single aggregation without grouping",
    "requires_comparison": false
  },
  "plan": { ... },
  "queries": ["SELECT SUM(amount) FROM transactions"],
  "results": [
    {
      "id": "sq_1",
      "sql": "...",
      "row_count": 1,
      "columns": ["total"],
      "preview_rows": [{"total": 45678}],
      "display_hint": "number",
      "chart_type": null,
      "units": "currency"
    }
  ],
  "comparison": null,
  "metadata": {
    "execution_time_ms": 234,
    "total_rows": 1,
    "tables_used": ["transactions"]
  },
  "warnings": [],
  "errors": []
}
```

## Visualization Hints

The backend augments results with rendering hints (non-binding):

- `display_hint`: `number` | `table` | `bar_chart` | `line_chart`
- `chart_type`: `bar` | `line` | null
- `x_axis`, `y_axis`: Column names for charts
- `units`: `currency` | `count` | `percentage` | null

The UI decides final rendering based on these hints.

## Constraints

- UI is a viewer, not a planner (read-only)
- All validations (A8-A11) remain unchanged
- No external dependencies required (local dev only)
- Designed for clarity + demo impact, not production polish

## Comparison UX (A11 Compliant)

For comparison queries:
- Backend extracts normalized comparison structure
- Narrator receives pre-computed values (no math in UI)
- UI displays:
  - Current period value
  - Comparison period value
  - Absolute delta
  - Percentage change (if base ≠ 0)
  - Direction indicator

Example:
```
Revenue this_year ($30,000) vs previous_year ($25,000)
↑ up by $5,000 (20%)
```

## Troubleshooting

### Database Not Found
```
Error: Database not found at ./data/haikugraph.duckdb
```
**Solution**: Run `haikugraph ingest` to create and populate the database.

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Stop other services on ports 8000 or 5173, or modify ports in config.

### Module Not Found
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Reinstall dependencies:
```bash
source .venv/bin/activate
uv pip install -e .
```

### Web Dependencies Error
```
Error: Cannot find module 'react'
```
**Solution**: Install web dependencies:
```bash
cd web
npm install
```

## Development

### Backend Changes
The backend server runs with `--reload`, so changes to Python files auto-reload.

### Frontend Changes
Vite dev server has HMR (Hot Module Replacement), so React changes update instantly.

### Adding New Visualizations
1. Update `_infer_visualization_hints()` in `api/server.py`
2. Add rendering logic in `VisualizationView.jsx`
3. Test with demo questions

## File Structure

```
haikugraph/
├── src/haikugraph/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py          # FastAPI backend
│   ├── planning/              # Intent, planner (unchanged)
│   ├── execution/             # Executor (unchanged)
│   └── explain/               # Narrator (unchanged)
├── web/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ComparisonCard.jsx
│   │   │   ├── VisualizationView.jsx
│   │   │   └── ExplainabilityTabs.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── start_server.sh            # Backend only
├── start_ui.sh                # Full stack
└── WEB_UI_README.md           # This file
```

## Known Limitations

- No authentication/authorization
- Limited to local development (not production-ready)
- Visualizations are basic (recharts defaults)
- No result caching or pagination
- Single database connection (configured in server.py)

## Next Steps

For production deployment:
1. Add authentication (OAuth, JWT)
2. Configure production database path
3. Add result pagination for large datasets
4. Implement query caching
5. Add rate limiting
6. Deploy with Docker/Kubernetes
7. Add monitoring and logging

## Support

For issues or questions:
1. Check this README
2. Review API docs at http://localhost:8000/docs
3. Check CLI documentation in main README.md
4. Verify database is set up correctly with `haikugraph doctor`
