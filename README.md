# HaikuGraph

**A hybrid AI data assistant that combines deterministic planning with LLM enhancement for natural language queries over structured data.**

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Structure](#system-structure)
- [CLI Commands Reference](#cli-commands-reference)
- [Web UI](#web-ui)
- [Service Management](#service-management)
- [Development](#development)

---

## Overview

HaikuGraph is a natural language data assistant that allows users to ask questions about their data in plain English and receive answers backed by SQL queries, visualizations, and explanations. It uses a **hybrid planning approach** that combines:

- **Deterministic planner**: Fast, reliable, graph-based query planning
- **LLM planner**: Flexible, context-aware query understanding with local Ollama models
- **Auto-fallback**: Gracefully falls back from LLM to deterministic when needed

### Key Features

✅ **Natural Language Queries**: "How many transactions in December?" → SQL + Results  
✅ **Smart Schema Discovery**: Automatically profiles data and discovers relationships  
✅ **Semantic Data Cards**: Annotates columns with types (money, timestamp, identifier, etc.)  
✅ **Ambiguity Resolution**: Handles unclear queries with LLM-powered disambiguation  
✅ **Follow-up Conversations**: Maintains context across related questions  
✅ **Rich Visualizations**: Auto-selects charts (bar, line, table) based on query intent  
✅ **Local-First AI**: Uses Ollama for privacy-preserving LLM operations  

---

## Architecture

### Why Hybrid Planning?

**The Problem**: Pure LLM planners are flexible but unreliable (hallucinate column names, miss filters). Pure deterministic planners are reliable but rigid (struggle with ambiguous language).

**Our Solution**: Start with LLM for flexibility, fall back to deterministic for reliability.

### Pipeline Overview

```
User Question
     ↓
[1] Intent Classification (LLM)
     ↓
[2] Query Planning (Hybrid)
     ├─→ Try LLM Planner (llama3.1:8b)
     │   └─→ If fails → Deterministic Planner
     ↓
[3] SQL Generation & Execution
     └─→ If schema errors → Retry with Deterministic
     ↓
[4] Natural Language Explanation (LLM)
     ↓
[5] Visualization Hints
     ↓
Answer + Charts + SQL
```

### Component Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Web UI (React)                    │
│  - Question input  - Charts  - Transparency panel   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/JSON
┌──────────────────────▼──────────────────────────────┐
│              Backend API (FastAPI)                   │
│  - /ask endpoint  - Intent → Plan → Execute → Explain│
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼───────┐ ┌───▼────┐ ┌──────▼────────┐
│ LLM Planner   │ │ Det.   │ │  Executor     │
│ (Ollama)      │ │ Planner│ │  (DuckDB SQL) │
│ llama3.1:8b   │ │ Graph  │ │               │
└───────────────┘ └────────┘ └───────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
           ┌───────────▼───────────┐
           │  Data Layer           │
           │  - DuckDB database    │
           │  - Graph (relations)  │
           │  - Semantic cards     │
           └───────────────────────┘
```

### Pros and Cons

#### ✅ **Advantages**

1. **Reliability**: Deterministic fallback ensures queries always execute
2. **Flexibility**: LLM handles natural language variations and ambiguity
3. **Transparency**: Users see SQL, intent classification, and confidence scores
4. **Privacy**: Local Ollama models keep data processing on-premises
5. **Smart Joins**: Graph-based relationship discovery finds multi-table paths
6. **Conversational**: Follow-up questions maintain context automatically

#### ❌ **Limitations**

1. **LLM Schema Awareness**: LLM doesn't always pick the right tables/columns (mitigated by fallback)
2. **Aggregation Ambiguity**: "Revenue split by platform" could mean count or sum (uses heuristics)
3. **Complex Joins**: Multi-hop joins (3+ tables) may need manual graph edges
4. **Performance**: LLM planning adds 2-5s latency vs pure deterministic
5. **Model Dependency**: Requires Ollama running locally with specific models

### When to Use Each Planner

| Scenario | Best Planner | Why |
|----------|-------------|-----|
| Simple aggregations | Deterministic | Faster, reliable |
| Ambiguous language | LLM | Handles "last month", "top platforms" |
| Multi-table queries | Deterministic | Better join path finding |
| Follow-up questions | Hybrid | LLM for context, fallback for safety |
| Complex filters | LLM | Better at parsing "in December excluding refunds" |

---

## Installation

### Prerequisites

1. **Python 3.11+**
2. **Ollama** (for LLM features)
   ```bash
   # Install Ollama from https://ollama.ai
   
   # Pull required models
   ollama pull llama3.1:8b        # Planner model
   ollama pull qwen2.5:7b-instruct # Narrator model
   ```
3. **Node.js 18+** (for Web UI)

### Install HaikuGraph

```bash
# Clone repository
cd ~/Desktop/dataAssistantGenAI/haikugraph

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows

# Install package
uv pip install -e .

# Verify installation
haikugraph --version
haikugraph doctor
```

---

## Quick Start

**Get from Excel to chatting with your data in 5 minutes:**

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Put Excel files in ./data directory
# (Or use your existing Excel files)

# 3. Ingest data
haikugraph ingest --data-dir ./data

# 4. Profile and build graph
haikugraph profile
haikugraph cards build
haikugraph graph build

# 5. Start Web UI
./start_ui.sh

# 6. Open browser and ask questions!
# http://localhost:3000
```

**Example questions to try:**
- "How many transactions do we have?"
- "What is the total revenue in December?"
- "Show me revenue split by platform"
- "Compare this month vs last month"

---

## System Structure

```
haikugraph/
├── src/haikugraph/          # Core Python package
│   ├── cli.py              # CLI entry point
│   ├── io/                 # Data ingestion & profiling
│   │   ├── ingestion.py   # Excel → DuckDB
│   │   └── profiler.py    # Schema profiling
│   ├── planning/           # Query planning
│   │   ├── plan.py        # Deterministic planner
│   │   ├── llm_planner.py # LLM-based planner
│   │   ├── intent.py      # Intent classification
│   │   ├── followups.py   # Conversation context
│   │   └── llm_resolver.py # Ambiguity resolution
│   ├── execution/          # SQL generation & execution
│   │   └── execute.py     # DuckDB query builder
│   ├── explain/            # Natural language output
│   │   └── narrator.py    # LLM-based explanation
│   ├── cards/              # Semantic annotations
│   │   ├── builder.py     # Card generation
│   │   └── loader.py      # Card loading
│   ├── graph/              # Relationship discovery
│   │   └── builder.py     # Graph construction
│   ├── api/                # Web API
│   │   └── server.py      # FastAPI backend
│   └── llm/                # LLM routing
│       └── router.py      # Ollama integration
├── ui/                      # React frontend
│   ├── src/
│   │   ├── components/
│   │   └── App.tsx
│   └── package.json
├── data/                    # Data directory (gitignored)
│   ├── *.xlsx              # Excel input files
│   ├── haikugraph.duckdb  # DuckDB database
│   ├── profile.json        # Schema profile
│   ├── graph.json          # Table relationships
│   ├── cards/              # Semantic annotations
│   └── plan.json           # Last query plan (for followups)
├── start_ui.sh             # UI launcher script
└── README.md
```

### Data Flow

```
Excel Files → [Ingest] → DuckDB
                ↓
         [Profile] → profile.json
                ↓
    [Cards Build] → data/cards/*.json
                ↓
    [Graph Build] → graph.json
                ↓
          [Ask Question]
                ↓
   Web UI / CLI Query Interface
```

---

## CLI Commands Reference

### Data Ingestion

#### `haikugraph ingest`

Ingest Excel files into DuckDB database.

**Usage:**

```bash
# Basic usage - ingest all Excel files from ./data
haikugraph ingest

# Specify custom data directory
haikugraph ingest --data-dir ./my-data

# Specify custom database path
haikugraph ingest --db-path ./output/my-database.duckdb

# Read a specific sheet by name or index
haikugraph ingest --sheet "Sheet2"
haikugraph ingest --sheet 1

# Combine options
haikugraph ingest --data-dir ./data --db-path ./data/main.duckdb --sheet 0
```

**What it does:**
- Finds all `.xlsx` and `.xls` files in data directory
- Creates tables with sanitized names from filenames
- Auto-fallback to string mode if mixed types detected
- Outputs: `./data/haikugraph.duckdb`

**Table Naming:** `Sales Data 2024.xlsx` → `sales_data_2024`

---

### Schema Profiling

#### `haikugraph profile`

Generate detailed JSON profile of database schema.

```bash
# Basic usage
haikugraph profile

# Custom paths
haikugraph profile --db-path ./data/main.duckdb --out ./reports/profile.json

# Control sampling
haikugraph profile --sample-rows 50000 --top-k 20
```

**Output:** `./data/profile.json` with column types, nulls, distinct counts, sample values

**View profile:**
```bash
python3 -m json.tool data/profile.json | less
jq '.tables | keys' data/profile.json  # List tables
```

---

### Semantic Cards

#### `haikugraph cards build`

Generate semantic annotations (TableCard, ColumnCard, RelationCard) from profile.

```bash
# Build cards
haikugraph cards build

# List cards
haikugraph cards list

# Show specific card
haikugraph cards show "table:test_1_1"
haikugraph cards show "column:test_1_1.customer_id"

# View table summary
haikugraph cards table test_3_1
```

**Output:** `./data/cards/*.json` - Semantic hints like "identifier", "timestamp", "money", "geography"

---

### Relationship Graph

#### `haikugraph graph build`

Discover table relationships based on column names and keys.

```bash
# Build relationship graph
haikugraph graph build

# Show graph structure
haikugraph graph show
```

**Output:** `./data/graph.json` - Nodes (tables), edges (join paths), confidence scores

---

### Query Planning (CLI)

#### `haikugraph ask`

Deterministic planner (fast, reliable).

```bash
# Plan only
haikugraph ask --question "How many transactions in December?"

# Plan + execute
haikugraph ask --question "Revenue by platform" --execute

# With LLM resolver for ambiguity
haikugraph ask --question "MT103 transactions" --use-llm-resolver --execute

# Follow-up questions (maintains context)
haikugraph ask --question "How many in November?" --followup --execute
```

**Output:** `./data/plan.json`, `./data/result.json`

#### `haikugraph ask-a6`

Hybrid planner with Ollama LLM.

```bash
# LLM-powered planning + execution
haikugraph ask-a6 --question "Revenue split by platform in December"
```

**Output:** `./data/plan_a6.json`, `./data/result_a6.json`

---

### Diagnostics

#### `haikugraph doctor`

Verify installation and environment.

```bash
haikugraph doctor
```

Shows: Python version, DuckDB version, data directory status, database tables

---

## Web UI

### Starting the UI

```bash
# Start both backend and frontend
./start_ui.sh
```

This launches:
- **Backend**: FastAPI server on `http://localhost:8000` (with auto-reload)
- **Frontend**: React dev server on `http://localhost:3000`

**Open in browser:** http://localhost:3000

### UI Features

💬 **Question Input**: Natural language query box with auto-complete  
📊 **Visualizations**: Auto-selects bar charts, line charts, tables, or numbers  
🔍 **Transparency Panel**: View intent, plan JSON, SQL queries, metadata  
📝 **Conversation History**: Maintains context for follow-up questions  
⚠️ **Warnings**: Shows LLM fallback status and errors  

### Example Queries in UI

```
❓ "How many transactions do we have?"
   → Returns count with number display

❓ "What is the revenue by platform?"
   → Returns bar chart grouped by platform

❓ "Compare this month vs last month revenue"
   → Returns comparison card with delta % and direction

❓ "Show me transactions in December"
   → Returns data table with filters applied
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Ask question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many transactions?"}'
```

**API Response:**
```json
{
  "final_answer": "There are 25,056 transactions.",
  "intent": {"type": "metric", "confidence": 0.95},
  "plan": {...},
  "queries": ["SELECT COUNT(DISTINCT ...) FROM ..."],
  "results": [{"row_count": 1, "preview_rows": [...]}],
  "metadata": {"execution_time_ms": 450},
  "warnings": [],
  "errors": []
}
```

---

## Service Management

### Checking Running Services

```bash
# Check if backend is running
curl http://localhost:8000/health

# Check if frontend is running
curl http://localhost:3000

# List processes
ps aux | grep uvicorn  # Backend
ps aux | grep "npm"     # Frontend
```

### Stopping Services

#### Option 1: Graceful Shutdown (if started in terminal)

```bash
# If you started ./start_ui.sh in foreground:
# Press Ctrl+C in the terminal
```

#### Option 2: Kill by Process Name

```bash
# Kill backend (FastAPI/uvicorn)
pkill -f "uvicorn haikugraph.api.server:app"

# Kill frontend (npm dev server)
pkill -f "npm run dev"

# Kill both at once
pkill -f "uvicorn haikugraph.api.server:app" && pkill -f "npm run dev"
```

#### Option 3: Kill by Port

```bash
# Find and kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Find and kill process on port 3000 (frontend)
lsof -ti:3000 | xargs kill -9
```

#### Option 4: Kill All Node/Python Processes (Nuclear Option)

```bash
# CAUTION: This kills ALL node and python processes!
pkill node
pkill python
```

### Restarting Services

```bash
# Stop services
pkill -f "uvicorn haikugraph.api.server:app" && pkill -f "npm run dev"

# Wait a moment
sleep 2

# Restart
./start_ui.sh
```

### Checking Logs

```bash
# Backend logs (if started with start_ui.sh)
tail -f nohup.out  # Backend output

# Frontend logs
tail -f ui/npm.log  # If logging to file

# Check for errors
grep -i error nohup.out
```

### Troubleshooting

**Port already in use:**
```bash
# Find what's using port 8000
lsof -i:8000

# Kill it
lsof -ti:8000 | xargs kill -9

# Restart
./start_ui.sh
```

**Ollama not running:**
```bash
# Check Ollama status
ollama list

# Start Ollama (if needed)
ollama serve &

# Test model
ollama run llama3.1:8b "Hello"
```

**Frontend won't start:**
```bash
# Install dependencies
cd ui
npm install

# Start manually
npm run dev
```

---

## Development

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/haikugraph
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_planning.py

# Run with coverage
pytest --cov=haikugraph tests/
```

### Project Structure

See [System Structure](#system-structure) section above for detailed directory layout.

### Key Technologies

- **Backend**: Python 3.11+, FastAPI, DuckDB, Pydantic
- **Frontend**: React, TypeScript, Vite, Recharts
- **LLM**: Ollama (llama3.1:8b, qwen2.5:7b-instruct)
- **Data**: Pandas, Openpyxl

### Environment Variables

```bash
# LLM model selection
export HG_PLANNER_MODEL="llama3.1:8b"       # Planner model
export HG_NARRATOR_MODEL="qwen2.5:7b-instruct"  # Narrator model

# Retry limits
export HG_MAX_RETRIES="2"                   # Plan generation retries
export HG_INTENT_MAX_RETRIES="1"            # Intent classification retries
```

---

## Contributing

Contributions welcome! Areas for improvement:

1. **Multi-hop joins**: Enhance graph traversal for 3+ table joins
2. **Query optimization**: Cache plans, optimize SQL generation
3. **Better LLM prompts**: Improve column/table selection accuracy
4. **More visualizations**: Add scatter plots, heatmaps, sankey diagrams
5. **Export features**: CSV/Excel export of results
6. **Saved queries**: Bookmark and replay common questions

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `ruff format .` and `ruff check .`
5. Submit a pull request

---

## License

See LICENSE file for details.

---

## Troubleshooting Guide

### Common Issues

**Q: "ModuleNotFoundError: No module named 'haikugraph'"**  
A: Activate virtual environment: `source .venv/bin/activate`

**Q: "Database not found" error**  
A: Run `haikugraph ingest` first to create database

**Q: "Ollama connection refused"**  
A: Start Ollama: `ollama serve` and pull models: `ollama pull llama3.1:8b`

**Q: LLM queries are slow**  
A: Use smaller models or switch to deterministic planner only

**Q: "Column X does not exist" errors**  
A: The hybrid system should auto-fallback. Check warnings in UI. May need to rebuild cards/graph.

**Q: UI won't connect to backend**  
A: Check `http://localhost:8000/health`. Verify ports 3000 and 8000 are available.

---

## Support

For issues, questions, or feature requests:
- Check [Troubleshooting Guide](#troubleshooting-guide) above
- Review logs: `tail -f nohup.out`
- Run `haikugraph doctor` for diagnostics

---

**Built with ❤️ for data teams who want AI assistance without sacrificing transparency or reliability.**
