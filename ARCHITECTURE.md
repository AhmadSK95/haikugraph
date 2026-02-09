# HaikuGraph POC - Architecture Overview

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         BROWSER UI                               │
│                    (http://localhost:5173)                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Question Input                                        │    │
│  │  "What is total revenue?"                              │    │
│  └────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Answer View                                           │    │
│  │  • Final answer in plain English                       │    │
│  │  • Execution metadata (time, rows, intent)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Comparison Card (if applicable)                       │    │
│  │  • Side-by-side values                                 │    │
│  │  • Delta & percentage                                  │    │
│  │  • Direction indicator                                 │    │
│  └────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Visualizations                                        │    │
│  │  • Number displays                                     │    │
│  │  • Bar charts                                          │    │
│  │  • Line charts                                         │    │
│  │  • Data tables                                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Explainability Tabs (collapsible)                     │    │
│  │  • Intent | Plan | SQL | Metadata                      │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                             ↕ HTTP (POST /ask)
┌──────────────────────────────────────────────────────────────────┐
│                      BACKEND API                                 │
│                   (http://localhost:8000)                        │
│                                                                  │
│  POST /ask                                                       │
│  ├─ Request:  { "question": "..." }                             │
│  └─ Response: {                                                  │
│       "final_answer": "...",                                     │
│       "intent": {...},                                           │
│       "plan": {...},                                             │
│       "queries": [...],                                          │
│       "results": [                                               │
│         {                                                        │
│           "id": "sq_1",                                          │
│           "sql": "...",                                          │
│           "preview_rows": [...],                                 │
│           "display_hint": "bar_chart",  ← Visualization hint     │
│           "chart_type": "bar",                                   │
│           "x_axis": "customer",                                  │
│           "y_axis": "revenue",                                   │
│           "units": "currency"                                    │
│         }                                                        │
│       ],                                                         │
│       "comparison": {...},                                       │
│       "metadata": {...},                                         │
│       "warnings": [],                                            │
│       "errors": []                                               │
│     }                                                            │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                   EXECUTION PIPELINE                             │
│                   (UNCHANGED - same as CLI)                      │
│                                                                  │
│  1. Intent Classification (A8)                                   │
│     ├─ Input: Question string                                    │
│     ├─ Classifier: Rule-based + LLM                              │
│     └─ Output: { type, confidence, rationale }                   │
│                                                                  │
│  2. Plan Generation (A6-A7)                                      │
│     ├─ Input: Question + Schema + Intent                         │
│     ├─ Planner: Ollama LLM                                       │
│     └─ Output: { subquestions, constraints, tables, ... }        │
│                                                                  │
│  3. SQL Execution (A4-A5)                                        │
│     ├─ Input: Plan                                               │
│     ├─ Generator: SQL builder                                    │
│     ├─ Executor: DuckDB query engine                             │
│     └─ Output: { rows, columns, row_count, ... }                 │
│                                                                  │
│  4. Comparison Extraction (A11) [if applicable]                  │
│     ├─ Input: Plan + Results                                     │
│     ├─ Logic: Normalize comparison structure                     │
│     └─ Output: { current, comparison, delta, delta_pct }         │
│                                                                  │
│  5. Narration (A9)                                               │
│     ├─ Input: Question + Intent + Plan + Results + Comparison    │
│     ├─ Narrator: Ollama LLM                                      │
│     └─ Output: Plain English explanation                         │
│                                                                  │
│  6. Visualization Hints (NEW)                                    │
│     ├─ Input: Results + Intent                                   │
│     ├─ Logic: Shape analysis (rows, cols, types)                 │
│     └─ Output: { display_hint, chart_type, axes, units }         │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│                                                                  │
│  ┌─────────────────┐         ┌──────────────────┐              │
│  │  DuckDB         │  ←──    │  Excel Files     │              │
│  │  (33 MB)        │         │  (data/*.xlsx)   │              │
│  │                 │         └──────────────────┘              │
│  │  Tables:        │                ↑                           │
│  │  • test_1_1     │         haikugraph ingest                  │
│  │  • test_2_1     │                                            │
│  │  • test_3_1     │         ┌──────────────────┐              │
│  │  • test_4_1     │  ←──    │  Profile JSON    │              │
│  │  • test_5_1     │         │  (134 KB)        │              │
│  └─────────────────┘         └──────────────────┘              │
│                                        ↓                         │
│                              ┌──────────────────┐               │
│                              │  Data Cards      │               │
│                              │  (145 files)     │               │
│                              └──────────────────┘               │
│                                        ↓                         │
│                              ┌──────────────────┐               │
│                              │  Graph JSON      │               │
│                              │  (42 KB)         │               │
│                              └──────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Frontend Components

```
App.jsx (Main Container)
  ├── QuestionForm
  │   ├── Input field
  │   └── Submit button
  │
  ├── DemoQuestions
  │   └── 6 clickable example questions
  │
  ├── ErrorCard (conditional)
  │   └── User-friendly error message
  │
  └── ResultSection (conditional)
      ├── AnswerCard
      │   ├── Final answer text
      │   └── Metadata badges
      │
      ├── ComparisonCard (if comparison)
      │   ├── Current period value
      │   ├── Comparison period value
      │   ├── Direction indicator
      │   └── Delta + percentage
      │
      ├── VisualizationView
      │   ├── NumberDisplay (single values)
      │   ├── ChartDisplay (bar/line)
      │   └── DataTable (multi-col)
      │
      └── ExplainabilityTabs
          ├── Intent Tab
          ├── Plan Tab
          ├── SQL Tab
          └── Metadata Tab
```

### Backend Flow

```
FastAPI Server (server.py)
  │
  ├── /health (GET)
  │   └── Returns: { "status": "ok", "db_exists": true }
  │
  └── /ask (POST)
      ├── Validate request
      ├── Check database exists
      │
      ├── [Stage 1] classify_intent(question)
      │   ├── Success → intent object
      │   └── Failure → warning, continue without intent
      │
      ├── [Stage 2] generate_or_patch_plan(question, schema, intent)
      │   ├── introspect_schema(db_path)
      │   └── LLM generates plan
      │
      ├── [Stage 3] execute_plan(plan, db_path)
      │   ├── Build SQL queries
      │   ├── Execute against DuckDB
      │   └── Return results + comparison (A11)
      │
      ├── [Stage 4] narrate_results(...)
      │   ├── Convert results to narrator format
      │   ├── Pass comparison if present
      │   └── LLM generates explanation
      │
      ├── [Stage 5] _infer_visualization_hints(result, intent)
      │   ├── Analyze data shape
      │   └── Suggest display type
      │
      └── Return AskResponse with all data
```

## Data Flow Example

### Example Query: "Show me revenue by customer"

```
1. USER INPUT
   "Show me revenue by customer"
   
2. BACKEND API receives POST /ask
   
3. INTENT CLASSIFICATION
   ├─ Type: grouped_metric
   ├─ Confidence: 0.87
   └─ Rationale: "Aggregation by dimension detected"
   
4. PLAN GENERATION
   ├─ Subquestions: [sq_1]
   ├─ Tables: ["test_1_1"]
   ├─ Columns: ["customer", "amount"]
   ├─ Group by: ["customer"]
   └─ Aggregations: [{"agg": "sum", "col": "amount"}]
   
5. SQL EXECUTION
   ├─ SQL: SELECT customer, SUM(amount) FROM test_1_1 GROUP BY customer
   ├─ Rows: [
        {"customer": "Alice", "sum": 15000},
        {"customer": "Bob", "sum": 12000},
        {"customer": "Charlie", "sum": 8000}
      ]
   └─ Row count: 3
   
6. NARRATION
   "Revenue by customer: Alice ($15,000), Bob ($12,000), Charlie ($8,000)."
   
7. VISUALIZATION HINTS
   ├─ display_hint: "bar_chart"
   ├─ chart_type: "bar"
   ├─ x_axis: "customer"
   ├─ y_axis: "sum"
   └─ units: "currency"
   
8. BACKEND RESPONSE (JSON)
   {
     "final_answer": "Revenue by customer: ...",
     "intent": { "type": "grouped_metric", ... },
     "plan": { ... },
     "queries": ["SELECT customer, ..."],
     "results": [
       {
         "id": "sq_1",
         "preview_rows": [...],
         "display_hint": "bar_chart",
         "chart_type": "bar",
         ...
       }
     ],
     "comparison": null,
     "metadata": { "execution_time_ms": 456, ... }
   }
   
9. FRONTEND RENDERING
   ├─ Answer: "Revenue by customer: ..."
   ├─ Visualization: Bar chart with 3 bars
   └─ Explainability: Available in tabs
```

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server with hot reload
- **Pydantic**: Data validation and serialization
- **DuckDB**: In-process analytical database
- **Ollama**: Local LLM for planning and narration

### Frontend
- **React 18**: UI library with hooks
- **Vite**: Fast dev server and build tool
- **Recharts**: Composable charting library
- **Lucide React**: Icon library

### Development
- **Python 3.11+**: Backend language
- **Node.js 18+**: Frontend tooling
- **npm**: Package management

## Deployment Architecture (Future)

```
┌────────────────────────────────────────────────────────┐
│                    PRODUCTION                          │
│                                                        │
│  ┌──────────────┐         ┌──────────────┐           │
│  │  CDN         │         │  Load        │           │
│  │  (Static UI) │ ←────── │  Balancer    │           │
│  └──────────────┘         └──────────────┘           │
│                                   ↓                    │
│                          ┌──────────────┐             │
│                          │  API         │             │
│                          │  Gateway     │             │
│                          │  (Auth)      │             │
│                          └──────────────┘             │
│                                   ↓                    │
│                 ┌─────────────────┴──────────────┐    │
│                 ↓                 ↓               ↓    │
│         ┌──────────────┐  ┌──────────────┐  ┌──────┐│
│         │  Backend     │  │  Backend     │  │ ... ││
│         │  Instance 1  │  │  Instance 2  │  └──────┘│
│         └──────────────┘  └──────────────┘           │
│                 ↓                 ↓                    │
│         ┌──────────────────────────────────┐         │
│         │  Database Pool                   │         │
│         │  (DuckDB / PostgreSQL)           │         │
│         └──────────────────────────────────┘         │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │
│  │  Monitoring  │  │  Logging     │  │  Cache   │   │
│  │  (Prometheus)│  │  (ELK)       │  │  (Redis) │   │
│  └──────────────┘  └──────────────┘  └──────────┘   │
└────────────────────────────────────────────────────────┘
```

## Security Considerations

- **No auth in POC**: Add OAuth/JWT for production
- **CORS**: Currently allows localhost only
- **SQL Injection**: DuckDB parameterized queries (safe)
- **Rate Limiting**: Not implemented (needed for prod)
- **Input Validation**: Pydantic schemas enforce types
- **Error Handling**: User-safe messages (no stack traces)

## Performance Characteristics

### Current (POC)
- **Latency**: 200-500ms per query (local LLM)
- **Throughput**: 1-5 concurrent users
- **Database**: In-process (single connection)
- **Caching**: None

### Target (Production)
- **Latency**: <100ms (cloud LLM + caching)
- **Throughput**: 100+ concurrent users
- **Database**: Connection pool
- **Caching**: Redis for plans + results

---

This architecture ensures:
- ✅ Clean separation of concerns
- ✅ Same pipeline for CLI and UI
- ✅ Zero changes to existing validators
- ✅ Production-ready API contract
- ✅ Scalability path for future growth
