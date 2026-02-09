# HaikuGraph Repository Map

## Critical Components for Hardening

### CLI Entry Points
- **`src/haikugraph/cli.py`**
  - `ask` (line 643): Deterministic planner CLI
  - `ask-a6` (line 878): Ollama LLM planner CLI
  - `ask-demo` (line 1146): Full pipeline demo with --debug flag

### Planning Layer (Query Understanding)
- **`src/haikugraph/planning/plan.py`**: Deterministic planner
  - `build_plan()`: Main entry - converts NL → structured plan
  - `detect_intent()`: Classify question type (metric/comparison/trend/etc)
  - `detect_entities()`: Find table/column references
  - `detect_metrics()`: Extract aggregations (sum/count/avg/etc)
  - `detect_constraints()`: Extract filters (time, status, values)
  - `build_subquestions()`: Generate executable subquestions with GROUP BY logic
  
- **`src/haikugraph/planning/llm_planner.py`**: LLM-based planner
  - `generate_or_patch_plan()`: Ollama-based plan generation
  
- **`src/haikugraph/planning/intent.py`**: Intent classification (A8)
- **`src/haikugraph/planning/followups.py`**: Conversation context handling

### Execution Layer (SQL Generation & Execution)
- **`src/haikugraph/execution/execute.py`**: Query builder
  - `execute_plan()`: Main entry - plan → SQL → results
  - `_build_sql_from_subquestion()`: Generates SELECT/WHERE/GROUP BY/ORDER BY
  - `_apply_constraints()`: Converts constraints to WHERE clauses
  - `_apply_time_filters()`: Handles time logic (CRITICAL FOR HARDENING)
  
- **`src/haikugraph/execution/comparison.py`**: Comparison query logic

### Explanation Layer
- **`src/haikugraph/explain/narrator.py`**: Natural language result narration

### Data Layer
- **`data/haikugraph.duckdb`**: DuckDB database
- **`data/profile.json`**: Schema profile
- **`data/graph.json`**: Table relationships
- **`data/cards/*.json`**: Semantic annotations
- **`data/plan.json`**: Last query plan (for followups)

### Test Infrastructure
- **`tests/`**: Existing test suite (18 test files)
  - Good: test_intent_classification_a8.py, test_comparison_*.py
  - Needs expansion: time windows, GROUP BY correctness, scalar vs series

## Key Weak Points Identified

### 1. Time Filtering (`src/haikugraph/execution/execute.py:_apply_time_filters()`)
- "this month" vs "last month" vs "last 30 days" semantics unclear
- Calendar vs rolling window logic inconsistent
- Month/year comparisons may not align properly

### 2. GROUP BY Logic (`src/haikugraph/planning/plan.py:build_subquestions()`)
- May over-group (adding GROUP BY when not needed)
- Grain selection (day/week/month) needs validation
- Scalar comparisons ("this vs last month") should not produce grouped series

### 3. Metric Detection (`src/haikugraph/planning/plan.py:detect_metrics()`)
- Only detects ONE metric per aggregation type (line 267: `break`)
- "count and volume by platform" → only gets volume, misses count

### 4. Constraint Scoping
- Fixed: timestamp column selection now prioritizes primary table
- May still have issues with complex multi-table queries

### 5. Intent Classification
- "Compare X vs Y" needs to prevent unwanted GROUP BY
- Trend vs scalar comparison distinction critical

## Database Schema (Current Test Data)
- **test_1_1_merged**: 162,932 rows
  - transaction_id, payment_amount, platform_name
  - Timestamps: created_at, updated_at, a2_form_created_at, mt103_created_at
- **test_3_1**: Related via quote_id
- **Other tables**: test_*, various structures

## Execution Flow
```
Question
  ↓
[Intent Classification] (optional, A8)
  ↓
[Plan Generation] 
  ├─ Detect intent (metric/comparison/trend/etc)
  ├─ Detect entities (tables/columns)
  ├─ Detect metrics (sum/count/avg)
  ├─ Detect constraints (time/filters)
  └─ Build subquestions (with GROUP BY logic)
  ↓
[SQL Generation]
  ├─ Build SELECT (columns + aggregations)
  ├─ Apply constraints → WHERE
  ├─ Apply GROUP BY (if breakdown)
  └─ Apply ORDER BY
  ↓
[Execute DuckDB Query]
  ↓
[Narrate Results] (optional)
```

## Testing Strategy

### Phase 1: Build Question Generator
Generate 200-500 questions from:
- **Intents**: revenue, count, unique customers, avg, top-k, breakdown
- **Time windows**: today, yesterday, this week, last week, this month, last month, 
                   this year, last year, last 7/30/90 days, last N months
- **Comparisons**: "X vs Y", "MoM", "YoY"
- **Breakdowns**: by day, week, month, platform, etc
- **Filters**: status, platform, date ranges

### Phase 2: Build Oracle Checks
Invariant validations:
- Scalar questions → no GROUP BY (unless multi-column select)
- "this month vs last month" → 2 values or delta, NOT grouped by month
- Time filters use correct SQL (calendar vs rolling)
- COUNT DISTINCT for "unique customers"
- Grain consistency (monthly breakdown → date_trunc('month'))

### Phase 3: Run & Classify Failures
- Auto-classify: planner errors, SQL errors, wrong shape, wrong time logic
- Generate failure report with buckets

### Phase 4: Fix in Tight Loops
- Pick top failure class
- Patch code
- Add regression test
- Re-run until bucket = 0

## Next Steps
1. Build question generator (combinatorial matrix)
2. Build CLI test runner with structured output capture
3. Build oracle/invariant checker
4. Run 200+ questions, classify failures
5. Begin fix loop on top failure classes
