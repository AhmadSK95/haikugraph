# HaikuGraph Hardening Test Infrastructure

Comprehensive test harness for systematically finding and fixing bugs in HaikuGraph's query planning and SQL generation.

## Overview

This test infrastructure generates 200-5000 natural language questions by combining different dimensions (metrics, time windows, breakdowns, filters), runs them through the actual CLI, and validates that the generated SQL and results match expected patterns.

## Architecture

### Components

1. **`question_generator.py`**: Combinatorial question generator
   - Generates questions from intent × metric × time_window × breakdown × filter combinations
   - Includes expected behavior (e.g., "scalar query should NOT have GROUP BY")
   - Supports 200-5000 questions depending on configuration

2. **`cli_runner.py`**: CLI test runner
   - Runs questions through actual `haikugraph ask` CLI
   - Captures plan JSON, SQL, execution results, and errors
   - Extracts metadata (GROUP BY presence, DISTINCT usage, row counts)

3. **`oracle.py`**: Invariant checker
   - Validates SQL against expected patterns based on question intent
   - Checks ~12 invariant rules (GROUP BY correctness, time filters, etc.)
   - Classifies failures into buckets for systematic fixing

4. **`run_matrix.py`**: Main test harness
   - Orchestrates the full pipeline
   - Generates detailed failure reports
   - Creates per-bucket JSON files for targeted fixing

## Usage

### Quick Start

```bash
# From repo root with virtualenv activated
cd /Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph
source .venv/bin/activate

# Run 200-question matrix (takes ~5-10 minutes)
python -m tests.hardening.run_matrix \\
  --max-questions 200 \\
  --planner deterministic

# Output goes to ./reports/hardening_TIMESTAMP/
```

### Command-Line Options

```bash
python -m tests.hardening.run_matrix \\
  --db-path ./data/haikugraph.duckdb \\   # Database path
  --data-dir ./data \\                     # Data directory
  --max-questions 200 \\                   # Number of questions (default: 200)
  --planner deterministic \\               # or "llm"
  --output-dir ./reports/my_run           # Custom output directory
```

### Output Files

After running, you'll find:

```
reports/hardening_TIMESTAMP/
├── summary.json                # High-level stats: pass rate, failure buckets
├── results.json                # All test results with SQL + violations
├── failures.json               # Failed tests only
└── buckets/                    # Per-bucket failure files
    ├── group_by_over_aggregation.json
    ├── time_filter_error.json
    ├── distinct_missing.json
    └── ...
```

## Invariant Rules Checked

The oracle validates these invariants:

### 1. GROUP BY Correctness
- **SCALAR_HAS_GROUP_BY**: Scalar queries (e.g., "total revenue") must NOT have GROUP BY
- **MISSING_GROUP_BY**: Breakdown queries (e.g., "revenue by platform") MUST have GROUP BY
- **COMPARISON_HAS_GROUP_BY**: Scalar comparisons (e.g., "this month vs last month") must NOT group by month

### 2. DISTINCT Usage
- **UNIQUE_MISSING_DISTINCT**: "unique customers" must use `COUNT(DISTINCT customer_id)`

### 3. Time Filters
- **MISSING_TIME_FILTER**: Time window specified but no SQL filter applied
- **WRONG_TIME_FILTER**: Filter type mismatch (e.g., rolling window instead of calendar month)

### 4. Shape Validation
- **WRONG_ROW_COUNT**: Scalar queries must return exactly 1 row
- **COMPARISON_WRONG_SHAPE**: Comparisons should return 2 scalars or delta, not grouped series

### 5. Time Grain
- **WRONG_TIME_GRAIN**: `date_trunc('month', ...)` for monthly trends, not daily

### 6. Execution
- **PLAN_ERROR**: Planning failed
- **SQL_ERROR**: SQL execution failed
- **TIMEOUT**: Query exceeded timeout

## Failure Buckets

Failures are auto-classified into buckets for systematic fixing:

1. **`group_by_over_aggregation`**: Scalar queries incorrectly have GROUP BY
2. **`group_by_missing`**: Breakdown queries missing GROUP BY
3. **`time_filter_error`**: Time filter missing or incorrect
4. **`distinct_missing`**: Unique count without DISTINCT
5. **`comparison_shape_error`**: Comparison returns wrong shape
6. **`planner_error`**: Planning stage failed
7. **`sql_error`**: SQL execution failed
8. **`other_error`**: Other issues

## Typical Workflow

### Phase 1: Run Matrix and Identify Top Issues

```bash
# Run 200 questions
python -m tests.hardening.run_matrix --max-questions 200

# Check summary
cat reports/hardening_*/summary.json | jq '.failure_buckets'
```

Example output:
```json
{
  "group_by_over_aggregation": 45,
  "time_filter_error": 28,
  "distinct_missing": 12,
  "comparison_shape_error": 8
}
```

### Phase 2: Fix Top Bucket

Pick the largest bucket (e.g., `group_by_over_aggregation` with 45 failures):

```bash
# Inspect failures in that bucket
cat reports/hardening_*/buckets/group_by_over_aggregation.json | jq '.[0]'
```

Example failure:
```json
{
  "question": "What is the total revenue this month?",
  "spec": {
    "intent": "scalar_metric",
    "expected_group_by": false
  },
  "result": {
    "sql": "SELECT SUM(...) FROM test_1 GROUP BY platform_name",
    "has_group_by": true
  },
  "violations": [
    {
      "type": "scalar_has_group_by",
      "expected": "No GROUP BY for scalar result",
      "actual": "GROUP BY found in SQL"
    }
  ]
}
```

### Phase 3: Patch Code

Identify root cause:
- **Planning layer** (`src/haikugraph/planning/plan.py`): `build_subquestions()` adds GROUP BY when it shouldn't
- **Execution layer** (`src/haikugraph/execution/execute.py`): SQL builder doesn't check if GROUP BY is needed

Fix the code:
```python
# In build_subquestions()
if intent_type == "metric" and not has_breakdown:
    # Scalar metric - no GROUP BY needed
    group_by = []
else:
    # Breakdown - add GROUP BY
    group_by = [breakdown_column]
```

### Phase 4: Add Regression Test

```python
# tests/test_scalar_group_by_fix.py
def test_scalar_revenue_no_group_by():
    spec = QuestionSpec(
        intent=IntentType.SCALAR_METRIC,
        metric="revenue",
        expected_group_by=False,
    )
    question = "What is the total revenue?"
    
    result = run_question_through_cli(question, ...)
    violations = check_oracle_invariants(spec, result)
    
    assert not result.has_group_by, "Scalar query should not have GROUP BY"
    assert len(violations) == 0
```

### Phase 5: Re-run Matrix

```bash
# Re-run to verify fix
python -m tests.hardening.run_matrix --max-questions 200

# Check if bucket is reduced
cat reports/hardening_*/summary.json | jq '.failure_buckets.group_by_over_aggregation'
# Expected: 0 (or much lower)
```

### Phase 6: Repeat

Continue fixing buckets in order of frequency until pass rate is ~100%.

## Question Matrix Dimensions

The generator combines:

**Intents:**
- `scalar_metric`: "What is total revenue?"
- `scalar_count`: "How many transactions?"
- `scalar_unique`: "How many unique customers?"
- `scalar_avg`: "What is average ticket size?"
- `breakdown`: "Revenue by platform"
- `trend`: "Revenue by month"
- `comparison_scalar`: "This month vs last month"
- `top_k`: "Top 10 platforms"

**Time Windows:**
- Calendar: today, yesterday, this week, last week, this month, last month, this year, last year
- Rolling: last 7/30/90 days, last 3/6/12 months
- Specific: December, January

**Breakdowns:**
- by day, by week, by month
- by platform, by status

**Metrics:**
- revenue, volume, transactions, customers

This produces ~200-500 unique question combinations.

## Extending the Generator

To add new dimensions:

```python
# In question_generator.py

# Add new intent type
class IntentType(Enum):
    TOP_K = "top_k"
    TOP_K_PERCENT = "top_k_percent"  # NEW

# Add to matrix generator
for metric in ["revenue", "transactions"]:
    questions.append(QuestionSpec(
        intent=IntentType.TOP_K_PERCENT,
        metric=metric,
        top_k_percent=20,  # Top 20%
        expected_shape="grouped",
        expected_group_by=True,
    ))
```

## Known Limitations

1. **Time semantics**: "this month" vs "last month" interpretation depends on clock injection (not yet implemented - see Phase G)
2. **Multi-metric queries**: "count and volume by platform" only generates one metric (known issue #3 in REPO_MAP.md)
3. **Complex joins**: Multi-table queries not fully covered
4. **Specific months**: "December" currently uses month=12 filter without year (may match multiple Decembers)

## Next Steps

See REPO_MAP.md "Next Steps" section for the full fix loop procedure.

Quick version:
1. Run matrix → identify top bucket
2. Fix code in planner/execution layer
3. Add regression test
4. Re-run matrix → verify bucket is reduced
5. Repeat until pass rate ~100%
