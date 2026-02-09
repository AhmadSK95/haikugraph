# A10: End-to-End Demo Interface

## Overview

The `ask-demo` command demonstrates the complete HaikuGraph pipeline in a single, simple entrypoint.

**Pipeline Flow:**
```
Question → Intent → Plan → Execute → Narrate → Answer
```

## Usage

### Basic Usage

```bash
haikugraph ask-demo "What is total revenue?"
```

**Output:**
```
======================================================================
Question: What is total revenue?
======================================================================

======================================================================
Answer:
======================================================================

Total revenue is $25,000
```

### Debug Mode

Show all intermediate steps:

```bash
haikugraph ask-demo "Revenue by barber" --debug
```

**Debug output includes:**
- **Intent classification** (A8): type, confidence, rationale
- **Plan JSON** (A6-A7): subquestions, constraints, join paths
- **SQL execution** (A4-A5): generated SQL + results per subquestion
- **Final narration** (A9): user-facing explanation

### Raw Mode

Get machine-readable JSON output:

```bash
haikugraph ask-demo "Compare this month vs last month" --raw
```

Returns the full execution result as JSON (no narration).

### Skip Intent Classification

Bypass intent classification for speed:

```bash
haikugraph ask-demo "Show recent appointments" --no-intent
```

## Command Options

| Flag | Description | Default |
|------|-------------|---------|
| `--db-path` | Path to DuckDB database | `./data/haikugraph.duckdb` |
| `--debug` | Print intent + plan + SQL | `False` |
| `--no-intent` | Skip A8 intent classification | `False` |
| `--raw` | Print raw execution results only | `False` |

## Exit Codes

The command uses stage-specific exit codes for clear failure diagnosis:

| Code | Meaning | Stage |
|------|---------|-------|
| `0` | Success | All stages completed |
| `1` | Planner failure | Stage 2 (Plan Generation) |
| `2` | Execution failure | Stage 3 (SQL Execution) |
| `3` | Narration failure | Stage 4 (Narration) |
| `130` | User interrupt | Ctrl+C pressed |
| `255` | Unexpected error | Unknown stage |

## Pipeline Stages

### Stage 1: Intent Classification (A8)

**Classifies user intent into:**
- `metric` - Single aggregated value
- `grouped_metric` - Aggregation by dimension
- `comparison` - Temporal/cohort comparison
- `lookup` - Raw rows listing
- `diagnostic` - Health/anomalies analysis
- `unknown` - Cannot confidently classify

**Failure behavior:** Non-fatal (warns and continues without intent)

### Stage 2: Plan Generation (A6-A7)

**Generates execution plan:**
- Introspects database schema
- Calls LLM planner (Ollama)
- Produces validated Plan JSON
- Includes intent context if available

**Failure behavior:** Fatal (exit code 1)

### Stage 3: SQL Execution (A4-A5)

**Executes each subquestion:**
- Generates SQL for each subquestion
- Executes against DuckDB
- Collects results and metadata
- Detects execution failures

**Failure behavior:** Fatal (exit code 2)

### Stage 4: Narration (A9)

**Converts results to natural language:**
- Intent-aware narration style
- Short-circuits if ANY subquestion failed
- Calls LLM narrator (Ollama)
- Produces user-facing text

**Failure behavior:** Fatal (exit code 3, shows raw results as fallback)

## Examples

### Example 1: Simple Metric

```bash
$ haikugraph ask-demo "How many appointments?"

======================================================================
Question: How many appointments?
======================================================================

======================================================================
Answer:
======================================================================

There are 127 appointments in the system.
```

### Example 2: Grouped Aggregation (Debug)

```bash
$ haikugraph ask-demo "Revenue by barber" --debug

======================================================================
Question: Revenue by barber
======================================================================

[1/4] Classifying intent...

✅ Intent:
  Type: grouped_metric
  Confidence: 0.93
  Rationale: Aggregation with 'by' grouping dimension
  Requires comparison: False

[2/4] Generating plan...

✅ Plan:
{
  "original_question": "Revenue by barber",
  "subquestions": [
    {
      "id": "SQ1",
      "tables": ["appointments"],
      "group_by": ["barber_name"],
      "aggregations": [{"agg": "sum", "col": "revenue"}]
    }
  ]
}

[3/4] Executing SQL...

✅ Execution Results:

  ✅ SQ1:
     SQL: SELECT barber_name, SUM(revenue) as sum_revenue FROM appointments GROUP BY barber_name
     Rows: 5
     Sample: [{'barber_name': 'Alice', 'sum_revenue': 10000}, ...]

[4/4] Generating narrative explanation...

======================================================================
Answer:
======================================================================

Revenue by barber: Alice ($10,000), Bob ($8,500), Charlie ($6,500), 
Dana ($5,200), Eve ($4,100).
```

### Example 3: Comparison Query

```bash
$ haikugraph ask-demo "Revenue this month vs last month"

======================================================================
Question: Revenue this month vs last month
======================================================================

======================================================================
Answer:
======================================================================

Revenue this month ($30,000) vs last month ($25,000) - increased by 
$5,000 (20%).
```

### Example 4: Execution Failure

```bash
$ haikugraph ask-demo "Revenue from nonexistent_table"

======================================================================
Question: Revenue from nonexistent_table
======================================================================

❌ Execution failed: Table 'nonexistent_table' not found

$ echo $?
2
```

## Requirements

- **Ollama running** with models:
  - Planner: `qwen2.5:7b-instruct` (or `HG_PLANNER_MODEL`)
  - Narrator: `llama3.1:8b` (or `HG_NARRATOR_MODEL`)
- **DuckDB database** with data ingested
- **Internet connection** (for first-time model pulls)

## Setup

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Start Ollama
   ollama serve
   ```

2. **Pull models:**
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull llama3.1:8b
   ```

3. **Ingest data:**
   ```bash
   haikugraph ingest --data-dir ./data
   ```

4. **Run demo:**
   ```bash
   haikugraph ask-demo "What is total revenue?"
   ```

## Customization

### Use different models:

```bash
export HG_PLANNER_MODEL=llama3:8b
export HG_NARRATOR_MODEL=mistral:latest

haikugraph ask-demo "Your question"
```

### Adjust temperatures:

```bash
export HG_PLANNER_TEMPERATURE=0    # Deterministic (default)
export HG_NARRATOR_TEMPERATURE=0.2 # Slightly creative (default)
```

### Change max retries:

```bash
export HG_MAX_RETRIES=2              # Planner retries
export HG_INTENT_MAX_RETRIES=1       # Intent retries (default)
export HG_NARRATOR_MAX_RETRIES=1     # Narrator retries (default)
```

## Architecture

The `ask-demo` command is the **single entrypoint** that demonstrates all HaikuGraph components working together:

```
┌─────────────────────────────────────────────────────────────┐
│                      ask-demo Command                        │
│                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐          │
│  │   Intent   │   │   Planner  │   │  Executor  │          │
│  │    (A8)    │ → │   (A6-A7)  │ → │   (A4-A5)  │          │
│  └────────────┘   └────────────┘   └────────────┘          │
│                                            ↓                 │
│                                     ┌────────────┐           │
│                                     │  Narrator  │           │
│                                     │    (A9)    │           │
│                                     └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Comparison with Other Commands

| Command | Purpose | LLM | Output |
|---------|---------|-----|--------|
| `ask` | Deterministic planning (no LLM) | ❌ | Plan JSON |
| `ask-llm` | OpenAI-based planning | OpenAI | Plan JSON + optional execution |
| `ask-a6` | Ollama A6 POC | Ollama | Execution + narrative |
| **`ask-demo`** | **Full A10 pipeline** | **Ollama** | **Clean answer** |

## Troubleshooting

### "Connection refused"

**Problem:** Ollama not running

**Solution:**
```bash
ollama serve
```

### "Model not found"

**Problem:** Models not pulled

**Solution:**
```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
```

### "Planner failed: validation error"

**Problem:** LLM returned invalid JSON

**Solution:** Check LLM output in debug mode:
```bash
haikugraph ask-demo "your question" --debug
```

### "Execution failed: table not found"

**Problem:** Data not ingested

**Solution:**
```bash
haikugraph ingest --data-dir ./data
```

## Demo Script

For a complete demo walkthrough:

```bash
# 1. Simple metric
haikugraph ask-demo "What is total revenue?"

# 2. Grouped aggregation
haikugraph ask-demo "Revenue by barber"

# 3. Comparison (intent-aware)
haikugraph ask-demo "Compare revenue this month vs last month"

# 4. Lookup query
haikugraph ask-demo "Show me recent appointments"

# 5. Diagnostic query
haikugraph ask-demo "Why did revenue drop?"

# 6. Debug mode (show all stages)
haikugraph ask-demo "Total appointments" --debug

# 7. Raw output (machine-readable)
haikugraph ask-demo "List services" --raw
```

## Screen Recording Tips

For demo videos:

1. **Clear terminal:**
   ```bash
   clear
   ```

2. **Run with clean output:**
   ```bash
   haikugraph ask-demo "What is total revenue?"
   ```

3. **Show debug for technical audience:**
   ```bash
   haikugraph ask-demo "Revenue by barber" --debug
   ```

4. **Demonstrate failure handling:**
   ```bash
   # Graceful degradation
   haikugraph ask-demo "Ambiguous query" --debug
   ```

## Success Criteria ✅

- ✅ One command demonstrates entire architecture
- ✅ Easy to screen-record / demo
- ✅ Easy to reason about failures (stage-specific exit codes)
- ✅ Proves HaikuGraph is coherent system, not just components
- ✅ Clean, readable output by default
- ✅ Debug mode for technical validation
- ✅ Raw mode for programmatic access
