# A6 POC: Local LLMs via Ollama (Planner + Narrator)

## Overview
A6 implements a split-LLM architecture using ONLY local Ollama models on MacBook. No OpenAI dependencies in runtime path.

**Architecture:**
- **Planner LLM**: Emits/repairs strict JSON Plan that passes Pydantic validation
- **Narrator LLM**: Converts results into human-readable explanations (NO SQL generation)

## Recommended Models (MacBook-friendly)

### Planner (structured output, reliable JSON)
- **Primary**: `qwen2.5:7b-instruct` - Good JSON discipline
- **Fallback**: `mistral:7b-instruct` or `qwen2.5:3b-instruct` (smaller Macs)

### Narrator (natural language explanations)
- **Primary**: `llama3.1:8b-instruct` - Good explanations
- **Fallback**: `mistral:7b-instruct`

## Setup

### 1. Install and Start Ollama
```bash
# macOS: Install Ollama app or use CLI
# Start Ollama service (if using app, it starts automatically)
ollama serve  # CLI method

# Pull recommended models
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b-instruct
```

### 2. Configure Environment Variables
```bash
export HG_LLM_PROVIDER=ollama
export HG_OLLAMA_BASE_URL=http://localhost:11434  # Default
export HG_PLANNER_MODEL=qwen2.5:7b-instruct       # Default
export HG_NARRATOR_MODEL=llama3.1:8b-instruct     # Default
export HG_PLANNER_TEMPERATURE=0                    # Default
export HG_NARRATOR_TEMPERATURE=0.4                 # Default
export HG_MAX_RETRIES=2                            # Default
```

### 3. Install Dependencies
```bash
pip install -e .
```

## Usage

### CLI Command: `haikugraph ask-a6`
```bash
# Basic usage
haikugraph ask-a6 --question "What is the total revenue?" --db-path ./data/haikugraph.duckdb

# Example with custom output
haikugraph ask-a6 -q "Show me top 10 customers by revenue" --db-path ./data/haikugraph.duckdb --out ./data/plan_a6.json
```

### What Happens:
1. **Planner LLM** generates Plan JSON (with auto-repair if needed)
2. Plan is validated against schema (raises error if invalid after retries)
3. SQL queries are built and executed (existing execution engine)
4. **Narrator LLM** explains results in natural language
5. Both plan and results are saved to disk

### Comparison Follow-ups
For comparison queries (e.g., "compare vs last month"), A6 automatically:
- Creates `SQ1_current` and `SQ2_comparison` subquestions
- Adds scoped time constraint: `{"type": "time", "expression": "previous_month", "applies_to": "SQ2_comparison"}`

## Architecture Details

### 1. Ollama Client (`src/haikugraph/llm/ollama_client.py`)
- POST requests to `{HG_OLLAMA_BASE_URL}/api/chat`
- Retry logic for transient failures (connection reset, 5xx errors)
- Exponential backoff: 0.5s, 1s, 2s

### 2. LLM Router (`src/haikugraph/llm/router.py`)
- Dispatches to appropriate model based on role (`planner` or `narrator`)
- Reads model and temperature from environment variables

### 3. Planner (`src/haikugraph/planning/llm_planner.py`)
- **Strict JSON-only prompts** (no markdown, no commentary)
- **Auto-repair loop** (max `HG_MAX_RETRIES` attempts):
  1. Call planner model
  2. Parse JSON (if fails → repair prompt)
  3. Validate against schema (if fails → repair prompt with errors)
  4. Return validated plan or raise ValueError
- **Comparison context**: Adds special instructions for comparison followups

### 4. Narrator (`src/haikugraph/explain/narrator.py`)
- Called AFTER execution (never before)
- Prompt explicitly forbids SQL output
- Formats results and metadata into human-readable text
- Structure:
  1. Answer (1-2 lines)
  2. Comparison delta (if applicable)
  3. Filters applied (bullet list)
  4. Suggested follow-up question

### 5. CLI Integration (`src/haikugraph/cli.py`)
- `ask-a6` command wires planner → execution → narrator
- Displays plan summary, executes queries, shows narrative explanation
- Error handling for Ollama connection issues

## Testing

### Run A6 Tests
```bash
python -m pytest tests/test_a6_ollama_split.py -v
```

### Test Coverage
- ✅ Planner generates valid plans
- ✅ Planner repairs JSON parse errors
- ✅ Planner repairs validation errors
- ✅ Planner fails after max retries
- ✅ Comparison followups create scoped constraints
- ✅ Unique subquestion IDs enforced
- ✅ Narrator called after execution
- ✅ Narrator never outputs SQL
- ✅ Narrator handles comparison deltas
- ✅ Planner before narrator order
- ✅ No OpenAI dependencies
- ✅ Constraint applies_to validation

## Guardrails

### Planner Guardrails
1. **Strict JSON**: No markdown, no commentary
2. **Required fields**: `original_question`, `subquestions` (non-empty)
3. **Unique IDs**: All subquestion IDs must be unique
4. **Scoped constraints**: `applies_to` must reference valid subquestion ID
5. **Time constraints**: Comparison followups MUST have scoped time constraint

### Narrator Guardrails
1. **No SQL output**: System prompt explicitly forbids SQL
2. **Results-only**: Can only explain provided results/meta (no planning)
3. **No plan modification**: Cannot change the plan

### Validation
- All plans MUST pass `validate_plan_or_raise()` from `schema.py`
- Pydantic models enforce:
  - Non-empty subquestion tables
  - Valid aggregation functions
  - Constraint types (time/filter)
  - Applies_to references

## Environment-Specific Notes

### MacBook Memory Considerations
- **8GB RAM**: Use 3B models (`qwen2.5:3b-instruct`)
- **16GB RAM**: Use 7B-8B models (recommended setup)
- **32GB+ RAM**: Can experiment with larger models

### Model Download Sizes
- `qwen2.5:3b-instruct`: ~2GB
- `qwen2.5:7b-instruct`: ~5GB
- `llama3.1:8b-instruct`: ~5GB
- `mistral:7b-instruct`: ~4GB

## Troubleshooting

### Ollama Not Running
```
❌ Connection Error: Cannot connect to Ollama at http://localhost:11434
```
**Solution**: Start Ollama app or run `ollama serve`

### Model Not Found
```
❌ Ollama API error (404): model 'qwen2.5:7b-instruct' not found
```
**Solution**: `ollama pull qwen2.5:7b-instruct`

### Plan Validation Fails
```
❌ Plan validation failed after 2 retries
```
**Solution**: 
- Check model quality (try different planner model)
- Increase `HG_MAX_RETRIES`
- Review validation errors in output

### Slow Inference
**Solution**:
- Use smaller models (3B instead of 7B)
- Reduce `HG_MAX_RETRIES` to fail faster
- Ensure no other heavy processes running

## Files Created/Modified

### New Files
- `src/haikugraph/llm/ollama_client.py` - Ollama HTTP client
- `src/haikugraph/llm/router.py` - LLM dispatcher
- `src/haikugraph/planning/llm_planner.py` - Planner with repair loop
- `src/haikugraph/explain/narrator.py` - Results narrator
- `tests/test_a6_ollama_split.py` - A6 test suite
- `A6_README.md` - This file

### Modified Files
- `pyproject.toml` - Added `requests` dependency
- `src/haikugraph/cli.py` - Added `ask-a6` command

## Success Criteria (Met ✅)
- ✅ `python -m pytest tests/ -q` passes
- ✅ Comparison follow-ups create scoped time constraints
- ✅ No OpenAI dependencies in runtime path
- ✅ Planner repairs invalid JSON/validation errors
- ✅ Narrator explains results without SQL
- ✅ Clean CLI output with narrative explanations
