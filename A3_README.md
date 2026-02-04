# A3: LLM-Powered Plan Generator for HaikuGraph

## Overview

A3 adds LLM-powered plan generation to HaikuGraph. Given a natural-language question and a DuckDB database, it generates a validated Plan JSON object that conforms to the canonical Pydantic schema.

## What Was Built

### 1. Core Modules

#### `src/haikugraph/llm/client.py`
- **Purpose**: Thin wrapper for OpenAI API calls
- **Key Functions**:
  - `call_openai()`: Makes API calls with temperature=0 for deterministic output
  - `parse_json_response()`: Handles JSON parsing with markdown code block support

#### `src/haikugraph/llm/plan_generator.py`
- **Purpose**: Main plan generation logic
- **Key Functions**:
  - `generate_plan(question, db_path, *, model=None)`: Main entry point
    1. Introspects database schema
    2. Calls LLM to generate Plan JSON
    3. Validates against schema
    4. Auto-repairs invalid plans (max 2 retries)
    5. Returns validated plan dict
  - `introspect_schema(db_path)`: Reads table/column metadata from DuckDB
  - `create_initial_plan_prompt()`: Generates structured prompt for LLM
  - `create_repair_prompt()`: Generates repair prompt with validation errors

### 2. CLI Integration

#### New Command: `haikugraph ask-llm`
```bash
haikugraph ask-llm --question "How many users signed up last month?" \
  --db-path ./data/haikugraph.duckdb \
  --out ./data/plan_llm.json \
  --execute
```

**Options**:
- `--question, -q`: Natural language question (required)
- `--db-path`: Path to DuckDB database (default: ./data/haikugraph.duckdb)
- `--out`: Output plan file (default: ./data/plan_llm.json)
- `--model`: LLM model to use (default: gpt-4o-mini)
- `--execute`: Execute the plan immediately after generation

#### Enhanced Command: `haikugraph ask`
The existing deterministic `ask` command now supports:
- `--execute`: Execute plan immediately after generation
- `--db-path`: Database path for execution

### 3. Tests

**File**: `tests/test_plan_generator_validation.py`

**Coverage**:
- ✅ Successful plan generation on first attempt
- ✅ Plan repair after validation errors
- ✅ Failure after max retries
- ✅ JSON parse error handling
- ✅ Markdown code block handling
- ✅ Database not found error
- ✅ Prompt generation validation

All 8 tests pass with mocked LLM responses.

## Key Features

### ✅ Strict JSON Validation
- Output must conform to `Plan` schema in `src/haikugraph/planning/schema.py`
- Automatic validation using `validate_plan_or_raise()`
- Required fields enforced:
  - `original_question` (string)
  - `subquestions` (at least 1, each with non-empty `tables` list)

### ✅ Auto-Repair Logic
- Max 2 repair attempts on validation failure
- Provides detailed error messages to LLM for correction
- Handles both JSON parse errors and schema validation errors

### ✅ Schema Introspection
- Reads table names and column types from DuckDB
- Samples rows (optional, minimal) for type hints
- Formats schema as readable text for LLM prompt

### ✅ Forward-Compatible Schema
- Uses Pydantic with `extra="allow"` for unknown fields
- Optional fields for future extensibility:
  - `intent`, `entities_detected`, `metrics_requested`
  - `join_paths`, `constraints`, `ambiguities`

## Usage Examples

### Basic Usage

```bash
# Set API key
export OPENAI_API_KEY='your-api-key-here'

# Generate a plan
haikugraph ask-llm -q "What's the total revenue by month?" \
  --db-path ./data/haikugraph.duckdb

# Generate and execute immediately
haikugraph ask-llm -q "Count active users" \
  --db-path ./data/haikugraph.duckdb \
  --execute

# Use a specific model
haikugraph ask-llm -q "Show top 10 products" \
  --model gpt-4o \
  --execute
```

### Programmatic Usage

```python
from pathlib import Path
from haikugraph.llm.plan_generator import generate_plan
from haikugraph.execution import execute_plan

# Generate plan
plan = generate_plan(
    question="How many orders were placed last week?",
    db_path=Path("./data/haikugraph.duckdb"),
    model="gpt-4o-mini"  # optional
)

# Execute plan
result = execute_plan(plan, Path("./data/haikugraph.duckdb"))
print(result["final_summary"])
```

## Environment Setup

### Requirements

1. **Install openai package** (not in default dependencies):
   ```bash
   pip install openai
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. **Verify installation**:
   ```bash
   haikugraph --version
   haikugraph ask-llm --help
   ```

## Plan JSON Structure

The LLM generates plans conforming to this structure:

```json
{
  "original_question": "How many users signed up last month?",
  "subquestions": [
    {
      "id": "SQ1",
      "description": "Count users with signup_date in last month",
      "tables": ["users"],
      "columns": ["id", "signup_date"],
      "group_by": null,
      "aggregations": [{"agg": "count", "col": "id"}]
    }
  ],
  "intent": {
    "type": "metric",
    "confidence": 0.9
  },
  "constraints": [
    {
      "type": "time",
      "expression": "users.signup_date in last_30_days"
    }
  ],
  "entities_detected": [
    {
      "name": "users",
      "mapped_to": ["users.id"],
      "confidence": 1.0
    }
  ],
  "metrics_requested": [
    {
      "name": "user_count",
      "mapped_columns": ["users.id"],
      "aggregation": "count",
      "confidence": 0.9
    }
  ],
  "plan_confidence": 0.85
}
```

## Implementation Guarantees

### ✅ No Breaking Changes
- Existing `haikugraph ask` command behavior unchanged
- Existing `haikugraph run` command works with both deterministic and LLM plans
- All existing tests pass

### ✅ Deterministic LLM Output
- Temperature set to 0.0 for consistency
- Validation ensures output conforms to schema

### ✅ Error Handling
- Clear error messages when API key missing
- Graceful failure with retry mechanism
- Database validation before schema introspection

## File Changes Summary

### New Files
1. `src/haikugraph/llm/client.py` - OpenAI API wrapper
2. `src/haikugraph/llm/plan_generator.py` - Plan generation logic
3. `tests/test_plan_generator_validation.py` - Comprehensive tests
4. `A3_README.md` - This documentation

### Modified Files
1. `src/haikugraph/cli.py` - Added `ask-llm` command and `--execute` flag

### Code Quality
- ✅ All code passes `ruff format`
- ✅ All code passes `ruff check`
- ✅ All 8 new tests pass
- ✅ Existing tests unaffected

## Development Workflow

### Run Tests
```bash
# Run plan generator tests
pytest tests/test_plan_generator_validation.py -v

# Run all tests
pytest tests/
```

### Format and Lint
```bash
# Format code
ruff format src/haikugraph/llm/ tests/test_plan_generator_validation.py

# Check linting
ruff check src/haikugraph/llm/ src/haikugraph/cli.py
```

## Limitations and Future Work

### Current Limitations
1. Requires OpenAI API key (not included in base dependencies)
2. Only supports OpenAI models (gpt-4o-mini, gpt-4o, etc.)
3. No streaming support for long-running LLM calls
4. Limited context window (schema introspection is minimal)

### Future Enhancements
1. Support for other LLM providers (Anthropic, local models)
2. Semantic search over database content for better context
3. Multi-turn conversation for ambiguity resolution
4. Caching of schema introspection results
5. Support for database-specific optimizations

## Troubleshooting

### Error: "OPENAI_API_KEY environment variable not set"
**Solution**: Set the API key before running:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Error: "openai package not installed"
**Solution**: Install the package:
```bash
pip install openai
```

### Error: "Plan validation failed after 2 retries"
**Possible causes**:
1. LLM unable to understand schema structure
2. Database schema is too complex
3. Question is ambiguous or underspecified

**Solutions**:
- Simplify your question
- Try with a different model (e.g., gpt-4o instead of gpt-4o-mini)
- Check that database contains relevant tables/columns

### LLM generates incorrect plans
**Solutions**:
1. Use more specific questions
2. Ensure database schema is well-structured
3. Try different models
4. Check that schema introspection is accurate

## Compliance with Requirements

### ✅ HARD RULES Satisfied
1. ✅ No execution logic changed (only added planning layer)
2. ✅ LLM outputs strict JSON (validated with schema)
3. ✅ Automatic repair on validation failure (max 2 retries)
4. ✅ All required plan fields present (original_question, subquestions)
5. ✅ Schema is forward-compatible (extra keys allowed)

### ✅ Deliverables Completed
1. ✅ `src/haikugraph/llm/plan_generator.py`
2. ✅ CLI wiring in `src/haikugraph/cli.py`
3. ✅ Tests in `tests/test_plan_generator_validation.py`
4. ✅ Code passes ruff format + check
5. ✅ All tests pass

## Contact and Support

For issues or questions:
1. Check this README
2. Review test cases in `tests/test_plan_generator_validation.py`
3. Examine prompt templates in `src/haikugraph/llm/plan_generator.py`
