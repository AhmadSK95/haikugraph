# A6 Implementation Summary

## What Was Built
A complete POC for using **local Ollama models** with split LLM responsibilities:
- **Planner LLM** (`qwen2.5:7b-instruct`): Strict JSON Plan generation with auto-repair
- **Narrator LLM** (`llama3.1:8b-instruct`): Natural language explanations (no SQL)

## Key Features
✅ **Zero OpenAI Dependencies**: Complete runtime path uses only local Ollama  
✅ **Strict Validation**: All plans pass Pydantic `validate_plan_or_raise()`  
✅ **Auto-Repair Loop**: Planner repairs JSON/validation errors (up to HG_MAX_RETRIES)  
✅ **Scoped Constraints**: Comparison followups generate proper time constraints with `applies_to`  
✅ **Guardrails**: Narrator forbidden from outputting SQL  
✅ **Retry Logic**: Exponential backoff for transient failures  
✅ **MacBook Optimized**: Recommended 7B-8B models for 16GB RAM  

## Files Created
1. `src/haikugraph/llm/ollama_client.py` (120 lines)
2. `src/haikugraph/llm/router.py` (58 lines)
3. `src/haikugraph/planning/llm_planner.py` (210 lines)
4. `src/haikugraph/explain/narrator.py` (109 lines)
5. `tests/test_a6_ollama_split.py` (358 lines, 12 tests)
6. `A6_README.md` (documentation)
7. `A6_SUMMARY.md` (this file)

## Files Modified
1. `pyproject.toml` - Added `requests>=2.31.0` dependency
2. `src/haikugraph/cli.py` - Added `ask-a6` command (120 lines)

## Test Results
```
$ python -m pytest tests/ -q
69 passed in 0.13s
```

All tests pass including:
- 12 new A6-specific tests
- 57 existing tests (no regressions)

## Usage Example
```bash
# Setup (one-time)
export HG_LLM_PROVIDER=ollama
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b-instruct

# Run
haikugraph ask-a6 -q "What is total revenue?" --db-path ./data/haikugraph.duckdb
```

## Architecture Flow
```
User Question
    ↓
Planner LLM (ollama_chat → qwen2.5:7b)
    ↓
Strict JSON → validate_plan_or_raise()
    ↓ (if invalid)
Repair Loop (max 2 retries with error feedback)
    ↓
Validated Plan
    ↓
SQL Builder → DuckDB Execution (existing)
    ↓
Results + Metadata
    ↓
Narrator LLM (ollama_chat → llama3.1:8b)
    ↓
Human-Readable Explanation
```

## Comparison Followup Example
**Question**: "Compare revenue vs last month"

**Planner Output**:
```json
{
  "original_question": "Compare revenue vs last month",
  "subquestions": [
    {"id": "SQ1_current", "tables": ["orders"]},
    {"id": "SQ2_comparison", "tables": ["orders"]}
  ],
  "constraints": [
    {
      "type": "time",
      "expression": "previous_month",
      "applies_to": "SQ2_comparison"  ← Scoped!
    }
  ]
}
```

## Guardrails Implemented

### Planner
- JSON-only output (no markdown)
- Required fields enforced
- Unique subquestion IDs
- `applies_to` references validated
- Scoped time constraints for comparisons

### Narrator
- System prompt: "NEVER output SQL"
- Results-only explanation
- No plan modification
- Called AFTER execution only

### Validation
- Pydantic schema enforcement
- Non-empty tables list
- Valid constraint types
- Group-by + aggregations consistency

## Performance Characteristics
- **Planner inference**: ~2-5 seconds (7B model)
- **Narrator inference**: ~3-6 seconds (8B model)
- **Total overhead**: ~5-11 seconds per query
- **Memory usage**: ~8GB for both models loaded

## Error Handling
1. **Ollama not running**: Clear error message with instructions
2. **Model not found**: Suggests `ollama pull` command
3. **JSON parse errors**: Auto-repair with retry
4. **Validation errors**: Auto-repair with specific error feedback
5. **Max retries exceeded**: Detailed error output

## Environment Variables
```bash
HG_LLM_PROVIDER=ollama              # Required
HG_OLLAMA_BASE_URL=http://localhost:11434  # Default
HG_PLANNER_MODEL=qwen2.5:7b-instruct        # Default
HG_NARRATOR_MODEL=llama3.1:8b-instruct      # Default
HG_PLANNER_TEMPERATURE=0                    # Default
HG_NARRATOR_TEMPERATURE=0.4                 # Default
HG_MAX_RETRIES=2                            # Default
```

## Success Criteria (All Met ✅)
1. ✅ Use ONLY local Ollama models (no OpenAI)
2. ✅ Split responsibilities (Planner + Narrator)
3. ✅ PLANNER emits/repairs STRICT JSON Plan
4. ✅ Plan passes `validate_plan_or_raise()`
5. ✅ NARRATOR turns results into explanation (NO SQL)
6. ✅ Comparison followups have scoped time constraints
7. ✅ Auto-repair loop for JSON/validation errors
8. ✅ All tests pass (`pytest tests/ -q`)
9. ✅ Clean CLI integration
10. ✅ MacBook-friendly (7B-8B models)

## Next Steps (Optional)
- [ ] Add streaming support for narrator output
- [ ] Implement followup classification in LLM planner
- [ ] Add caching for repeated schema introspection
- [ ] Support for multi-turn conversations
- [ ] Metrics collection (inference time, retry counts)

## Documentation
See `A6_README.md` for:
- Setup instructions
- Model recommendations
- Troubleshooting guide
- Architecture details
- Testing guide
