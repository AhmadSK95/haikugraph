# A4: Interactive Ambiguity Resolution for HaikuGraph

## Overview

A4 adds interactive ambiguity resolution to HaikuGraph. When plans contain unresolved or low-confidence ambiguities, users can interactively clarify their intent before execution. This human-in-the-loop approach ensures plans are executed with high confidence without requiring full plan regeneration.

## What Was Built

### 1. Core Module: `src/haikugraph/planning/ambiguity.py`

#### Key Functions

##### `get_unresolved_ambiguities(plan, *, confidence_threshold=0.7)`
Identifies ambiguities that need user resolution.

An ambiguity is unresolved if:
- `recommended` is `None`, OR
- `confidence < confidence_threshold`

```python
unresolved = get_unresolved_ambiguities(plan)
# Returns list of ambiguity dicts that need resolution
```

##### `ambiguity_to_question(ambiguity)`
Converts an ambiguity into a human-readable question.

```python
question_data = ambiguity_to_question({
    "issue": "Entity 'customer' found in multiple tables",
    "options": ["customers", "orders"]
})
# Returns:
# {
#     "issue": "Entity 'customer' found in multiple tables",
#     "question": "Which table should be used for 'customer'?",
#     "options": ["customers", "orders"],
#     "type": "single_choice"
# }
```

**Smart Question Generation:**
- Detects common patterns (entity, column, join path, time period)
- Generates user-friendly questions without SQL jargon
- Falls back to generic question format for unknown patterns

##### `apply_user_resolution(plan, issue, chosen)`
Applies a user's resolution to a plan.

```python
updated_plan = apply_user_resolution(
    plan, 
    "Entity 'customer' found in multiple tables",
    "customers"
)
# Sets recommended="customers", confidence=1.0
# Returns new plan (does not modify original)
```

##### `validate_no_unresolved_ambiguities(plan, *, confidence_threshold=0.7)`
Validates that a plan has no unresolved ambiguities.

```python
validate_no_unresolved_ambiguities(plan)
# Raises ValueError if unresolved ambiguities exist
```

### 2. CLI Integration

#### Enhanced Command: `haikugraph ask-llm --interactive`

```bash
haikugraph ask-llm -q "How many customers signed up?" \
  --db-path ./data/haikugraph.duckdb \
  --interactive \
  --execute
```

**Flow:**
1. Generate plan using LLM (A3)
2. Check for unresolved ambiguities
3. For each unresolved ambiguity:
   - Display human-readable question
   - Show numbered options
   - Accept user input
   - Apply resolution to plan
4. Save updated plan with resolutions
5. Execute plan (if `--execute` flag present)

**Example Interactive Session:**
```
🔍 Found 2 unresolved ambiguity/ambiguities

======================================================================

[1/2] Which table should be used for 'customer'?

Options:
  1. customers
  2. orders

Your choice (number): 1
✓ Selected: customers

[2/2] Which time period should be used?

Options:
  1. last_30_days
  2. last_7_days

Your choice (number): 2
✓ Selected: last_7_days

======================================================================

✅ All ambiguities resolved!
```

### 3. Execution Guardrail

Added validation in `execute_plan()` to prevent execution with unresolved ambiguities:

```python
# In src/haikugraph/execution/execute.py
from haikugraph.planning.ambiguity import validate_no_unresolved_ambiguities

def execute_plan(plan: dict, db_path: Path) -> dict:
    validate_plan_or_raise(plan)
    validate_no_unresolved_ambiguities(plan)  # NEW: Blocks execution if unresolved
    # ... rest of execution
```

**Error Message:**
```
ValueError: Unresolved ambiguities remain. Run with --interactive to resolve them.
Unresolved issues:
  - Entity 'customer' found in multiple tables
  - Multiple tables contain column name
```

### 4. Tests

**File:** `tests/test_ambiguity_resolution.py`

**Coverage (20 tests):**
- ✅ Detection of unresolved ambiguities (no recommendation, low confidence, thresholds)
- ✅ Question generation for various patterns
- ✅ User resolution application (success, errors, edge cases)
- ✅ Validation of plans (success, failures)
- ✅ Sequential resolution workflow
- ✅ Multiple ambiguities handling

All 20 tests pass.

## Key Features

### ✅ Human-in-the-Loop (Not Chatty)
- Only prompts when necessary (unresolved ambiguities)
- Clear, numbered options for easy selection
- No LLM calls during resolution (deterministic)

### ✅ Incremental Resolution
- Does NOT regenerate full plan
- Only updates specific ambiguities
- Preserves all other plan components

### ✅ Confidence-Based Auto-Apply
- High confidence (≥ 0.7) → auto-applied
- Low confidence (< 0.7) → asks user
- Configurable threshold

### ✅ Smart Question Generation
Recognizes common patterns:
- Entity ambiguity: "Which table should be used for 'customer'?"
- Column ambiguity: "Which column should be used?"
- Join path: "Which join path should be used?"
- Time period: "Which time period should be used?"
- Fallback: Generic question for unknown patterns

### ✅ Execution Safety
- Blocks execution if unresolved ambiguities exist
- Clear error messages guide user to use `--interactive`
- Prevents silent failures from ambiguous queries

## Usage Examples

### Basic Interactive Resolution

```bash
# Generate plan with interactive resolution
haikugraph ask-llm -q "Show customer orders" \
  --interactive

# Generate, resolve, and execute in one command
haikugraph ask-llm -q "Count active users" \
  --interactive \
  --execute
```

### Programmatic Usage

```python
from pathlib import Path
from haikugraph.llm.plan_generator import generate_plan
from haikugraph.planning.ambiguity import (
    get_unresolved_ambiguities,
    ambiguity_to_question,
    apply_user_resolution
)
from haikugraph.execution import execute_plan

# Generate plan
plan = generate_plan("How many customers?", Path("./data/db.duckdb"))

# Check for unresolved ambiguities
unresolved = get_unresolved_ambiguities(plan)

if unresolved:
    for amb in unresolved:
        question = ambiguity_to_question(amb)
        print(question["question"])
        for i, opt in enumerate(question["options"], 1):
            print(f"  {i}. {opt}")
        
        # Get user input (simplified)
        choice = int(input("Choice: ")) - 1
        chosen = question["options"][choice]
        
        # Apply resolution
        plan = apply_user_resolution(plan, question["issue"], chosen)

# Execute resolved plan
result = execute_plan(plan, Path("./data/db.duckdb"))
```

### Custom Confidence Threshold

```python
# Use stricter threshold (0.8)
unresolved = get_unresolved_ambiguities(plan, confidence_threshold=0.8)
# More ambiguities will require user input

# Use looser threshold (0.5)
unresolved = get_unresolved_ambiguities(plan, confidence_threshold=0.5)
# Fewer ambiguities will require user input
```

## Design Principles

### 1. Human-in-the-Loop, Not Chatty
- Only prompt when necessary
- Clear, actionable questions
- No verbose explanations

### 2. No LLM Calls
- All resolution logic is deterministic
- Pattern matching for question generation
- Fast and predictable

### 3. Incremental Updates
- Never regenerate full plan
- Only update specific ambiguities
- Preserve LLM-generated content

### 4. CLI-First UX
- Simple numbered selection
- Clear feedback after each choice
- Progress indicators (e.g., [1/2])

### 5. Testable and Deterministic
- Pure functions for all core logic
- No side effects
- Easy to mock and test

## Resolution Patterns

### Pattern 1: Entity Ambiguity
**Issue:** `"Entity 'customer' found in multiple tables"`
**Question:** `"Which table should be used for 'customer'?"`

### Pattern 2: Column Ambiguity
**Issue:** `"Multiple tables contain column name"`
**Question:** `"Which table should be used for column name?"`

### Pattern 3: Join Path Ambiguity
**Issue:** `"Multiple possible joins between tables"`
**Question:** `"Which join path should be used?"`

### Pattern 4: Time Period Ambiguity
**Issue:** `"Time constraint ambiguous"`
**Question:** `"Which time period should be used?"`

### Pattern 5: Already a Question
**Issue:** `"Which metric should be calculated?"`
**Question:** `"Which metric should be calculated?"` (unchanged)

### Pattern 6: Fallback
**Issue:** `"Some custom ambiguity"`
**Question:** `"Which option should be used for: Some custom ambiguity?"`

## Error Handling

### Unresolved Ambiguities at Execution
```python
# When trying to execute plan with unresolved ambiguities
try:
    execute_plan(plan, db_path)
except ValueError as e:
    # Error: Unresolved ambiguities remain. Run with --interactive to resolve them.
    # Unresolved issues:
    #   - Entity 'customer' found in multiple tables
```

### Invalid User Choice
```python
# When user chooses option not in list
try:
    apply_user_resolution(plan, issue, "invalid_option")
except ValueError as e:
    # Error: Chosen option 'invalid_option' not in available options: ['a', 'b']
```

### Issue Not Found
```python
# When trying to resolve non-existent issue
try:
    apply_user_resolution(plan, "Nonexistent issue", "a")
except ValueError as e:
    # Error: Ambiguity with issue 'Nonexistent issue' not found in plan
```

## File Changes Summary

### New Files
1. `src/haikugraph/planning/ambiguity.py` - Core resolution functions
2. `tests/test_ambiguity_resolution.py` - Comprehensive tests (20 tests)
3. `A4_README.md` - This documentation

### Modified Files
1. `src/haikugraph/cli.py` - Added `--interactive` flag and resolution flow
2. `src/haikugraph/execution/execute.py` - Added execution guardrail

### Code Quality
- ✅ All code passes `ruff format`
- ✅ All code passes `ruff check`
- ✅ All 20 new tests pass
- ✅ All 24 existing tests still pass (no breaking changes)

## Implementation Compliance

### ✅ HARD RULES Satisfied

1. ✅ **No SQL execution logic changed** - Only added validation before execution
2. ✅ **No full plan regeneration** - Only incremental updates to ambiguities
3. ✅ **Incremental and deterministic** - All resolution logic is pure functions
4. ✅ **Auto-apply high confidence** - Ambiguities with confidence ≥ threshold are not prompted
5. ✅ **Ask user for low confidence** - Ambiguities with confidence < threshold require user input

### ✅ Deliverables Completed

1. ✅ Ambiguity classifier (`get_unresolved_ambiguities`)
2. ✅ Question generator (`ambiguity_to_question`)
3. ✅ Plan patch function (`apply_user_resolution`)
4. ✅ CLI integration (`--interactive` flag)
5. ✅ Execution guardrail (`validate_no_unresolved_ambiguities`)
6. ✅ Comprehensive tests (20 tests)

## Development Workflow

### Run Tests
```bash
# Run ambiguity resolution tests
pytest tests/test_ambiguity_resolution.py -v

# Run all tests
pytest tests/
```

### Format and Lint
```bash
# Format code
ruff format src/haikugraph/planning/ambiguity.py

# Check linting
ruff check src/haikugraph/planning/ambiguity.py
```

### Interactive Testing
```bash
# Test interactive flow (requires database)
haikugraph ask-llm -q "Your question" --interactive
```

## Integration with A3

A4 seamlessly integrates with A3 (LLM Plan Generator):

1. **A3 generates plan** with potential ambiguities
2. **A4 detects unresolved** ambiguities (recommended=None or low confidence)
3. **A4 asks user** to clarify
4. **A4 updates plan** with user's choices
5. **Plan executes** with high confidence

**No LLM calls needed during resolution!**

## Limitations and Future Work

### Current Limitations
1. Only supports single-choice questions (no multi-select)
2. Pattern matching is basic (could be enhanced)
3. CLI-only (no GUI or web interface)
4. No undo/redo for resolution choices
5. No explanation of why ambiguity exists

### Future Enhancements
1. Multi-select for ambiguities with multiple valid options
2. More sophisticated pattern matching (ML-based)
3. Web UI for interactive resolution
4. Undo/redo functionality
5. Explanations from LLM about ambiguity sources
6. Save/load resolution preferences for repeated questions
7. Batch resolution for related ambiguities

## Troubleshooting

### Issue: "Unresolved ambiguities remain" error
**Cause:** Plan has ambiguities with `recommended=None` or low confidence

**Solution:**
```bash
# Run with --interactive to resolve
haikugraph ask-llm -q "Your question" --interactive --execute
```

### Issue: Interactive prompt not working
**Possible causes:**
1. Not running in interactive terminal
2. Using output redirection (e.g., `> output.txt`)

**Solution:**
- Run in normal terminal without redirection
- Or resolve ambiguities programmatically

### Issue: Want to change confidence threshold
**Solution:**
```python
# In code
unresolved = get_unresolved_ambiguities(plan, confidence_threshold=0.8)

# Currently no CLI flag - can be added if needed
```

## Examples

### Example 1: Entity Ambiguity
```bash
$ haikugraph ask-llm -q "Show customer details" --interactive

🔍 Found 1 unresolved ambiguity/ambiguities

======================================================================

[1/1] Which table should be used for 'customer'?

Options:
  1. customers
  2. customer_orders
  3. customer_profiles

Your choice (number): 1
✓ Selected: customers

======================================================================

✅ All ambiguities resolved!
```

### Example 2: Multiple Ambiguities
```bash
$ haikugraph ask-llm -q "Total sales by region last month" --interactive

🔍 Found 2 unresolved ambiguity/ambiguities

======================================================================

[1/2] Which table should be used for 'sales'?

Options:
  1. sales
  2. orders

Your choice (number): 1
✓ Selected: sales

[2/2] Which time period should be used?

Options:
  1. last_30_days
  2. last_month_calendar

Your choice (number): 2
✓ Selected: last_month_calendar

======================================================================

✅ All ambiguities resolved!
```

### Example 3: High Confidence (Auto-Applied)
```bash
$ haikugraph ask-llm -q "Count users" --interactive

✅ No unresolved ambiguities found.
```
(Plan had ambiguities but all with confidence ≥ 0.7, so auto-applied)

## Comparison: A3 vs A4

| Feature | A3 (Plan Generator) | A4 (Ambiguity Resolver) |
|---------|-------------------|------------------------|
| **Purpose** | Generate plans from questions | Resolve ambiguities in plans |
| **Input** | Natural language question | Plan with ambiguities |
| **Output** | Plan JSON | Updated plan JSON |
| **LLM Usage** | Yes (multiple calls) | No (deterministic) |
| **User Input** | Question only | Interactive choices |
| **Speed** | Slower (LLM calls) | Fast (no LLM) |
| **Deterministic** | No (LLM variance) | Yes (pure functions) |

## Summary

A4 provides a lightweight, deterministic, human-in-the-loop solution for resolving ambiguities in HaikuGraph plans. By avoiding full plan regeneration and LLM calls, it offers a fast, predictable, and user-friendly way to clarify intent before execution.

**Key Benefits:**
- ✅ Fast (no LLM calls)
- ✅ Deterministic (testable)
- ✅ User-friendly (clear questions)
- ✅ Safe (execution guardrail)
- ✅ Incremental (no plan regeneration)

For issues or questions, check test cases in `tests/test_ambiguity_resolution.py` for usage examples.
