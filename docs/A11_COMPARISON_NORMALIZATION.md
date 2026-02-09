# A11: Comparison Normalization - First-Class Comparisons

## Overview

A11 makes comparison queries a **first-class, normalized concept** with strict output invariants. Comparison results are now structurally correct, semantically safe, and narrator-proof.

**Core Principle**: All comparison math happens BEFORE narration. The narrator NEVER computes deltas or infers trends.

## Problem Statement

**Before A11:**
- Comparison results were ad-hoc
- Narrator computed deltas from raw SQL results
- No validation of comparison invariants
- Division by zero not handled consistently
- Narrator could guess/infer direction

**Issues:**
1. Narrator doing math → unreliable narration
2. No structural validation → silent bugs
3. Ad-hoc format → hard to extend
4. Zero-division errors → crashes or wrong percentages

## Solution

**A11 introduces a normalized comparison contract:**

```python
{
  "metric": "sum_revenue",
  "current": {
    "value": 1000.0,
    "period": "this_year",
    "subquestion_id": "SQ1_current",
    "row_count": 1
  },
  "comparison": {
    "value": 800.0,
    "period": "previous_year",
    "subquestion_id": "SQ2_comparison",
    "row_count": 1
  },
  "delta": 200.0,
  "delta_pct": 25.0,
  "direction": "up"
}
```

### Strict Invariants

1. **Exactly 2 operands**: current + comparison
2. **Values from scoped subquestions only**: SQ1_current and SQ2_comparison
3. **Delta correctness**: `delta = current.value - comparison.value`
4. **Percentage correctness**: `delta_pct = null` if `comparison.value == 0`, else `(delta / comparison.value) * 100`
5. **Direction correctness**: `"up"` if `delta > 0`, `"down"` if `delta < 0`, `"flat"` if `delta == 0`

### Fail-Fast Validation

All invariants are enforced by Pydantic validators. Invalid comparisons **cannot be constructed**.

## Architecture

### 1. ComparisonResult Schema (`comparison.py`)

```python
class ComparisonOperand(BaseModel):
    value: float
    period: str
    subquestion_id: str
    row_count: int

class ComparisonResult(BaseModel):
    metric: str
    current: ComparisonOperand
    comparison: ComparisonOperand
    delta: float
    delta_pct: float | None  # null for division by zero
    direction: Literal["up", "down", "flat"]
    
    @model_validator(mode="after")
    def validate_comparison_invariants(self):
        # Validates delta, delta_pct, and direction
        ...
```

**Key Features:**
- Pydantic validation enforces all invariants
- Impossible to create invalid ComparisonResult
- Explicit zero-division handling (null vs float)

### 2. Executor Integration (`execute.py`)

```python
def execute_plan(plan: dict, db_path: Path) -> dict:
    # Execute subquestions...
    
    # A11: Extract and normalize comparison
    comparison_result = extract_comparison_from_results(plan, subquestion_results)
    
    # Include in execution result
    if comparison_result:
        result["comparison"] = comparison_result.model_dump()
    
    return result
```

**Detection Logic:**
- Checks for `SQ1_current` and `SQ2_comparison` subquestions
- Validates exactly 2 subquestions
- Extracts metric value from first numeric column
- Fails fast if structure invalid

### 3. Narrator Simplification (`narrator.py`)

**Before A11:**
```python
# Narrator computed delta from raw results
current_value = results["SQ1_current"][0]["sum_revenue"]
comp_value = results["SQ2_comparison"][0]["sum_revenue"]
delta = current_value - comp_value  # WRONG: Math in narrator!
```

**After A11:**
```python
# Narrator receives pre-computed comparison
def narrate_results(..., comparison: dict | None = None):
    if comparison:
        summary = _build_comparison_summary(comparison)
        # Narrator ONLY formats, NEVER computes
```

**Narrator is FORBIDDEN from:**
- Computing deltas
- Inferring trends
- Calculating percentages
- Comparing raw SQL outputs

## Implementation Details

### normalize_comparison()

Utility function that computes all comparison fields with strict validation:

```python
def normalize_comparison(
    metric: str,
    current_value: float,
    current_period: str,
    current_sq_id: str,
    current_row_count: int,
    comparison_value: float,
    comparison_period: str,
    comparison_sq_id: str,
    comparison_row_count: int,
) -> ComparisonResult:
    # Compute delta
    delta = current_value - comparison_value
    
    # Compute delta_pct (null if division by zero)
    if comparison_value == 0:
        delta_pct = None
    else:
        delta_pct = (delta / comparison_value) * 100
    
    # Compute direction
    if delta > epsilon:
        direction = "up"
    elif delta < -epsilon:
        direction = "down"
    else:
        direction = "flat"
    
    # Returns validated ComparisonResult
```

### extract_comparison_from_results()

Detects comparison plans and produces normalized ComparisonResult:

```python
def extract_comparison_from_results(
    plan: dict,
    subquestion_results: list[dict],
) -> ComparisonResult | None:
    # Detect comparison plan
    if not ("SQ1_current" in sq_ids and "SQ2_comparison" in sq_ids):
        return None  # Not a comparison
    
    # Validate structure
    if len(subquestions) != 2:
        raise ValueError("must have exactly 2 subquestions")
    
    # Check for failures
    if any failed:
        raise ValueError("subquestion failed")
    
    # Extract single-row results
    if len(current_rows) != 1:
        raise ValueError("must return exactly 1 row")
    
    # Extract metric values
    current_value = float(current_row[metric_col])
    comparison_value = float(comparison_row[metric_col])
    
    # Normalize and validate
    return normalize_comparison(...)
```

## Test Coverage

### Test Categories (24 tests)

1. **ComparisonOperand Validation** (3 tests)
   - Valid operand passes
   - Empty period fails
   - Negative row count fails

2. **ComparisonResult Invariants** (8 tests)
   - Valid comparison (up, down, flat)
   - Zero division requires null pct
   - Invalid delta fails
   - Invalid delta_pct fails
   - Invalid direction fails
   - Zero division with non-null pct fails

3. **normalize_comparison()** (4 tests)
   - Increase (delta > 0, direction="up")
   - Decrease (delta < 0, direction="down")
   - Flat (delta == 0, direction="flat")
   - Zero division (delta_pct = null)

4. **extract_comparison_from_results()** (6 tests)
   - Extract valid comparison
   - Non-comparison returns None
   - Wrong subquestion count fails
   - Failed subquestion fails
   - Multiple rows fails
   - Metric mismatch fails

5. **Narrator Integration** (3 tests)
   - Narrator receives normalized comparison
   - Handles zero division
   - Handles flat comparison

### Test Results

```
============================= 164 passed in 12.21s ==============================
```

- 140 existing tests (unchanged)
- 24 new A11 tests
- 100% passing

## Examples

### Example 1: Increase

**Query:** "Revenue this year vs last year"

**Normalized Result:**
```json
{
  "metric": "sum_revenue",
  "current": {"value": 30000.0, "period": "this_year", ...},
  "comparison": {"value": 25000.0, "period": "previous_year", ...},
  "delta": 5000.0,
  "delta_pct": 20.0,
  "direction": "up"
}
```

**Narration:**
```
Revenue this_year ($30,000) vs previous_year ($25,000) - up by $5,000 (20%)
```

### Example 2: Decrease

**Query:** "Sales this month vs last month"

**Normalized Result:**
```json
{
  "metric": "count_sales",
  "current": {"value": 80.0, "period": "this_month", ...},
  "comparison": {"value": 100.0, "period": "previous_month", ...},
  "delta": -20.0,
  "delta_pct": -20.0,
  "direction": "down"
}
```

**Narration:**
```
Sales this_month (80) vs previous_month (100) - down by 20 (20%)
```

### Example 3: Flat (No Change)

**Query:** "Orders this week vs last week"

**Normalized Result:**
```json
{
  "metric": "count_orders",
  "current": {"value": 500.0, "period": "this_week", ...},
  "comparison": {"value": 500.0, "period": "previous_week", ...},
  "delta": 0.0,
  "delta_pct": 0.0,
  "direction": "flat"
}
```

**Narration:**
```
Orders this_week (500) vs previous_week (500) - flat
```

### Example 4: Zero Division

**Query:** "Revenue this year vs last year" (last year = $0)

**Normalized Result:**
```json
{
  "metric": "sum_revenue",
  "current": {"value": 1000.0, "period": "this_year", ...},
  "comparison": {"value": 0.0, "period": "previous_year", ...},
  "delta": 1000.0,
  "delta_pct": null,
  "direction": "up"
}
```

**Narration:**
```
Revenue this_year ($1,000) vs previous_year ($0) - up by $1,000
```

## Failure Modes

### Failure Mode 1: Wrong Subquestion Count

```python
plan = {
    "subquestions": [
        {"id": "SQ1_current", ...},
        {"id": "SQ2_comparison", ...},
        {"id": "SQ3", ...},  # Extra subquestion!
    ]
}

# Fails with:
ValueError: Comparison plan must have exactly 2 subquestions, got 3
```

### Failure Mode 2: Multiple Rows Returned

```python
subquestion_results = [
    {
        "id": "SQ1_current",
        "preview_rows": [{"sum_revenue": 1000}, {"sum_revenue": 500}]  # Multiple!
    }
]

# Fails with:
ValueError: SQ1_current must return exactly 1 row, got 2
```

### Failure Mode 3: Metric Mismatch

```python
subquestion_results = [
    {"id": "SQ1_current", "preview_rows": [{"sum_revenue": 1000}]},
    {"id": "SQ2_comparison", "preview_rows": [{"count_orders": 100}]}  # Different metric!
]

# Fails with:
ValueError: Metric 'sum_revenue' not found in SQ2_comparison results
```

### Failure Mode 4: Invalid Invariant

```python
ComparisonResult(
    ...,
    delta=200.0,
    delta_pct=50.0,  # Wrong! Should be 25.0
    direction="up"
)

# Fails with:
ValueError: Invalid delta_pct: expected 25.00, got 50.00
```

## Benefits

### 1. **Bug Prevention**

Invariants enforced by Pydantic make comparison bugs **impossible**, not unlikely.

### 2. **Clear Separation of Concerns**

- Executor: Computes comparison math
- Narrator: Formats pre-computed values
- No overlap, no confusion

### 3. **Fail-Fast**

Invalid comparisons detected immediately, with clear error messages.

### 4. **Narrator Simplification**

Narrator no longer needs to understand comparison logic - just format what's provided.

### 5. **Type Safety**

Pydantic models provide runtime type checking and IDE autocompletion.

### 6. **Extensibility**

Easy to add new comparison types (cohort, multi-period) with same pattern.

## Design Decisions

### Why Pydantic?

- Runtime validation (not just type hints)
- Custom validators for invariants
- Automatic JSON serialization
- IDE support

### Why Epsilon for Float Comparison?

Floating-point arithmetic needs epsilon (1e-9) to handle rounding errors:

```python
epsilon = 1e-9
if delta > epsilon:
    direction = "up"
elif delta < -epsilon:
    direction = "down"
else:
    direction = "flat"
```

### Why Null for Zero Division?

- `None` (null) is semantically correct for "cannot compute"
- Avoids confusing `0.0` with "no change"
- Forces explicit handling in narrator
- Prevents silent bugs (e.g., treating null as 0%)

### Why Single Metric Column?

Comparison requires comparing **one metric**. Multiple metrics would need:
- Multiple delta/delta_pct fields
- Unclear which metric to narrate
- Complex validation

Better to require:
- Single aggregation in comparison subquestions
- Clear metric identity
- Simple validation

## Migration Guide

### Existing Code

No changes required for existing non-comparison queries.

### New Comparison Consumers

If you're consuming comparison results:

**Before A11:**
```python
results = execute_plan(plan, db_path)
current = results["subquestion_results"][0]["preview_rows"][0]
comparison = results["subquestion_results"][1]["preview_rows"][0]
delta = current["sum_revenue"] - comparison["sum_revenue"]  # Manual math
```

**After A11:**
```python
results = execute_plan(plan, db_path)
comparison = results.get("comparison")  # Normalized structure
if comparison:
    print(f"Delta: {comparison['delta']}")
    print(f"Percentage: {comparison['delta_pct']}")
    print(f"Direction: {comparison['direction']}")
```

## Files Changed

| File | Lines | Purpose |
|------|-------|---------|
| `src/haikugraph/execution/comparison.py` | +328 (new) | ComparisonResult schema + normalization |
| `src/haikugraph/execution/execute.py` | +24 | Executor integration |
| `src/haikugraph/explain/narrator.py` | +56 | Narrator update |
| `tests/test_comparison_normalization_a11.py` | +586 (new) | Comprehensive test suite |

**Total:** ~994 lines added

## Future Enhancements

### Multi-Period Comparisons

Extend to 3+ periods:

```json
{
  "metric": "sum_revenue",
  "periods": [
    {"period": "this_year", "value": 30000},
    {"period": "previous_year", "value": 25000},
    {"period": "two_years_ago", "value": 20000}
  ],
  "deltas": [5000, 5000],
  "direction": "up"
}
```

### Cohort Comparisons

Compare non-temporal cohorts:

```json
{
  "metric": "avg_revenue",
  "cohort_dimension": "product_category",
  "cohorts": [
    {"name": "electronics", "value": 500},
    {"name": "clothing", "value": 300}
  ],
  "delta": 200,
  "delta_pct": 66.67,
  "direction": "up"
}
```

### Grouped Comparisons

Compare aggregations by group:

```json
{
  "metric": "sum_revenue",
  "groups": [
    {
      "group_key": {"barber": "Alice"},
      "current": 10000,
      "comparison": 8000,
      "delta": 2000,
      "delta_pct": 25.0,
      "direction": "up"
    },
    ...
  ]
}
```

## Success Criteria ✅

All criteria met:

- ✅ Normalized comparison structure defined
- ✅ Strict invariants enforced (delta, delta_pct, direction)
- ✅ Zero-division handled (delta_pct = null)
- ✅ Executor integration complete
- ✅ Narrator consumes normalized comparison (no math)
- ✅ Comprehensive tests (24 tests, all passing)
- ✅ All existing tests pass (164/164)
- ✅ Fail-fast error messages
- ✅ Type-safe with Pydantic
- ✅ Documentation complete

---

**Status**: ✅ Complete  
**Tests**: ✅ 164/164 passing (140 existing + 24 new)  
**Date**: 2026-02-05
