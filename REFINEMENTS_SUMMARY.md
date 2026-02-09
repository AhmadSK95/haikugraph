# HaikuGraph Refinements Summary

## Overview
This document summarizes the refinements made to fix VARCHAR timestamp handling and ensure comparison queries produce scalar results by default (not time-series).

## Changes Made

### 1. VARCHAR Timestamp Handling ✅

**Problem**: DuckDB's `date_trunc()` requires DATE/TIMESTAMP types, but time columns may be stored as VARCHAR. This caused queries with time bucketing or time constraints to fail on VARCHAR timestamp columns.

**Solution**: Implemented a robust timestamp expression utility that:
- Introspects column types via DuckDB's information schema
- For VARCHAR/TEXT columns: wraps with `TRY_CAST(col AS TIMESTAMP)`
- For TIMESTAMP/DATE columns: uses column as-is
- Falls back to `TRY_CAST` if schema introspection fails

**Files Changed**:
- `src/haikugraph/execution/execute.py`:
  - Added `get_timestamp_expression()` utility (lines 167-209)
  - Updated `build_sql()` to accept optional `conn` parameter
  - Updated `execute_subquestion()` to pass connection to `build_sql()`
  - Updated time bucketing in SELECT clause (lines 254-261)
  - Updated time bucketing in GROUP BY clause (lines 384-390)
  - Updated `translate_time_constraint()` to accept `conn` and use utility (lines 490-517)

**Test Coverage** (12 new tests):
- `tests/test_varchar_timestamps_and_scalar_comparisons.py`:
  - `TestVarcharTimestampHandling`: 4 tests for VARCHAR vs TIMESTAMP columns
  - Full end-to-end tests with real DuckDB databases

**Examples**:
```python
# Before (FAILS for VARCHAR):
date_trunc('month', "orders"."created_at")

# After (WORKS for VARCHAR):
date_trunc('month', TRY_CAST("orders"."created_at" AS TIMESTAMP))

# After (OPTIMAL for TIMESTAMP):
date_trunc('month', "orders"."created_at")
```

### 2. Scalar Comparison Queries by Default ✅

**Problem**: Comparison queries like "revenue this month vs last month" were incorrectly generating monthly time-series with GROUP BY, when users expected two scalar totals for side-by-side comparison.

**Solution**: Updated planner prompts to explicitly distinguish:
- **Scalar comparisons** (default): "X this month vs last month" → TWO scalar values, NO group_by
- **Time-series comparisons** (explicit): "monthly X this year vs last year" → monthly GROUP BY

**Files Changed**:
- `src/haikugraph/planning/llm_planner.py`:
  - Updated main prompt rule #6 (lines 46-54) with scalar-first guidance and examples
  - Updated comparison context (lines 168-193) with explicit scalar example

**Prompt Rules Added**:
```
6. For comparison queries, create exactly TWO subquestions:
   - SQ1_current (current period/cohort)
   - SQ2_comparison (comparison period/cohort)
   - DEFAULT: NO GROUP BY (produces two scalar totals for comparison)
   - ONLY add group_by when explicitly asked for time-series ("monthly trend", "by month", "over time")
   - Examples:
     * "revenue this month vs last month" → TWO scalars (no group_by)
     * "this year vs last year revenue" → TWO scalars (no group_by)
     * "monthly revenue this year vs last year" → monthly group_by (time-series comparison)
```

**Test Coverage** (8 new tests):
- `tests/test_varchar_timestamps_and_scalar_comparisons.py`:
  - `TestScalarComparisonQueries`: 5 tests for scalar vs time-series behavior
  - `TestComparisonTimeScoping`: 3 tests for symmetric time constraint enforcement

**Examples**:
```json
// "Revenue this month vs last month" (SCALAR - no group_by):
{
  "subquestions": [
    {"id": "SQ1_current", "tables": ["orders"], "aggregations": [{"agg": "sum", "col": "revenue"}]},
    {"id": "SQ2_comparison", "tables": ["orders"], "aggregations": [{"agg": "sum", "col": "revenue"}]}
  ],
  "constraints": [
    {"type": "time", "expression": "orders.created_at in this_month", "applies_to": "SQ1_current"},
    {"type": "time", "expression": "orders.created_at in previous_month", "applies_to": "SQ2_comparison"}
  ]
}

// "Monthly revenue this year vs last year" (TIME-SERIES - with group_by):
{
  "subquestions": [
    {
      "id": "SQ1_current",
      "tables": ["orders"],
      "group_by": [{"type": "time_bucket", "grain": "month", "col": "created_at"}],
      "aggregations": [{"agg": "sum", "col": "revenue"}]
    }
  ]
}
```

### 3. Maintained Features ✅

**Symmetric Time Scoping** (from previous fix):
- All comparison queries still enforce scoped time constraints
- Validation ensures both `SQ1_current` and `SQ2_comparison` have `applies_to`
- Tests confirm this remains working

**DISTINCT Counts** (from previous fix):
- `COUNT(DISTINCT col)` generation remains correct
- Schema validation still prevents SQL keyword hacks
- Tests confirm this remains working

## Test Results

**Total Tests**: 191 (179 existing + 12 new)
**Status**: ✅ All passing
**Runtime**: ~12.65 seconds

**New Test Files**:
- `tests/test_varchar_timestamps_and_scalar_comparisons.py` (12 tests)

**Breakdown**:
- VARCHAR timestamp handling: 4 tests
- Scalar comparison behavior: 5 tests
- Comparison time scoping: 3 tests

## Impact Summary

### Query Types Fixed

1. **"Monthly unique customers"**
   - Before: Failed with VARCHAR timestamps
   - After: Works with automatic TRY_CAST

2. **"This month vs last month revenue"**
   - Before: Generated monthly time-series (GROUP BY month)
   - After: Generates two scalar totals (no GROUP BY)

3. **"This year vs last year revenue"**
   - Before: May have generated yearly time-series
   - After: Generates two scalar totals (no GROUP BY)

4. **"Monthly revenue this year vs last year"**
   - Before: Worked (if TIMESTAMP), but query pattern unclear
   - After: Explicitly generates monthly GROUP BY for time-series comparison

### Backward Compatibility

✅ **Fully backward compatible**:
- Existing TIMESTAMP columns work exactly as before (no TRY_CAST overhead)
- VARCHAR columns now work where they previously failed
- All 179 existing tests still pass
- No breaking changes to schema or API

### Performance

- **VARCHAR timestamps**: Minimal overhead from `TRY_CAST` (only when needed)
- **TIMESTAMP columns**: Zero overhead (used as-is)
- **Schema introspection**: Cached by DuckDB, minimal impact

## Demo Commands

### Test VARCHAR Timestamp Handling
```bash
# Run VARCHAR timestamp tests
pytest tests/test_varchar_timestamps_and_scalar_comparisons.py::TestVarcharTimestampHandling -v
```

### Test Scalar Comparison Behavior
```bash
# Run scalar comparison tests
pytest tests/test_varchar_timestamps_and_scalar_comparisons.py::TestScalarComparisonQueries -v
```

### Run All Tests
```bash
# Verify all 191 tests pass
pytest tests/ -v
```

### Integration Test Examples

Create a test database with VARCHAR timestamps:
```python
import duckdb

conn = duckdb.connect("test.db")
conn.execute("""
    CREATE TABLE orders (
        id INTEGER,
        created_at VARCHAR,  -- Note: VARCHAR, not TIMESTAMP
        customer_id VARCHAR,
        revenue DOUBLE
    )
""")

conn.execute("""
    INSERT INTO orders VALUES
    (1, '2024-01-15', 'C1', 100.0),
    (2, '2024-02-20', 'C2', 200.0),
    (3, '2024-01-25', 'C1', 150.0)
""")
```

Then test queries:
- "Monthly revenue" → Should work with automatic TRY_CAST
- "Revenue this month vs last month" → Should produce 2 scalar values
- "Monthly unique customers" → Should produce time-series with GROUP BY

## Key Takeaways

### For Users
1. ✅ VARCHAR timestamp columns now work automatically
2. ✅ Comparison queries produce intuitive scalar results by default
3. ✅ Explicit time-series comparisons still work when requested

### For Developers
1. Always pass `conn` to `build_sql()` for VARCHAR handling
2. Planner now distinguishes scalar vs time-series comparisons
3. Test coverage includes real DuckDB integration tests

### For Future Work
1. Consider adding `TRY_STRPTIME` for non-ISO date formats
2. Add planner examples for time-series comparison patterns
3. Consider UI hints distinguishing scalar vs time-series results

## Related Documentation
- Previous fixes: `FIXES_SUMMARY.md` (DISTINCT and time bucketing)
- Quick reference: `QUICK_REFERENCE.md` (usage guide)
- Architecture: `ARCHITECTURE.md` (system design)
