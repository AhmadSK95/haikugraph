# HaikuGraph End-to-End Hardening Summary

**Date**: February 9, 2026  
**Status**: Infrastructure Complete + Initial Bug Discovery  
**Pass Rate**: 0% (20 failures out of 20 questions) → **Ready for systematic fixing**

---

## What Was Accomplished

### Phase A: Repository Discovery & Mapping ✅

Created comprehensive repository map (`REPO_MAP.md`) identifying:

**Critical Components**:
- **Planning Layer**: `src/haikugraph/planning/plan.py` (deterministic planner)
  - `detect_intent()`, `detect_entities()`, `detect_metrics()`, `detect_constraints()`
  - `build_subquestions()` - generates GROUP BY logic
  
- **Execution Layer**: `src/haikugraph/execution/execute.py` (SQL builder)
  - `_build_sql_from_subquestion()` - generates SELECT/WHERE/GROUP BY
  - `_apply_constraints()` - time filter logic
  
- **Test Infrastructure**: 18 existing test files (some coverage, needs expansion)

**Key Weak Points Identified**:
1. GROUP BY over-aggregation (scalar queries incorrectly grouped)
2. Time filtering inconsistencies ("this month" vs "last month" semantics)
3. Multi-metric detection (only detects 1 metric per type)
4. Constraint scoping issues (partially fixed)
5. Intent classification for comparisons

---

### Phase B: Test Infrastructure Development ✅

Built comprehensive hardening test harness with 4 components:

#### 1. **Question Generator** (`tests/hardening/question_generator.py`)
- Combinatorial matrix generator: Intent × Metric × Time × Breakdown × Filter
- Generates 200-5000 natural language questions
- Includes expected behavior (GROUP BY, DISTINCT, time filters, shape)

**Intents Covered**:
- `scalar_metric`: "What is total revenue?" → NO GROUP BY expected
- `scalar_count`: "How many transactions?" → NO GROUP BY expected
- `scalar_unique`: "How many unique customers?" → COUNT(DISTINCT) expected
- `scalar_avg`: "What is average ticket?" → NO GROUP BY expected
- `breakdown`: "Revenue by platform" → GROUP BY expected
- `trend`: "Revenue by month" → GROUP BY + date_trunc expected
- `comparison_scalar`: "This month vs last month" → NO GROUP BY, 2 scalars expected
- `top_k`: "Top 10 platforms" → GROUP BY + ORDER BY + LIMIT expected

**Time Windows Covered**:
- Calendar: today, yesterday, this/last week, this/last month, this/last year
- Rolling: last 7/30/90 days, last 3/6/12 months
- Specific: December, January

**Total Combinations**: ~200-500 questions depending on flags

#### 2. **CLI Test Runner** (`tests/hardening/cli_runner.py`)
- Executes questions through actual `haikugraph ask` CLI
- Captures plan JSON, SQL, execution results, errors
- Extracts metadata:
  - GROUP BY presence
  - COUNT(DISTINCT) usage
  - Row count (1 for scalar, N for grouped)
  - Column count
  - Time filter SQL

#### 3. **Oracle/Invariant Checker** (`tests/hardening/oracle.py`)
- Validates 12+ invariant rules:
  - **SCALAR_HAS_GROUP_BY**: Scalar queries must NOT have GROUP BY
  - **MISSING_GROUP_BY**: Breakdown queries MUST have GROUP BY
  - **UNIQUE_MISSING_DISTINCT**: "unique X" must use COUNT(DISTINCT)
  - **MISSING_TIME_FILTER**: Time window specified but no SQL filter
  - **WRONG_TIME_FILTER**: Filter type mismatch (calendar vs rolling)
  - **WRONG_ROW_COUNT**: Scalar queries must return 1 row
  - **WRONG_TIME_GRAIN**: date_trunc grain must match breakdown
  - **COMPARISON_WRONG_SHAPE**: Comparisons should be 2 scalars, not series

- Auto-classifies failures into buckets for systematic fixing

#### 4. **Main Test Harness** (`tests/hardening/run_matrix.py`)
- Orchestrates full pipeline: generate → execute → validate
- Generates detailed reports with per-bucket JSON files
- Provides CLI interface: `python -m tests.hardening.run_matrix`

**Output Structure**:
```
reports/hardening_TIMESTAMP/
├── summary.json                # Pass rate + failure buckets
├── results.json                # All test results
├── failures.json               # Failed tests only
└── buckets/                    # Per-bucket failure files
    ├── group_by_over_aggregation.json
    ├── time_filter_error.json
    └── ...
```

---

### Phase C: Initial Matrix Run ✅

**Test Configuration**:
- Questions: 20 (small test)
- Planner: Deterministic
- Database: `data/haikugraph.duckdb` (162,932 rows)

**Results**:
- **Total**: 20 questions
- **Passed**: 0 (0%)
- **Failed**: 20 (100%)

**Failure Distribution**:
```
group_by_over_aggregation: 20 (100%)
```

**Example Failure**:
```
Question: "What is the total revenue?"
Expected: Scalar result, NO GROUP BY
Actual: SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name
Violation: Scalar query incorrectly has GROUP BY

This returns 4 rows (one per platform) instead of 1 total!
```

**Root Cause**: The deterministic planner's `build_subquestions()` function is adding GROUP BY clauses even for scalar aggregation queries.

---

## Critical Bugs Discovered

### Bug #1: GROUP BY Over-Aggregation (100% of test failures)
**Severity**: CRITICAL  
**Impact**: User asks for single scalar, gets grouped result instead

**Examples**:
- "What is the total revenue?" → Returns 4 rows grouped by platform (wrong!)
- "How many transactions?" → Returns multiple rows instead of 1 total
- "What is the total volume this month?" → Groups by platform even though not requested

**Location**: `src/haikugraph/planning/plan.py:build_subquestions()`

**Fix Required**: Add logic to distinguish scalar vs breakdown intents:
```python
# Pseudocode fix
if intent_type == "metric" and not has_explicit_breakdown:
    # Scalar query - no GROUP BY
    group_by = []
else:
    # Breakdown query - add GROUP BY
    group_by = [breakdown_column]
```

**Affected Queries**: All scalar metric/count queries (estimated ~40% of real-world queries)

---

## Next Steps: The Fix Loop

### Immediate Next Steps (Phase D)

1. **Fix Bug #1: GROUP BY Over-Aggregation**
   - Modify `build_subquestions()` to check for breakdown keywords
   - Add intent-based GROUP BY decision logic
   - Ensure scalar queries (no "by" keyword) don't get GROUP BY

2. **Add Regression Test**
   ```python
   def test_scalar_revenue_no_group_by():
       result = run_query("What is the total revenue?")
       assert not result.has_group_by
       assert result.row_count == 1
   ```

3. **Re-run Matrix (20 questions)**
   - Verify `group_by_over_aggregation` bucket goes to 0
   - Check if new failure buckets emerge

### Subsequent Phases (E-H)

4. **Phase E: Run 200-Question Matrix**
   - Increase to 200 questions for broader coverage
   - Identify top 3 failure buckets
   - Expected new buckets:
     - `time_filter_error` (missing "this month" filters)
     - `distinct_missing` (unique queries without DISTINCT)
     - `comparison_shape_error` (comparisons return series instead of 2 scalars)

5. **Phase F: Fix Top 3 Buckets**
   - For each bucket:
     - Analyze failures
     - Patch planner/executor
     - Add regression tests
     - Re-run matrix

6. **Phase G: Time Semantics Definition**
   - Define and document time window semantics:
     - "this month" = current calendar month (2026-02-01 to 2026-02-29)
     - "last month" = previous calendar month (2026-01-01 to 2026-01-31)
     - "last 30 days" = rolling window from today - 30 days
   - Inject fixed clock for deterministic tests (use `freezegun`)
   - Implement calendar vs rolling logic in `_apply_constraints()`

7. **Phase H: Full 500-Question Matrix**
   - Run complete matrix with all dimensions
   - Target: 95%+ pass rate
   - Document remaining known limitations

8. **Phase I: Comparison Queries**
   - Fix comparison logic to return 2 scalars, not series
   - Ensure "this month vs last month" generates 2 separate queries
   - Implement delta/pct_change calculation

9. **Phase J: Multi-Metric Support**
   - Fix `detect_metrics()` to detect multiple metrics in one question
   - "count and volume by platform" should generate: COUNT + SUM, not just SUM

---

## Files Added/Modified

### New Files Created
```
REPO_MAP.md                               # Repository structure map
HARDENING_SUMMARY.md                      # This file
tests/hardening/                          # Test infrastructure
├── __init__.py
├── README.md                             # Usage documentation
├── question_generator.py                 # Combinatorial question generator
├── cli_runner.py                         # CLI test runner
├── oracle.py                             # Invariant checker
└── run_matrix.py                         # Main test harness
```

### Bug Fixes Applied
```
src/haikugraph/planning/plan.py           # Fix timestamp column selection (committed)
tests/hardening/cli_runner.py             # Fix subprocess returncode bug (committed)
```

### Test Reports Generated
```
reports/hardening_20260209_135142/        # Latest test run
├── summary.json                          # Pass rate: 0%, group_by_over_aggregation: 20
├── results.json                          # All 20 failures with SQL
├── failures.json                         # Same as results (all failed)
└── buckets/group_by_over_aggregation.json  # All 20 failures in detail
```

---

## How to Use This Infrastructure

### Quick Start
```bash
# Activate virtualenv
source .venv/bin/activate

# Run 20-question smoke test
python -m tests.hardening.run_matrix --max-questions 20

# Run 200-question comprehensive test
python -m tests.hardening.run_matrix --max-questions 200

# Check results
cat reports/hardening_*/summary.json | jq
```

### Inspect Failures
```bash
# See failure buckets
cat reports/hardening_*/summary.json | jq '.failure_buckets'

# Inspect top failure bucket
cat reports/hardening_*/buckets/group_by_over_aggregation.json | jq '.[0]'
```

### Fix Loop
1. Identify top failure bucket from summary
2. Inspect failures in bucket JSON
3. Fix code in planner/executor
4. Add regression test
5. Re-run matrix
6. Verify bucket count reduced
7. Repeat

See `tests/hardening/README.md` for detailed workflow examples.

---

## Success Metrics

### Current State
- **Pass Rate**: 0% (0/20)
- **Critical Bugs**: 1 identified (GROUP BY over-aggregation)
- **Test Coverage**: 20 questions (scalar metrics only)

### Target State (Phase H)
- **Pass Rate**: ≥95% (190+/200)
- **Critical Bugs**: 0 (all P0 issues fixed)
- **Test Coverage**: 200-500 questions (all intents, time windows, breakdowns)
- **Regression Tests**: 20+ tests covering fixed bugs

### Acceptance Criteria
1. ✅ Infrastructure complete and verified
2. ⏳ GROUP BY over-aggregation fixed (next)
3. ⏳ Time filter logic implemented correctly
4. ⏳ DISTINCT usage for unique queries
5. ⏳ Comparison queries return correct shape
6. ⏳ Multi-metric detection working
7. ⏳ Pass rate ≥95% on 200-question matrix

---

## Technical Debt & Limitations

### Known Limitations
1. **Time injection**: No fixed clock for deterministic time tests (use `freezegun`)
2. **Multi-table queries**: Limited coverage of complex joins
3. **LLM planner**: Infrastructure only tests deterministic planner so far
4. **Specific months**: "December" matches all Decembers (no year constraint)

### Future Enhancements
1. Add LLM planner testing support
2. Expand to 5000-question mega-matrix
3. Add performance benchmarks (query latency)
4. Test hybrid planner fallback behavior
5. Add data validation (result correctness, not just SQL correctness)

---

## Conclusion

The hardening infrastructure is **complete and operational**. Initial testing successfully detected a critical bug (GROUP BY over-aggregation affecting 100% of scalar queries). The system is now ready for the systematic fix loop:

**Immediate Priority**: Fix GROUP BY over-aggregation bug, add regression test, re-run 20-question matrix to verify fix.

**Next Priority**: Scale to 200 questions, identify and fix top 3 failure buckets.

**End Goal**: Achieve 95%+ pass rate on comprehensive 200-500 question matrix, ensuring HaikuGraph generates correct SQL for all common query patterns.

---

## References

- **Infrastructure**: `tests/hardening/README.md`
- **Repository Map**: `REPO_MAP.md`
- **Latest Test Report**: `reports/hardening_20260209_135142/summary.json`
- **CLI Usage**: `python -m tests.hardening.run_matrix --help`
