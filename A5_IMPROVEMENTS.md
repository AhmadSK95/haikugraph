# A5 Tiny Improvements - Implementation Notes

## Overview
Following the initial A5 hardening, we implemented additional robustness improvements based on best practices for validator determinism and test resilience.

---

## Improvements Made

### 1. Stricter Test Assertions for Error Messages ✅

**Problem**: Original assertion was too permissive
```python
# BEFORE (could pass with incomplete error message)
assert "SQ1" in error_str or "SQ2" in error_str
```

**Solution**: Made assertion more deterministic and complete
```python
# AFTER (ensures error message contains all expected parts)
assert "Valid IDs:" in error_str
assert "SQ1" in error_str and "SQ2" in error_str
```

**Benefits**:
- Catches formatting changes that might omit parts of the error message
- Ensures error message structure remains consistent
- More explicit about what we're testing

**File**: `tests/test_plan_schema.py` (line 391-393)

---

### 2. Future-Proof SQL Assertions ✅

**Problem**: Tests checked SQL text directly, which is brittle
```python
# BEFORE (breaks if SQL formatter changes)
assert "WHERE" not in sql1.upper()
assert "date_trunc" in sql2 and "month" in sql2
```

**Issue Identified**: If we later add default safety filters (tenant_id, soft deletes), tests would fail even though feature is correct.

**Solution**: Check constraint metadata instead of SQL text
```python
# AFTER (robust to SQL generation changes)
# Check metadata, not SQL text
time_constraints_sq1 = [c for c in meta1["constraints_applied"] if c.get("type") == "time"]
assert len(time_constraints_sq1) == 0, "Current period should have no time constraints"

# Check original plan constraint (before SQL translation)
plan_time_constraints = [c for c in patched_plan["constraints"] if c.get("type") == "time"]
assert len(plan_time_constraints) == 1
assert "previous_month" in plan_time_constraints[0]["expression"]
```

**Benefits**:
- Resilient to SQL formatter changes
- Won't break when adding tenant scoping or soft deletes
- Tests intent, not implementation details
- Checks both metadata tracking AND original constraint

**Files Modified**:
- `tests/test_followups_comparison.py::test_comparison_followup_full_chain()` (lines 71-87)
- `tests/test_followups_comparison.py::test_comparison_to_previous_year()` (lines 112-120)

---

### 3. Added Validator for Duplicate Subquestion IDs ✅

**Problem**: Duplicate subquestion IDs would cause the `applies_to` validator to work incorrectly (using a set silently deduplicates)

**Solution**: Added explicit validator before constraint checking
```python
@model_validator(mode="after")
def validate_subquestion_ids_unique(self) -> "Plan":
    """Validate that subquestion IDs are unique."""
    sq_ids = [sq.id for sq in self.subquestions]
    if len(sq_ids) != len(set(sq_ids)):
        # Find duplicates
        seen = set()
        duplicates = set()
        for sq_id in sq_ids:
            if sq_id in seen:
                duplicates.add(sq_id)
            seen.add(sq_id)
        raise ValueError(
            f"Duplicate subquestion IDs found: {sorted(duplicates)}. "
            f"All subquestion IDs must be unique."
        )
    return self
```

**Benefits**:
- Catches buggy plans early
- Clear error message identifies which IDs are duplicated
- Prevents subtle bugs where applies_to matching would be unreliable
- Validator runs BEFORE applies_to validator (correct order)

**File**: `src/haikugraph/planning/schema.py` (lines 199-215)

---

### 4. Added Test for Duplicate ID Validation ✅

**Test Coverage**: New test ensures duplicate detection works
```python
def test_duplicate_subquestion_ids_fails():
    """Test that duplicate subquestion IDs fail validation."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
            {"id": "SQ2", "tables": ["customers"]},
            {"id": "SQ1", "tables": ["products"]},  # Duplicate SQ1
        ],
    }
    
    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    # Check error message
    error_str = " ".join(errors)
    assert "Duplicate subquestion IDs" in error_str
    assert "SQ1" in error_str
    assert "unique" in error_str.lower()
```

**File**: `tests/test_plan_schema.py` (lines 449-469)

---

## Impact Summary

### Test Resilience

| Aspect | Before | After |
|--------|--------|-------|
| SQL assertion | Checks exact SQL strings | Checks constraint metadata |
| Error message checks | Permissive OR condition | Strict AND conditions |
| Duplicate ID detection | None | Explicit validator |

### Validator Robustness

- ✅ **Deterministic error messages**: All validators use `sorted()` for consistency
- ✅ **Ordering matters**: Duplicate check runs before applies_to check
- ✅ **Clear error messages**: All errors include helpful context

### Future-Proofing

**Protected Against**:
- SQL formatter changes
- Adding default WHERE clauses (tenant scoping, soft deletes)
- Equivalent query rewrites
- Plan generation bugs (duplicate IDs)

**Will NOT Break When**:
- SQL generation is optimized
- Multi-tenant enforcement is added
- Soft delete filtering is implemented
- Query optimizer changes output format

---

## Test Results

### Before Improvements
- 56 tests passing

### After Improvements
- **57 tests passing** (+1 new test for duplicate IDs)
- 0 failures
- 0 errors
- All lint checks pass

### New Test Added
- ✓ `test_duplicate_subquestion_ids_fails` - Validates duplicate ID detection

---

## Files Modified

### Source Code (1 file)
1. **src/haikugraph/planning/schema.py** (+16 lines)
   - Added `validate_subquestion_ids_unique` validator
   - Lines 199-215

### Tests (2 files)
1. **tests/test_plan_schema.py** (+22 lines)
   - Stricter assertion for applies_to error message (line 392-393)
   - New test for duplicate subquestion IDs (lines 449-469)

2. **tests/test_followups_comparison.py** (~30 lines changed)
   - Replace SQL text checks with metadata checks
   - Check plan constraints instead of translated SQL
   - Two test functions updated

**Total**: ~68 lines changed/added across 3 files

---

## Code Quality

### Linting
```bash
$ ruff format
1 file reformatted, 2 files left unchanged

$ ruff check
All checks passed!
```

### Tests
```bash
$ python -m pytest tests/ -v
============================== 57 passed in 0.08s ===============================
```

---

## Design Decisions

### 1. Why Check Metadata Instead of SQL?
**Decision**: Use `meta["constraints_applied"]` instead of parsing SQL  
**Rationale**:
- SQL text can change for many valid reasons
- Metadata tracks what constraints were actually applied
- More stable API surface for testing
- Easier to debug when tests fail

### 2. Why Add Duplicate ID Validator?
**Decision**: Explicit validator for unique subquestion IDs  
**Rationale**:
- Using `set(sq_ids)` in applies_to validator silently masks duplicates
- Duplicate IDs are a serious bug, not a warning
- Error message helps developers fix the issue quickly
- Validator ordering matters - check uniqueness first

### 3. Why Check Both Plan AND Metadata Constraints?
**Decision**: Assert on both `patched_plan["constraints"]` and `meta["constraints_applied"]`  
**Rationale**:
- Plan constraints show original symbolic expression (previous_month)
- Metadata shows translated SQL expression (date_trunc...)
- Checking both ensures full pipeline works correctly
- Plan constraint is more stable for testing

---

## Backward Compatibility

### 100% Backward Compatible ✅

- All existing tests still pass
- No breaking changes to API
- No changes to runtime behavior for valid plans
- Only rejects plans that were already buggy (duplicate IDs)

---

## Lessons Learned

### Test Brittleness Sources
1. **SQL text matching** - Breaks on formatting changes
2. **Permissive assertions** - Pass even when incomplete
3. **Order assumptions** - Fragile when list order changes

### Validator Best Practices
1. **Deterministic output** - Always sort collections in error messages
2. **Validation order** - Check preconditions before dependent validations
3. **Clear error messages** - Include actual values and valid options
4. **Fail fast** - Validate at schema level, not execution time

### Future Safeguards
1. **Test metadata, not output** - More stable test API
2. **Test intent, not implementation** - Survives refactoring
3. **Explicit validators** - Don't rely on implicit set deduplication

---

## Summary

These tiny improvements make the codebase more robust without changing any runtime behavior:

✅ **Tests are more resilient** to implementation changes  
✅ **Validators are more deterministic** and catch more bugs  
✅ **Error messages are more helpful** for debugging  
✅ **Future-proofed** against common maintenance tasks  

All improvements follow the principle of **testing intent rather than implementation**, making the codebase easier to maintain and evolve.
