"""Tests for plan schema validation."""

import pytest

from haikugraph.planning.schema import (
    validate_plan,
    validate_plan_or_raise,
    validate_plan_with_warnings,
)


def test_valid_plan_passes():
    """Test that a valid plan passes validation."""
    # Based on _demo_apply_resolutions test_plan structure
    valid_plan = {
        "original_question": "Test question",
        "ambiguities": [
            {
                "issue": "Entity 'customer' found in multiple tables",
                "recommended": "test_1_1",
                "options": ["test_1_1", "test_2_1", "test_4_1"],
            },
            {
                "issue": "Multiple tables contain payment_amount",
                "recommended": "test_2_1",
                "options": ["test_1_1", "test_2_1"],
            },
        ],
        "entities_detected": [
            {
                "name": "customer",
                "mapped_to": [
                    "test_1_1.customer_id",
                    "test_2_1.customer_id",
                    "test_4_1.customer_id",
                ],
            }
        ],
        "metrics_requested": [
            {
                "name": "sum_payment_amount",
                "mapped_columns": ["test_1_1.payment_amount", "test_2_1.payment_amount"],
            }
        ],
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1", "test_2_1"],
                "columns": ["customer_id", "payment_amount"],
                "aggregations": [{"agg": "sum", "col": "payment_amount"}],
            },
            {
                "id": "SQ2",
                "tables": ["test_3_1"],
                "columns": ["transaction_id"],
            },
        ],
        "join_paths": [],
        "constraints": [],
        "intent": {"type": "metric", "confidence": 0.8},
        "plan_confidence": 0.75,
    }

    is_valid, errors = validate_plan(valid_plan)
    assert is_valid, f"Expected valid plan to pass, got errors: {errors}"
    assert len(errors) == 0

    # Should not raise
    validate_plan_or_raise(valid_plan)


def test_missing_original_question_fails():
    """Test that missing original_question fails validation."""
    invalid_plan = {
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("original_question" in err for err in errors)

    with pytest.raises(ValueError, match="Invalid plan"):
        validate_plan_or_raise(invalid_plan)


def test_missing_subquestions_fails():
    """Test that missing subquestions fails validation."""
    invalid_plan = {
        "original_question": "Test question",
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("subquestions" in err for err in errors)

    with pytest.raises(ValueError, match="Invalid plan"):
        validate_plan_or_raise(invalid_plan)


def test_empty_subquestions_fails():
    """Test that empty subquestions list fails validation."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("at least 1" in err.lower() for err in errors)


def test_subquestion_missing_tables_fails():
    """Test that subquestion without tables fails validation."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "columns": ["customer_id"],
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("tables" in err for err in errors)


def test_subquestion_empty_tables_fails():
    """Test that subquestion with empty tables list fails validation."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": [],
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("at least 1" in err.lower() for err in errors)


def test_ambiguity_recommended_not_in_options_fails():
    """Test that ambiguity with recommended not in options fails."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
            }
        ],
        "ambiguities": [
            {
                "issue": "Test ambiguity",
                "recommended": "test_3_1",  # Not in options
                "options": ["test_1_1", "test_2_1"],
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("not in options" in err for err in errors)


def test_group_by_without_aggregations_fails():
    """Test that group_by with empty aggregations fails."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
                "group_by": ["customer_id"],
                "aggregations": [],  # Empty when group_by present
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("aggregations empty" in err for err in errors)


def test_group_by_with_aggregations_passes():
    """Test that group_by with aggregations passes."""
    valid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
                "group_by": ["customer_id"],
                "aggregations": [{"agg": "sum", "col": "amount"}],
            }
        ],
    }

    is_valid, errors = validate_plan(valid_plan)
    assert is_valid, f"Expected valid, got errors: {errors}"


def test_unknown_top_level_key_warns_not_fails():
    """Test that unknown top-level keys produce warnings, not errors."""
    plan_with_unknown_key = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
            }
        ],
        "unknown_future_field": "some value",
    }

    is_valid, errors, warnings = validate_plan_with_warnings(plan_with_unknown_key)
    assert is_valid
    assert len(errors) == 0
    assert any("unknown_future_field" in warn.lower() for warn in warnings)


def test_unknown_constraint_type_warns():
    """Test that unknown constraint types produce warnings."""
    plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
            }
        ],
        "constraints": [
            {"type": "custom_type", "expression": "some_expr"},
        ],
    }

    is_valid, errors, warnings = validate_plan_with_warnings(plan)
    assert is_valid
    assert len(errors) == 0
    assert any("constraint type" in warn.lower() for warn in warnings)


def test_empty_optional_list_warns():
    """Test that empty optional lists produce warnings."""
    plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
            }
        ],
        "ambiguities": [],
        "constraints": [],
    }

    is_valid, errors, warnings = validate_plan_with_warnings(plan)
    assert is_valid
    assert len(errors) == 0
    assert any("empty" in warn.lower() for warn in warnings)
    assert len(warnings) >= 2  # At least ambiguities and constraints


def test_join_path_validation():
    """Test join path validation."""
    valid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1", "test_2_1"],
            }
        ],
        "join_paths": [
            {
                "from": "test_1_1",
                "to": "test_2_1",
                "via": ["customer_id"],
                "confidence": 0.9,
            }
        ],
    }

    is_valid, errors = validate_plan(valid_plan)
    assert is_valid, f"Expected valid, got errors: {errors}"


def test_join_path_missing_via_fails():
    """Test that join path without via fails."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
            }
        ],
        "join_paths": [
            {
                "from": "test_1_1",
                "to": "test_2_1",
                # Missing via
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid
    assert any("via" in err for err in errors)


def test_minimal_valid_plan():
    """Test the absolute minimum valid plan."""
    minimal_plan = {
        "original_question": "Q",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["t1"],
            }
        ],
    }

    is_valid, errors = validate_plan(minimal_plan)
    assert is_valid, f"Expected minimal plan to pass, got errors: {errors}"


def test_complex_nested_structure():
    """Test validation of complex nested structures."""
    complex_plan = {
        "original_question": "Complex question",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1"],
                "columns": ["col1", "col2"],
                "group_by": ["col1"],
                "aggregations": [
                    {"agg": "sum", "col": "col2"},
                    {"agg": "avg", "col": "col3"},
                ],
            }
        ],
        "ambiguities": [
            {
                "issue": "Multiple matches",
                "recommended": None,  # None is allowed
                "options": ["opt1", "opt2"],
            }
        ],
        "constraints": [
            {"type": "time", "expression": "last_30_days"},
            {"type": "filter", "expression": "status='active'"},
        ],
    }

    is_valid, errors = validate_plan(complex_plan)
    assert is_valid, f"Expected complex plan to pass, got errors: {errors}"


def test_constraint_applies_to_invalid_subquestion_fails():
    """Test that constraint with invalid applies_to fails validation."""
    invalid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
            {"id": "SQ2", "tables": ["customers"]},
        ],
        "constraints": [
            {
                "type": "time",
                "expression": "orders.created_at in last_month",
                "applies_to": "SQ_DOES_NOT_EXIST",  # Invalid ID
            }
        ],
    }

    is_valid, errors = validate_plan(invalid_plan)
    assert not is_valid, "Plan with invalid applies_to should fail"
    # Check error message contains the invalid ID and valid IDs
    error_str = " ".join(errors)
    assert "SQ_DOES_NOT_EXIST" in error_str
    assert "SQ1" in error_str or "SQ2" in error_str

    with pytest.raises(ValueError, match="applies_to.*SQ_DOES_NOT_EXIST"):
        validate_plan_or_raise(invalid_plan)


def test_constraint_applies_to_valid_subquestion_passes():
    """Test that constraint with valid applies_to passes validation."""
    valid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {"id": "SQ1_current", "tables": ["orders"]},
            {"id": "SQ2_comparison", "tables": ["orders"]},
        ],
        "constraints": [
            {
                "type": "time",
                "expression": "orders.created_at in previous_month",
                "applies_to": "SQ2_comparison",  # Valid ID
            }
        ],
    }

    is_valid, errors = validate_plan(valid_plan)
    assert is_valid, f"Plan with valid applies_to should pass, got errors: {errors}"
    assert len(errors) == 0

    # Should not raise
    validate_plan_or_raise(valid_plan)


def test_constraint_without_applies_to_passes():
    """Test that constraint without applies_to (unscoped) passes validation."""
    valid_plan = {
        "original_question": "Test question",
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
            {"id": "SQ2", "tables": ["customers"]},
        ],
        "constraints": [
            {
                "type": "filter",
                "expression": "orders.status = 'completed'",
                # No applies_to - should apply to all
            }
        ],
    }

    is_valid, errors = validate_plan(valid_plan)
    assert is_valid, f"Plan with unscoped constraint should pass, got errors: {errors}"
    assert len(errors) == 0

    # Should not raise
    validate_plan_or_raise(valid_plan)


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise basic execution
    pytest.main([__file__, "-v"])
