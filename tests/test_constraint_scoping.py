"""Tests for constraint scoping with applies_to in executor.

This module tests the A5 feature where constraints can be scoped to specific
subquestions using the applies_to field, enabling correct comparison queries.
"""

from haikugraph.execution.execute import build_sql


def test_scoped_constraint_applies_only_to_matching_subquestion():
    """Test that scoped constraints only apply to subquestions with matching ID."""
    # Plan with scoped constraint
    plan = {
        "constraints": [
            {
                "type": "time",
                "expression": "orders.created_at in previous_month",
                "applies_to": "SQ2_comparison",
            }
        ],
    }

    # Subquestion 1: Current period (should not get the constraint)
    sq1 = {
        "id": "SQ1_current",
        "tables": ["orders"],
        "columns": ["id", "created_at"],
    }

    # Subquestion 2: Comparison period (should get the constraint)
    sq2 = {
        "id": "SQ2_comparison",
        "tables": ["orders"],
        "columns": ["id", "created_at"],
    }

    # Build SQL for both
    sql1, meta1 = build_sql(sq1, plan)
    sql2, meta2 = build_sql(sq2, plan)

    # SQ1 should have 0 constraints applied
    assert len(meta1["constraints_applied"]) == 0, "SQ1 should not receive scoped constraint"
    assert "previous_month" not in sql1, "SQ1 SQL should not contain time filter"

    # SQ2 should have 1 constraint applied
    assert len(meta2["constraints_applied"]) == 1, "SQ2 should receive scoped constraint"
    assert "date_trunc" in sql2, "SQ2 SQL should contain translated time filter"


def test_unscoped_constraint_applies_to_all_subquestions():
    """Test that constraints without applies_to field apply to all subquestions."""
    # Plan with unscoped constraint
    plan = {
        "constraints": [
            {"type": "filter", "expression": "orders.status = 'completed'"}
            # No applies_to field
        ],
    }

    # Two subquestions with different IDs
    sq1 = {"id": "SQ1", "tables": ["orders"], "columns": ["id", "status"]}
    sq2 = {"id": "SQ2", "tables": ["orders"], "columns": ["id", "status"]}

    # Build SQL for both
    sql1, meta1 = build_sql(sq1, plan)
    sql2, meta2 = build_sql(sq2, plan)

    # Both should have 1 constraint applied
    assert len(meta1["constraints_applied"]) == 1, "SQ1 should receive unscoped constraint"
    assert "completed" in sql1, "SQ1 SQL should contain filter"

    assert len(meta2["constraints_applied"]) == 1, "SQ2 should receive unscoped constraint"
    assert "completed" in sql2, "SQ2 SQL should contain filter"


def test_mixed_scoped_and_unscoped_constraints():
    """Test combination of scoped and unscoped constraints."""
    # Plan with both scoped and unscoped constraints
    plan = {
        "constraints": [
            {"type": "filter", "expression": "orders.status = 'completed'"},  # Unscoped
            {
                "type": "time",
                "expression": "orders.created_at in previous_month",
                "applies_to": "SQ2",
            },  # Scoped to SQ2
        ],
    }

    sq1 = {"id": "SQ1", "tables": ["orders"], "columns": ["id", "status", "created_at"]}
    sq2 = {"id": "SQ2", "tables": ["orders"], "columns": ["id", "status", "created_at"]}

    sql1, meta1 = build_sql(sq1, plan)
    sql2, meta2 = build_sql(sq2, plan)

    # SQ1 should only get unscoped constraint (1 constraint)
    assert len(meta1["constraints_applied"]) == 1, "SQ1 should get only unscoped constraint"
    assert "completed" in sql1
    assert "previous_month" not in sql1

    # SQ2 should get both constraints (2 constraints)
    assert len(meta2["constraints_applied"]) == 2, "SQ2 should get both constraints"
    assert "completed" in sql2
    assert "date_trunc" in sql2


def test_scoped_constraint_ignored_by_non_matching_subquestion():
    """Test that subquestions with different IDs ignore scoped constraints."""
    plan = {
        "constraints": [
            {
                "type": "time",
                "expression": "users.signup_date in last_7_days",
                "applies_to": "SQ_special",
            }
        ],
    }

    # Subquestion with different ID
    sq = {"id": "SQ_normal", "tables": ["users"], "columns": ["id", "signup_date"]}

    sql, meta = build_sql(sq, plan)

    # Should have no constraints applied
    assert len(meta["constraints_applied"]) == 0
    assert "last_7_days" not in sql
    assert "interval" not in sql
