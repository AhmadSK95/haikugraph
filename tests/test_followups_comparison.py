"""Integration tests for A5 comparison follow-ups.

This module tests the full A5 chain for comparison queries:
classify_followup → patch_plan → validate_plan_or_raise → build_sql
"""

from haikugraph.execution.execute import build_sql
from haikugraph.planning.followups import classify_followup, patch_plan
from haikugraph.planning.schema import validate_plan_or_raise


def test_comparison_followup_full_chain():
    """Test complete flow: classify, patch, validate, execute comparison follow-up."""
    # Initial plan
    prev_plan = {
        "original_question": "How many orders?",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["orders"],
                "columns": ["id", "created_at", "status"],
            }
        ],
    }

    prev_question = "How many orders?"
    followup_question = "compare to previous month"

    # Step 1: Classify follow-up
    classification = classify_followup(followup_question, prev_question, prev_plan)

    assert classification["is_followup"] is True, "Should be classified as follow-up"
    assert classification["type"] == "comparison", "Should be comparison type"
    assert classification["confidence"] >= 0.8, "Should have high confidence"

    # Step 2: Patch plan
    patched_plan = patch_plan(prev_plan, classification, followup_question)

    # Verify plan structure
    assert "subquestions" in patched_plan
    assert len(patched_plan["subquestions"]) == 2, "Should have 2 subquestions"

    # Check subquestion IDs
    sq_ids = [sq["id"] for sq in patched_plan["subquestions"]]
    assert "SQ1_current" in sq_ids, "Should have current period subquestion"
    assert "SQ2_comparison" in sq_ids, "Should have comparison period subquestion"

    # Check constraints
    assert "constraints" in patched_plan
    assert len(patched_plan["constraints"]) > 0, "Should have at least one constraint"

    # Find scoped constraint
    scoped_constraints = [
        c for c in patched_plan["constraints"] if c.get("applies_to") == "SQ2_comparison"
    ]
    assert len(scoped_constraints) == 1, "Should have one constraint scoped to SQ2"
    assert scoped_constraints[0]["type"] == "time", "Scoped constraint should be time type"
    assert "previous_month" in scoped_constraints[0]["expression"]

    # Step 3: Validate schema
    validate_plan_or_raise(patched_plan)  # Should not raise

    # Step 4: Build SQL for both subquestions (select by ID, not order)
    sq_by_id = {sq["id"]: sq for sq in patched_plan["subquestions"]}
    sq1 = sq_by_id["SQ1_current"]
    sq2 = sq_by_id["SQ2_comparison"]

    sql1, meta1 = build_sql(sq1, patched_plan)
    sql2, meta2 = build_sql(sq2, patched_plan)

    # Verify SQL for SQ1 (current) - should NOT have time constraint
    assert len(meta1["constraints_applied"]) == 0, "Current period should have no constraints"
    # Verify no time constraints in metadata (more robust than checking SQL text)
    time_constraints_sq1 = [c for c in meta1["constraints_applied"] if c.get("type") == "time"]
    assert len(time_constraints_sq1) == 0, "Current period should have no time constraints"

    # Verify SQL for SQ2 (comparison) - SHOULD have time constraint
    assert len(meta2["constraints_applied"]) == 1, "Comparison period should have 1 constraint"
    # Verify time constraint is present in metadata
    time_constraints_sq2 = [c for c in meta2["constraints_applied"] if c.get("type") == "time"]
    assert len(time_constraints_sq2) == 1, "Comparison period should have exactly 1 time constraint"
    # Check the plan's original constraint (before SQL translation)
    # Filter by both type AND applies_to to be resilient to future default time filters
    plan_time_constraints = [
        c
        for c in patched_plan["constraints"]
        if c.get("type") == "time" and c.get("applies_to") == "SQ2_comparison"
    ]
    assert len(plan_time_constraints) == 1
    assert (
        plan_time_constraints[0].get("applies_to") == "SQ2_comparison"
    ), "Time constraint must be scoped to SQ2_comparison"
    assert (
        "previous_month" in plan_time_constraints[0]["expression"]
    ), "Original constraint should reference previous_month"


def test_comparison_to_previous_year():
    """Test comparison to previous year."""
    prev_plan = {
        "original_question": "Total sales?",
        "subquestions": [{"id": "SQ1", "tables": ["sales"], "columns": ["amount", "date"]}],
    }

    classification = classify_followup("compare to previous year", "Total sales?", prev_plan)
    patched_plan = patch_plan(prev_plan, classification, "compare to previous year")

    # Check constraint has previous_year
    constraints = patched_plan["constraints"]
    time_constraint = next(c for c in constraints if c["type"] == "time")
    assert "previous_year" in time_constraint["expression"]

    # Validate
    validate_plan_or_raise(patched_plan)

    # Build SQL and verify (select by ID, not order)
    sq_by_id = {sq["id"]: sq for sq in patched_plan["subquestions"]}
    sq2 = sq_by_id["SQ2_comparison"]
    sql2, meta2 = build_sql(sq2, patched_plan)
    # Verify time constraint is applied via metadata (more robust)
    time_constraints = [c for c in meta2["constraints_applied"] if c.get("type") == "time"]
    assert len(time_constraints) == 1, "Should have one time constraint"
    # Check the plan's original constraint (before SQL translation)
    # Filter by both type AND applies_to to be resilient to future default time filters
    plan_time_constraints = [
        c
        for c in patched_plan["constraints"]
        if c.get("type") == "time" and c.get("applies_to") == "SQ2_comparison"
    ]
    assert len(plan_time_constraints) == 1
    assert (
        plan_time_constraints[0].get("applies_to") == "SQ2_comparison"
    ), "Time constraint must be scoped to SQ2_comparison"
    assert (
        "previous_year" in plan_time_constraints[0]["expression"]
    ), "Original constraint should reference previous_year"


def test_comparison_vs_pattern():
    """Test 'vs' pattern for comparison."""
    prev_plan = {
        "original_question": "Revenue",
        "subquestions": [{"id": "SQ1", "tables": ["revenue"], "columns": ["amount"]}],
    }

    classification = classify_followup("vs previous week", "Revenue", prev_plan)

    assert classification["is_followup"] is True
    assert classification["type"] == "comparison"

    patched_plan = patch_plan(prev_plan, classification, "vs previous week")
    assert len(patched_plan["subquestions"]) == 2

    # Check for previous_week in constraint
    time_constraint = next(c for c in patched_plan["constraints"] if c.get("type") == "time")
    assert "previous_week" in time_constraint["expression"]


def test_comparison_merged_question():
    """Test that original_question is properly merged for comparison."""
    prev_plan = {
        "original_question": "How many users signed up?",
        "subquestions": [{"id": "SQ1", "tables": ["users"], "columns": ["id", "created_at"]}],
    }

    classification = classify_followup(
        "compare to previous month", "How many users signed up?", prev_plan
    )
    patched_plan = patch_plan(prev_plan, classification, "compare to previous month")

    # Check merged question
    assert (
        "comparison" in patched_plan["original_question"].lower()
        or "previous" in patched_plan["original_question"].lower()
    )


def test_comparison_with_existing_constraint():
    """Test comparison when plan already has constraints."""
    prev_plan = {
        "original_question": "Orders for product X?",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["orders"],
                "columns": ["id", "product_id", "created_at"],
            }
        ],
        "constraints": [
            {"type": "filter", "expression": "orders.product_id = 'X'"}
            # Unscoped filter - should apply to both
        ],
    }

    classification = classify_followup("compare to last month", "Orders for product X?", prev_plan)
    patched_plan = patch_plan(prev_plan, classification, "compare to last month")

    # Should have 2 constraints: 1 filter (unscoped) + 1 time (scoped)
    assert len(patched_plan["constraints"]) == 2

    # Build SQL for both (select by ID, not order)
    sq_by_id = {sq["id"]: sq for sq in patched_plan["subquestions"]}
    sq1 = sq_by_id["SQ1_current"]
    sq2 = sq_by_id["SQ2_comparison"]

    sql1, meta1 = build_sql(sq1, patched_plan)
    sql2, meta2 = build_sql(sq2, patched_plan)

    # Both should have the product filter
    assert "product_id" in sql1
    assert "product_id" in sql2

    # Only SQ2 should have time filter
    assert len(meta1["constraints_applied"]) == 1  # Just filter
    assert len(meta2["constraints_applied"]) == 2  # Filter + time
