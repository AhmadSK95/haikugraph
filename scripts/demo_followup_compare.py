#!/usr/bin/env python3
"""Demo script for A5 comparison follow-ups.

This demonstrates the full A5 chain:
1. Start with an initial plan
2. Ask a comparison follow-up
3. Show how the plan is patched
4. Execute both subquestions with scoped constraints
"""

import json
from haikugraph.execution.execute import build_sql
from haikugraph.planning.followups import classify_followup, patch_plan
from haikugraph.planning.schema import validate_plan_or_raise


def demo_comparison_followup():
    """Demonstrate comparison follow-up with constraint scoping."""
    print("=" * 80)
    print("A5 Comparison Follow-up Demo")
    print("=" * 80)

    # Initial plan: "How many orders?"
    initial_plan = {
        "original_question": "How many orders?",
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["orders"],
                "columns": ["id", "created_at", "status"],
                "aggregations": [{"agg": "count", "col": "id"}],
            }
        ],
    }

    print("\n1. INITIAL PLAN")
    print("-" * 80)
    print(f"Question: {initial_plan['original_question']}")
    print(f"Subquestions: {len(initial_plan['subquestions'])}")
    print(json.dumps(initial_plan, indent=2))

    # Follow-up question
    followup_q = "compare to previous month"
    prev_q = initial_plan["original_question"]

    print("\n2. FOLLOW-UP QUESTION")
    print("-" * 80)
    print(f"User asks: '{followup_q}'")

    # Classify
    classification = classify_followup(followup_q, prev_q, initial_plan)
    print("\n3. CLASSIFICATION")
    print("-" * 80)
    print(f"Is follow-up: {classification['is_followup']}")
    print(f"Type: {classification['type']}")
    print(f"Confidence: {classification['confidence']:.2f}")

    # Patch plan
    patched = patch_plan(initial_plan, classification, followup_q)

    print("\n4. PATCHED PLAN")
    print("-" * 80)
    print(f"New question: {patched['original_question']}")
    print(f"Subquestions: {len(patched['subquestions'])}")
    print(f"  - {patched['subquestions'][0]['id']}: current period")
    print(f"  - {patched['subquestions'][1]['id']}: comparison period")
    print(f"\nConstraints: {len(patched.get('constraints', []))}")
    for i, c in enumerate(patched.get("constraints", []), 1):
        applies = c.get("applies_to", "all")
        print(f"  {i}. {c['type']}: {c['expression']} (applies_to: {applies})")

    # Validate
    print("\n5. SCHEMA VALIDATION")
    print("-" * 80)
    try:
        validate_plan_or_raise(patched)
        print("✓ Plan is schema-valid")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return

    # Build SQL for both subquestions
    print("\n6. SQL GENERATION")
    print("-" * 80)

    sq1 = patched["subquestions"][0]
    sq2 = patched["subquestions"][1]

    print(f"\nSubquestion 1: {sq1['id']} (current period)")
    sql1, meta1 = build_sql(sq1, patched)
    print(f"Constraints applied: {len(meta1['constraints_applied'])}")
    print(f"SQL:\n{sql1}\n")

    print(f"\nSubquestion 2: {sq2['id']} (comparison period)")
    sql2, meta2 = build_sql(sq2, patched)
    print(f"Constraints applied: {len(meta2['constraints_applied'])}")
    print(f"SQL:\n{sql2}\n")

    # Summary
    print("\n7. SUMMARY")
    print("-" * 80)
    print(f"✓ Follow-up detected: {classification['type']}")
    print(f"✓ Plan patched: {len(initial_plan['subquestions'])} → {len(patched['subquestions'])} subquestions")
    print(f"✓ Constraints scoped: SQ1 has {len(meta1['constraints_applied'])}, SQ2 has {len(meta2['constraints_applied'])}")
    print(f"✓ SQL generated for both periods")
    print("\nComparison query ready for execution!")
    print("=" * 80)


if __name__ == "__main__":
    demo_comparison_followup()
