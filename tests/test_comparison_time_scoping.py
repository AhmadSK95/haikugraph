"""Tests for A10.x comparison time scoping fix.

This test suite validates that comparison queries ALWAYS produce
explicit, symmetric time constraints for every comparison subquestion.
"""

import json
from unittest.mock import patch

import pytest

from haikugraph.planning.llm_planner import generate_or_patch_plan
from haikugraph.planning.intent import Intent, IntentType
from haikugraph.planning.schema import validate_plan_or_raise


class TestComparisonTimeScoping:
    """Validate comparison queries produce symmetric time constraints."""
    
    def test_comparison_must_have_scoped_time_constraints(self):
        """Comparison plan MUST have scoped time constraint for EVERY subquestion."""
        
        # Mock LLM to return a comparison plan with proper time scoping
        valid_plan = {
            "original_question": "Revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "this_year",
                    "applies_to": "SQ1_current"
                },
                {
                    "type": "time",
                    "expression": "previous_year",
                    "applies_to": "SQ2_comparison"
                }
            ]
        }
        
        # Mock intent classification
        mock_intent = Intent(
            type=IntentType.COMPARISON,
            confidence=0.92,
            rationale="Temporal comparison: this year vs last year",
            requires_comparison=True
        )
        
        with patch("haikugraph.planning.llm_planner.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps(valid_plan)
            
            schema = "Table: orders (id, payment_amount, created_at)"
            plan = generate_or_patch_plan(
                question="Revenue this year vs last year",
                schema=schema,
                intent=mock_intent
            )
            
            # Validate plan structure
            validate_plan_or_raise(plan)
            
            # Verify both subquestions exist
            assert len(plan["subquestions"]) == 2
            sq_ids = {sq["id"] for sq in plan["subquestions"]}
            assert "SQ1_current" in sq_ids
            assert "SQ2_comparison" in sq_ids
            
            # Verify time constraints exist and are scoped
            assert "constraints" in plan
            assert len(plan["constraints"]) == 2
            
            time_constraints = [c for c in plan["constraints"] if c["type"] == "time"]
            assert len(time_constraints) == 2
            
            # Every time constraint must have applies_to
            for constraint in time_constraints:
                assert "applies_to" in constraint
                assert constraint["applies_to"] is not None
                assert constraint["applies_to"] in sq_ids
    
    def test_comparison_rejects_unscoped_time_constraint(self):
        """Plan with unscoped time constraint in comparison should fail validation."""
        
        # Invalid: one time constraint without applies_to
        invalid_plan = {
            "original_question": "Revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "this_year"
                    # Missing applies_to!
                },
                {
                    "type": "time",
                    "expression": "previous_year",
                    "applies_to": "SQ2_comparison"
                }
            ]
        }
        
        # Should fail validation for comparison plans
        with pytest.raises(ValueError, match="unscoped time constraint"):
            validate_plan_or_raise(invalid_plan)
    
    def test_comparison_rejects_only_one_side_scoped(self):
        """Plan with only one subquestion scoped should fail validation."""
        
        # Invalid: only SQ2_comparison has time constraint
        invalid_plan = {
            "original_question": "Revenue this year vs last year",
            "subquestions": [
                {
                    "id": "SQ1_current",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                },
                {
                    "id": "SQ2_comparison",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "previous_year",
                    "applies_to": "SQ2_comparison"
                }
                # Missing constraint for SQ1_current!
            ]
        }
        
        # Should fail validation
        with pytest.raises(ValueError, match="missing time constraint"):
            validate_plan_or_raise(invalid_plan)
    
    def test_non_comparison_allows_unscoped_constraints(self):
        """Non-comparison queries can have unscoped constraints."""
        
        # Valid: non-comparison query with unscoped time constraint
        valid_plan = {
            "original_question": "What is total revenue?",
            "subquestions": [
                {
                    "id": "SQ1",
                    "tables": ["orders"],
                    "aggregations": [{"agg": "sum", "col": "payment_amount"}]
                }
            ],
            "constraints": [
                {
                    "type": "time",
                    "expression": "this_month"
                    # No applies_to is OK for non-comparison
                }
            ]
        }
        
        # Should pass validation
        validate_plan_or_raise(valid_plan)
    
    def test_planner_prompt_enforces_comparison_time_scoping(self):
        """Planner prompt should explicitly require symmetric time constraints."""
        
        # Mock LLM that initially returns invalid plan, then repairs it
        call_count = {"count": 0}
        
        def mock_llm_side_effect(messages, **kwargs):
            call_count["count"] += 1
            
            if call_count["count"] == 1:
                # First attempt: invalid (unscoped constraint)
                return json.dumps({
                    "original_question": "Revenue this year vs last year",
                    "subquestions": [
                        {"id": "SQ1_current", "tables": ["orders"], "aggregations": [{"agg": "sum", "col": "payment_amount"}]},
                        {"id": "SQ2_comparison", "tables": ["orders"], "aggregations": [{"agg": "sum", "col": "payment_amount"}]}
                    ],
                    "constraints": [
                        {"type": "time", "expression": "this_year"},  # Unscoped!
                        {"type": "time", "expression": "previous_year", "applies_to": "SQ2_comparison"}
                    ]
                })
            else:
                # Repair: add applies_to
                return json.dumps({
                    "original_question": "Revenue this year vs last year",
                    "subquestions": [
                        {"id": "SQ1_current", "tables": ["orders"], "aggregations": [{"agg": "sum", "col": "payment_amount"}]},
                        {"id": "SQ2_comparison", "tables": ["orders"], "aggregations": [{"agg": "sum", "col": "payment_amount"}]}
                    ],
                    "constraints": [
                        {"type": "time", "expression": "this_year", "applies_to": "SQ1_current"},
                        {"type": "time", "expression": "previous_year", "applies_to": "SQ2_comparison"}
                    ]
                })
        
        mock_intent = Intent(
            type=IntentType.COMPARISON,
            confidence=0.92,
            rationale="Temporal comparison",
            requires_comparison=True
        )
        
        with patch("haikugraph.planning.llm_planner.call_llm") as mock_llm:
            mock_llm.side_effect = mock_llm_side_effect
            
            schema = "Table: orders (id, payment_amount, created_at)"
            plan = generate_or_patch_plan(
                question="Revenue this year vs last year",
                schema=schema,
                intent=mock_intent
            )
            
            # Should have triggered repair
            assert call_count["count"] == 2
            
            # Final plan should be valid
            validate_plan_or_raise(plan)
            
            # Both constraints should be scoped
            for constraint in plan["constraints"]:
                if constraint["type"] == "time":
                    assert constraint.get("applies_to") is not None
