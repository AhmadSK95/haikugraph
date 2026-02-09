"""Tests for A11 comparison normalization.

This test suite validates that comparison queries produce normalized,
structurally correct results with strict invariants enforced.

Test Categories:
1. Normalized Structure - ComparisonResult schema validation
2. Delta/Delta_pct Correctness - Math invariants
3. Zero Division Handling - delta_pct = null when comparison.value == 0
4. Flat Comparison - direction = "flat" when delta == 0
5. Narrator Integration - Narrator receives normalized comparison, no math
"""

import pytest

from haikugraph.execution.comparison import (
    ComparisonOperand,
    ComparisonResult,
    normalize_comparison,
    extract_comparison_from_results,
)


class TestComparisonOperand:
    """Test ComparisonOperand validation."""
    
    def test_valid_operand(self):
        """Valid operand passes validation."""
        operand = ComparisonOperand(
            value=1000.0,
            period="this_year",
            subquestion_id="SQ1_current",
            row_count=1
        )
        
        assert operand.value == 1000.0
        assert operand.period == "this_year"
        assert operand.subquestion_id == "SQ1_current"
        assert operand.row_count == 1
    
    def test_period_cannot_be_empty(self):
        """Empty period raises ValidationError."""
        with pytest.raises(ValueError, match="Period cannot be empty"):
            ComparisonOperand(
                value=1000.0,
                period="",
                subquestion_id="SQ1_current",
                row_count=1
            )
    
    def test_negative_row_count_fails(self):
        """Negative row count raises ValidationError."""
        with pytest.raises(ValueError):
            ComparisonOperand(
                value=1000.0,
                period="this_year",
                subquestion_id="SQ1_current",
                row_count=-1
            )


class TestComparisonResultInvariants:
    """Test ComparisonResult strict invariants."""
    
    def test_valid_comparison_up(self):
        """Valid comparison with increase passes all invariants."""
        result = ComparisonResult(
            metric="sum_revenue",
            current=ComparisonOperand(
                value=1000.0,
                period="this_year",
                subquestion_id="SQ1_current",
                row_count=1
            ),
            comparison=ComparisonOperand(
                value=800.0,
                period="previous_year",
                subquestion_id="SQ2_comparison",
                row_count=1
            ),
            delta=200.0,
            delta_pct=25.0,
            direction="up"
        )
        
        assert result.metric == "sum_revenue"
        assert result.current.value == 1000.0
        assert result.comparison.value == 800.0
        assert result.delta == 200.0
        assert result.delta_pct == 25.0
        assert result.direction == "up"
    
    def test_valid_comparison_down(self):
        """Valid comparison with decrease passes invariants."""
        result = ComparisonResult(
            metric="sum_revenue",
            current=ComparisonOperand(
                value=800.0,
                period="this_year",
                subquestion_id="SQ1_current",
                row_count=1
            ),
            comparison=ComparisonOperand(
                value=1000.0,
                period="previous_year",
                subquestion_id="SQ2_comparison",
                row_count=1
            ),
            delta=-200.0,
            delta_pct=-20.0,
            direction="down"
        )
        
        assert result.delta == -200.0
        assert result.delta_pct == -20.0
        assert result.direction == "down"
    
    def test_valid_comparison_flat(self):
        """Valid comparison with no change passes invariants."""
        result = ComparisonResult(
            metric="sum_revenue",
            current=ComparisonOperand(
                value=1000.0,
                period="this_year",
                subquestion_id="SQ1_current",
                row_count=1
            ),
            comparison=ComparisonOperand(
                value=1000.0,
                period="previous_year",
                subquestion_id="SQ2_comparison",
                row_count=1
            ),
            delta=0.0,
            delta_pct=0.0,
            direction="flat"
        )
        
        assert result.delta == 0.0
        assert result.delta_pct == 0.0
        assert result.direction == "flat"
    
    def test_zero_division_requires_null_pct(self):
        """Comparison with zero base requires delta_pct = null."""
        result = ComparisonResult(
            metric="sum_revenue",
            current=ComparisonOperand(
                value=100.0,
                period="this_year",
                subquestion_id="SQ1_current",
                row_count=1
            ),
            comparison=ComparisonOperand(
                value=0.0,
                period="previous_year",
                subquestion_id="SQ2_comparison",
                row_count=1
            ),
            delta=100.0,
            delta_pct=None,  # Must be null
            direction="up"
        )
        
        assert result.comparison.value == 0.0
        assert result.delta_pct is None
        assert result.direction == "up"
    
    def test_invalid_delta_fails(self):
        """Invalid delta (doesn't match current - comparison) fails validation."""
        with pytest.raises(ValueError, match="Invalid delta"):
            ComparisonResult(
                metric="sum_revenue",
                current=ComparisonOperand(
                    value=1000.0,
                    period="this_year",
                    subquestion_id="SQ1_current",
                    row_count=1
                ),
                comparison=ComparisonOperand(
                    value=800.0,
                    period="previous_year",
                    subquestion_id="SQ2_comparison",
                    row_count=1
                ),
                delta=100.0,  # Wrong! Should be 200.0
                delta_pct=12.5,
                direction="up"
            )
    
    def test_invalid_delta_pct_fails(self):
        """Invalid delta_pct (doesn't match formula) fails validation."""
        with pytest.raises(ValueError, match="Invalid delta_pct"):
            ComparisonResult(
                metric="sum_revenue",
                current=ComparisonOperand(
                    value=1000.0,
                    period="this_year",
                    subquestion_id="SQ1_current",
                    row_count=1
                ),
                comparison=ComparisonOperand(
                    value=800.0,
                    period="previous_year",
                    subquestion_id="SQ2_comparison",
                    row_count=1
                ),
                delta=200.0,
                delta_pct=50.0,  # Wrong! Should be 25.0
                direction="up"
            )
    
    def test_invalid_direction_fails(self):
        """Invalid direction (doesn't match delta) fails validation."""
        with pytest.raises(ValueError, match="Invalid direction"):
            ComparisonResult(
                metric="sum_revenue",
                current=ComparisonOperand(
                    value=1000.0,
                    period="this_year",
                    subquestion_id="SQ1_current",
                    row_count=1
                ),
                comparison=ComparisonOperand(
                    value=800.0,
                    period="previous_year",
                    subquestion_id="SQ2_comparison",
                    row_count=1
                ),
                delta=200.0,
                delta_pct=25.0,
                direction="down"  # Wrong! Should be "up"
            )
    
    def test_zero_division_with_non_null_pct_fails(self):
        """Zero division with non-null delta_pct fails validation."""
        with pytest.raises(ValueError, match="delta_pct must be null"):
            ComparisonResult(
                metric="sum_revenue",
                current=ComparisonOperand(
                    value=100.0,
                    period="this_year",
                    subquestion_id="SQ1_current",
                    row_count=1
                ),
                comparison=ComparisonOperand(
                    value=0.0,
                    period="previous_year",
                    subquestion_id="SQ2_comparison",
                    row_count=1
                ),
                delta=100.0,
                delta_pct=50.0,  # Wrong! Must be null
                direction="up"
            )


class TestNormalizeComparison:
    """Test normalize_comparison utility function."""
    
    def test_normalize_comparison_increase(self):
        """Normalize comparison with increase."""
        result = normalize_comparison(
            metric="sum_revenue",
            current_value=1000.0,
            current_period="this_year",
            current_sq_id="SQ1_current",
            current_row_count=1,
            comparison_value=800.0,
            comparison_period="previous_year",
            comparison_sq_id="SQ2_comparison",
            comparison_row_count=1,
        )
        
        assert result.delta == 200.0
        assert result.delta_pct == 25.0
        assert result.direction == "up"
    
    def test_normalize_comparison_decrease(self):
        """Normalize comparison with decrease."""
        result = normalize_comparison(
            metric="sum_revenue",
            current_value=700.0,
            current_period="this_month",
            current_sq_id="SQ1_current",
            current_row_count=1,
            comparison_value=1000.0,
            comparison_period="previous_month",
            comparison_sq_id="SQ2_comparison",
            comparison_row_count=1,
        )
        
        assert result.delta == -300.0
        assert result.delta_pct == -30.0
        assert result.direction == "down"
    
    def test_normalize_comparison_flat(self):
        """Normalize comparison with no change."""
        result = normalize_comparison(
            metric="count_orders",
            current_value=500.0,
            current_period="this_week",
            current_sq_id="SQ1_current",
            current_row_count=1,
            comparison_value=500.0,
            comparison_period="previous_week",
            comparison_sq_id="SQ2_comparison",
            comparison_row_count=1,
        )
        
        assert result.delta == 0.0
        assert result.delta_pct == 0.0
        assert result.direction == "flat"
    
    def test_normalize_comparison_zero_division(self):
        """Normalize comparison with zero base (division by zero)."""
        result = normalize_comparison(
            metric="sum_revenue",
            current_value=100.0,
            current_period="this_year",
            current_sq_id="SQ1_current",
            current_row_count=1,
            comparison_value=0.0,
            comparison_period="previous_year",
            comparison_sq_id="SQ2_comparison",
            comparison_row_count=1,
        )
        
        assert result.delta == 100.0
        assert result.delta_pct is None  # Division by zero
        assert result.direction == "up"


class TestExtractComparisonFromResults:
    """Test extract_comparison_from_results function."""
    
    def test_extract_valid_comparison(self):
        """Extract valid comparison from execution results."""
        plan = {
            "subquestions": [
                {"id": "SQ1_current", "tables": ["orders"]},
                {"id": "SQ2_comparison", "tables": ["orders"]},
            ],
            "constraints": [
                {"type": "time", "expression": "orders.created_at in this_year", "applies_to": "SQ1_current"},
                {"type": "time", "expression": "orders.created_at in previous_year", "applies_to": "SQ2_comparison"},
            ]
        }
        
        subquestion_results = [
            {
                "id": "SQ1_current",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"sum_revenue": 1000.0}]
            },
            {
                "id": "SQ2_comparison",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"sum_revenue": 800.0}]
            }
        ]
        
        result = extract_comparison_from_results(plan, subquestion_results)
        
        assert result is not None
        assert result.metric == "sum_revenue"
        assert result.current.value == 1000.0
        assert result.current.period == "this_year"
        assert result.comparison.value == 800.0
        assert result.comparison.period == "previous_year"
        assert result.delta == 200.0
        assert result.delta_pct == 25.0
        assert result.direction == "up"
    
    def test_non_comparison_returns_none(self):
        """Non-comparison query returns None."""
        plan = {
            "subquestions": [
                {"id": "SQ1", "tables": ["orders"]},
            ]
        }
        
        subquestion_results = [
            {
                "id": "SQ1",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"sum_revenue": 1000.0}]
            }
        ]
        
        result = extract_comparison_from_results(plan, subquestion_results)
        
        assert result is None
    
    def test_comparison_with_wrong_subquestion_count_fails(self):
        """Comparison with more than 2 subquestions fails."""
        plan = {
            "subquestions": [
                {"id": "SQ1_current", "tables": ["orders"]},
                {"id": "SQ2_comparison", "tables": ["orders"]},
                {"id": "SQ3", "tables": ["customers"]},  # Extra subquestion
            ]
        }
        
        subquestion_results = []
        
        with pytest.raises(ValueError, match="must have exactly 2 subquestions"):
            extract_comparison_from_results(plan, subquestion_results)
    
    def test_comparison_with_failed_subquestion_fails(self):
        """Comparison with failed subquestion raises error."""
        plan = {
            "subquestions": [
                {"id": "SQ1_current", "tables": ["orders"]},
                {"id": "SQ2_comparison", "tables": ["orders"]},
            ]
        }
        
        subquestion_results = [
            {
                "id": "SQ1_current",
                "status": "failed",
                "error": "Table not found"
            },
            {
                "id": "SQ2_comparison",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"sum_revenue": 800.0}]
            }
        ]
        
        with pytest.raises(ValueError, match="SQ1_current failed"):
            extract_comparison_from_results(plan, subquestion_results)
    
    def test_comparison_with_multiple_rows_fails(self):
        """Comparison with multiple rows in result fails."""
        plan = {
            "subquestions": [
                {"id": "SQ1_current", "tables": ["orders"]},
                {"id": "SQ2_comparison", "tables": ["orders"]},
            ]
        }
        
        subquestion_results = [
            {
                "id": "SQ1_current",
                "status": "success",
                "row_count": 2,
                "preview_rows": [{"sum_revenue": 1000.0}, {"sum_revenue": 500.0}]  # Multiple rows!
            },
            {
                "id": "SQ2_comparison",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"sum_revenue": 800.0}]
            }
        ]
        
        with pytest.raises(ValueError, match="must return exactly 1 row"):
            extract_comparison_from_results(plan, subquestion_results)
    
    def test_comparison_with_metric_mismatch_fails(self):
        """Comparison with different metrics fails."""
        plan = {
            "subquestions": [
                {"id": "SQ1_current", "tables": ["orders"]},
                {"id": "SQ2_comparison", "tables": ["orders"]},
            ]
        }
        
        subquestion_results = [
            {
                "id": "SQ1_current",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"sum_revenue": 1000.0}]
            },
            {
                "id": "SQ2_comparison",
                "status": "success",
                "row_count": 1,
                "preview_rows": [{"count_orders": 100}]  # Different metric!
            }
        ]
        
        with pytest.raises(ValueError, match="not found in SQ2_comparison"):
            extract_comparison_from_results(plan, subquestion_results)


class TestNarratorIntegration:
    """Test narrator integration with normalized comparisons."""
    
    def test_narrator_receives_normalized_comparison(self):
        """Narrator receives normalized comparison, not raw results."""
        from haikugraph.explain.narrator import _build_comparison_summary
        
        comparison = {
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
        
        summary = _build_comparison_summary(comparison)
        
        # Verify summary contains all required fields
        assert "sum_revenue" in summary
        assert "this_year" in summary
        assert "1000.0" in summary
        assert "previous_year" in summary
        assert "800.0" in summary
        assert "200.0" in summary
        assert "25.00%" in summary
        assert "up" in summary
    
    def test_narrator_handles_zero_division(self):
        """Narrator correctly handles division by zero."""
        from haikugraph.explain.narrator import _build_comparison_summary
        
        comparison = {
            "metric": "sum_revenue",
            "current": {
                "value": 100.0,
                "period": "this_month",
                "subquestion_id": "SQ1_current",
                "row_count": 1
            },
            "comparison": {
                "value": 0.0,
                "period": "previous_month",
                "subquestion_id": "SQ2_comparison",
                "row_count": 1
            },
            "delta": 100.0,
            "delta_pct": None,  # Division by zero
            "direction": "up"
        }
        
        summary = _build_comparison_summary(comparison)
        
        # Verify N/A for percentage
        assert "N/A" in summary or "division by zero" in summary.lower()
    
    def test_narrator_handles_flat_comparison(self):
        """Narrator correctly handles flat (no change) comparison."""
        from haikugraph.explain.narrator import _build_comparison_summary
        
        comparison = {
            "metric": "count_orders",
            "current": {
                "value": 500.0,
                "period": "this_week",
                "subquestion_id": "SQ1_current",
                "row_count": 1
            },
            "comparison": {
                "value": 500.0,
                "period": "previous_week",
                "subquestion_id": "SQ2_comparison",
                "row_count": 1
            },
            "delta": 0.0,
            "delta_pct": 0.0,
            "direction": "flat"
        }
        
        summary = _build_comparison_summary(comparison)
        
        # Verify flat direction
        assert "0.0" in summary or "0.00" in summary
        assert "flat" in summary
