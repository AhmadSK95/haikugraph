"""Normalized comparison result structures and utilities.

This module defines the first-class comparison concept with strict invariants.
Comparison results are normalized BEFORE narration to ensure structural correctness
and semantic safety.

A11 Requirements:
- Normalized internal structure for comparison outputs
- Strict invariants (exactly 2 operands, no narrator math, zero-division handling)
- Fail-fast on invalid comparisons
"""

from typing import Literal, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ComparisonOperand(BaseModel):
    """A single operand in a comparison (current or comparison period).
    
    Attributes:
        value: The numeric metric value (must be castable to float)
        period: Time period expression (e.g., "this_year", "previous_month")
        subquestion_id: Source subquestion ID (SQ1_current or SQ2_comparison)
        row_count: Number of rows used to compute this value
    """
    value: float = Field(..., description="Metric value")
    period: str = Field(..., description="Time period expression")
    subquestion_id: str = Field(..., description="Source subquestion ID")
    row_count: int = Field(..., ge=0, description="Number of rows")
    
    @field_validator("period")
    @classmethod
    def validate_period_not_empty(cls, v: str) -> str:
        """Ensure period is non-empty."""
        if not v or not v.strip():
            raise ValueError("Period cannot be empty")
        return v.strip()


class ComparisonResult(BaseModel):
    """Normalized comparison result with strict invariants.
    
    This structure MUST be produced before narration for all comparison queries.
    
    Invariants:
    - Exactly 2 operands (current + comparison)
    - Values from scoped subquestions only
    - delta = current.value - comparison.value
    - delta_pct = null if comparison.value == 0, else (delta / comparison.value) * 100
    - direction = "up" if delta > 0, "down" if delta < 0, "flat" if delta == 0
    
    Attributes:
        metric: Metric name (e.g., "sum_revenue", "count_orders")
        current: Current period operand
        comparison: Comparison period operand
        delta: Absolute difference (current - comparison)
        delta_pct: Percentage change (null if division by zero)
        direction: Comparison direction ("up", "down", or "flat")
    """
    metric: str = Field(..., description="Metric name")
    current: ComparisonOperand = Field(..., description="Current period value")
    comparison: ComparisonOperand = Field(..., description="Comparison period value")
    delta: float = Field(..., description="Absolute difference")
    delta_pct: float | None = Field(None, description="Percentage change (null if division by zero)")
    direction: Literal["up", "down", "flat"] = Field(..., description="Change direction")
    
    @field_validator("metric")
    @classmethod
    def validate_metric_not_empty(cls, v: str) -> str:
        """Ensure metric is non-empty."""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()
    
    @model_validator(mode="after")
    def validate_comparison_invariants(self) -> "ComparisonResult":
        """Validate strict comparison invariants.
        
        Ensures:
        - delta = current - comparison
        - delta_pct computed correctly or null
        - direction matches delta
        """
        # Validate delta
        expected_delta = self.current.value - self.comparison.value
        # Use small epsilon for float comparison
        epsilon = 1e-9
        if abs(self.delta - expected_delta) > epsilon:
            raise ValueError(
                f"Invalid delta: expected {expected_delta}, got {self.delta}"
            )
        
        # Validate delta_pct
        if self.comparison.value == 0:
            if self.delta_pct is not None:
                raise ValueError(
                    "delta_pct must be null when comparison value is 0 "
                    "(division by zero)"
                )
        else:
            expected_pct = (self.delta / self.comparison.value) * 100
            if self.delta_pct is None:
                raise ValueError(
                    f"delta_pct must be {expected_pct:.2f} when comparison value is non-zero"
                )
            if abs(self.delta_pct - expected_pct) > epsilon:
                raise ValueError(
                    f"Invalid delta_pct: expected {expected_pct:.2f}, got {self.delta_pct:.2f}"
                )
        
        # Validate direction
        if self.delta > epsilon:
            expected_direction = "up"
        elif self.delta < -epsilon:
            expected_direction = "down"
        else:
            expected_direction = "flat"
        
        if self.direction != expected_direction:
            raise ValueError(
                f"Invalid direction: delta={self.delta:.2f} but direction={self.direction}, "
                f"expected {expected_direction}"
            )
        
        return self


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
    """Create a normalized comparison result from raw values.
    
    This function computes delta, delta_pct, and direction according to
    strict invariants. It fails fast on invalid inputs.
    
    Args:
        metric: Metric name
        current_value: Current period value
        current_period: Current period expression
        current_sq_id: Current subquestion ID
        current_row_count: Current period row count
        comparison_value: Comparison period value
        comparison_period: Comparison period expression
        comparison_sq_id: Comparison subquestion ID
        comparison_row_count: Comparison period row count
    
    Returns:
        Validated ComparisonResult
    
    Raises:
        ValueError: If inputs are invalid or invariants cannot be satisfied
    """
    # Compute delta
    delta = current_value - comparison_value
    
    # Compute delta_pct (null if division by zero)
    if comparison_value == 0:
        delta_pct = None
    else:
        delta_pct = (delta / comparison_value) * 100
    
    # Compute direction
    epsilon = 1e-9
    if delta > epsilon:
        direction = "up"
    elif delta < -epsilon:
        direction = "down"
    else:
        direction = "flat"
    
    # Create operands
    current_operand = ComparisonOperand(
        value=current_value,
        period=current_period,
        subquestion_id=current_sq_id,
        row_count=current_row_count,
    )
    
    comparison_operand = ComparisonOperand(
        value=comparison_value,
        period=comparison_period,
        subquestion_id=comparison_sq_id,
        row_count=comparison_row_count,
    )
    
    # Create and validate comparison result
    return ComparisonResult(
        metric=metric,
        current=current_operand,
        comparison=comparison_operand,
        delta=delta,
        delta_pct=delta_pct,
        direction=direction,
    )


def extract_comparison_from_results(
    plan: dict,
    subquestion_results: list[dict],
) -> ComparisonResult | None:
    """Extract and normalize a comparison from execution results.
    
    Detects comparison plans by checking for SQ1_current/SQ2_comparison subquestions.
    If detected, validates structure and produces normalized ComparisonResult.
    
    Args:
        plan: Validated plan dict
        subquestion_results: List of subquestion execution results
    
    Returns:
        ComparisonResult if this is a comparison query, None otherwise
    
    Raises:
        ValueError: If comparison plan is structurally invalid
    """
    # Detect comparison plan
    subquestions = plan.get("subquestions", [])
    sq_ids = {sq["id"] for sq in subquestions}
    
    has_comparison = (
        "SQ1_current" in sq_ids and "SQ2_comparison" in sq_ids
    )
    
    if not has_comparison:
        # Not a comparison query
        return None
    
    # Validate exactly 2 subquestions for comparison
    if len(subquestions) != 2:
        raise ValueError(
            f"Comparison plan must have exactly 2 subquestions, got {len(subquestions)}"
        )
    
    # Find results by ID
    results_by_id = {result["id"]: result for result in subquestion_results}
    
    if "SQ1_current" not in results_by_id:
        raise ValueError("Missing result for SQ1_current")
    if "SQ2_comparison" not in results_by_id:
        raise ValueError("Missing result for SQ2_comparison")
    
    current_result = results_by_id["SQ1_current"]
    comparison_result = results_by_id["SQ2_comparison"]
    
    # Check for execution failures
    if current_result.get("status") != "success":
        raise ValueError(
            f"SQ1_current failed: {current_result.get('error', 'unknown error')}"
        )
    if comparison_result.get("status") != "success":
        raise ValueError(
            f"SQ2_comparison failed: {comparison_result.get('error', 'unknown error')}"
        )
    
    # Extract metric value from results
    # Expect single-row aggregation result
    current_rows = current_result.get("preview_rows", [])
    comparison_rows = comparison_result.get("preview_rows", [])
    
    if len(current_rows) != 1:
        raise ValueError(
            f"SQ1_current must return exactly 1 row, got {len(current_rows)}"
        )
    if len(comparison_rows) != 1:
        raise ValueError(
            f"SQ2_comparison must return exactly 1 row, got {len(comparison_rows)}"
        )
    
    current_row = current_rows[0]
    comparison_row = comparison_rows[0]
    
    # Find aggregation column (first non-null numeric column)
    metric_col = None
    for col, val in current_row.items():
        if val is not None and isinstance(val, (int, float)):
            metric_col = col
            break
    
    if not metric_col:
        raise ValueError("No numeric metric column found in SQ1_current results")
    
    # Validate same metric in comparison
    if metric_col not in comparison_row:
        raise ValueError(
            f"Metric '{metric_col}' not found in SQ2_comparison results"
        )
    
    current_value = float(current_row[metric_col])
    comparison_value = float(comparison_row[metric_col])
    
    # Extract periods from constraints
    constraints = plan.get("constraints", [])
    
    current_period = "current_period"
    comparison_period = "previous_period"
    
    for constraint in constraints:
        if constraint.get("type") == "time":
            if constraint.get("applies_to") == "SQ1_current":
                # Extract period from expression (e.g., "orders.created_at in this_year" -> "this_year")
                expr = constraint.get("expression", "")
                if " in " in expr:
                    current_period = expr.split(" in ")[-1].strip()
            elif constraint.get("applies_to") == "SQ2_comparison":
                expr = constraint.get("expression", "")
                if " in " in expr:
                    comparison_period = expr.split(" in ")[-1].strip()
    
    # Create normalized comparison
    return normalize_comparison(
        metric=metric_col,
        current_value=current_value,
        current_period=current_period,
        current_sq_id="SQ1_current",
        current_row_count=current_result.get("row_count", 1),
        comparison_value=comparison_value,
        comparison_period=comparison_period,
        comparison_sq_id="SQ2_comparison",
        comparison_row_count=comparison_result.get("row_count", 1),
    )
