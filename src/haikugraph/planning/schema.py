"""Plan JSON schema and validation for HaikuGraph.

This module defines the canonical schema for plan dictionaries used throughout
the execution pipeline. It uses Pydantic for runtime validation and provides
a clear contract for what valid plans must contain.
"""

from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class AggregationSpec(BaseModel):
    """Specification for an aggregation operation.

    Attributes:
        agg: Aggregation function (sum, avg, min, max, count, count_distinct)
        col: Column name to aggregate
        distinct: Whether to apply DISTINCT (optional, for count)
    """

    agg: str = Field(..., description="Aggregation function")
    col: str = Field(..., description="Column to aggregate")
    distinct: Optional[bool] = Field(None, description="Apply DISTINCT (for count)")

    @field_validator("agg")
    @classmethod
    def validate_agg_function(cls, v: str) -> str:
        """Validate aggregation function is one of known types."""
        allowed = {"sum", "avg", "min", "max", "count", "count_distinct"}
        if v.lower() not in allowed:
            # Warn but don't fail for unknown agg functions
            pass
        return v
    
    @field_validator("col")
    @classmethod
    def validate_col_no_hacks(cls, v: str) -> str:
        """Reject SQL keyword hacks in column names.
        
        This prevents invalid SQL generation from hacks like:
        - {"agg": "count", "col": "DISTINCT customer_id"}
        - {"agg": "sum", "col": "amount * 2"}
        
        Column names must be simple identifiers.
        """
        if not v:
            raise ValueError("Column name cannot be empty")
        
        # Check for SQL keywords that indicate hacks
        col_lower = v.lower().strip()
        forbidden_keywords = ["distinct", "select", "from", "where", "join", "union"]
        
        for keyword in forbidden_keywords:
            if keyword in col_lower:
                raise ValueError(
                    f"Column name contains forbidden SQL keyword '{keyword}': {v}. "
                    f"Use 'distinct' field for DISTINCT counts, not in column name."
                )
        
        # Check for spaces (indicates multi-word hack)
        if " " in v.strip():
            raise ValueError(
                f"Column name contains spaces: '{v}'. "
                f"Column names must be simple identifiers without spaces."
            )
        
        return v


class Subquestion(BaseModel):
    """A subquestion within a plan.

    Attributes:
        id: Unique identifier (e.g., 'SQ1')
        description: Optional human-readable description
        tables: Non-empty list of table names
        columns: Optional list of column names
        group_by: Optional list of columns to group by (strings) or time buckets (dicts)
        aggregations: Optional list of aggregation specs
        required_joins: Optional list (kept for backward compatibility)
        constraints: Optional list (kept for backward compatibility)
        confidence: Optional confidence score
    """

    id: str = Field(..., description="Subquestion ID")
    description: Optional[str] = Field(None, description="Description")
    tables: list[str] = Field(..., min_length=1, description="Table names (at least one)")
    columns: Optional[list[str]] = Field(None, description="Column names")
    group_by: Optional[list[Union[str, dict]]] = Field(
        None, 
        description="Group by columns (strings) or time buckets (dicts with type='time_bucket')"
    )
    aggregations: Optional[list[AggregationSpec]] = Field(
        None, description="Aggregation operations"
    )
    required_joins: Optional[list] = Field(None, description="Required joins (legacy)")
    constraints: Optional[list] = Field(None, description="Constraints (legacy)")
    confidence: Optional[float] = Field(None, description="Confidence score")

    @model_validator(mode="after")
    def validate_group_by_aggregations(self) -> "Subquestion":
        """If group_by exists and aggregations is set (not None), aggregations must be non-empty.
        
        Note: This allows group_by with aggregations=None (for potential future use like
        "distinct per group"), but prevents the confusing case of group_by with aggregations=[].
        """
        if self.group_by and self.aggregations is not None:
            if len(self.aggregations) == 0:
                raise ValueError(f"Subquestion {self.id}: group_by present but aggregations empty")
        return self


class Ambiguity(BaseModel):
    """An ambiguity detected during planning.

    Attributes:
        issue: Description of the ambiguity
        recommended: Recommended resolution (may be None)
        options: List of possible resolution options
        confidence: Optional confidence score
    """

    issue: str = Field(..., description="Ambiguity description")
    recommended: Optional[str] = Field(None, description="Recommended option")
    options: list[str] = Field(..., description="Available options")
    confidence: Optional[float] = Field(None, description="Confidence score")

    @model_validator(mode="after")
    def validate_recommended_in_options(self) -> "Ambiguity":
        """If recommended is set, it must be in options."""
        if self.recommended and self.recommended not in self.options:
            raise ValueError(f"Recommended '{self.recommended}' not in options {self.options}")
        return self


class Constraint(BaseModel):
    """A constraint to apply to queries.

    Attributes:
        type: Constraint type ('time', 'filter', or other)
        expression: Constraint expression
        applies_to: Optional subquestion ID this constraint applies to (for scoping)
    """

    type: str = Field(..., description="Constraint type")
    expression: str = Field(..., description="Constraint expression")
    applies_to: Optional[str] = Field(
        None, description="Subquestion ID this constraint applies to (optional scoping)"
    )


class JoinPath(BaseModel):
    """A join path between tables.

    Attributes:
        from_: Source table
        to: Destination table
        via: List of join column names
        confidence: Optional confidence score
        cardinality: Optional cardinality description
    """

    from_: str = Field(..., alias="from", description="Source table")
    to: str = Field(..., description="Destination table")
    via: list[str] = Field(..., min_length=1, description="Join columns")
    confidence: Optional[float] = Field(None, description="Confidence score")
    cardinality: Optional[str] = Field(None, description="Cardinality")

    model_config = {"populate_by_name": True}


class EntityDetected(BaseModel):
    """An entity detected in the question.

    Attributes:
        name: Entity name
        mapped_to: List of table.column references
        confidence: Optional confidence score
    """

    name: str = Field(..., description="Entity name")
    mapped_to: list[str] = Field(..., description="Mapped table.column references")
    confidence: Optional[float] = Field(None, description="Confidence score")


class MetricRequested(BaseModel):
    """A metric requested in the question.

    Attributes:
        name: Metric name
        mapped_columns: List of table.column references
        aggregation: Optional aggregation type
        confidence: Optional confidence score
    """

    name: str = Field(..., description="Metric name")
    mapped_columns: list[str] = Field(..., description="Column references")
    aggregation: Optional[str] = Field(None, description="Aggregation type")
    confidence: Optional[float] = Field(None, description="Confidence score")


class Intent(BaseModel):
    """Intent classification for the question.

    Attributes:
        type: Intent type (metric, lookup, diagnostic, comparison, etc.)
        confidence: Confidence score
    """

    type: str = Field(..., description="Intent type")
    confidence: float = Field(..., description="Confidence score")


class Plan(BaseModel):
    """Complete plan schema for HaikuGraph execution.

    This is the canonical schema that all plans must conform to.
    Required fields ensure the executor has minimum information needed.
    """

    original_question: str = Field(..., description="Original user question")
    subquestions: list[Subquestion] = Field(
        ..., min_length=1, description="At least one subquestion required"
    )

    # Optional but commonly used fields
    ambiguities: Optional[list[Ambiguity]] = Field(None, description="Ambiguities")
    constraints: Optional[list[Constraint]] = Field(None, description="Constraints")
    join_paths: Optional[list[JoinPath]] = Field(None, description="Join paths")
    entities_detected: Optional[list[EntityDetected]] = Field(None, description="Detected entities")
    metrics_requested: Optional[list[MetricRequested]] = Field(
        None, description="Requested metrics"
    )

    # Metadata fields
    intent: Optional[Intent] = Field(None, description="Intent classification")
    plan_confidence: Optional[float] = Field(None, description="Overall confidence")

    model_config = {"extra": "allow"}  # Allow unknown fields (for evolution)

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

    @model_validator(mode="after")
    def validate_constraint_applies_to(self) -> "Plan":
        """Validate that constraint applies_to references valid subquestion IDs."""
        if not self.constraints:
            return self

        # Collect valid subquestion IDs
        valid_sq_ids = {sq.id for sq in self.subquestions}

        # Check each constraint with applies_to
        for constraint in self.constraints:
            if constraint.applies_to and constraint.applies_to not in valid_sq_ids:
                raise ValueError(
                    f"Constraint applies_to='{constraint.applies_to}' "
                    f"does not match any subquestion ID. "
                    f"Valid IDs: {sorted(valid_sq_ids)}"
                )

        return self
    
    @model_validator(mode="after")
    def validate_comparison_time_scoping(self) -> "Plan":
        """Validate symmetric time constraints for comparison queries.
        
        When a plan contains comparison subquestions (IDs ending with '_current' 
        or '_comparison'), ALL time constraints MUST be scoped (have applies_to).
        
        This prevents the bug where comparison queries return identical results
        because the 'current' subquestion runs unscoped over all data.
        """
        if not self.constraints:
            return self
        
        # Detect comparison subquestions
        sq_ids = {sq.id for sq in self.subquestions}
        has_comparison_sqs = any(
            sq_id.endswith("_current") or sq_id.endswith("_comparison")
            for sq_id in sq_ids
        )
        
        if not has_comparison_sqs:
            # Non-comparison queries can have unscoped constraints
            return self
        
        # For comparison queries, check time constraint scoping
        time_constraints = [c for c in self.constraints if c.type == "time"]
        
        if not time_constraints:
            # No time constraints at all - likely missing constraints
            comparison_sq_ids = [
                sq_id for sq_id in sq_ids 
                if sq_id.endswith("_current") or sq_id.endswith("_comparison")
            ]
            raise ValueError(
                f"Comparison plan has subquestions {sorted(comparison_sq_ids)} "
                f"but missing time constraints. Each comparison subquestion "
                f"MUST have an explicit time constraint with applies_to."
            )
        
        # Check for unscoped time constraints
        unscoped_time = [c for c in time_constraints if not c.applies_to]
        if unscoped_time:
            raise ValueError(
                f"Comparison plan has unscoped time constraint(s): "
                f"{[c.expression for c in unscoped_time]}. "
                f"For comparison queries, EVERY time constraint MUST have "
                f"applies_to to ensure symmetric scoping."
            )
        
        # Check that each comparison subquestion has a time constraint
        comparison_sq_ids = [
            sq_id for sq_id in sq_ids 
            if sq_id.endswith("_current") or sq_id.endswith("_comparison")
        ]
        
        scoped_sq_ids = {c.applies_to for c in time_constraints if c.applies_to}
        missing_constraints = set(comparison_sq_ids) - scoped_sq_ids
        
        if missing_constraints:
            raise ValueError(
                f"Comparison subquestion(s) {sorted(missing_constraints)} "
                f"missing time constraint. Each comparison subquestion "
                f"MUST have its own scoped time constraint with applies_to."
            )
        
        return self


def validate_plan(plan: dict) -> tuple[bool, list[str]]:
    """Validate a plan dictionary against the schema.

    Args:
        plan: Plan dictionary to validate

    Returns:
        Tuple of (is_valid, errors) where errors is a list of error messages
    """
    try:
        Plan.model_validate(plan)
        return True, []
    except Exception as e:
        # Parse pydantic validation errors into readable messages
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = " -> ".join(str(x) for x in err["loc"])
                msg = err["msg"]
                errors.append(f"{loc}: {msg}")
        else:
            errors.append(str(e))
        return False, errors


def validate_plan_with_warnings(
    plan: dict,
) -> tuple[bool, list[str], list[str]]:
    """Validate a plan and also return warnings.

    Args:
        plan: Plan dictionary to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    is_valid, errors = validate_plan(plan)
    warnings = []

    if is_valid:
        # Check for warnings on valid plans
        validated = Plan.model_validate(plan)

        # Warn about unknown constraint types
        if validated.constraints:
            known_types = {"time", "filter"}
            for constraint in validated.constraints:
                if constraint.type not in known_types:
                    warnings.append(
                        f"Unknown constraint type: '{constraint.type}' (known: {known_types})"
                    )

        # Warn about empty optional lists
        if validated.ambiguities is not None and len(validated.ambiguities) == 0:
            warnings.append("Empty ambiguities list (consider omitting)")
        if validated.constraints is not None and len(validated.constraints) == 0:
            warnings.append("Empty constraints list (consider omitting)")
        if validated.join_paths is not None and len(validated.join_paths) == 0:
            warnings.append("Empty join_paths list (consider omitting)")

        # Warn about unknown top-level keys
        known_keys = set(Plan.model_fields.keys())
        actual_keys = set(plan.keys())
        unknown_keys = actual_keys - known_keys
        if unknown_keys:
            warnings.append(f"Unknown top-level keys: {sorted(unknown_keys)}")

    return is_valid, errors, warnings


def validate_plan_or_raise(plan: dict) -> None:
    """Validate a plan and raise ValueError if invalid.

    Args:
        plan: Plan dictionary to validate

    Raises:
        ValueError: If plan is invalid, with detailed error messages
    """
    is_valid, errors = validate_plan(plan)
    if not is_valid:
        error_msg = "Invalid plan:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_msg)
