"""Pydantic schemas for multi-agent data assistant contracts.

This module defines the structured JSON output contracts for each agent
in the analyst loop. All agents must return outputs conforming to these schemas.

Agent Output Contracts:
- IntakeResult: Goal clarification with metrics/dimensions/time window
- SchemaResult: Schema introspection with semantic catalog reference
- QueryPlanResult: SQL plan with execution results
- AuditResult: Validation results with pass/fail and reasons
- NarrationResult: Final answer with evidence, confidence, and suggestions
- RunTrace: Full execution trace for debugging
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence level for agent outputs."""
    
    HIGH = "high"  # >0.8 confidence
    MEDIUM = "medium"  # 0.5-0.8 confidence
    LOW = "low"  # 0.3-0.5 confidence
    UNCERTAIN = "uncertain"  # <0.3 confidence


class AuditCheckStatus(str, Enum):
    """Status of an audit check."""
    
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


class AgentStatus(str, Enum):
    """Status of an agent execution."""
    
    SUCCESS = "success"
    PARTIAL = "partial"  # Partial success with warnings
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


# =============================================================================
# Intake Agent Contract
# =============================================================================

class ExtractedMetric(BaseModel):
    """A metric extracted from the user's goal."""
    
    name: str = Field(..., description="Metric name (e.g., 'total_revenue', 'count_transactions')")
    aggregation: str = Field(..., description="Aggregation type: sum, count, avg, min, max, count_distinct")
    column_hint: str | None = Field(None, description="Suggested column name if inferable")
    definition: str | None = Field(None, description="User-provided or inferred definition")


class ExtractedDimension(BaseModel):
    """A dimension (group by) extracted from the user's goal."""
    
    name: str = Field(..., description="Dimension name (e.g., 'customer', 'month', 'platform')")
    column_hint: str | None = Field(None, description="Suggested column name if inferable")
    is_time_dimension: bool = Field(False, description="Whether this is a time-based dimension")
    time_grain: str | None = Field(None, description="Time grain if time dimension: day, week, month, quarter, year")


class ExtractedTimeWindow(BaseModel):
    """Time window extracted from the user's goal."""
    
    has_time_filter: bool = Field(..., description="Whether a time filter was specified")
    period_type: str | None = Field(None, description="Type: absolute, relative, comparison")
    start_date: str | None = Field(None, description="Start date (ISO format) if absolute")
    end_date: str | None = Field(None, description="End date (ISO format) if absolute")
    relative_period: str | None = Field(None, description="Relative period: today, yesterday, this_week, last_month, etc.")
    comparison_period: str | None = Field(None, description="Comparison period for vs queries")


class IntakeResult(BaseModel):
    """Output contract for IntakeAgent."""
    
    status: AgentStatus = Field(..., description="Agent execution status")
    
    # Extracted components
    original_goal: str = Field(..., description="User's original goal/question")
    clarified_goal: str = Field(..., description="Clarified/normalized version of the goal")
    intent_type: str = Field(..., description="Detected intent: metric, grouped_metric, comparison, lookup, diagnostic")
    
    metrics: list[ExtractedMetric] = Field(default_factory=list, description="Extracted metrics")
    dimensions: list[ExtractedDimension] = Field(default_factory=list, description="Extracted dimensions")
    time_window: ExtractedTimeWindow | None = Field(None, description="Extracted time window")
    filters: list[dict[str, Any]] = Field(default_factory=list, description="Extracted filter conditions")
    
    # Clarification
    needs_clarification: bool = Field(False, description="Whether user clarification is needed")
    clarification_questions: list[str] = Field(default_factory=list, description="Questions to ask user")
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of extraction logic")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")


# =============================================================================
# Schema Agent Contract
# =============================================================================

class ColumnProfile(BaseModel):
    """Profile of a database column."""
    
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Database data type")
    semantic_type: str | None = Field(None, description="Semantic type: identifier, timestamp, money, rate, text, etc.")
    null_rate: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of null values")
    distinct_count: int = Field(0, ge=0, description="Number of distinct values")
    sample_values: list[str] = Field(default_factory=list, description="Sample values (sanitized)")
    min_value: str | None = Field(None, description="Minimum value (as string)")
    max_value: str | None = Field(None, description="Maximum value (as string)")
    is_likely_key: bool = Field(False, description="Whether column appears to be a key/ID")
    is_likely_metric: bool = Field(False, description="Whether column appears to be a numeric metric")


class TableProfile(BaseModel):
    """Profile of a database table."""
    
    name: str = Field(..., description="Table name")
    row_count: int = Field(0, ge=0, description="Number of rows")
    columns: list[ColumnProfile] = Field(default_factory=list, description="Column profiles")
    primary_key_columns: list[str] = Field(default_factory=list, description="Detected primary key columns")
    timestamp_columns: list[str] = Field(default_factory=list, description="Detected timestamp columns")
    metric_columns: list[str] = Field(default_factory=list, description="Detected metric columns")


class JoinEdge(BaseModel):
    """An inferred join relationship between tables."""
    
    from_table: str = Field(..., description="Source table")
    to_table: str = Field(..., description="Target table")
    from_column: str = Field(..., description="Source column")
    to_column: str = Field(..., description="Target column")
    join_type: str = Field("inner", description="Suggested join type: inner, left, right")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this relationship")
    evidence: str = Field(..., description="Why this join was inferred")


class SchemaResult(BaseModel):
    """Output contract for SchemaAgent."""
    
    status: AgentStatus = Field(..., description="Agent execution status")
    
    # Schema information
    tables: list[TableProfile] = Field(default_factory=list, description="Profiled tables")
    join_graph: list[JoinEdge] = Field(default_factory=list, description="Inferred join relationships")
    
    # Relevance to goal
    relevant_tables: list[str] = Field(default_factory=list, description="Tables relevant to the goal")
    relevant_columns: list[str] = Field(default_factory=list, description="Columns relevant to the goal (table.column)")
    suggested_metrics: list[str] = Field(default_factory=list, description="Suggested metric columns")
    suggested_dimensions: list[str] = Field(default_factory=list, description="Suggested dimension columns")
    suggested_time_column: str | None = Field(None, description="Suggested time column for filtering")
    
    # Warnings
    warnings: list[str] = Field(default_factory=list, description="Schema warnings")
    
    # Metadata
    catalog_version: str | None = Field(None, description="Semantic catalog version if cached")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of schema analysis")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")


# =============================================================================
# Query Agent Contract
# =============================================================================

class QueryStep(BaseModel):
    """A step in the query plan."""
    
    step_id: str = Field(..., description="Unique step identifier")
    description: str = Field(..., description="What this step does")
    sql: str = Field(..., description="SQL query for this step")
    depends_on: list[str] = Field(default_factory=list, description="Step IDs this depends on")
    is_probe: bool = Field(False, description="Whether this is a probing query")


class QueryExecution(BaseModel):
    """Result of executing a query step."""
    
    step_id: str = Field(..., description="Step identifier")
    success: bool = Field(..., description="Whether execution succeeded")
    sql_executed: str = Field(..., description="Actual SQL that was executed")
    row_count: int = Field(0, description="Number of rows returned")
    columns: list[str] = Field(default_factory=list, description="Column names")
    sample_rows: list[dict[str, Any]] = Field(default_factory=list, description="Sample rows (max 10)")
    execution_time_ms: float = Field(0.0, description="Execution time in milliseconds")
    error: str | None = Field(None, description="Error message if failed")
    warnings: list[str] = Field(default_factory=list, description="Execution warnings")


class QueryPlanResult(BaseModel):
    """Output contract for QueryAgent."""
    
    status: AgentStatus = Field(..., description="Agent execution status")
    
    # Plan
    plan_steps: list[QueryStep] = Field(default_factory=list, description="Query plan steps")
    final_sql: str = Field(..., description="Final SQL query")
    
    # Execution
    executions: list[QueryExecution] = Field(default_factory=list, description="Execution results")
    final_result: QueryExecution | None = Field(None, description="Final query result")
    
    # Data summary
    result_summary: dict[str, Any] = Field(default_factory=dict, description="Summary of results")
    
    # Metadata
    tables_used: list[str] = Field(default_factory=list, description="Tables used in query")
    joins_used: list[str] = Field(default_factory=list, description="Joins performed")
    filters_applied: list[str] = Field(default_factory=list, description="Filters applied")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of query strategy")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")


# =============================================================================
# Audit Agent Contract
# =============================================================================

class AuditCheck(BaseModel):
    """Result of a single audit check."""
    
    check_name: str = Field(..., description="Name of the check")
    check_type: str = Field(..., description="Type: join_validity, time_filter, null_rate, duplicates, outliers, cardinality")
    status: AuditCheckStatus = Field(..., description="Check status")
    message: str = Field(..., description="Human-readable result message")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed check results")
    severity: str = Field("info", description="Severity: info, warning, error")
    remediation: str | None = Field(None, description="Suggested remediation if failed")


class AuditResult(BaseModel):
    """Output contract for AuditAgent."""
    
    status: AgentStatus = Field(..., description="Agent execution status")
    
    # Check results
    checks: list[AuditCheck] = Field(default_factory=list, description="Individual check results")
    
    # Summary
    passed: int = Field(0, description="Number of passed checks")
    warned: int = Field(0, description="Number of warnings")
    failed: int = Field(0, description="Number of failed checks")
    skipped: int = Field(0, description="Number of skipped checks")
    
    overall_pass: bool = Field(..., description="Whether audit passed overall")
    requires_refinement: bool = Field(False, description="Whether query needs refinement")
    refinement_suggestions: list[str] = Field(default_factory=list, description="Suggestions for refinement")
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in audit results")
    reasoning: str = Field(..., description="Explanation of audit findings")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")


# =============================================================================
# Narrator Agent Contract
# =============================================================================

class EvidenceItem(BaseModel):
    """A piece of evidence supporting the answer."""
    
    description: str = Field(..., description="What this evidence shows")
    value: str = Field(..., description="The evidence value")
    source: str = Field(..., description="Source: query result, definition, calculation")
    sql_reference: str | None = Field(None, description="SQL that produced this evidence")


class SanityCheck(BaseModel):
    """A sanity check performed on the answer."""
    
    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether check passed")
    message: str = Field(..., description="Check result message")


class NarrationResult(BaseModel):
    """Output contract for NarratorAgent."""
    
    status: AgentStatus = Field(..., description="Agent execution status")
    
    # Answer
    answer_markdown: str = Field(..., description="Final answer in markdown format")
    answer_summary: str = Field(..., description="One-line summary of the answer")
    
    # Evidence and transparency
    definition_used: str = Field(..., description="Definition/interpretation used for the metric")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Evidence supporting the answer")
    sanity_checks: list[SanityCheck] = Field(default_factory=list, description="Sanity checks performed")
    
    # Confidence
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numeric confidence score")
    confidence_factors: list[str] = Field(default_factory=list, description="Factors affecting confidence")
    
    # Follow-up
    suggested_questions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")
    caveats: list[str] = Field(default_factory=list, description="Caveats and limitations")
    
    # Metadata
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")


# =============================================================================
# Run Trace Contract
# =============================================================================

class AgentTrace(BaseModel):
    """Trace of a single agent execution."""
    
    agent_name: str = Field(..., description="Name of the agent")
    started_at: datetime = Field(..., description="When agent started")
    completed_at: datetime = Field(..., description="When agent completed")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    status: AgentStatus = Field(..., description="Execution status")
    input_summary: dict[str, Any] = Field(default_factory=dict, description="Summary of input")
    output_summary: dict[str, Any] = Field(default_factory=dict, description="Summary of output")
    error: str | None = Field(None, description="Error message if failed")


class RunTrace(BaseModel):
    """Full execution trace for a query run."""
    
    trace_id: str = Field(..., description="Unique trace identifier")
    started_at: datetime = Field(..., description="When run started")
    completed_at: datetime | None = Field(None, description="When run completed")
    total_duration_ms: float = Field(0.0, description="Total duration in milliseconds")
    
    # Input
    original_goal: str = Field(..., description="User's original goal")
    db_connection_id: str = Field(..., description="Database connection identifier")
    constraints: dict[str, Any] = Field(default_factory=dict, description="User-provided constraints")
    
    # Agent traces
    agents: list[AgentTrace] = Field(default_factory=list, description="Individual agent traces")
    refinement_count: int = Field(0, description="Number of refinement iterations")
    
    # Final status
    final_status: AgentStatus = Field(..., description="Overall run status")
    final_answer: str | None = Field(None, description="Final answer if successful")
    final_error: str | None = Field(None, description="Error message if failed")
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Assistant Query Request/Response (API Contract)
# =============================================================================

class AssistantQueryRequest(BaseModel):
    """Request for /api/assistant/query endpoint."""
    
    goal: str = Field(..., min_length=1, max_length=2000, description="User's goal/question")
    db_connection_id: str = Field(..., description="Database connection identifier")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Optional constraints")


class AssistantQueryResponse(BaseModel):
    """Response from /api/assistant/query endpoint."""
    
    success: bool = Field(..., description="Whether query succeeded")
    
    # Answer
    answer_markdown: str = Field(..., description="Final answer in markdown")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numeric confidence")
    
    # Evidence
    definition_used: str = Field(..., description="Definition used")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Evidence items")
    sanity_checks: list[SanityCheck] = Field(default_factory=list, description="Sanity checks")
    
    # SQL
    sql: str | None = Field(None, description="Final SQL executed")
    row_count: int | None = Field(None, description="Number of rows returned by final SQL")
    columns: list[str] = Field(default_factory=list, description="Final result column names")
    sample_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sample rows from final query result",
    )
    execution_time_ms: float | None = Field(
        None,
        description="Execution time of final SQL in milliseconds",
    )
    
    # Trace
    trace_id: str = Field(..., description="Trace ID for debugging")
    runtime: dict[str, Any] = Field(default_factory=dict, description="Runtime metadata")
    agent_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-agent execution trace for the run",
    )
    chart_spec: dict[str, Any] | None = Field(
        None,
        description="Suggested visualization specification",
    )
    evidence_packets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured evidence objects produced by agents",
    )
    data_quality: dict[str, Any] = Field(
        default_factory=dict,
        description="Data quality and audit score summary",
    )
    
    # Statistical analysis
    stats_analysis: dict[str, Any] = Field(
        default_factory=dict,
        description="Statistical analysis results (distributions, correlations, outliers, trends)",
    )

    # Agent contribution map
    contribution_map: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-agent contribution summary showing what each agent added or changed.",
    )
    confidence_reasoning: str = Field(
        default="",
        description="Explanation for the assigned confidence level.",
    )

    # Errors
    error: str | None = Field(None, description="Error message if failed")

    # Warnings
    warnings: list[str] = Field(
        default_factory=list,
        description="User-visible warnings about query interpretation (e.g., dropped dimensions).",
    )

    # Suggestions
    suggested_questions: list[str] = Field(default_factory=list, description="Follow-up questions")

    # BRD: Semantic contract (FR-1, FR-6, closure criterion #1)
    contract_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic contract binding: metric, domain, dimensions, time_scope, filters, exclusions.",
    )
    contract_validation: dict[str, Any] = Field(
        default_factory=dict,
        description="Contract-vs-SQL validation result: valid, violations, checks.",
    )

    # BRD: Decision flow (FR-6, closure criterion #6)
    decision_flow: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Explain-yourself decision timeline: each entry is a step "
            "(question, contract, rejected_alternatives, sql, audit, confidence_decomposition)."
        ),
    )
