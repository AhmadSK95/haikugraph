"""Core typed contracts for the v2 runtime pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TableProfileV2(BaseModel):
    table: str
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
    id_like_columns: list[str] = Field(default_factory=list)
    top_null_columns: list[dict[str, Any]] = Field(default_factory=list)


class JoinEdgeV2(BaseModel):
    left_table: str
    right_table: str
    key_column: str
    left_coverage_pct: float = 0.0
    right_coverage_pct: float = 0.0
    confidence: float = 0.0
    risk: str = "unknown"


class SemanticCatalogV2(BaseModel):
    dataset_signature: str
    schema_signature: str = ""
    tables: list[TableProfileV2] = Field(default_factory=list)
    join_edges: list[JoinEdgeV2] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_fields: list[str] = Field(default_factory=list)
    quality_summary: dict[str, float] = Field(default_factory=dict)


class ConversationStateV2(BaseModel):
    session_id: str
    turn_index: int = 1
    prior_goal: str = ""
    prior_sql: str = ""
    prior_slice_signature: str = ""
    grouped: bool = False
    prior_group_dimensions: list[str] = Field(default_factory=list)
    prior_primary_metric: str = ""
    prior_secondary_metric: str = ""
    prior_time_scope: str = ""
    prior_denominator: str = ""


class IntentSpecV2(BaseModel):
    goal: str
    is_followup: bool = False
    operations: list[str] = Field(default_factory=list)
    requires_validity_guard: bool = False
    strategy: str = "metric"
    requested_metrics: list[str] = Field(default_factory=list)
    requested_dimensions: list[str] = Field(default_factory=list)
    requested_time_scope: str = ""
    denominator_semantics: str = ""
    output_mode: str = "answer"


class ObjectiveScoreV2(BaseModel):
    objective_id: str
    score: float = 0.0
    passed: bool = False
    weight: float = 1.0
    detail: str = ""


class PlanCandidateV2(BaseModel):
    candidate_id: str
    strategy: str
    objective_coverage_pct: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)
    objective_scores: list[ObjectiveScoreV2] = Field(default_factory=list)
    hard_gate_pass: bool = True
    hard_gate_failures: list[str] = Field(default_factory=list)
    sql_complexity_score: float = 0.0


class PlanSetV2(BaseModel):
    selected_id: str = ""
    candidates: list[PlanCandidateV2] = Field(default_factory=list)


class QueryPlanV2(BaseModel):
    sql_hint: str = ""
    sql: str = ""
    guardrails: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    primary_metric: str = "metric_value"
    secondary_metric: str = ""
    grouping: list[str] = Field(default_factory=list)
    time_dimension: str = ""
    grain_signature: str = ""


class ExecutionResultV2(BaseModel):
    success: bool
    row_count: int = 0
    latency_ms: float = 0.0
    columns: list[str] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class StageEventV2(BaseModel):
    sequence: int
    stage: str
    status: Literal["started", "completed", "failed", "skipped"]
    elapsed_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class ProvenanceEntryV2(BaseModel):
    field: str
    source_table: str = ""
    source_column: str = ""
    expression: str = ""
    confidence: float = 0.0
    note: str = ""


class RecommendationOptionV2(BaseModel):
    action: str
    expected_impact: str = ""
    risk: str = "medium"
    effort: str = "medium"
    rationale: str = ""


class ContradictionReportV2(BaseModel):
    detected: bool = False
    conflicts: list[str] = Field(default_factory=list)
    clarification_prompt: str = ""
    severity: str = "none"


class DecisionMemoV2(BaseModel):
    title: str = "Decision memo"
    summary: str = ""
    drivers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationOptionV2] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    certainty_tags: list[str] = Field(default_factory=list)


class QualityReportV2(BaseModel):
    truth_score: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    provider_effective: str = ""
    fallback_used: dict[str, Any] = Field(default_factory=dict)
    certainty_tags: list[str] = Field(default_factory=list)
    grain_signature: str = ""
    denominator_semantics: str = ""


class InsightReportV2(BaseModel):
    assumptions: list[str] = Field(default_factory=list)
    certainty_tags: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceEntryV2] = Field(default_factory=list)
    recommendations: list[RecommendationOptionV2] = Field(default_factory=list)
    contradiction: ContradictionReportV2 | None = None
    decision_memo: DecisionMemoV2 | None = None
    summary_markdown: str = ""


class AssistantResponseV2(BaseModel):
    analysis_version: str = "v2"
    slice_signature: str = ""
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    stage_events: list[StageEventV2] = Field(default_factory=list)
    semantic_catalog: SemanticCatalogV2 | None = None
    conversation_state: ConversationStateV2 | None = None
    intent: IntentSpecV2 | None = None
    plan_set: PlanSetV2 | None = None
    query_plan: QueryPlanV2 | None = None
    execution: ExecutionResultV2 | None = None
    quality: QualityReportV2 | None = None
    insight: InsightReportV2 | None = None
