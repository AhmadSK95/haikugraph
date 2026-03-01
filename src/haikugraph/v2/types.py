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


class IntentSpecV2(BaseModel):
    goal: str
    is_followup: bool = False
    operations: list[str] = Field(default_factory=list)
    requires_validity_guard: bool = False
    strategy: str = "metric"


class PlanCandidateV2(BaseModel):
    candidate_id: str
    strategy: str
    objective_coverage_pct: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)


class PlanSetV2(BaseModel):
    selected_id: str = ""
    candidates: list[PlanCandidateV2] = Field(default_factory=list)


class QueryPlanV2(BaseModel):
    sql_hint: str = ""
    guardrails: list[str] = Field(default_factory=list)


class ExecutionResultV2(BaseModel):
    success: bool
    row_count: int = 0
    latency_ms: float = 0.0


class StageEventV2(BaseModel):
    sequence: int
    stage: str
    status: Literal["started", "completed", "failed", "skipped"]
    elapsed_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class QualityReportV2(BaseModel):
    truth_score: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    provider_effective: str = ""
    fallback_used: dict = Field(default_factory=dict)


class InsightReportV2(BaseModel):
    assumptions: list[str] = Field(default_factory=list)


class AssistantResponseV2(BaseModel):
    analysis_version: str = "v2"
    slice_signature: str = ""
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    stage_events: list[StageEventV2] = Field(default_factory=list)
    semantic_catalog: SemanticCatalogV2 | None = None
    intent: IntentSpecV2 | None = None
    plan_set: PlanSetV2 | None = None
    query_plan: QueryPlanV2 | None = None
    execution: ExecutionResultV2 | None = None
    quality: QualityReportV2 | None = None
    insight: InsightReportV2 | None = None
