"""FastAPI server for dataDa agentic POC."""

from __future__ import annotations

import base64
import concurrent.futures
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
import re
import socket
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from haikugraph.api.connection_registry import ConnectionRegistry
from haikugraph.api.runtime_store import RuntimeStore
from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.io.document_ingest import ingest_documents_to_duckdb
from haikugraph.io.onboarding_profile import load_or_create_onboarding_profile
from haikugraph.io.stream_snapshot import ingest_stream_snapshot_to_duckdb
from haikugraph.llm.router import DEFAULT_MODELS
from haikugraph.poc import AgenticAnalyticsTeam
from haikugraph.poc.source_truth import run_source_truth_suite
from haikugraph.services import (
    CorrectionsService,
    RulesService,
    ScenarioService,
    ToolsmithService,
    TrustService,
)
from haikugraph.v2.exceptions import (
    ContradictionDetectedError,
    PlanningError,
    PolicyViolationError,
    ProviderDegradedError,
    QueryCompilationError,
    QueryExecutionError,
)
from haikugraph.v2 import SemanticProfileCache, V2Orchestrator, apply_v2_compat_fields, profile_dataset
from haikugraph.v2.runtime import AutonomyConfig, RuntimeSelection, load_dotenv_file


DEFAULT_DB_CANDIDATES = (
    Path("./data/haikugraph.db"),
    Path("./data/datada.duckdb"),
    Path("./data/haikugraph.duckdb"),
)

_PROVIDER_SNAPSHOT_LOCK = threading.Lock()
_PROVIDER_SNAPSHOT_CACHE: ProvidersResponse | None = None
_PROVIDER_SNAPSHOT_CACHE_TS = 0.0


class LLMMode(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DETERMINISTIC = "deterministic"


class ProviderCheck(BaseModel):
    available: bool
    reason: str


class ProvidersResponse(BaseModel):
    default_mode: LLMMode
    recommended_mode: LLMMode
    checks: dict[str, ProviderCheck]


class QueryRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)
    db_connection_id: str = Field(default="default")
    constraints: dict[str, Any] = Field(default_factory=dict)
    scenario_set_id: str | None = Field(default=None, max_length=128)
    llm_mode: LLMMode = Field(default=LLMMode.AUTO)
    local_model: str | None = Field(default=None)
    local_narrator_model: str | None = Field(default=None)
    openai_model: str | None = Field(default=None)
    openai_narrator_model: str | None = Field(default=None)
    anthropic_model: str | None = Field(default=None)
    anthropic_narrator_model: str | None = Field(default=None)
    session_id: str | None = Field(default=None, max_length=128)
    storyteller_mode: bool = Field(default=False)
    autonomy_mode: str = Field(default="bounded", max_length=32)
    auto_correction: bool = Field(default=True)
    strict_truth: bool = Field(default=True)
    max_refinement_rounds: int = Field(default=2, ge=0, le=6)
    max_candidate_plans: int = Field(default=5, ge=1, le=12)
    tenant_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    role: str | None = Field(default=None, max_length=32)
    api_key: str | None = Field(default=None, max_length=512)


class DatasetProfileRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    tenant_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    role: str | None = Field(default=None, max_length=32)
    api_key: str | None = Field(default=None, max_length=512)


class FeedbackRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    trace_id: str | None = None
    session_id: str | None = None
    goal: str | None = None
    issue: str = Field(..., min_length=5, max_length=2000)
    suggested_fix: str | None = None
    severity: str = Field(default="medium", max_length=16)
    keyword: str | None = None
    target_table: str | None = None
    target_metric: str | None = None
    target_dimensions: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: str | None = None
    correction_id: str | None = None


class CorrectionRuleInfo(BaseModel):
    correction_id: str
    created_at: str = ""
    source: str = ""
    keyword: str
    target_table: str
    target_metric: str
    target_dimensions: list[str] = Field(default_factory=list)
    notes: str = ""
    weight: float = 1.0
    enabled: bool = True


class CorrectionsResponse(BaseModel):
    db_connection_id: str
    rules: list[CorrectionRuleInfo] = Field(default_factory=list)


class CorrectionToggleRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    correction_id: str = Field(..., min_length=1, max_length=128)
    enabled: bool = True


class CorrectionToggleResponse(BaseModel):
    success: bool
    message: str
    db_connection_id: str
    correction_id: str
    enabled: bool


class CorrectionRollbackRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    correction_id: str = Field(..., min_length=1, max_length=128)


class BusinessRuleInfo(BaseModel):
    rule_id: str
    created_at: str = ""
    updated_at: str = ""
    tenant_id: str = ""
    domain: str = ""
    name: str
    rule_type: str
    triggers: list[str] = Field(default_factory=list)
    action_payload: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    priority: float = 1.0
    status: str = "draft"
    source: str = ""
    created_by: str = ""
    approved_by: str = ""
    version: int = 1


class BusinessRulesResponse(BaseModel):
    db_connection_id: str
    rules: list[BusinessRuleInfo] = Field(default_factory=list)


class BusinessRuleCreateRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    domain: str = Field(default="general", max_length=64)
    name: str = Field(..., min_length=2, max_length=180)
    rule_type: str = Field(default="plan_override", max_length=64)
    triggers: list[str] = Field(default_factory=list)
    action_payload: dict[str, Any] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=500)
    priority: float = Field(default=1.0, ge=0.0, le=10.0)
    status: str = Field(default="draft", max_length=32)


class BusinessRuleStatusRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    rule_id: str = Field(..., min_length=1, max_length=128)
    status: str = Field(..., min_length=4, max_length=32)
    note: str = Field(default="", max_length=300)


class BusinessRuleUpdateRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    rule_id: str = Field(..., min_length=1, max_length=128)
    domain: str | None = Field(default=None, max_length=64)
    name: str | None = Field(default=None, min_length=2, max_length=180)
    rule_type: str | None = Field(default=None, max_length=64)
    triggers: list[str] | None = None
    action_payload: dict[str, Any] | None = None
    notes: str | None = Field(default=None, max_length=500)
    priority: float | None = Field(default=None, ge=0.0, le=10.0)
    status: str | None = Field(default=None, min_length=4, max_length=32)
    note: str = Field(default="", max_length=300)


class BusinessRuleRollbackRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    rule_id: str = Field(..., min_length=1, max_length=128)


class BusinessRuleActionResponse(BaseModel):
    success: bool
    message: str
    db_connection_id: str
    rule_id: str = ""
    status: str = ""


class FixRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    trace_id: str | None = None
    session_id: str | None = None
    goal: str | None = None
    issue: str = Field(..., min_length=5, max_length=2000)
    keyword: str = Field(..., min_length=2, max_length=180)
    domain: str = Field(default="general", max_length=64)
    target_table: str = Field(..., min_length=2, max_length=180)
    target_metric: str = Field(..., min_length=2, max_length=180)
    target_dimensions: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=500)


class FixResponse(BaseModel):
    success: bool
    message: str
    feedback_id: str = ""
    correction_id: str = ""
    rule_id: str = ""


class ToolCandidateInfo(BaseModel):
    tool_id: str
    created_at: str = ""
    updated_at: str = ""
    status: str
    source: str = ""
    title: str = ""
    sql_text: str = ""
    test_sql_text: str = ""
    test_success: bool = False
    test_message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCandidatesResponse(BaseModel):
    db_connection_id: str
    tools: list[ToolCandidateInfo] = Field(default_factory=list)


class ToolCandidateActionRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    tool_id: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(default="", max_length=280)


class ToolCandidateActionResponse(BaseModel):
    success: bool
    message: str
    db_connection_id: str
    tool_id: str
    status: str = ""


class TrustModeMetric(BaseModel):
    mode: str
    runs: int
    success_rate: float
    avg_confidence: float
    avg_execution_ms: float


class TrustFailureSample(BaseModel):
    created_at: str
    connection_id: str
    llm_mode: str
    goal: str = ""
    warning_terms: list[str] = Field(default_factory=list)


class TrustDashboardResponse(BaseModel):
    generated_at: str
    tenant_id: str
    window_hours: int
    runs: int
    success_runs: int
    success_rate: float
    avg_confidence: float
    avg_execution_ms: float
    p95_execution_ms: float
    total_warnings: int
    by_mode: list[TrustModeMetric] = Field(default_factory=list)
    parity_summary: dict[str, Any] = Field(default_factory=dict)
    recent_failures: list[TrustFailureSample] = Field(default_factory=list)


class SLOBreachMetric(BaseModel):
    metric: str
    actual: float
    target: float
    direction: str = "min"
    delta: float


class SLOEvaluationResponse(BaseModel):
    generated_at: str
    tenant_id: str
    window_hours: int
    runs: int
    status: str
    burn_rate: float
    success_rate: float
    p95_execution_ms: float
    warning_rate: float
    targets: dict[str, Any] = Field(default_factory=dict)
    breaches: list[SLOBreachMetric] = Field(default_factory=list)


class IncidentEvent(BaseModel):
    incident_id: str
    created_at: str
    updated_at: str
    tenant_id: str
    severity: str
    status: str
    source: str
    title: str
    summary: str
    fingerprint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IncidentsResponse(BaseModel):
    incidents: list[IncidentEvent] = Field(default_factory=list)


class IncidentAcknowledgeRequest(BaseModel):
    incident_id: str = Field(..., min_length=1, max_length=128)
    status: str = Field(default="acknowledged", max_length=32)
    note: str = Field(default="", max_length=400)


class AsyncQueryRequest(QueryRequest):
    priority: str = Field(default="normal", max_length=24)


class AsyncQueryAccepted(BaseModel):
    success: bool
    message: str
    job_id: str
    status: str
    db_connection_id: str
    session_id: str
    tenant_id: str


class AsyncJobStatusResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    db_connection_id: str
    session_id: str
    tenant_id: str
    runtime_ms: float = 0.0
    response: dict[str, Any] | None = None
    error: str | None = None


class SessionClearRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    session_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=128)


class SourceTruthCaseResult(BaseModel):
    case_id: str
    question: str
    note: str = ""
    status: str
    reason: str = ""
    exact_match: bool
    latency_ms: float
    expected_sql: str
    actual_sql: str
    expected_cols: list[str] = Field(default_factory=list)
    actual_cols: list[str] = Field(default_factory=list)
    expected_rows: list[Any] = Field(default_factory=list)
    actual_rows: list[Any] = Field(default_factory=list)


class SourceTruthResponse(BaseModel):
    db_connection_id: str
    mode_requested: str
    mode_actual: str
    provider: str = ""
    cases: int
    evaluated_cases: int
    exact_matches: int
    accuracy_pct: float
    avg_latency_ms: float
    parity_summary: dict[str, Any] = Field(default_factory=dict)
    runs: list[SourceTruthCaseResult] = Field(default_factory=list)


class ConnectionInfo(BaseModel):
    id: str
    kind: str
    path: str
    description: str = ""
    enabled: bool = True
    is_default: bool = False
    exists: bool = False
    db_size_bytes: int = 0


class ConnectionsResponse(BaseModel):
    default_connection_id: str
    connections: list[ConnectionInfo] = Field(default_factory=list)


class ConnectorCapability(BaseModel):
    kind: str
    query_routing_supported: bool
    mirror_ingest_supported: bool
    notes: str = ""


class ConnectorsResponse(BaseModel):
    connectors: list[ConnectorCapability] = Field(default_factory=list)


class ConnectionUpsertRequest(BaseModel):
    connection_id: str = Field(..., min_length=1, max_length=64)
    kind: str = Field(default="duckdb", min_length=1, max_length=32)
    path: str = Field(..., min_length=1, max_length=4096)
    description: str = Field(default="", max_length=280)
    enabled: bool = True
    set_default: bool = False
    validate_connection: bool = True


class ConnectionSetDefaultRequest(BaseModel):
    connection_id: str = Field(..., min_length=1, max_length=64)


class ConnectionTestRequest(BaseModel):
    connection_id: str | None = None
    kind: str = Field(default="duckdb", min_length=1, max_length=32)
    path: str | None = None


class ConnectionActionResponse(BaseModel):
    success: bool
    message: str
    connection: ConnectionInfo | None = None


class ScenarioSetInfo(BaseModel):
    scenario_set_id: str
    created_at: str = ""
    updated_at: str = ""
    tenant_id: str = ""
    connection_id: str = ""
    name: str
    assumptions: list[str] = Field(default_factory=list)
    status: str = "active"
    version: int = 1


class ScenarioSetUpsertRequest(BaseModel):
    db_connection_id: str = Field(default="default")
    scenario_set_id: str | None = Field(default=None, max_length=128)
    name: str = Field(..., min_length=2, max_length=180)
    assumptions: list[str] = Field(default_factory=list)
    status: str = Field(default="active", max_length=32)


class ScenarioSetActionResponse(BaseModel):
    success: bool
    message: str
    db_connection_id: str
    scenario_set: ScenarioSetInfo | None = None


class ScenarioSetsResponse(BaseModel):
    db_connection_id: str
    scenario_sets: list[ScenarioSetInfo] = Field(default_factory=list)


class LegacyAskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class LegacyAskResponse(BaseModel):
    final_answer: str
    intent: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    queries: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    db_exists: bool
    db_path: str
    db_size_bytes: int = 0
    semantic_ready: bool = False
    default_connection_id: str = "default"
    available_connections: int = 1
    active_connection_kind: str = "duckdb"
    onboarding_profile_version: str = ""
    onboarding_profile_path: str = ""
    runtime_store_path: str = ""
    version: str = "2.0.0-poc"


class AgentInfo(BaseModel):
    name: str
    role: str
    description: str
    inputs: list[str]
    outputs: list[str]


class ArchitectureResponse(BaseModel):
    system_name: str = "dataDa Agentic Analytics Team"
    version: str = "2.0.0-poc"
    description: str = "Hierarchical multi-agent analytics and data-engineering team"
    pipeline_flow: list[str] = [
        "1. Chief Analyst Agent - Supervises team and decomposes mission",
        "2. Connection Router - Resolves db_connection_id to a governed data source",
        "3. Memory Agent - Recalls similar runs + learned correction rules",
        "4. Blackboard - Shares artifacts between agents with explicit producer/consumer edges",
        "5. Intake Agent - Clarifies intent, metrics, filters, time scope",
        "6. Discovery Planner + Catalog Profiler - Deep dataset-overview intelligence for broad/vague questions",
        "7. Document Retrieval Agent - Citation-grounded chunk retrieval for text-heavy sources",
        "8. Semantic Retrieval Agent - Maps query to semantic marts",
        "9. Planning Agent - Produces task graph and metric definitions",
        "10. Specialist Agents - Transactions, Customers, Revenue, Risk",
        "11. Query Engineer + Execution Agents - Compile and run SQL",
        "12. Audit Agent - Validates consistency, grounding, replay checks",
        "13. Autonomy Agent - Evaluates hypotheses with confidence decomposition + contradiction resolution",
        "14. Toolsmith Agent - Captures probe intelligence into staged/promoted reusable tools",
        "15. Narrative + Visualization Agents - Final insight and chart spec",
        "16. Trust Agent - Records reliability telemetry and drift indicators",
        "17. SLO/Incident Agent - Evaluates reliability targets and emits governed incident hooks",
    ]
    guardrails: list[str] = [
        "Read-only SQL only",
        "Blocked destructive keywords",
        "Bounded result sizes",
        "Bounded autonomy controls (candidate/iteration caps)",
        "Tenant-aware session isolation",
        "Per-tenant query budgets",
        "Role-gated mutation endpoints (analyst/admin)",
        "Tenant-scoped memory, corrections, and toolsmith state",
        "Structured evidence packets",
        "Per-agent trace for every answer",
        "Runtime mode transparency (auto/local/openai/deterministic)",
    ]
    response_includes: list[str] = [
        "answer_markdown",
        "confidence + audit checks",
        "executed SQL",
        "sample rows",
        "agent_trace",
        "chart_spec",
    ]
    agents: list[AgentInfo] = []


class CapabilityScoreItem(BaseModel):
    capability_id: str
    name: str
    category: str
    status: str
    evidence: str = ""
    gap_to_close: str = ""
    requirement: str = ""


class CapabilityScoreCounts(BaseModel):
    total: int
    done: int
    partial: int
    gap: int
    np_strict: float
    np_reality: float


class CapabilityScoreboardResponse(BaseModel):
    tracker_path: str
    tracker_last_updated: str = ""
    generated_at_epoch_ms: int
    counts: CapabilityScoreCounts
    remaining: list[CapabilityScoreItem] = Field(default_factory=list)
    capabilities: list[CapabilityScoreItem] = Field(default_factory=list)
    tracker_score: float | None = None
    truth_score: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    score_drift: float | None = None


class QualityRunSummary(BaseModel):
    run_id: str
    generated_at: str = ""
    kind: str = ""
    overall_pass_rate: float | None = None
    truth_score: float | None = None
    path: str


class QualityLatestResponse(BaseModel):
    generated_at_epoch_ms: int
    latest_runs: list[QualityRunSummary] = Field(default_factory=list)
    composite_truth_score: float | None = None


class QualityRunDetailResponse(BaseModel):
    run_id: str
    path: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DatasetProfileResponse(BaseModel):
    db_connection_id: str
    dataset_signature: str
    schema_signature: str = ""
    table_count: int
    high_risk_join_edges: int
    sparse_table_count: int
    semantic_cache_hit: bool = False
    schema_drift_detected: bool = False
    profile: dict[str, Any] = Field(default_factory=dict)


class StageSLOResponse(BaseModel):
    generated_at_epoch_ms: int
    stage_budget_ms: dict[str, int] = Field(default_factory=dict)
    observed_p95_ms: dict[str, float] = Field(default_factory=dict)


class CutoverArtifactStatus(BaseModel):
    name: str
    path: str
    exists: bool


class CutoverReadinessResponse(BaseModel):
    generated_at_epoch_ms: int
    default_runtime_version: str
    canary_ready: bool
    release_gate_passed: bool
    latest_truth_report: str = ""
    composite_truth_score: float | None = None
    floor_violations: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[CutoverArtifactStatus] = Field(default_factory=list)


class LocalModelOption(BaseModel):
    name: str
    installed: bool
    tier: str
    recommended_for: str


class LocalModelsResponse(BaseModel):
    available: bool
    base_url: str
    active_intent_model: str | None = None
    active_narrator_model: str | None = None
    options: list[LocalModelOption] = Field(default_factory=list)
    reason: str = ""


class LocalModelSelectRequest(BaseModel):
    model: str = Field(..., min_length=1)
    narrator_model: str | None = None


class LocalModelPullRequest(BaseModel):
    model: str = Field(..., min_length=1)
    activate_after_download: bool = True


class LocalModelActionResponse(BaseModel):
    success: bool
    message: str
    active_intent_model: str | None = None
    active_narrator_model: str | None = None


class CloudModelOption(BaseModel):
    name: str
    tier: str
    recommended_for: str


class CloudModelsResponse(BaseModel):
    provider: str
    available: bool
    reason: str = ""
    active_intent_model: str | None = None
    active_narrator_model: str | None = None
    options: list[CloudModelOption] = Field(default_factory=list)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


OLLAMA_MODEL_HINTS: dict[str, tuple[str, str]] = {
    "qwen2.5:7b-instruct": ("balanced", "default local reasoning"),
    "llama3.1:8b": ("balanced", "strong narrative responses"),
    "mistral:7b": ("balanced", "fast and stable local inference"),
}

# Intentionally removed from active catalog due poor benchmark profile
# (latency/quality trade-off for this project baseline).
UNSUPPORTED_LOCAL_MODELS = {
    "qwen2.5:14b-instruct",
    "llama3.2:latest",
}


def _is_supported_local_model(model_name: str) -> bool:
    return str(model_name or "").strip().lower() not in {
        m.lower() for m in UNSUPPORTED_LOCAL_MODELS
    }


def _filter_supported_local_models(models: list[str]) -> list[str]:
    return [m for m in models if _is_supported_local_model(m)]

OPENAI_MODEL_HINTS: dict[str, tuple[str, str]] = {
    "gpt-4o": ("high", "best quality for difficult query decomposition"),
    "gpt-4.1": ("high", "strong reasoning and longer context handling"),
    "gpt-4o-mini": ("balanced", "cost-efficient default for most analytics prompts"),
    "gpt-4.1-mini": ("balanced", "fast structured reasoning with lower latency"),
    "o4-mini": ("balanced", "deep tool-use style reasoning"),
    "o3-mini": ("balanced", "strong chain-of-thought style planning"),
}

ANTHROPIC_MODEL_HINTS: dict[str, tuple[str, str]] = {
    "claude-sonnet-4-6": ("high", "best Claude quality for planning and analyst narration"),
    "claude-haiku-4-5-20251001": ("fast", "low-latency Claude responses"),
}


def _model_tier(model_name: str) -> str:
    lower = model_name.lower()
    if any(x in lower for x in ["70b", "34b", "32b", "27b", "14b"]):
        return "high"
    if any(x in lower for x in ["8b", "7b", "6b"]):
        return "balanced"
    if any(x in lower for x in ["3b", "2b", "1b"]):
        return "fast"
    return "balanced"


def _build_local_model_options(installed: list[str]) -> list[LocalModelOption]:
    installed = _filter_supported_local_models(installed)
    installed_set = {m.lower() for m in installed}
    ordered_names: list[str] = []

    for model_name in OLLAMA_MODEL_HINTS:
        if _is_supported_local_model(model_name) and model_name not in ordered_names:
            ordered_names.append(model_name)
    for model_name in installed:
        if model_name not in ordered_names:
            ordered_names.append(model_name)

    options: list[LocalModelOption] = []
    for model_name in ordered_names:
        hint = OLLAMA_MODEL_HINTS.get(model_name)
        tier = hint[0] if hint else _model_tier(model_name)
        recommended_for = hint[1] if hint else "general local analytics"
        options.append(
            LocalModelOption(
                name=model_name,
                installed=model_name.lower() in installed_set,
                tier=tier,
                recommended_for=recommended_for,
            )
        )
    return options


def _build_cloud_model_options(hints: dict[str, tuple[str, str]]) -> list[CloudModelOption]:
    return [
        CloudModelOption(
            name=name,
            tier=tier,
            recommended_for=recommended_for,
        )
        for name, (tier, recommended_for) in hints.items()
    ]


def _fetch_ollama_models(base_url: str) -> list[str]:
    response = requests.get(f"{base_url}/api/tags", timeout=2.5)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    names = []
    for model in models:
        name = model.get("name") or model.get("model")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _pick_ollama_model(available_models: list[str], preferred: list[str]) -> str | None:
    if not available_models:
        return None
    lowered = {m.lower(): m for m in available_models}
    for choice in preferred:
        if choice.lower() in lowered:
            return lowered[choice.lower()]
    return available_models[0]


def _configure_ollama_models(available_models: list[str]) -> tuple[str | None, str | None]:
    available_models = _filter_supported_local_models(available_models)
    # Prefer instruction-tuned model for intent extraction.
    intent_choice = _pick_ollama_model(
        available_models,
        [
            "qwen2.5:7b-instruct",
            "llama3.1:8b",
            "mistral:7b",
        ],
    )
    # Prefer fluent summarizer for narrative generation.
    narrator_choice = _pick_ollama_model(
        available_models,
        [
            "llama3.1:8b",
            "qwen2.5:7b-instruct",
            "mistral:7b",
        ],
    )

    if intent_choice:
        os.environ["HG_OLLAMA_INTENT_MODEL"] = intent_choice
    if narrator_choice:
        os.environ["HG_OLLAMA_NARRATOR_MODEL"] = narrator_choice

    return intent_choice, narrator_choice


def _activate_local_models(intent_model: str, narrator_model: str | None = None) -> None:
    clean_intent = intent_model.strip()
    clean_narrator = (narrator_model or intent_model).strip()
    if clean_intent and not _is_supported_local_model(clean_intent):
        raise ValueError(
            f"Model '{clean_intent}' is disabled in this build due benchmark underperformance."
        )
    if clean_narrator and not _is_supported_local_model(clean_narrator):
        raise ValueError(
            f"Narrator model '{clean_narrator}' is disabled in this build due benchmark underperformance."
        )
    if clean_intent:
        os.environ["HG_OLLAMA_INTENT_MODEL"] = clean_intent
    if clean_narrator:
        os.environ["HG_OLLAMA_NARRATOR_MODEL"] = clean_narrator


def _get_local_models_state() -> LocalModelsResponse:
    base_url = os.environ.get("HG_OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        installed_all = _fetch_ollama_models(base_url)
        installed = _filter_supported_local_models(installed_all)
        if not installed:
            disabled_installed = [
                m for m in installed_all if not _is_supported_local_model(m)
            ]
            disabled_note = (
                f" (disabled models detected: {', '.join(disabled_installed)})"
                if disabled_installed
                else ""
            )
            return LocalModelsResponse(
                available=False,
                base_url=base_url,
                options=_build_local_model_options([]),
                reason=f"Ollama reachable but no supported models installed.{disabled_note}",
            )
        intent_model, narrator_model = _configure_ollama_models(installed)
        return LocalModelsResponse(
            available=True,
            base_url=base_url,
            active_intent_model=os.environ.get("HG_OLLAMA_INTENT_MODEL") or intent_model,
            active_narrator_model=os.environ.get("HG_OLLAMA_NARRATOR_MODEL") or narrator_model,
            options=_build_local_model_options(installed),
            reason=f"connected ({len(installed)} models)",
        )
    except Exception as exc:  # pragma: no cover
        return LocalModelsResponse(
            available=False,
            base_url=base_url,
            options=_build_local_model_options([]),
            reason=str(exc),
        )


def _get_openai_models_state() -> CloudModelsResponse:
    check = _openai_check()
    default_intent = DEFAULT_MODELS.get("openai", {}).get("intent", "gpt-4o-mini")
    default_narrator = DEFAULT_MODELS.get("openai", {}).get("narrator", "gpt-4o-mini")
    return CloudModelsResponse(
        provider="openai",
        available=check.available,
        reason=check.reason,
        active_intent_model=(os.environ.get("HG_OPENAI_INTENT_MODEL") or default_intent),
        active_narrator_model=(os.environ.get("HG_OPENAI_NARRATOR_MODEL") or default_narrator),
        options=_build_cloud_model_options(OPENAI_MODEL_HINTS),
    )


def _get_anthropic_models_state() -> CloudModelsResponse:
    check = _anthropic_check()
    default_intent = DEFAULT_MODELS.get("anthropic", {}).get("intent", "claude-haiku-4-5-20251001")
    default_narrator = DEFAULT_MODELS.get("anthropic", {}).get("narrator", "claude-haiku-4-5-20251001")
    return CloudModelsResponse(
        provider="anthropic",
        available=check.available,
        reason=check.reason,
        active_intent_model=(os.environ.get("HG_ANTHROPIC_INTENT_MODEL") or default_intent),
        active_narrator_model=(os.environ.get("HG_ANTHROPIC_NARRATOR_MODEL") or default_narrator),
        options=_build_cloud_model_options(ANTHROPIC_MODEL_HINTS),
    )


def _get_db_path() -> Path:
    env_path = os.environ.get("HG_DB_PATH")
    if env_path:
        return Path(env_path)
    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_DB_CANDIDATES[-1]


def _get_connection_registry_path() -> Path:
    env_path = os.environ.get("HG_CONNECTION_REGISTRY_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path("./data/connections.json")


def _get_runtime_store_path() -> Path:
    env_path = os.environ.get("HG_RUNTIME_STORE_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path("./data/runtime_store.duckdb")


def _get_product_gap_tracker_path() -> Path:
    env_path = os.environ.get("HG_PRODUCT_GAP_TRACKER_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[3] / "PRODUCT_GAP_TRACKER.md"


def _get_reports_dir() -> Path:
    env_path = os.environ.get("HG_REPORTS_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[3] / "reports"


def _repo_root_path() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_version() -> str:
    raw = str(os.environ.get("HG_RUNTIME_VERSION", "v2")).strip().lower()
    if raw == "v2":
        return raw
    return "v2"


def _latest_report_file(pattern: str) -> Path | None:
    reports_dir = _get_reports_dir()
    if not reports_dir.exists():
        return None
    matches = list(reports_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_quality_summary(
    *,
    path: Path,
    kind: str,
    pass_rate: float | None,
    truth_score: float | None,
    generated_at: str,
) -> QualityRunSummary:
    return QualityRunSummary(
        run_id=path.stem,
        generated_at=generated_at,
        kind=kind,
        overall_pass_rate=pass_rate,
        truth_score=truth_score,
        path=str(path),
    )


def _latest_quality_runs() -> list[QualityRunSummary]:
    runs: list[QualityRunSummary] = []
    truth_report = _latest_report_file("v2_qa_truth_report_*.json")
    if truth_report:
        payload = _load_json(truth_report) or {}
        summary = payload.get("summary") or {}
        score = summary.get("composite_truth_score")
        if not isinstance(score, (int, float)):
            score = payload.get("truth_score")
        pass_rate = summary.get("overall_pass_rate_pct")
        generated_at = str(payload.get("generated_at") or summary.get("generated_at") or "")
        runs.append(
            _build_quality_summary(
                path=truth_report,
                kind="v2_truth_qa",
                pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
                truth_score=float(score) if isinstance(score, (int, float)) else None,
                generated_at=generated_at,
            )
        )

    round11 = _latest_report_file("qa_round11_blackbox_fresh_*.json")
    if round11:
        payload = _load_json(round11) or {}
        summary = ((payload.get("summary") or {}).get("overall") or {})
        pass_rate = summary.get("overall_pass_rate")
        generated_at = str(payload.get("generated_at") or "")
        runs.append(
            _build_quality_summary(
                path=round11,
                kind="blackbox_round11",
                pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
                truth_score=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
                generated_at=generated_at,
            )
        )

    semantic = _latest_report_file("blackbox_semantic_probe_*.json")
    if semantic:
        payload = _load_json(semantic) or {}
        summary = payload.get("summary") or {}
        pass_rate = summary.get("expectation_pass_rate_pct")
        generated_at = str(payload.get("generated_at") or summary.get("generated_at") or "")
        runs.append(
            _build_quality_summary(
                path=semantic,
                kind="semantic_probe",
                pass_rate=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
                truth_score=float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
                generated_at=generated_at,
            )
        )

    latency = _latest_report_file("latency_optimization_check_*.json")
    if latency:
        payload = _load_json(latency) or {}
        generated_at = str(payload.get("generated_at") or "")
        runs.append(
            _build_quality_summary(
                path=latency,
                kind="latency_check",
                pass_rate=None,
                truth_score=None,
                generated_at=generated_at,
            )
        )
    return runs


def _composite_truth_score(runs: list[QualityRunSummary]) -> float | None:
    v2_truth = next((r for r in runs if r.kind == "v2_truth_qa" and r.truth_score is not None), None)
    if v2_truth is not None and v2_truth.truth_score is not None:
        return round(float(v2_truth.truth_score), 2)

    # Weight black-box higher than semantic probes for release realism.
    weighted: list[tuple[float, float]] = []
    for run in runs:
        if run.truth_score is None:
            continue
        if run.kind == "blackbox_round11":
            weighted.append((float(run.truth_score), 0.7))
        elif run.kind == "semantic_probe":
            weighted.append((float(run.truth_score), 0.3))
    if not weighted:
        return None
    denom = sum(w for _, w in weighted)
    score = sum(val * w for val, w in weighted) / max(denom, 1e-9)
    return round(score, 2)


def _stage_slo_budget_ms() -> dict[str, int]:
    executor_budget = int(os.environ.get("HG_STAGE_BUDGET_EXECUTOR_MS", "6000"))
    evaluator_budget = int(os.environ.get("HG_STAGE_BUDGET_EVALUATOR_MS", "1200"))
    insight_budget = int(os.environ.get("HG_STAGE_BUDGET_INSIGHT_ENGINE_MS", "1200"))
    return {
        "semantic_profiler": int(os.environ.get("HG_STAGE_BUDGET_SEMANTIC_PROFILER_MS", "900")),
        "intent_engine": int(os.environ.get("HG_STAGE_BUDGET_INTENT_ENGINE_MS", "900")),
        "planner": int(os.environ.get("HG_STAGE_BUDGET_PLANNER_MS", "1500")),
        "query_compiler": int(os.environ.get("HG_STAGE_BUDGET_QUERY_COMPILER_MS", "1200")),
        "executor": executor_budget,
        "executor_delegate": executor_budget,
        "evaluator": evaluator_budget,
        "insight_engine": insight_budget,
        "evaluator_insight": evaluator_budget + insight_budget,
    }


def _stage_slo_breaches(stage_timings_ms: dict[str, float] | None) -> list[dict[str, Any]]:
    if not isinstance(stage_timings_ms, dict) or not stage_timings_ms:
        return []
    budget = _stage_slo_budget_ms()
    breaches: list[dict[str, Any]] = []
    for stage, limit_ms in budget.items():
        raw = stage_timings_ms.get(stage)
        if raw is None:
            continue
        try:
            actual = float(raw)
        except Exception:
            continue
        if actual > float(limit_ms):
            breaches.append(
                {
                    "stage": stage,
                    "actual_ms": round(actual, 2),
                    "budget_ms": int(limit_ms),
                    "delta_ms": round(actual - float(limit_ms), 2),
                }
            )
    return breaches


def _profile_dataset_cached(
    app: FastAPI,
    *,
    db_path: Path,
) -> tuple[Any, dict[str, Any]]:
    cache: SemanticProfileCache | None = getattr(app.state, "semantic_profile_cache", None)
    if cache is None:
        profile = profile_dataset(str(db_path))
        return profile, {
            "cache_hit": False,
            "cache_key": str(profile.dataset_signature or ""),
            "dataset_signature": str(profile.dataset_signature or ""),
            "schema_signature": str(getattr(profile, "schema_signature", "") or ""),
        }
    profile, meta = cache.get_or_build(str(db_path), profile_dataset)
    clean_meta = dict(meta or {})
    clean_meta["dataset_signature"] = str(clean_meta.get("dataset_signature") or profile.dataset_signature or "")
    clean_meta["schema_signature"] = str(
        clean_meta.get("schema_signature") or getattr(profile, "schema_signature", "") or ""
    )
    return profile, clean_meta


def _record_schema_drift_if_any(
    app: FastAPI,
    *,
    tenant_id: str,
    connection_id: str,
    profile: Any,
    cache_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_signature = str(getattr(profile, "dataset_signature", "") or "")
    schema_signature = str(getattr(profile, "schema_signature", "") or "")
    if not dataset_signature and cache_meta:
        dataset_signature = str(cache_meta.get("dataset_signature") or "")
    if not schema_signature and cache_meta:
        schema_signature = str(cache_meta.get("schema_signature") or "")
    if not dataset_signature and not schema_signature:
        return {"drift_detected": False}
    drift = app.state.runtime_store.record_schema_signature(
        tenant_id=tenant_id,
        connection_id=connection_id,
        dataset_signature=dataset_signature,
        schema_signature=schema_signature,
        metadata={
            "cache_hit": bool((cache_meta or {}).get("cache_hit")),
            "table_count": int((getattr(profile, "quality_summary", {}) or {}).get("table_count") or 0),
            "high_risk_join_edges": int(
                (getattr(profile, "quality_summary", {}) or {}).get("high_risk_join_edges") or 0
            ),
        },
    )
    if not bool(drift.get("drift_detected")):
        return drift
    incident = app.state.runtime_store.record_incident(
        tenant_id=tenant_id,
        severity="medium",
        source="schema_drift",
        title="Dataset schema drift detected",
        summary=f"Schema signature changed for connection '{connection_id}'.",
        fingerprint=f"schema_drift:{tenant_id}:{connection_id}",
        metadata={
            "connection_id": connection_id,
            "dataset_signature": dataset_signature,
            "schema_signature": schema_signature,
            "previous_schema_signature": str(drift.get("previous_schema_signature") or ""),
        },
        dedupe_window_minutes=int(app.state.incident_dedupe_minutes),
    )
    if incident.get("created"):
        _emit_incident_webhook(
            app,
            {
                "event": "incident_created",
                "tenant_id": tenant_id,
                "connection_id": connection_id,
                "severity": "medium",
                "title": "Dataset schema drift detected",
                "incident_id": incident.get("incident_id"),
            },
        )
    return drift


def _quality_run_by_id(run_id: str) -> dict[str, Any] | None:
    clean = str(run_id or "").strip()
    if not clean:
        return None
    reports_dir = _get_reports_dir()
    if not reports_dir.exists():
        return None
    safe_pattern = re.sub(r"[^A-Za-z0-9_\-\.]", "", clean)
    if not safe_pattern:
        return None
    for path in reports_dir.glob("*.json"):
        if path.stem == safe_pattern:
            payload = _load_json(path)
            if payload is None:
                continue
            return {"path": str(path), "payload": payload}
    return None


def _cutover_artifacts() -> list[CutoverArtifactStatus]:
    root = _repo_root_path()
    latest_baseline = _latest_report_file("v2_baseline_lock_*.json")
    latest_cutover_drill = _latest_report_file("v2_cutover_drill_*.json")
    latest_drift_alarm = _latest_report_file("v2_quality_drift_alarm_*.json")
    rows = [
        ("cutover_runbook", root / "docs" / "v2_cutover_runbook.md"),
        ("incident_playbook", root / "docs" / "v2_incident_playbook.md"),
        ("quality_review_cadence", root / "docs" / "v2_quality_review_cadence.md"),
        ("v1_decommission_criteria", root / "docs" / "v1_decommission_criteria.md"),
        ("decommission_postmortem", root / "docs" / "v2_decommission_postmortem.md"),
        (
            "baseline_lock_manifest",
            latest_baseline or (root / "reports" / "v2_baseline_lock_*.json"),
        ),
        (
            "cutover_drill_report",
            latest_cutover_drill or (root / "reports" / "v2_cutover_drill_*.json"),
        ),
        (
            "quality_drift_alarm_report",
            latest_drift_alarm or (root / "reports" / "v2_quality_drift_alarm_*.json"),
        ),
    ]
    return [
        CutoverArtifactStatus(name=name, path=str(path), exists=path.exists())
        for name, path in rows
    ]


def _normalize_capability_status(raw_status: str) -> str:
    status = str(raw_status or "").strip().upper()
    if status in {"DONE", "PARTIAL", "GAP"}:
        return status
    return "GAP"


def _parse_capability_row(line: str) -> CapabilityScoreItem | None:
    text = str(line or "").strip()
    if not text.startswith("|"):
        return None
    cells = [cell.strip() for cell in text.strip("|").split("|")]
    if len(cells) < 7:
        return None
    capability_id = str(cells[0] or "").strip().upper()
    if (
        len(capability_id) != 3
        or capability_id[0] not in {"A", "T"}
        or not capability_id[1:].isdigit()
    ):
        return None
    return CapabilityScoreItem(
        capability_id=capability_id,
        name=str(cells[1] or "").strip(),
        category="analyst_skill" if capability_id.startswith("A") else "team_capability",
        status=_normalize_capability_status(str(cells[3] or "")),
        evidence=str(cells[4] or "").strip(),
        gap_to_close=str(cells[5] or "").strip(),
        requirement=str(cells[6] or "").strip(),
    )


def _read_capability_scoreboard() -> CapabilityScoreboardResponse:
    tracker_path = _get_product_gap_tracker_path()
    if not tracker_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Capability tracker not found at {tracker_path}",
        )
    try:
        raw = tracker_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to read capability tracker: {exc}")

    tracker_last_updated = ""
    capabilities: list[CapabilityScoreItem] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not tracker_last_updated and stripped.lower().startswith("last updated:"):
            tracker_last_updated = stripped.split(":", 1)[1].strip()
        item = _parse_capability_row(stripped)
        if item is not None:
            capabilities.append(item)

    if not capabilities:
        raise HTTPException(
            status_code=503,
            detail="Capability tracker did not contain parseable A##/T## capability rows.",
        )

    done = sum(1 for row in capabilities if row.status == "DONE")
    partial = sum(1 for row in capabilities if row.status == "PARTIAL")
    gap = sum(1 for row in capabilities if row.status == "GAP")
    total = len(capabilities)
    np_strict = round((done / total) * 100.0, 2) if total else 0.0
    np_reality = round(((done + (0.5 * partial)) / total) * 100.0, 2) if total else 0.0
    remaining = [row for row in capabilities if row.status != "DONE"]
    quality_runs = _latest_quality_runs()
    truth_score = _composite_truth_score(quality_runs)
    tracker_score = np_reality

    return CapabilityScoreboardResponse(
        tracker_path=str(tracker_path),
        tracker_last_updated=tracker_last_updated,
        generated_at_epoch_ms=int(time.time() * 1000),
        counts=CapabilityScoreCounts(
            total=total,
            done=done,
            partial=partial,
            gap=gap,
            np_strict=np_strict,
            np_reality=np_reality,
        ),
        remaining=remaining,
        capabilities=capabilities,
        tracker_score=tracker_score,
        truth_score=truth_score,
        evidence_refs=[run.path for run in quality_runs],
        score_drift=(
            round(float(truth_score) - float(tracker_score), 2)
            if truth_score is not None
            else None
        ),
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _query_cache_enabled(app: FastAPI) -> bool:
    return bool(getattr(app.state, "query_response_cache_enabled", False))


def _query_cache_key(
    *,
    request: QueryRequest,
    tenant_id: str,
    connection_id: str,
    runtime: RuntimeSelection,
    autonomy: AutonomyConfig,
    history_turns: int,
) -> str:
    # Follow-up questions are context-dependent; do not cache them.
    if history_turns > 0:
        return ""
    payload = {
        "runtime_version": _runtime_version(),
        "tenant_id": tenant_id,
        "connection_id": connection_id,
        "goal": str(request.goal or "").strip(),
        "llm_mode": str(request.llm_mode.value),
        "constraints": dict(request.constraints or {}),
        "scenario_set_id": str(request.scenario_set_id or ""),
        "runtime_mode": str(runtime.mode),
        "provider": str(runtime.provider or ""),
        "model_overrides": {
            "local_model": request.local_model,
            "local_narrator_model": request.local_narrator_model,
            "openai_model": request.openai_model,
            "openai_narrator_model": request.openai_narrator_model,
            "anthropic_model": request.anthropic_model,
            "anthropic_narrator_model": request.anthropic_narrator_model,
        },
        "autonomy": {
            "mode": autonomy.mode,
            "auto_correction": bool(autonomy.auto_correction),
            "strict_truth": bool(autonomy.strict_truth),
            "max_refinement_rounds": int(autonomy.max_refinement_rounds),
            "max_candidate_plans": int(autonomy.max_candidate_plans),
        },
        "storyteller_mode": bool(request.storyteller_mode),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _query_cache_get(app: FastAPI, key: str) -> dict[str, Any] | None:
    if not key or not _query_cache_enabled(app):
        return None
    ttl_seconds = float(getattr(app.state, "query_response_cache_ttl_seconds", 0.0) or 0.0)
    if ttl_seconds <= 0:
        return None
    now = time.time()
    with app.state.query_response_cache_lock:
        cache: OrderedDict[str, dict[str, Any]] = app.state.query_response_cache
        stale_keys = [
            item_key
            for item_key, item in cache.items()
            if (now - float(item.get("created_at") or 0.0)) > ttl_seconds
        ]
        for stale in stale_keys:
            cache.pop(stale, None)
        hit = cache.get(key)
        if not hit:
            return None
        cache.move_to_end(key)
        payload = hit.get("response_payload")
        return dict(payload) if isinstance(payload, dict) else None


def _query_cache_set(app: FastAPI, key: str, response_payload: dict[str, Any]) -> None:
    if not key or not _query_cache_enabled(app):
        return
    max_entries = max(8, int(getattr(app.state, "query_response_cache_max_entries", 128)))
    with app.state.query_response_cache_lock:
        cache: OrderedDict[str, dict[str, Any]] = app.state.query_response_cache
        cache[key] = {
            "created_at": time.time(),
            "response_payload": dict(response_payload),
        }
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)


def _load_api_key_policies() -> dict[str, dict[str, str]]:
    raw = os.environ.get("HG_API_KEYS_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}

    out: dict[str, dict[str, str]] = {}
    for key, cfg in parsed.items():
        token = str(key or "").strip()
        if not token:
            continue
        if isinstance(cfg, str):
            out[token] = {"tenant_id": cfg.strip() or "public", "role": "analyst"}
            continue
        if isinstance(cfg, dict):
            tenant_id = str(cfg.get("tenant_id") or cfg.get("tenant") or "public").strip() or "public"
            role = str(cfg.get("role") or "analyst").strip().lower() or "analyst"
            out[token] = {"tenant_id": tenant_id, "role": role}
    return out


def _decode_bearer_claims(authorization_header: str | None) -> dict[str, Any]:
    raw = (authorization_header or "").strip()
    if not raw.lower().startswith("bearer "):
        return {}
    token = raw.split(" ", 1)[1].strip()
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1].strip()
    if not payload:
        return {}
    # JWT payload uses base64url encoding.
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("utf-8")).decode("utf-8")
        parsed = json.loads(decoded)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resolve_access_context(
    app: FastAPI,
    *,
    tenant_id: str | None,
    role: str | None,
    user_id: str | None = None,
    api_key_body: str | None,
    api_key_header: str | None,
    tenant_header: str | None = None,
    role_header: str | None = None,
    user_header: str | None = None,
    authorization_header: str | None = None,
) -> dict[str, str]:
    keys = app.state.api_key_policies
    require_key = bool(app.state.require_api_key)
    trust_headers = bool(getattr(app.state, "trust_identity_headers", False))
    trust_bearer = bool(getattr(app.state, "trust_bearer_claims", False))
    provided_key = (api_key_body or api_key_header or "").strip()

    key_policy = None
    if keys:
        if not provided_key:
            raise HTTPException(status_code=401, detail="Missing API key.")
        key_policy = keys.get(provided_key)
        if not key_policy:
            raise HTTPException(status_code=401, detail="Invalid API key.")
    elif require_key and not provided_key:
        raise HTTPException(status_code=401, detail="Missing API key.")

    claims = _decode_bearer_claims(authorization_header) if trust_bearer else {}
    claim_tenant = str(
        claims.get("tenant_id")
        or claims.get("tenant")
        or claims.get("org")
        or ""
    ).strip()
    claim_role = str(claims.get("role") or claims.get("roles") or "").strip().lower()
    claim_user = str(claims.get("sub") or claims.get("user_id") or "").strip()

    header_tenant = (tenant_header or "").strip() if trust_headers else ""
    header_role = (role_header or "").strip().lower() if trust_headers else ""
    header_user = (user_header or "").strip() if trust_headers else ""

    body_tenant = (tenant_id or "").strip()
    body_role = (role or "").strip().lower()
    body_user = (user_id or "").strip()

    if key_policy:
        key_tenant = str((key_policy or {}).get("tenant_id") or "public").strip() or "public"
        key_role = str((key_policy or {}).get("role") or "analyst").strip().lower() or "analyst"
        if body_tenant and body_tenant != key_tenant:
            raise HTTPException(status_code=403, detail="tenant_id mismatch with API key policy.")
        if header_tenant and header_tenant != key_tenant:
            raise HTTPException(status_code=403, detail="Identity tenant mismatch with API key policy.")
        resolved_tenant = key_tenant
        resolved_role = key_role
        resolved_user = body_user or header_user or claim_user or "api-user"
        auth_source = "api_key_policy"
    else:
        resolved_tenant = body_tenant or header_tenant or claim_tenant or "public"
        resolved_role = body_role or header_role or claim_role or "analyst"
        resolved_user = body_user or header_user or claim_user or "anonymous"
        auth_source = "identity_headers" if (header_tenant or header_role or header_user) else "request_body"
        if claim_tenant and not (header_tenant or body_tenant):
            auth_source = "bearer_claims"

    if resolved_role not in {"viewer", "analyst", "admin"}:
        resolved_role = "analyst"
    return {
        "tenant_id": resolved_tenant,
        "role": resolved_role,
        "user_id": resolved_user,
        "auth_source": auth_source,
    }


def _connection_info_from_entry(entry: dict[str, Any], *, is_default: bool | None = None) -> ConnectionInfo:
    kind = str(entry.get("kind") or "duckdb").lower()
    raw_path = str(entry.get("path") or "")
    if kind in {"duckdb", "documents"}:
        path = Path(raw_path)
        exists = path.exists()
        size = path.stat().st_size if exists and path.is_file() else 0
        display_path = str(path)
    else:
        exists = bool(raw_path.strip())
        size = 0
        display_path = raw_path
    return ConnectionInfo(
        id=str(entry.get("id") or ""),
        kind=kind,
        path=display_path,
        description=str(entry.get("description") or ""),
        enabled=bool(entry.get("enabled", True)),
        is_default=bool(entry.get("is_default")) if is_default is None else bool(is_default),
        exists=exists,
        db_size_bytes=int(size),
    )


def _find_working_duckdb_connection(app: FastAPI) -> dict[str, Any] | None:
    listed = app.state.connection_registry.list_connections()
    for item in listed.get("connections", []):
        if not bool(item.get("enabled", True)):
            continue
        if str(item.get("kind") or "").lower() != "duckdb":
            continue
        p = Path(str(item.get("path") or "")).expanduser()
        if p.exists():
            return item
    return None


def _is_ephemeral_test_db_path(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    if "pytest-of-" in text or "/pytest-" in text:
        return True
    name = path.name.lower()
    if not name.startswith("tmp"):
        return False
    if text.startswith("/tmp/") or text.startswith("/private/tmp/"):
        return True
    if "/var/folders/" in text and "/t/tmp" in text:
        return True
    return False


def _duckdb_has_semantic_marts(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    conn = None
    try:
        conn = duckdb.connect(str(path), read_only=True)
        row = conn.execute(
            """
            SELECT COUNT(*) AS metric_value
            FROM information_schema.tables
            WHERE table_schema='main' AND table_name LIKE 'datada_mart_%'
            """
        ).fetchone()
        return int((row or [0])[0] or 0) > 0
    except Exception:
        return False
    finally:
        if conn is not None:
            conn.close()


def _self_heal_default_connection(app: FastAPI) -> dict[str, Any] | None:
    default_entry = app.state.connection_registry.resolve("default")
    explicit_override = bool(getattr(app.state, "explicit_db_override", False))
    default_path = Path(str((default_entry or {}).get("path") or "")).expanduser()
    if (
        default_entry
        and bool(default_entry.get("enabled", True))
        and str(default_entry.get("kind") or "").lower() == "duckdb"
        and default_path.exists()
    ):
        if not explicit_override:
            stable_candidate = _get_db_path().expanduser()
            default_ready = _duckdb_has_semantic_marts(default_path)
            stable_ready = (
                stable_candidate.exists()
                and stable_candidate != default_path
                and _duckdb_has_semantic_marts(stable_candidate)
            )
            should_swap = _is_ephemeral_test_db_path(default_path) or (not default_ready and stable_ready)
            if should_swap and stable_candidate.exists() and stable_candidate != default_path:
                try:
                    app.state.connection_registry.upsert(
                        connection_id="default",
                        kind="duckdb",
                        path=str(stable_candidate),
                        description="Primary local DuckDB connection",
                        enabled=True,
                        set_default=True,
                    )
                except Exception:
                    pass
                healed = app.state.connection_registry.resolve("default")
                if healed is not None:
                    return healed
        return default_entry

    fallback = _find_working_duckdb_connection(app)
    if fallback:
        fallback_id = str(fallback.get("id") or "default")
        try:
            app.state.connection_registry.set_default(fallback_id)
        except Exception:
            pass
        return app.state.connection_registry.resolve("default")

    candidate = _get_db_path().expanduser()
    if candidate.exists():
        try:
            app.state.connection_registry.upsert(
                connection_id="default",
                kind="duckdb",
                path=str(candidate),
                description="Primary local DuckDB connection",
                enabled=True,
                set_default=True,
            )
        except Exception:
            return None
        return app.state.connection_registry.resolve("default")

    return default_entry


def _resolve_documents_duckdb_path(app: FastAPI, entry: dict[str, Any]) -> Path:
    source_path = Path(str(entry.get("path") or "")).expanduser()
    connection_id = str(entry.get("id") or "documents")
    if source_path.is_file() and source_path.suffix.lower() in {".duckdb", ".db"}:
        return source_path

    if not source_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Document source path not found for connection '{connection_id}' at {source_path}",
        )
    if not source_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Documents connection '{connection_id}' expects a directory or DuckDB path; "
                f"received {source_path}."
            ),
        )

    mirror_dir = app.state.db_path.parent / "document_mirrors"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_db = mirror_dir / f"{connection_id}.duckdb"
    result = ingest_documents_to_duckdb(
        docs_dir=source_path,
        db_path=mirror_db,
        force=False,
    )
    if not bool(result.get("success")):
        raise HTTPException(
            status_code=503,
            detail=str(result.get("message") or "Failed to ingest documents for runtime mirror."),
        )
    return mirror_db


def _resolve_stream_duckdb_path(app: FastAPI, entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    stream_uri = str(entry.get("path") or "").strip()
    connection_id = str(entry.get("id") or "stream")
    if not stream_uri:
        raise HTTPException(
            status_code=400,
            detail=f"Stream connection '{connection_id}' is missing a stream URI.",
        )

    mirror_dir = app.state.db_path.parent / "stream_mirrors"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_db = mirror_dir / f"{connection_id}.duckdb"
    try:
        snapshot = ingest_stream_snapshot_to_duckdb(
            stream_uri=stream_uri,
            db_path=mirror_db,
            connection_id=connection_id,
            force=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to ingest stream snapshot for '{connection_id}': {exc}",
        )
    if not bool(snapshot.get("success")):
        raise HTTPException(
            status_code=503,
            detail=f"Failed to ingest stream snapshot for '{connection_id}'.",
        )
    return mirror_db, snapshot


def _resolve_onboarding_profile_meta(app: FastAPI, *, connection_id: str, db_path: Path) -> dict[str, Any]:
    profile_dir = Path(
        os.environ.get(
            "HG_ONBOARDING_PROFILE_DIR",
            str(app.state.db_path.parent / "onboarding_profiles"),
        )
    ).expanduser()
    profile_path = profile_dir / f"{connection_id}.json"
    try:
        profile_meta = load_or_create_onboarding_profile(
            db_path=db_path,
            profile_path=profile_path,
            include_datada_views=False,
        )
    except Exception:
        return {
            "path": str(profile_path),
            "version": "",
            "table_count": 0,
            "generated_at": "",
        }
    return {
        "path": str(profile_meta.get("path") or profile_path),
        "version": str(profile_meta.get("version") or ""),
        "table_count": int(profile_meta.get("table_count") or 0),
        "generated_at": str(profile_meta.get("generated_at") or ""),
    }


def _resolve_connection_db_for_query(
    app: FastAPI,
    connection_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    requested = (connection_id or "default").strip() or "default"
    if requested == "default":
        entry = _self_heal_default_connection(app)
    else:
        entry = app.state.connection_registry.resolve(requested)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown db_connection_id '{requested}'.")

    if not bool(entry.get("enabled", True)):
        raise HTTPException(
            status_code=400,
            detail=f"Connection '{entry.get('id')}' is disabled.",
        )

    kind = str(entry.get("kind") or "duckdb").lower()
    if kind not in {"duckdb", "documents", "stream"}:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Connection kind '{kind}' is registered but not directly query-routable yet. "
                "Use mirrored ingestion into a DuckDB serving layer for bounded-autonomy runtime execution."
            ),
        )

    stream_snapshot_meta: dict[str, Any] = {}
    if kind == "documents":
        db_path = _resolve_documents_duckdb_path(app, entry)
    elif kind == "stream":
        db_path, stream_snapshot_meta = _resolve_stream_duckdb_path(app, entry)
    else:
        db_path = Path(str(entry.get("path") or "")).expanduser()
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found for connection '{entry.get('id')}' at {db_path}",
        )

    cid = str(entry["id"])
    default_id = app.state.connection_registry.default_connection_id()
    if cid == default_id:
        app.state.db_path = db_path

    enriched_entry = dict(entry)
    if stream_snapshot_meta:
        enriched_entry["_stream_snapshot"] = stream_snapshot_meta
    enriched_entry["_onboarding_profile"] = _resolve_onboarding_profile_meta(
        app,
        connection_id=cid,
        db_path=db_path,
    )
    return db_path, enriched_entry


def _resolve_team_for_connection(
    app: FastAPI,
    connection_id: str | None,
) -> tuple[AgenticAnalyticsTeam, Path, dict[str, Any]]:
    db_path, enriched_entry = _resolve_connection_db_for_query(app, connection_id)
    cid = str(enriched_entry["id"])
    kind = str(enriched_entry.get("kind") or "duckdb").lower()
    force_refresh = kind == "documents"
    with app.state.teams_lock:
        cached = app.state.teams.get(cid)
        if cached is not None:
            cached_path = Path(str(cached["db_path"]))
            if cached_path == db_path and not force_refresh:
                team = cached["team"]
            else:
                try:
                    cached["team"].close()
                except Exception:
                    pass
                team = AgenticAnalyticsTeam(db_path)
                app.state.teams[cid] = {"db_path": db_path, "team": team}
        else:
            team = AgenticAnalyticsTeam(db_path)
            app.state.teams[cid] = {"db_path": db_path, "team": team}

    default_id = app.state.connection_registry.default_connection_id()
    if cid == default_id:
        app.state.db_path = db_path
        app.state.team = team

    return team, db_path, enriched_entry


def _role_rank(role: str) -> int:
    clean = (role or "").strip().lower()
    if clean == "admin":
        return 3
    if clean == "analyst":
        return 2
    if clean == "viewer":
        return 1
    return 0


def _require_min_role(role: str, minimum: str) -> None:
    if _role_rank(role) < _role_rank(minimum):
        raise HTTPException(
            status_code=403,
            detail=f"Operation requires {minimum} role.",
        )


def _tenant_session_scope(tenant_id: str, connection_id: str, session_id: str) -> str:
    return f"{tenant_id}:{connection_id}:{session_id}"


def _emit_incident_webhook(app: FastAPI, incident: dict[str, Any]) -> None:
    url = str(getattr(app.state, "incident_webhook_url", "") or "").strip()
    if not url:
        return
    try:
        requests.post(url, json=incident, timeout=2.5)
    except Exception:
        return


def _maybe_record_runtime_incident(
    app: FastAPI,
    *,
    tenant_id: str,
    connection_id: str,
    goal: str,
    response: AssistantQueryResponse,
) -> None:
    # Immediate incident on hard failure.
    if not bool(response.success):
        incident = app.state.runtime_store.record_incident(
            tenant_id=tenant_id,
            severity="high",
            source="query_failure",
            title="Query execution failure",
            summary=f"Query failed on connection '{connection_id}'",
            fingerprint=f"query_failure:{tenant_id}:{connection_id}",
            metadata={
                "goal": goal[:240],
                "trace_id": response.trace_id,
                "error": response.error or "",
            },
            dedupe_window_minutes=int(app.state.incident_dedupe_minutes),
        )
        if incident.get("created"):
            _emit_incident_webhook(
                app,
                {
                    "event": "incident_created",
                    "tenant_id": tenant_id,
                    "connection_id": connection_id,
                    "severity": "high",
                    "title": "Query execution failure",
                    "trace_id": response.trace_id,
                    "incident_id": incident.get("incident_id"),
                },
            )

    if bool(getattr(app.state, "stage_slo_incident_enabled", True)):
        stage_breaches = list((response.runtime or {}).get("stage_slo_breaches") or [])
        if stage_breaches:
            stages = sorted({str(item.get("stage") or "") for item in stage_breaches if str(item.get("stage") or "")})
            fp = f"stage_slo_breach:{tenant_id}:{connection_id}:{','.join(stages)}"
            incident = app.state.runtime_store.record_incident(
                tenant_id=tenant_id,
                severity="low",
                source="stage_slo_monitor",
                title="Stage SLO breach detected",
                summary=f"Stage budget exceeded in {len(stages)} stage(s) for connection '{connection_id}'.",
                fingerprint=fp,
                metadata={"stage_breaches": stage_breaches, "trace_id": response.trace_id},
                dedupe_window_minutes=int(app.state.incident_dedupe_minutes),
            )
            if incident.get("created"):
                _emit_incident_webhook(
                    app,
                    {
                        "event": "incident_created",
                        "tenant_id": tenant_id,
                        "connection_id": connection_id,
                        "severity": "low",
                        "title": "Stage SLO breach detected",
                        "incident_id": incident.get("incident_id"),
                        "stage_breaches": stage_breaches,
                    },
                )

    slo = app.state.runtime_store.evaluate_slo(
        tenant_id=tenant_id,
        hours=int(app.state.slo_window_hours),
        success_rate_target=float(app.state.slo_success_rate_target),
        p95_execution_ms_target=float(app.state.slo_p95_ms_target),
        warning_rate_target=float(app.state.slo_warning_rate_target),
        min_runs=int(app.state.slo_min_runs),
    )
    if str(slo.get("status") or "") != "breach":
        return
    breach_metrics = [str(item.get("metric") or "") for item in slo.get("breaches", [])]
    fp = f"slo_breach:{tenant_id}:{','.join(sorted(breach_metrics))}"
    incident = app.state.runtime_store.record_incident(
        tenant_id=tenant_id,
        severity="medium",
        source="slo_monitor",
        title="SLO breach detected",
        summary=f"SLO breach across {len(breach_metrics)} metric(s) in the last {slo.get('window_hours')}h.",
        fingerprint=fp,
        metadata={
            "breaches": slo.get("breaches", []),
            "burn_rate": slo.get("burn_rate"),
            "runs": slo.get("runs"),
        },
        dedupe_window_minutes=int(app.state.incident_dedupe_minutes),
    )
    if incident.get("created"):
        _emit_incident_webhook(
            app,
            {
                "event": "incident_created",
                "tenant_id": tenant_id,
                "connection_id": connection_id,
                "severity": "medium",
                "title": "SLO breach detected",
                "incident_id": incident.get("incident_id"),
                "breaches": slo.get("breaches", []),
                "burn_rate": slo.get("burn_rate"),
            },
        )


def _execute_query_request(
    app: FastAPI,
    request: QueryRequest,
    *,
    tenant_id: str,
    role: str,
) -> AssistantQueryResponse:
    del role  # role gates are enforced at route layer.

    budget = app.state.runtime_store.consume_budget(
        tenant_id=tenant_id,
        limit_per_hour=int(app.state.query_budget_per_hour),
    )
    if not budget.get("allowed", False):
        raise HTTPException(status_code=429, detail=str(budget.get("message") or "Query budget exceeded."))

    db_path, connection_entry = _resolve_connection_db_for_query(
        app,
        request.db_connection_id,
    )

    if request.local_model:
        state = _get_local_models_state()
        installed = {opt.name.lower() for opt in state.options if opt.installed}
        if request.local_model.lower() in installed:
            if request.local_narrator_model and request.local_narrator_model.lower() not in installed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Local narrator model '{request.local_narrator_model}' is not installed. "
                        "Download it first."
                    ),
                )
            _activate_local_models(
                request.local_model,
                request.local_narrator_model or request.local_model,
            )
        elif request.llm_mode in {LLMMode.LOCAL, LLMMode.AUTO}:
            raise HTTPException(
                status_code=400,
                detail=f"Local model '{request.local_model}' is not installed. Download it first.",
            )

    runtime = _resolve_runtime(
        request.llm_mode,
        goal=request.goal,
        local_model=request.local_model,
        local_narrator_model=request.local_narrator_model,
        openai_model=request.openai_model,
        openai_narrator_model=request.openai_narrator_model,
        anthropic_model=request.anthropic_model,
        anthropic_narrator_model=request.anthropic_narrator_model,
    )
    autonomy = AutonomyConfig(
        mode=request.autonomy_mode.strip().lower() or "bounded",
        auto_correction=bool(request.auto_correction),
        strict_truth=bool(request.strict_truth),
        max_refinement_rounds=int(request.max_refinement_rounds),
        max_candidate_plans=int(request.max_candidate_plans),
    )
    session_id = (request.session_id or "default").strip()[:128] or "default"
    connection_id = str(connection_entry.get("id") or "default")
    session_scope = _tenant_session_scope(tenant_id, connection_id, session_id)
    semantic_cache_meta: dict[str, Any] = {}
    schema_drift: dict[str, Any] = {"drift_detected": False}
    dataset_signature = ""
    schema_signature = ""
    try:
        profile, semantic_cache_meta = _profile_dataset_cached(app, db_path=db_path)
        dataset_signature = str(getattr(profile, "dataset_signature", "") or semantic_cache_meta.get("dataset_signature") or "")
        schema_signature = str(getattr(profile, "schema_signature", "") or semantic_cache_meta.get("schema_signature") or "")
        schema_drift = _record_schema_drift_if_any(
            app,
            tenant_id=tenant_id,
            connection_id=connection_id,
            profile=profile,
            cache_meta=semantic_cache_meta,
        )
    except Exception as exc:
        semantic_cache_meta = {"cache_error": f"{type(exc).__name__}: {exc}"}
        schema_drift = {"drift_detected": False, "error": f"{type(exc).__name__}: {exc}"}
    scenario_context: dict[str, Any] | None = None
    if request.scenario_set_id:
        scenario_row = app.state.runtime_store.get_scenario_set(
            scenario_set_id=request.scenario_set_id,
            tenant_id=tenant_id,
        )
        if scenario_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario set '{request.scenario_set_id}' not found for tenant '{tenant_id}'.",
            )
        if str(scenario_row.get("connection_id") or connection_id) != connection_id:
            raise HTTPException(
                status_code=400,
                detail="Scenario set belongs to a different db_connection_id.",
            )
        scenario_context = {
            "scenario_set_id": str(scenario_row.get("scenario_set_id") or ""),
            "name": str(scenario_row.get("name") or ""),
            "assumptions": list(scenario_row.get("assumptions") or []),
        }

    history = app.state.runtime_store.load_session_turns(session_scope, limit=20)
    cache_key = _query_cache_key(
        request=request,
        tenant_id=tenant_id,
        connection_id=connection_id,
        runtime=runtime,
        autonomy=autonomy,
        history_turns=len(history),
    )
    cache_hit = False
    cached_payload = _query_cache_get(app, cache_key)

    started = time.perf_counter()
    if isinstance(cached_payload, dict):
        response = AssistantQueryResponse(**cached_payload)
        cache_hit = True
    else:
        try:
            v2_run = V2Orchestrator(
                semantic_cache=app.state.semantic_profile_cache,
            ).run(
                goal=request.goal,
                runtime=runtime,
                db_path=str(db_path),
                history=history,
                tenant_id=tenant_id,
                storyteller_mode=request.storyteller_mode,
                autonomy=autonomy,
                scenario_context=scenario_context,
                session_id=session_id,
            )
            response = apply_v2_compat_fields(v2_run.response, v2_run.v2_payload, analysis_version="v2")
        except (
            PolicyViolationError,
            PlanningError,
            QueryCompilationError,
            QueryExecutionError,
            ProviderDegradedError,
            ContradictionDetectedError,
        ) as exc:
            response = AssistantQueryResponse(
                success=False,
                answer_markdown=f"Execution error: {type(exc).__name__}: {exc}",
                confidence="low",
                confidence_score=0.2,
                definition_used="runtime_error",
                evidence=[],
                sanity_checks=[],
                sql=None,
                row_count=0,
                columns=[],
                sample_rows=[],
                execution_time_ms=0.0,
                trace_id=str(uuid.uuid4()),
                runtime={"mode": str(runtime.mode), "provider": str(runtime.provider or "")},
                error=f"{type(exc).__name__}: {exc}",
                warnings=["Runtime error captured in v2 pipeline."],
                analysis_version="v2",
            )
        except Exception as exc:
            response = AssistantQueryResponse(
                success=False,
                answer_markdown=f"Unhandled runtime error: {type(exc).__name__}: {exc}",
                confidence="uncertain",
                confidence_score=0.1,
                definition_used="runtime_error_untyped",
                evidence=[],
                sanity_checks=[],
                sql=None,
                row_count=0,
                columns=[],
                sample_rows=[],
                execution_time_ms=0.0,
                trace_id=str(uuid.uuid4()),
                runtime={"mode": str(runtime.mode), "provider": str(runtime.provider or "")},
                error=f"{type(exc).__name__}: {exc}",
                warnings=["Unhandled runtime exception."],
                quality_flags=["runtime_exception_untyped"],
                analysis_version="v2",
            )
        if response.success:
            _query_cache_set(app, cache_key, response.model_dump())

    # Explicit-mode safety: if selected provider did not produce any effective
    # LLM step, surface a clear user-facing degradation notice.
    if request.llm_mode in {LLMMode.LOCAL, LLMMode.OPENAI, LLMMode.ANTHROPIC} and runtime.use_llm:
        rt_payload = dict(response.runtime or {})
        llm_effective = bool(rt_payload.get("llm_effective"))
        if not llm_effective:
            provider_label = {
                "ollama": "Local model",
                "openai": "OpenAI",
                "anthropic": "Anthropic",
            }.get(str(runtime.provider or "").lower(), str(runtime.provider or "Selected provider"))
            last_error = str(rt_payload.get("llm_last_error") or "").strip()
            reason = last_error or "provider returned no usable response."
            notice = (
                f"{provider_label} was unavailable for this run ({reason}). "
                "Answered with deterministic fallback."
            )
            warnings = list(response.warnings or [])
            if notice not in warnings:
                warnings.append(notice)
            response.warnings = warnings
            if not str(response.answer_markdown or "").lower().startswith("**provider notice:**"):
                response.answer_markdown = (
                    f"**Provider notice:** {provider_label} is resting right now, so I used deterministic mode.\n\n"
                    f"{response.answer_markdown}"
                )
            response.runtime = {
                **rt_payload,
                "llm_degraded": True,
                "llm_degraded_provider": str(runtime.provider or ""),
                "llm_degraded_reason": reason,
            }

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    current_quality_flags = list(response.quality_flags or [])
    stage_breaches = _stage_slo_breaches(response.stage_timings_ms or {})
    if stage_breaches:
        current_quality_flags.append("stage_slo_breach")
    if bool(schema_drift.get("drift_detected")):
        current_quality_flags.append("schema_drift")
    response.quality_flags = sorted(set(current_quality_flags))

    runtime_payload = dict(response.runtime or {})
    runtime_payload["stage_slo_breaches"] = stage_breaches
    runtime_payload["dataset_signature"] = dataset_signature
    runtime_payload["schema_signature"] = schema_signature
    runtime_payload["semantic_cache_hit"] = bool(semantic_cache_meta.get("cache_hit"))
    runtime_payload["schema_drift"] = dict(schema_drift or {})
    response.runtime = runtime_payload

    app.state.runtime_store.append_session_turn(
        session_scope=session_scope,
        connection_id=connection_id,
        session_id=session_id,
        goal=request.goal,
        answer_markdown=response.answer_markdown,
        success=bool(response.success),
        sql=response.sql,
        confidence_score=float(response.confidence_score or 0.0),
        metadata={
            "trace_id": response.trace_id,
            "user_id": request.user_id or "",
            "tenant_id": tenant_id,
            "role": request.role or "",
            "analysis_version": str(response.analysis_version or "v2"),
            "slice_signature": str(response.slice_signature or ""),
            "stage_timings_ms": dict(response.stage_timings_ms or {}),
            "quality_flags": list(response.quality_flags or []),
            "metric": str((response.contract_spec or {}).get("metric") or ""),
            "secondary_metric": (
                "secondary_metric_value"
                if "secondary_metric_value" in (response.columns or [])
                else ""
            ),
            "group_dimensions": list((response.contract_spec or {}).get("dimensions") or []),
            "time_scope": str((response.contract_spec or {}).get("time_scope") or ""),
            "denominator": str(response.denominator_semantics or ""),
            "grain_signature": str(response.grain_signature or ""),
        },
    )
    conversation_turns = len(history) + 1

    warning_terms = (
        ((response.data_quality or {}).get("grounding") or {}).get("goal_term_misses")
        or []
    )
    app.state.runtime_store.record_run_metric(
        tenant_id=tenant_id,
        connection_id=connection_id,
        session_scope=session_scope,
        success=bool(response.success),
        confidence_score=float(response.confidence_score or 0.0),
        execution_ms=float(response.execution_time_ms or elapsed_ms),
        llm_mode=str(runtime.mode or "deterministic"),
        provider=str(runtime.provider or "none"),
        row_count=int(response.row_count or 0),
        warning_count=len(response.warnings or []),
        metadata={
            "goal": request.goal,
            "trace_id": response.trace_id,
            "scenario_set_id": str(request.scenario_set_id or ""),
            "analysis_version": str(response.analysis_version or "v2"),
            "contract_signature": json.dumps(
                {
                    "metric": str((response.contract_spec or {}).get("metric") or ""),
                    "table": str((response.contract_spec or {}).get("table") or ""),
                    "dimensions": list((response.contract_spec or {}).get("dimensions") or []),
                    "time_scope": str((response.contract_spec or {}).get("time_scope") or ""),
                },
                sort_keys=True,
                default=str,
            ),
            "stage_timings_ms": dict(response.stage_timings_ms or {}),
            "quality_flags": list(response.quality_flags or []),
            "provider_effective": str(response.provider_effective or runtime.provider or ""),
            "fallback_used": dict(response.fallback_used or {}),
            "certainty_tags": list(response.certainty_tags or []),
            "grain_signature": str(response.grain_signature or ""),
            "denominator_semantics": str(response.denominator_semantics or ""),
            "warning_terms": warning_terms if isinstance(warning_terms, list) else [],
            "dataset_signature": dataset_signature,
            "schema_signature": schema_signature,
            "semantic_cache_hit": bool(semantic_cache_meta.get("cache_hit")),
            "schema_drift": dict(schema_drift or {}),
            "stage_slo_breaches": stage_breaches,
        },
    )

    response.runtime = {
        **(response.runtime or {}),
        "session_id": session_id,
        "session_scope": session_scope,
        "db_connection_id": connection_id,
        "db_path": str(db_path),
        "db_kind": str(connection_entry.get("kind") or "duckdb"),
        "conversation_turns": conversation_turns,
        "tenant_id": tenant_id,
        "role": request.role or "",
        "budget_remaining": int(budget.get("remaining") or 0),
        "budget_limit_per_hour": int(budget.get("limit_per_hour") or 0),
        "response_cache_hit": bool(cache_hit),
        "response_cache_key": cache_key[:12] if cache_key else "",
        "scenario_set_id": str((scenario_context or {}).get("scenario_set_id") or ""),
        "onboarding_profile_version": str(
            ((connection_entry.get("_onboarding_profile") or {}).get("version") or "")
        ),
        "onboarding_profile_path": str(
            ((connection_entry.get("_onboarding_profile") or {}).get("path") or "")
        ),
        "stream_snapshot": dict(connection_entry.get("_stream_snapshot") or {}),
    }
    if not response.provider_effective:
        response.provider_effective = str(
            (response.runtime or {}).get("provider") or runtime.provider or "deterministic"
        )
    runtime_payload = dict(response.runtime or {})
    degraded_used = bool(runtime_payload.get("llm_degraded"))
    degraded_reason = str(runtime_payload.get("llm_degraded_reason") or "")
    normalized_fallback = dict(response.fallback_used or {})
    if degraded_used and not bool(normalized_fallback.get("used")):
        normalized_fallback["used"] = True
    if degraded_reason and not str(normalized_fallback.get("reason") or "").strip():
        normalized_fallback["reason"] = degraded_reason
    if not normalized_fallback:
        normalized_fallback = {"used": degraded_used, "reason": degraded_reason}
    response.fallback_used = normalized_fallback
    if response.stage_timings_ms is None:
        response.stage_timings_ms = {}
    if response.quality_flags is None:
        response.quality_flags = []

    _maybe_record_runtime_incident(
        app,
        tenant_id=tenant_id,
        connection_id=connection_id,
        goal=request.goal,
        response=response,
    )
    return response


def _close_all_teams(app: FastAPI) -> None:
    with app.state.teams_lock:
        for item in app.state.teams.values():
            try:
                item["team"].close()
            except Exception:
                pass
        app.state.teams = {}


def _ollama_check() -> ProviderCheck:
    base_url = os.environ.get("HG_OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        models = _fetch_ollama_models(base_url)
        if not models:
            return ProviderCheck(available=False, reason=f"reachable at {base_url}, but no models installed")
        intent_model, narrator_model = _configure_ollama_models(models)
        return ProviderCheck(
            available=True,
            reason=(
                f"reachable at {base_url}; models={len(models)}; "
                f"intent={intent_model or 'auto'}; narrator={narrator_model or 'auto'}"
            ),
        )
    except Exception as exc:  # pragma: no cover
        return ProviderCheck(available=False, reason=str(exc))


def _openai_check() -> ProviderCheck:
    if not _has_module("openai"):
        return ProviderCheck(available=False, reason="openai package not installed")
    key = os.environ.get("HG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ProviderCheck(available=False, reason="missing OPENAI_API_KEY")
    try:
        with socket.create_connection(("api.openai.com", 443), timeout=1.5):
            pass
    except Exception as exc:  # pragma: no cover
        return ProviderCheck(available=False, reason=f"openai network unreachable: {exc}")
    return ProviderCheck(available=True, reason="package + key detected")


def _anthropic_check() -> ProviderCheck:
    if not _has_module("anthropic"):
        return ProviderCheck(available=False, reason="anthropic package not installed")
    key = os.environ.get("HG_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ProviderCheck(available=False, reason="missing ANTHROPIC_API_KEY")
    try:
        with socket.create_connection(("api.anthropic.com", 443), timeout=1.5):
            pass
    except Exception as exc:  # pragma: no cover
        return ProviderCheck(available=False, reason=f"anthropic network unreachable: {exc}")
    return ProviderCheck(available=True, reason="package + key detected")


def _providers_snapshot(*, force_refresh: bool = False) -> ProvidersResponse:
    global _PROVIDER_SNAPSHOT_CACHE
    global _PROVIDER_SNAPSHOT_CACHE_TS
    ttl_seconds = max(0.0, float(os.environ.get("HG_PROVIDER_SNAPSHOT_TTL_SECONDS", "8") or 8.0))
    if not force_refresh and ttl_seconds > 0:
        with _PROVIDER_SNAPSHOT_LOCK:
            if _PROVIDER_SNAPSHOT_CACHE is not None and (time.time() - _PROVIDER_SNAPSHOT_CACHE_TS) <= ttl_seconds:
                return _PROVIDER_SNAPSHOT_CACHE

    raw_default = os.environ.get("HG_DEFAULT_LLM_MODE", LLMMode.AUTO.value).lower()
    default_mode = LLMMode(raw_default) if raw_default in {m.value for m in LLMMode} else LLMMode.AUTO

    checks = {
        "ollama": _ollama_check(),
        "openai": _openai_check(),
        "anthropic": _anthropic_check(),
    }

    order_map = {
        "openai": LLMMode.OPENAI,
        "anthropic": LLMMode.ANTHROPIC,
        "ollama": LLMMode.LOCAL,
    }
    pref_raw = os.environ.get("HG_AUTO_MODE_PRIORITY", "openai,anthropic,ollama")
    pref = [p.strip().lower() for p in pref_raw.split(",") if p.strip()]
    recommended = LLMMode.DETERMINISTIC
    for provider in pref:
        if provider in checks and checks[provider].available:
            recommended = order_map.get(provider, LLMMode.DETERMINISTIC)
            break

    snap = ProvidersResponse(default_mode=default_mode, recommended_mode=recommended, checks=checks)
    with _PROVIDER_SNAPSHOT_LOCK:
        _PROVIDER_SNAPSHOT_CACHE = snap
        _PROVIDER_SNAPSHOT_CACHE_TS = time.time()
    return snap


def _auto_prefers_deterministic_fast_path(goal: str) -> bool:
    if os.environ.get("HG_AUTO_DETERMINISTIC_FAST_PATH", "true").lower() not in {"1", "true", "yes", "on"}:
        return False
    text = (goal or "").strip().lower()
    if not text:
        return False
    # Keep analytical/narrative asks on LLM providers.
    llm_pref_keywords = (
        "insight",
        "why",
        "explain",
        "story",
        "narrative",
        "recommend",
        "strategy",
        "root cause",
        "forecast",
        "scenario",
        "correlation",
        "anomaly",
        "cohort",
        "funnel",
        "glossary",
        "dictionary",
        "schema",
        "what kind of data",
        "document",
        "pdf",
    )
    if any(k in text for k in llm_pref_keywords):
        return False
    # Fast-path simple metric asks to deterministic for latency.
    simple_metric_words = ("count", "total", "sum", "average", "avg", "split", "by", "month", "platform")
    has_simple_metric = any(k in text for k in simple_metric_words)
    return has_simple_metric and len(text) <= 240


def _resolve_runtime(
    mode: LLMMode,
    *,
    goal: str | None = None,
    local_model: str | None = None,
    local_narrator_model: str | None = None,
    openai_model: str | None = None,
    openai_narrator_model: str | None = None,
    anthropic_model: str | None = None,
    anthropic_narrator_model: str | None = None,
) -> RuntimeSelection:
    providers = _providers_snapshot()
    local_intent_model = (
        (local_model or "").strip()
        or os.environ.get("HG_OLLAMA_INTENT_MODEL")
        or DEFAULT_MODELS.get("ollama", {}).get("intent", "qwen2.5:7b-instruct")
    )
    local_narrator_model_final = (
        (local_narrator_model or "").strip()
        or os.environ.get("HG_OLLAMA_NARRATOR_MODEL")
        or local_intent_model
    )
    openai_intent_model = (
        (openai_model or "").strip()
        or os.environ.get("HG_OPENAI_INTENT_MODEL")
        or DEFAULT_MODELS.get("openai", {}).get("intent", "gpt-4o-mini")
    )
    openai_narrator_model_final = (
        (openai_narrator_model or "").strip()
        or os.environ.get("HG_OPENAI_NARRATOR_MODEL")
        or DEFAULT_MODELS.get("openai", {}).get("narrator", "gpt-4o-mini")
        or openai_intent_model
    )
    anthropic_intent_model = (
        (anthropic_model or "").strip()
        or os.environ.get("HG_ANTHROPIC_INTENT_MODEL")
        or DEFAULT_MODELS.get("anthropic", {}).get("intent", "claude-haiku-4-5-20251001")
    )
    anthropic_narrator_model_final = (
        (anthropic_narrator_model or "").strip()
        or os.environ.get("HG_ANTHROPIC_NARRATOR_MODEL")
        or DEFAULT_MODELS.get("anthropic", {}).get("narrator", "claude-haiku-4-5-20251001")
        or anthropic_intent_model
    )

    if mode == LLMMode.DETERMINISTIC:
        return RuntimeSelection(
            requested_mode=mode.value,
            mode=mode.value,
            use_llm=False,
            provider=None,
            reason="forced deterministic",
            intent_model=None,
            narrator_model=None,
        )

    # ── Explicit LLM modes: fail if provider unavailable (no silent fallback) ──
    _explicit_modes = {
        LLMMode.LOCAL: ("ollama", local_intent_model, local_narrator_model_final),
        LLMMode.OPENAI: ("openai", openai_intent_model, openai_narrator_model_final),
        LLMMode.ANTHROPIC: ("anthropic", anthropic_intent_model, anthropic_narrator_model_final),
    }
    if mode in _explicit_modes:
        provider_key, intent_m, narrator_m = _explicit_modes[mode]
        check = providers.checks[provider_key]
        if check.available:
            return RuntimeSelection(
                requested_mode=mode.value,
                mode=mode.value,
                use_llm=True,
                provider=provider_key if provider_key != "ollama" else "ollama",
                reason=f"{provider_key} selected",
                intent_model=intent_m,
                narrator_model=narrator_m,
            )
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider '{mode.value}' is unavailable: {check.reason}. "
                f"Fix the provider configuration or use 'deterministic' mode."
            ),
        )

    # auto mode — policy-controlled provider preference.
    if _auto_prefers_deterministic_fast_path(goal or ""):
        return RuntimeSelection(
            requested_mode=mode.value,
            mode=LLMMode.DETERMINISTIC.value,
            use_llm=False,
            provider=None,
            reason="auto fast-path deterministic (simple metric ask)",
            intent_model=None,
            narrator_model=None,
        )

    pref_raw = os.environ.get("HG_AUTO_MODE_PRIORITY", "openai,anthropic,ollama")
    pref = [p.strip().lower() for p in pref_raw.split(",") if p.strip()]
    options = {
        "openai": ("openai", LLMMode.OPENAI, openai_intent_model, openai_narrator_model_final),
        "anthropic": ("anthropic", LLMMode.ANTHROPIC, anthropic_intent_model, anthropic_narrator_model_final),
        "ollama": ("ollama", LLMMode.LOCAL, local_intent_model, local_narrator_model_final),
    }
    _auto_priority = [options[p] for p in pref if p in options]
    if not _auto_priority:
        _auto_priority = [
            ("openai", LLMMode.OPENAI, openai_intent_model, openai_narrator_model_final),
            ("anthropic", LLMMode.ANTHROPIC, anthropic_intent_model, anthropic_narrator_model_final),
            ("ollama", LLMMode.LOCAL, local_intent_model, local_narrator_model_final),
        ]
    for prov_key, llm_mode, intent_m, narrator_m in _auto_priority:
        if providers.checks[prov_key].available:
            return RuntimeSelection(
                requested_mode=mode.value,
                mode=llm_mode.value,
                use_llm=True,
                provider=prov_key if prov_key != "ollama" else "ollama",
                reason=f"auto selected {prov_key} (best available)",
                intent_model=intent_m,
                narrator_model=narrator_m,
            )

    return RuntimeSelection(
        requested_mode=mode.value,
        mode=LLMMode.DETERMINISTIC.value,
        use_llm=False,
        provider=None,
        reason="auto fallback to deterministic (no provider available)",
        intent_model=None,
        narrator_model=None,
    )


def create_app(db_path: Path | None = None) -> FastAPI:
    load_dotenv_file(".env")

    app = FastAPI(
        title="dataDa Agentic API",
        description="Agentic analytics team over enterprise data",
        version="2.0.0-poc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    initial_db_path = (db_path or _get_db_path()).expanduser()
    registry_path = _get_connection_registry_path()
    if db_path is not None and not os.environ.get("HG_CONNECTION_REGISTRY_PATH"):
        # Isolate one-off/test app instances from mutating the canonical project
        # registry unless an explicit registry path is requested.
        registry_path = initial_db_path.with_name(f"{initial_db_path.stem}_connections.json")
    app.state.connection_registry = ConnectionRegistry(registry_path, initial_db_path)
    app.state.explicit_db_override = db_path is not None
    if db_path is not None:
        # Explicit db_path should deterministically become the default connection
        # for this app instance (important for tests and one-off runs).
        default_entry = app.state.connection_registry.upsert(
            connection_id="default",
            kind="duckdb",
            path=str(initial_db_path),
            description="Primary local DuckDB connection",
            enabled=True,
            set_default=True,
        )
    else:
        default_entry = app.state.connection_registry.resolve("default")
        if default_entry is None:
            default_entry = app.state.connection_registry.upsert(
                connection_id="default",
                kind="duckdb",
                path=str(initial_db_path),
                description="Primary local DuckDB connection",
                enabled=True,
                set_default=True,
            )

    if db_path is None:
        healed = _self_heal_default_connection(app)
        if healed is not None:
            default_entry = healed

    if default_entry is None:
        default_entry = app.state.connection_registry.upsert(
            connection_id="default",
            kind="duckdb",
            path=str(initial_db_path),
            description="Primary local DuckDB connection",
            enabled=True,
            set_default=True,
        )

    resolved_db_path = Path(str(default_entry.get("path") or initial_db_path)).expanduser()
    if not resolved_db_path.parent.exists():
        # Stale temp paths (for example from /tmp test runs) can break app boot.
        # Re-anchor default routing to a stable path so startup remains resilient.
        fallback_path = initial_db_path
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        default_entry = app.state.connection_registry.upsert(
            connection_id="default",
            kind="duckdb",
            path=str(fallback_path),
            description="Primary local DuckDB connection",
            enabled=True,
            set_default=True,
        )
        resolved_db_path = Path(str(default_entry.get("path") or fallback_path)).expanduser()

    app.state.db_path = resolved_db_path
    app.state.team = AgenticAnalyticsTeam(app.state.db_path)
    app.state.teams: dict[str, dict[str, Any]] = {
        str(default_entry.get("id", "default")): {
            "db_path": app.state.db_path,
            "team": app.state.team,
        }
    }
    app.state.teams_lock = threading.RLock()
    runtime_store_path = (
        _get_runtime_store_path()
        if os.environ.get("HG_RUNTIME_STORE_PATH")
        else app.state.db_path.with_name(f"{app.state.db_path.stem}_runtime.duckdb")
    )
    app.state.runtime_store = RuntimeStore(runtime_store_path)
    app.state.query_budget_per_hour = max(1, _env_int("HG_QUERY_BUDGET_PER_HOUR", 300))
    app.state.require_api_key = _env_bool("HG_REQUIRE_API_KEY", False)
    app.state.api_key_policies = _load_api_key_policies()
    app.state.trust_identity_headers = _env_bool("HG_TRUST_IDENTITY_HEADERS", True)
    app.state.trust_bearer_claims = _env_bool("HG_TRUST_BEARER_CLAIMS", False)
    app.state.slo_window_hours = max(1, _env_int("HG_SLO_WINDOW_HOURS", 24))
    app.state.slo_min_runs = max(1, _env_int("HG_SLO_MIN_RUNS", 20))
    app.state.slo_success_rate_target = float(os.environ.get("HG_SLO_SUCCESS_RATE_TARGET", "0.95"))
    app.state.slo_p95_ms_target = float(os.environ.get("HG_SLO_P95_MS_TARGET", "3500"))
    app.state.slo_warning_rate_target = float(os.environ.get("HG_SLO_WARNING_RATE_TARGET", "0.15"))
    app.state.incident_webhook_url = os.environ.get("HG_INCIDENT_WEBHOOK_URL", "").strip()
    app.state.incident_dedupe_minutes = max(1, _env_int("HG_INCIDENT_DEDUPE_MINUTES", 60))
    app.state.query_response_cache_enabled = _env_bool("HG_QUERY_RESPONSE_CACHE_ENABLED", True)
    app.state.query_response_cache_ttl_seconds = max(0, _env_int("HG_QUERY_RESPONSE_CACHE_TTL_SECONDS", 120))
    app.state.query_response_cache_max_entries = max(8, _env_int("HG_QUERY_RESPONSE_CACHE_MAX_ENTRIES", 256))
    app.state.query_response_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
    app.state.query_response_cache_lock = threading.RLock()
    app.state.semantic_profile_cache = SemanticProfileCache(
        max_entries=max(4, _env_int("HG_SEMANTIC_CACHE_MAX_ENTRIES", 24)),
        ttl_seconds=max(30, _env_int("HG_SEMANTIC_CACHE_TTL_SECONDS", 900)),
    )
    app.state.stage_slo_incident_enabled = _env_bool("HG_STAGE_SLO_INCIDENT_ENABLED", True)
    app.state.async_max_inflight = max(1, _env_int("HG_ASYNC_MAX_INFLIGHT", 64))
    app.state.async_max_inflight_per_tenant = max(1, _env_int("HG_ASYNC_MAX_INFLIGHT_PER_TENANT", 16))
    app.state.async_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, _env_int("HG_ASYNC_WORKERS", 4)),
        thread_name_prefix="datada-async",
    )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        _close_all_teams(app)
        try:
            app.state.async_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def ui() -> str:
        return get_ui_html()

    @app.get("/ui/assets/{asset_name}")
    async def ui_asset(asset_name: str) -> FileResponse:
        allowed_assets = {
            "ui.css": "text/css; charset=utf-8",
            "ui.js": "application/javascript; charset=utf-8",
        }
        if asset_name not in allowed_assets:
            raise HTTPException(status_code=404, detail="UI asset not found.")
        path = Path(__file__).with_name(asset_name)
        if not path.exists():
            raise HTTPException(status_code=404, detail="UI asset file missing.")
        return FileResponse(
            path,
            media_type=allowed_assets[asset_name],
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/assistant/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        default_entry = _self_heal_default_connection(app)
        if default_entry is None:
            db_path = app.state.db_path
            exists = db_path.exists()
            semantic_ready = False
            return HealthResponse(
                status="no_database",
                db_exists=exists,
                db_path=str(db_path),
                db_size_bytes=db_path.stat().st_size if exists else 0,
                semantic_ready=semantic_ready,
                default_connection_id="default",
                available_connections=0,
                active_connection_kind="duckdb",
                onboarding_profile_version="",
                onboarding_profile_path="",
                runtime_store_path=str(app.state.runtime_store.db_path),
            )

        kind = str(default_entry.get("kind") or "duckdb").lower()
        db_path = Path(str(default_entry.get("path") or app.state.db_path))
        if kind in {"documents", "stream"}:
            try:
                _, resolved_path, resolved_entry = _resolve_team_for_connection(
                    app, str(default_entry.get("id") or "default")
                )
                db_path = resolved_path
                default_entry = resolved_entry
            except Exception:
                pass
        exists = db_path.exists()
        semantic_ready = False
        if exists:
            try:
                team, _, _ = _resolve_team_for_connection(
                    app, str(default_entry.get("id") or "default")
                )
                team.semantic.prepare()
                semantic_ready = True
            except Exception:
                semantic_ready = False

        listed = app.state.connection_registry.list_connections()
        onboarding_meta = _resolve_onboarding_profile_meta(
            app,
            connection_id=str(default_entry.get("id") or "default"),
            db_path=db_path,
        )
        return HealthResponse(
            status="ok" if exists else "no_database",
            db_exists=exists,
            db_path=str(db_path),
            db_size_bytes=db_path.stat().st_size if exists else 0,
            semantic_ready=semantic_ready,
            default_connection_id=str(listed.get("default_connection_id") or "default"),
            available_connections=len(listed.get("connections", [])),
            active_connection_kind=str(default_entry.get("kind") or "duckdb"),
            onboarding_profile_version=str(onboarding_meta.get("version") or ""),
            onboarding_profile_path=str(onboarding_meta.get("path") or ""),
            runtime_store_path=str(app.state.runtime_store.db_path),
        )

    @app.get("/api/assistant/connections", response_model=ConnectionsResponse)
    async def connections(
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectionsResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        listed = app.state.connection_registry.list_connections()
        rows = [
            _connection_info_from_entry(item, is_default=bool(item.get("is_default")))
            for item in listed.get("connections", [])
        ]
        return ConnectionsResponse(
            default_connection_id=str(listed.get("default_connection_id") or "default"),
            connections=rows,
        )

    @app.get("/api/assistant/connectors", response_model=ConnectorsResponse)
    async def connectors(
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectorsResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        return ConnectorsResponse(
            connectors=[
                ConnectorCapability(
                    kind="duckdb",
                    query_routing_supported=True,
                    mirror_ingest_supported=True,
                    notes="Native runtime path. Also used as mirrored serving layer for other sources.",
                ),
                ConnectorCapability(
                    kind="postgres",
                    query_routing_supported=False,
                    mirror_ingest_supported=True,
                    notes="Use connector sync into DuckDB mirror for bounded-autonomy execution.",
                ),
                ConnectorCapability(
                    kind="snowflake",
                    query_routing_supported=False,
                    mirror_ingest_supported=True,
                    notes="Connector metadata + mirror workflow supported; direct pushdown is roadmap.",
                ),
                ConnectorCapability(
                    kind="bigquery",
                    query_routing_supported=False,
                    mirror_ingest_supported=True,
                    notes="Connector metadata + mirror workflow supported; direct pushdown is roadmap.",
                ),
                ConnectorCapability(
                    kind="stream",
                    query_routing_supported=False,
                    mirror_ingest_supported=True,
                    notes="Bounded stream snapshots (Kafka/Kinesis URI registration).",
                ),
                ConnectorCapability(
                    kind="documents",
                    query_routing_supported=True,
                    mirror_ingest_supported=True,
                    notes="Routes via managed DuckDB mirror with chunk-level citation retrieval.",
                ),
            ]
        )

    @app.post("/api/assistant/connections/upsert", response_model=ConnectionActionResponse)
    async def upsert_connection(
        request: ConnectionUpsertRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectionActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        if request.validate_connection:
            ok, reason = app.state.connection_registry.test(kind=request.kind, path=request.path)
            if not ok:
                raise HTTPException(status_code=400, detail=reason)

        try:
            entry = app.state.connection_registry.upsert(
                connection_id=request.connection_id,
                kind=request.kind,
                path=request.path,
                description=request.description,
                enabled=request.enabled,
                set_default=request.set_default,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        cid = str(entry.get("id") or request.connection_id)
        with app.state.teams_lock:
            cached = app.state.teams.get(cid)
            if cached:
                cached_path = Path(str(cached.get("db_path")))
                new_path = Path(str(entry.get("path")))
                if cached_path != new_path:
                    try:
                        cached["team"].close()
                    except Exception:
                        pass
                    del app.state.teams[cid]

        if request.set_default:
            _resolve_team_for_connection(app, cid)

        info = _connection_info_from_entry(
            entry,
            is_default=cid == app.state.connection_registry.default_connection_id(),
        )
        return ConnectionActionResponse(
            success=True,
            message=f"Connection '{cid}' saved.",
            connection=info,
        )

    @app.post("/api/assistant/connections/default", response_model=ConnectionActionResponse)
    async def set_default_connection(
        request: ConnectionSetDefaultRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectionActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        try:
            entry = app.state.connection_registry.set_default(request.connection_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Prime and bind default runtime eagerly so health/UI reflect immediately.
        _resolve_team_for_connection(app, request.connection_id)

        info = _connection_info_from_entry(entry, is_default=True)
        return ConnectionActionResponse(
            success=True,
            message=f"Default connection set to '{request.connection_id}'.",
            connection=info,
        )

    @app.post("/api/assistant/connections/test", response_model=ConnectionActionResponse)
    async def test_connection(
        request: ConnectionTestRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectionActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        if request.connection_id:
            entry = app.state.connection_registry.resolve(request.connection_id)
            if entry is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown connection_id '{request.connection_id}'.",
                )
            ok, reason = app.state.connection_registry.test(
                kind=str(entry.get("kind") or "duckdb"),
                path=str(entry.get("path") or ""),
            )
            if not ok:
                raise HTTPException(status_code=400, detail=reason)
            return ConnectionActionResponse(
                success=True,
                message=reason,
                connection=_connection_info_from_entry(
                    entry,
                    is_default=str(entry.get("id")) == app.state.connection_registry.default_connection_id(),
                ),
            )

        if not request.path:
            raise HTTPException(
                status_code=400,
                detail="Provide either connection_id or path for test.",
            )

        ok, reason = app.state.connection_registry.test(kind=request.kind, path=request.path)
        if not ok:
            raise HTTPException(status_code=400, detail=reason)
        temp = {
            "id": "_adhoc",
            "kind": request.kind.lower(),
            "path": str(Path(request.path).expanduser()),
            "description": "Ad-hoc test target",
            "enabled": True,
        }
        return ConnectionActionResponse(
            success=True,
            message=reason,
            connection=_connection_info_from_entry(temp, is_default=False),
        )

    @app.get("/api/assistant/providers", response_model=ProvidersResponse)
    async def providers() -> ProvidersResponse:
        return _providers_snapshot(force_refresh=True)

    @app.get("/api/assistant/model-health")
    async def model_health():
        """Check health of all configured LLM models."""
        from haikugraph.llm.router import check_model_health, DEFAULT_MODELS
        providers_snap = _providers_snapshot(force_refresh=True)
        results = {}
        for provider_name in ("anthropic", "openai", "ollama"):
            check = providers_snap.checks.get(provider_name)
            if not check or not check.available:
                results[provider_name] = {"available": False, "reason": check.reason if check else "unknown"}
                continue
            provider_health = {"available": True, "models": {}}
            for role in ("planner", "intent", "narrator"):
                model_id = DEFAULT_MODELS.get(provider_name, {}).get(role)
                if model_id:
                    provider_health["models"][role] = {"model": model_id, "status": "configured"}
            results[provider_name] = provider_health
        return results

    @app.get("/api/assistant/models/local", response_model=LocalModelsResponse)
    async def local_models() -> LocalModelsResponse:
        return _get_local_models_state()

    @app.get("/api/assistant/models/openai", response_model=CloudModelsResponse)
    async def openai_models() -> CloudModelsResponse:
        return _get_openai_models_state()

    @app.get("/api/assistant/models/anthropic", response_model=CloudModelsResponse)
    async def anthropic_models() -> CloudModelsResponse:
        return _get_anthropic_models_state()

    @app.post("/api/assistant/models/local/select", response_model=LocalModelActionResponse)
    async def select_local_model(request: LocalModelSelectRequest) -> LocalModelActionResponse:
        if not _is_supported_local_model(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is disabled in this build due benchmark underperformance.",
            )
        if request.narrator_model and not _is_supported_local_model(request.narrator_model):
            raise HTTPException(
                status_code=400,
                detail=f"Narrator model '{request.narrator_model}' is disabled in this build due benchmark underperformance.",
            )
        state = _get_local_models_state()
        if not state.available:
            raise HTTPException(status_code=503, detail=f"Ollama unavailable: {state.reason}")
        installed = {opt.name.lower() for opt in state.options if opt.installed}
        if request.model.lower() not in installed:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not installed. Download it first.",
            )
        if request.narrator_model and request.narrator_model.lower() not in installed:
            raise HTTPException(
                status_code=400,
                detail=f"Narrator model '{request.narrator_model}' is not installed.",
            )
        _activate_local_models(request.model, request.narrator_model)
        return LocalModelActionResponse(
            success=True,
            message=f"Active local model set to {request.model}",
            active_intent_model=os.environ.get("HG_OLLAMA_INTENT_MODEL"),
            active_narrator_model=os.environ.get("HG_OLLAMA_NARRATOR_MODEL"),
        )

    @app.post("/api/assistant/models/local/pull", response_model=LocalModelActionResponse)
    async def pull_local_model(request: LocalModelPullRequest) -> LocalModelActionResponse:
        if not _is_supported_local_model(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is disabled in this build due benchmark underperformance.",
            )
        base_url = os.environ.get("HG_OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            response = requests.post(
                f"{base_url}/api/pull",
                json={"name": request.model, "stream": False},
                timeout=3600,
            )
            response.raise_for_status()
            payload = response.json()
            status_msg = payload.get("status") or "downloaded"
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to pull model '{request.model}': {exc}")

        if request.activate_after_download:
            _activate_local_models(request.model, request.model)
            message = f"{request.model} pulled and activated ({status_msg})."
        else:
            message = f"{request.model} pulled ({status_msg})."
        return LocalModelActionResponse(
            success=True,
            message=message,
            active_intent_model=os.environ.get("HG_OLLAMA_INTENT_MODEL"),
            active_narrator_model=os.environ.get("HG_OLLAMA_NARRATOR_MODEL"),
        )

    @app.get("/api/assistant/architecture", response_model=ArchitectureResponse)
    async def architecture() -> ArchitectureResponse:
        agents = [
            AgentInfo(
                name="ChiefAnalystAgent",
                role="Supervisor",
                description="Coordinates specialist team and mission",
                inputs=["goal", "runtime", "catalog"],
                outputs=["mission brief"],
            ),
            AgentInfo(
                name="DataEngineeringTeam",
                role="Semantic data builder",
                description="Builds typed semantic marts and data quality signals",
                inputs=["raw source tables"],
                outputs=["semantic marts", "catalog"],
            ),
            AgentInfo(
                name="ConnectionRouter",
                role="Connection resolver",
                description="Resolves db_connection_id to active source and team runtime",
                inputs=["db_connection_id", "connection registry"],
                outputs=["resolved data source", "bound team instance"],
            ),
            AgentInfo(
                name="MemoryAgent",
                role="Episodic + procedural memory",
                description="Recalls prior runs and correction rules; persists outcomes for future improvements",
                inputs=["goal", "trace", "feedback"],
                outputs=["memory hints", "learned correction rules"],
            ),
            AgentInfo(
                name="OrganizationalKnowledgeAgent",
                role="Domain governance curator",
                description="Maintains organization rulebook/domain views and feeds governance context into planning",
                inputs=["goal", "catalog", "matched rules", "memory store"],
                outputs=["required domains", "governance directives", "knowledge views"],
            ),
            AgentInfo(
                name="BlackboardAgent",
                role="Artifact exchange bus",
                description="Publishes and routes structured artifacts between agents with transparent producer/consumer edges",
                inputs=["agent outputs"],
                outputs=["shared artifacts", "diagnostic flow graph"],
            ),
            AgentInfo(
                name="IntakeAgent",
                role="Intent parser",
                description="Extracts intent, metric, dimensions, filters, time scope",
                inputs=["natural language goal"],
                outputs=["structured query intent"],
            ),
            AgentInfo(
                name="DiscoveryPlannerAgent",
                role="Exploration planner",
                description="For broad discovery questions, plans which marts and distributions to profile",
                inputs=["catalog", "goal"],
                outputs=["overview profiling plan"],
            ),
            AgentInfo(
                name="CatalogProfilerAgent",
                role="Data profiler",
                description="Builds deeper overview stats (top segments, timelines, signal presence)",
                inputs=["overview plan", "semantic marts"],
                outputs=["profile snapshots"],
            ),
            AgentInfo(
                name="DocumentRetrievalAgent",
                role="Citation retriever",
                description="Retrieves document chunks with citation spans and lexical relevance scoring",
                inputs=["document question", "datada_document_chunks"],
                outputs=["citation-ranked snippets", "source coverage"],
            ),
            AgentInfo(
                name="SemanticRetrievalAgent",
                role="Domain mapper",
                description="Maps goal to mart + schema context",
                inputs=["intake", "catalog"],
                outputs=["table + metric catalog"],
            ),
            AgentInfo(
                name="PlanningAgent",
                role="Task planner",
                description="Builds executable analytics plan",
                inputs=["intake", "semantic context"],
                outputs=["plan graph"],
            ),
            AgentInfo(
                name="QueryEngineerAgent",
                role="SQL compiler",
                description="Compiles task plan to guarded SQL",
                inputs=["plan", "specialist findings"],
                outputs=["SQL"],
            ),
            AgentInfo(
                name="ExecutionAgent",
                role="SQL executor",
                description="Executes SQL via safe read-only engine",
                inputs=["SQL"],
                outputs=["rows + timing"],
            ),
            AgentInfo(
                name="AuditAgent",
                role="Quality checker",
                description="Scores result reliability and warnings",
                inputs=["execution result"],
                outputs=["checks + confidence score"],
            ),
            AgentInfo(
                name="AutonomyAgent",
                role="Candidate reconciler",
                description="Generates plan variants, validates alternatives, and self-corrects when evidence improves",
                inputs=["base plan + audit", "memory hints", "correction rules"],
                outputs=["selected plan", "correction rationale", "probe findings"],
            ),
            AgentInfo(
                name="ToolsmithAgent",
                role="Procedural tool lifecycle",
                description="Turns successful probe SQL into governed candidates with stage/promote/rollback states",
                inputs=["probe findings", "policy gates"],
                outputs=["tool candidates", "staged/promoted tools"],
            ),
            AgentInfo(
                name="TrustAgent",
                role="Reliability telemetry",
                description="Aggregates run quality, confidence, latency, and drift metrics for enterprise trust dashboard",
                inputs=["run metrics", "audit outputs"],
                outputs=["trust dashboard", "failure samples"],
            ),
            AgentInfo(
                name="SLOIncidentAgent",
                role="Reliability governance",
                description="Evaluates SLO compliance and emits deduplicated incident events/webhook hooks",
                inputs=["trust aggregates", "SLO targets"],
                outputs=["slo evaluation", "incident events"],
            ),
            AgentInfo(
                name="NarrativeAgent",
                role="Answer writer",
                description="Writes final business answer",
                inputs=["result + audit"],
                outputs=["answer markdown"],
            ),
            AgentInfo(
                name="VisualizationAgent",
                role="Chart recommender",
                description="Suggests chart spec from result shape",
                inputs=["result"],
                outputs=["chart_spec"],
            ),
        ]
        return ArchitectureResponse(agents=agents)

    @app.get("/api/assistant/capability/scoreboard", response_model=CapabilityScoreboardResponse)
    async def capability_scoreboard() -> CapabilityScoreboardResponse:
        return _read_capability_scoreboard()

    @app.get("/api/assistant/quality/latest", response_model=QualityLatestResponse)
    async def quality_latest() -> QualityLatestResponse:
        latest_runs = _latest_quality_runs()
        return QualityLatestResponse(
            generated_at_epoch_ms=int(time.time() * 1000),
            latest_runs=latest_runs,
            composite_truth_score=_composite_truth_score(latest_runs),
        )

    @app.get("/api/assistant/quality/runs/{run_id}", response_model=QualityRunDetailResponse)
    async def quality_run_detail(run_id: str) -> QualityRunDetailResponse:
        run = _quality_run_by_id(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown quality run_id '{run_id}'.")
        return QualityRunDetailResponse(
            run_id=run_id,
            path=str(run.get("path") or ""),
            payload=dict(run.get("payload") or {}),
        )

    @app.post("/api/assistant/datasets/profile", response_model=DatasetProfileResponse)
    async def datasets_profile(
        request: DatasetProfileRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> DatasetProfileResponse:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id,
            role=request.role,
            user_id=request.user_id,
            api_key_body=request.api_key,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        db_path, connection_entry = _resolve_connection_db_for_query(
            app,
            request.db_connection_id,
        )
        profile, cache_meta = _profile_dataset_cached(app, db_path=db_path)
        drift = _record_schema_drift_if_any(
            app,
            tenant_id=access["tenant_id"],
            connection_id=str(connection_entry.get("id") or request.db_connection_id),
            profile=profile,
            cache_meta=cache_meta,
        )
        quality_summary = dict(profile.quality_summary or {})
        dataset_signature = str(profile.dataset_signature or cache_meta.get("dataset_signature") or "")
        schema_signature = str(profile.schema_signature or cache_meta.get("schema_signature") or "")
        if bool(drift.get("drift_detected")):
            quality_summary["schema_drift_detected"] = 1.0
        return DatasetProfileResponse(
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            dataset_signature=dataset_signature,
            schema_signature=schema_signature,
            table_count=int(quality_summary.get("table_count") or len(profile.tables)),
            high_risk_join_edges=int(quality_summary.get("high_risk_join_edges") or 0),
            sparse_table_count=int(quality_summary.get("sparse_table_count") or 0),
            semantic_cache_hit=bool(cache_meta.get("cache_hit")),
            schema_drift_detected=bool(drift.get("drift_detected")),
            profile=profile.model_dump(),
        )

    @app.get("/api/assistant/runtime/stage-slo", response_model=StageSLOResponse)
    async def runtime_stage_slo(
        hours: int = 24,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> StageSLOResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        snapshot = app.state.runtime_store.stage_slo_snapshot(
            tenant_id=access["tenant_id"],
            hours=max(1, min(24 * 7, int(hours))),
        )
        return StageSLOResponse(
            generated_at_epoch_ms=int(time.time() * 1000),
            stage_budget_ms=_stage_slo_budget_ms(),
            observed_p95_ms=dict(snapshot.get("observed_p95_ms") or {}),
        )

    @app.get("/api/assistant/runtime/readiness", response_model=CutoverReadinessResponse)
    @app.get("/api/assistant/runtime/cutover/readiness", response_model=CutoverReadinessResponse)
    async def runtime_cutover_readiness(
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> CutoverReadinessResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")

        latest_truth = _latest_report_file("v2_qa_truth_report_*.json")
        payload = _load_json(latest_truth) if latest_truth else None
        summary = dict((payload or {}).get("summary") or {})
        release_gate_passed = bool(summary.get("release_gate_passed"))
        floor_violations = list(summary.get("floor_violations") or [])
        composite_truth_score = summary.get("composite_truth_score")
        if not isinstance(composite_truth_score, (int, float)):
            composite_truth_score = None

        artifacts = _cutover_artifacts()
        canary_ready = release_gate_passed and all(item.exists for item in artifacts)

        return CutoverReadinessResponse(
            generated_at_epoch_ms=int(time.time() * 1000),
            default_runtime_version=_runtime_version(),
            canary_ready=bool(canary_ready),
            release_gate_passed=release_gate_passed,
            latest_truth_report=str(latest_truth) if latest_truth else "",
            composite_truth_score=float(composite_truth_score) if composite_truth_score is not None else None,
            floor_violations=floor_violations,
            artifacts=artifacts,
        )

    @app.post("/api/assistant/query", response_model=AssistantQueryResponse)
    async def query(
        request: QueryRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> AssistantQueryResponse:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id,
            role=request.role,
            user_id=request.user_id,
            api_key_body=request.api_key,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        request.tenant_id = access["tenant_id"]
        request.role = access["role"]
        request.user_id = access["user_id"]
        return _execute_query_request(
            app,
            request,
            tenant_id=access["tenant_id"],
            role=access["role"],
        )

    @app.post("/api/assistant/feedback", response_model=FeedbackResponse)
    async def feedback(
        request: FeedbackRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> FeedbackResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        backend, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        corrections_service = CorrectionsService(backend)
        saved = corrections_service.record_feedback(
            {
                "trace_id": request.trace_id,
                "session_id": request.session_id,
                "goal": request.goal,
                "issue": request.issue,
                "suggested_fix": request.suggested_fix,
                "severity": request.severity,
                "keyword": request.keyword,
                "target_table": request.target_table,
                "target_metric": request.target_metric,
                "target_dimensions": request.target_dimensions,
            },
            tenant_id=access["tenant_id"],
        )
        has_rule = bool(saved.get("correction_id"))
        return FeedbackResponse(
            success=True,
            message=(
                (
                    f"Feedback saved and correction rule registered on connection "
                    f"'{connection_entry.get('id', 'default')}'."
                )
                if has_rule
                else f"Feedback saved on connection '{connection_entry.get('id', 'default')}'."
            ),
            feedback_id=saved.get("feedback_id"),
            correction_id=saved.get("correction_id") or None,
        )

    @app.get("/api/assistant/corrections", response_model=CorrectionsResponse)
    async def list_corrections(
        db_connection_id: str = "default",
        include_disabled: bool = True,
        limit: int = 120,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> CorrectionsResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        backend, _, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        corrections_service = CorrectionsService(backend)
        rows = corrections_service.list_corrections(
            tenant_id=access["tenant_id"],
            limit=limit,
        )
        if not include_disabled:
            rows = [row for row in rows if bool(row.get("enabled", True))]
        rules = [CorrectionRuleInfo(**row) for row in rows]
        return CorrectionsResponse(
            db_connection_id=str(connection_entry.get("id") or db_connection_id),
            rules=rules,
        )

    @app.post("/api/assistant/corrections/toggle", response_model=CorrectionToggleResponse)
    async def toggle_correction(
        request: CorrectionToggleRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> CorrectionToggleResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        backend, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        corrections_service = CorrectionsService(backend)
        ok = corrections_service.set_correction_enabled(
            correction_id=request.correction_id,
            enabled=request.enabled,
            tenant_id=access["tenant_id"],
        )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown correction_id '{request.correction_id}'.",
            )
        state = "enabled" if request.enabled else "disabled"
        return CorrectionToggleResponse(
            success=True,
            message=f"Correction rule {request.correction_id} {state}.",
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            correction_id=request.correction_id,
            enabled=request.enabled,
        )

    @app.post("/api/assistant/corrections/rollback", response_model=CorrectionToggleResponse)
    async def rollback_correction(
        request: CorrectionRollbackRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> CorrectionToggleResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        backend, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        corrections_service = CorrectionsService(backend)
        result = corrections_service.rollback_correction(request.correction_id, tenant_id=access["tenant_id"])
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rollback failed."))
        return CorrectionToggleResponse(
            success=True,
            message=str(result.get("message") or "Correction rollback applied."),
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            correction_id=request.correction_id,
            enabled=bool(result.get("enabled", True)),
        )

    @app.get("/api/assistant/rules", response_model=BusinessRulesResponse)
    async def list_business_rules(
        db_connection_id: str = "default",
        status: str | None = None,
        domain: str | None = None,
        limit: int = 200,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> BusinessRulesResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, entry = _resolve_team_for_connection(app, db_connection_id)
        rules_service = RulesService(backend)
        rows = rules_service.list_rules(
            tenant_id=access["tenant_id"],
            limit=limit,
        )
        if status:
            rows = [row for row in rows if str(row.get("status") or "").lower() == str(status).lower()]
        if domain:
            rows = [row for row in rows if str(row.get("domain") or "").lower() == str(domain).lower()]
        return BusinessRulesResponse(
            db_connection_id=str(entry.get("id") or db_connection_id),
            rules=[BusinessRuleInfo(**row) for row in rows],
        )

    @app.post("/api/assistant/rules", response_model=BusinessRuleActionResponse)
    async def create_business_rule(
        request: BusinessRuleCreateRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> BusinessRuleActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, entry = _resolve_team_for_connection(app, request.db_connection_id)
        rules_service = RulesService(backend)
        rule_id = rules_service.create_rule(
            {
                "domain": request.domain,
                "name": request.name,
                "rule_type": request.rule_type,
                "triggers": request.triggers,
                "action_payload": request.action_payload,
                "notes": request.notes,
                "priority": request.priority,
                "status": request.status,
                "source": "admin_ui",
                "approved_by": access["user_id"] if request.status.lower() == "active" else "",
            },
            tenant_id=access["tenant_id"],
            created_by=access["user_id"],
        )
        if not rule_id:
            raise HTTPException(status_code=400, detail="Rule creation failed or duplicate active rule exists.")
        return BusinessRuleActionResponse(
            success=True,
            message=f"Business rule '{request.name}' saved.",
            db_connection_id=str(entry.get("id") or request.db_connection_id),
            rule_id=rule_id,
            status=request.status.lower(),
        )

    @app.post("/api/assistant/rules/status", response_model=BusinessRuleActionResponse)
    async def update_business_rule_status(
        request: BusinessRuleStatusRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> BusinessRuleActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, entry = _resolve_team_for_connection(app, request.db_connection_id)
        rules_service = RulesService(backend)
        result = rules_service.set_rule_status(
            rule_id=request.rule_id,
            status=request.status,
            note=request.note,
            tenant_id=access["tenant_id"],
            approved_by=access["user_id"],
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rule status update failed."))
        return BusinessRuleActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(entry.get("id") or request.db_connection_id),
            rule_id=request.rule_id,
            status=str(result.get("status") or request.status),
        )

    @app.post("/api/assistant/rules/update", response_model=BusinessRuleActionResponse)
    async def update_business_rule(
        request: BusinessRuleUpdateRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> BusinessRuleActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, entry = _resolve_team_for_connection(app, request.db_connection_id)
        rules_service = RulesService(backend)
        result = rules_service.update_rule(
            {
                "rule_id": request.rule_id,
                "domain": request.domain,
                "name": request.name,
                "rule_type": request.rule_type,
                "triggers": request.triggers,
                "action_payload": request.action_payload,
                "notes": request.notes,
                "priority": request.priority,
                "status": request.status,
                "note": request.note,
            },
            tenant_id=access["tenant_id"],
            updated_by=access["user_id"],
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rule update failed."))
        return BusinessRuleActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(entry.get("id") or request.db_connection_id),
            rule_id=request.rule_id,
            status=str(result.get("status") or request.status or ""),
        )

    @app.post("/api/assistant/rules/rollback", response_model=BusinessRuleActionResponse)
    async def rollback_business_rule(
        request: BusinessRuleRollbackRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> BusinessRuleActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, entry = _resolve_team_for_connection(app, request.db_connection_id)
        rules_service = RulesService(backend)
        result = rules_service.rollback_rule(
            request.rule_id,
            tenant_id=access["tenant_id"],
            rolled_back_by=access["user_id"],
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rule rollback failed."))
        return BusinessRuleActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(entry.get("id") or request.db_connection_id),
            rule_id=request.rule_id,
            status=str(result.get("status") or ""),
        )

    @app.get("/api/assistant/scenarios", response_model=ScenarioSetsResponse)
    async def list_scenarios(
        db_connection_id: str = "default",
        status: str = "active",
        limit: int = 100,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ScenarioSetsResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        _, connection_entry = _resolve_connection_db_for_query(app, db_connection_id)
        connection_id = str(connection_entry.get("id") or db_connection_id)
        scenario_service = ScenarioService(app.state.runtime_store)
        rows = scenario_service.list_scenarios(
            tenant_id=access["tenant_id"],
            connection_id=connection_id,
            status=status,
            limit=limit,
        )
        return ScenarioSetsResponse(
            db_connection_id=connection_id,
            scenario_sets=[ScenarioSetInfo(**row) for row in rows],
        )

    @app.post("/api/assistant/scenarios", response_model=ScenarioSetActionResponse)
    async def upsert_scenario(
        request: ScenarioSetUpsertRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ScenarioSetActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        _, connection_entry = _resolve_connection_db_for_query(app, request.db_connection_id)
        connection_id = str(connection_entry.get("id") or request.db_connection_id)
        scenario_service = ScenarioService(app.state.runtime_store)
        saved = scenario_service.upsert_scenario(
            {
                "name": request.name,
                "assumptions": request.assumptions,
                "scenario_set_id": request.scenario_set_id,
                "status": request.status,
            },
            tenant_id=access["tenant_id"],
            connection_id=connection_id,
        )
        return ScenarioSetActionResponse(
            success=True,
            message="Scenario set saved.",
            db_connection_id=connection_id,
            scenario_set=ScenarioSetInfo(**saved),
        )

    @app.get("/api/assistant/scenarios/{scenario_set_id}", response_model=ScenarioSetActionResponse)
    async def get_scenario(
        scenario_set_id: str,
        db_connection_id: str = "default",
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ScenarioSetActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        _, connection_entry = _resolve_connection_db_for_query(app, db_connection_id)
        connection_id = str(connection_entry.get("id") or db_connection_id)
        scenario_service = ScenarioService(app.state.runtime_store)
        row = scenario_service.get_scenario(
            scenario_set_id=scenario_set_id,
            tenant_id=access["tenant_id"],
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"Scenario set '{scenario_set_id}' not found.")
        return ScenarioSetActionResponse(
            success=True,
            message="Scenario set loaded.",
            db_connection_id=connection_id,
            scenario_set=ScenarioSetInfo(**row),
        )

    @app.post("/api/assistant/fix", response_model=FixResponse)
    async def fix_answer(
        request: FixRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> FixResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, _ = _resolve_team_for_connection(app, request.db_connection_id)
        toolsmith_service = ToolsmithService(backend)
        saved = toolsmith_service.record_fix(
            {
                "trace_id": request.trace_id,
                "session_id": request.session_id,
                "goal": request.goal,
                "issue": request.issue,
                "keyword": request.keyword,
                "domain": request.domain,
                "target_table": request.target_table,
                "target_metric": request.target_metric,
                "target_dimensions": request.target_dimensions,
                "notes": request.notes,
                "actor": access["user_id"],
            },
            tenant_id=access["tenant_id"],
        )
        return FixResponse(
            success=True,
            message="Fix captured: feedback logged, correction updated, and admin rule activated.",
            feedback_id=str(saved.get("feedback_id") or ""),
            correction_id=str(saved.get("correction_id") or ""),
            rule_id=str(saved.get("rule_id") or ""),
        )

    @app.get("/api/assistant/toolsmith", response_model=ToolCandidatesResponse)
    async def list_toolsmith(
        db_connection_id: str = "default",
        status: str | None = None,
        limit: int = 120,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ToolCandidatesResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        backend, _, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        toolsmith_service = ToolsmithService(backend)
        rows = toolsmith_service.list_tool_candidates(
            tenant_id=access["tenant_id"],
            limit=limit,
        )
        if status:
            rows = [row for row in rows if str(row.get("status") or "").lower() == str(status).lower()]
        tools = [ToolCandidateInfo(**row) for row in rows]
        return ToolCandidatesResponse(
            db_connection_id=str(connection_entry.get("id") or db_connection_id),
            tools=tools,
        )

    @app.post("/api/assistant/toolsmith/stage", response_model=ToolCandidateActionResponse)
    async def stage_toolsmith(
        request: ToolCandidateActionRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ToolCandidateActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        backend, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        toolsmith_service = ToolsmithService(backend)
        result = toolsmith_service.stage_tool_candidate(request.tool_id, tenant_id=access["tenant_id"])
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Stage failed."))
        return ToolCandidateActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            tool_id=request.tool_id,
            status="staged",
        )

    @app.post("/api/assistant/toolsmith/promote", response_model=ToolCandidateActionResponse)
    async def promote_toolsmith(
        request: ToolCandidateActionRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ToolCandidateActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        toolsmith_service = ToolsmithService(backend)
        result = toolsmith_service.promote_tool_candidate(request.tool_id, tenant_id=access["tenant_id"])
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Promote failed."))
        return ToolCandidateActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            tool_id=request.tool_id,
            status="promoted",
        )

    @app.post("/api/assistant/toolsmith/rollback", response_model=ToolCandidateActionResponse)
    async def rollback_toolsmith(
        request: ToolCandidateActionRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ToolCandidateActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "admin")
        backend, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        toolsmith_service = ToolsmithService(backend)
        result = toolsmith_service.rollback_tool_candidate(request.tool_id, tenant_id=access["tenant_id"])
        if request.reason and result and not str(result.get("reason") or ""):
            result["reason"] = request.reason
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rollback failed."))
        return ToolCandidateActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            tool_id=request.tool_id,
            status="rolled_back",
        )

    # ------------------------------------------------------------------
    # Glossary endpoints
    # ------------------------------------------------------------------

    @app.get("/api/assistant/glossary")
    async def list_glossary(
        tenant_id: str | None = None,
        x_datada_tenant_id: str | None = Header(default=None),
    ):
        tid = (tenant_id or x_datada_tenant_id or "public").strip() or "public"
        backend, _, _ = _resolve_team_for_connection(app, None)
        toolsmith_service = ToolsmithService(backend)
        terms = toolsmith_service.list_glossary(tenant_id=tid)
        return {"terms": terms, "count": len(terms)}

    @app.post("/api/assistant/glossary")
    async def upsert_glossary(
        request: Request,
        x_datada_tenant_id: str | None = Header(default=None),
    ):
        body = await request.json()
        tid = (body.get("tenant_id") or x_datada_tenant_id or "public").strip() or "public"
        backend, _, _ = _resolve_team_for_connection(app, body.get("db_connection_id"))
        toolsmith_service = ToolsmithService(backend)
        result = toolsmith_service.upsert_glossary_term(
            {
                "term": body.get("term", ""),
                "definition": body.get("definition", ""),
                "sql_expression": body.get("sql_expression", ""),
                "target_table": body.get("target_table", ""),
                "target_column": body.get("target_column", ""),
                "examples": body.get("examples"),
                "contributed_by": body.get("contributed_by", "user"),
            },
            tenant_id=tid,
        )
        return result

    # ------------------------------------------------------------------
    # Teaching endpoints
    # ------------------------------------------------------------------

    @app.get("/api/assistant/teachings")
    async def list_teachings(
        tenant_id: str | None = None,
        x_datada_tenant_id: str | None = Header(default=None),
    ):
        tid = (tenant_id or x_datada_tenant_id or "public").strip() or "public"
        backend, _, _ = _resolve_team_for_connection(app, None)
        toolsmith_service = ToolsmithService(backend)
        teachings = toolsmith_service.list_teachings(tenant_id=tid)
        return {"teachings": teachings, "count": len(teachings)}

    @app.post("/api/assistant/teach")
    async def add_teaching(
        request: Request,
        x_datada_tenant_id: str | None = Header(default=None),
    ):
        body = await request.json()
        tid = (body.get("tenant_id") or x_datada_tenant_id or "public").strip() or "public"
        backend, _, _ = _resolve_team_for_connection(app, body.get("db_connection_id"))
        toolsmith_service = ToolsmithService(backend)
        result = toolsmith_service.add_teaching(
            {
                "teaching_text": body.get("teaching_text", ""),
                "expert_name": body.get("expert_name", "anonymous"),
                "keyword": body.get("keyword", ""),
                "target_table": body.get("target_table", ""),
                "target_metric": body.get("target_metric", ""),
                "target_dimensions": body.get("target_dimensions"),
            },
            tenant_id=tid,
        )
        return result

    @app.get("/api/assistant/trust/dashboard", response_model=TrustDashboardResponse)
    async def trust_dashboard(
        tenant_id: str | None = None,
        hours: int = 168,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> TrustDashboardResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        requested_tenant = (tenant_id or "").strip()
        if access["role"] != "admin":
            if requested_tenant and requested_tenant != access["tenant_id"]:
                raise HTTPException(status_code=403, detail="Cross-tenant trust view requires admin role.")
            requested_tenant = access["tenant_id"]
        trust_service = TrustService(app.state.runtime_store)
        payload = trust_service.trust_dashboard(tenant_id=requested_tenant or None, hours=hours)
        return TrustDashboardResponse(
            generated_at=str(payload.get("generated_at") or ""),
            tenant_id=str(payload.get("tenant_id") or "all"),
            window_hours=int(payload.get("window_hours") or 0),
            runs=int(payload.get("runs") or 0),
            success_runs=int(payload.get("success_runs") or 0),
            success_rate=float(payload.get("success_rate") or 0.0),
            avg_confidence=float(payload.get("avg_confidence") or 0.0),
            avg_execution_ms=float(payload.get("avg_execution_ms") or 0.0),
            p95_execution_ms=float(payload.get("p95_execution_ms") or 0.0),
            total_warnings=int(payload.get("total_warnings") or 0),
            by_mode=[TrustModeMetric(**row) for row in payload.get("by_mode", [])],
            parity_summary=dict(payload.get("parity_summary") or {}),
            recent_failures=[TrustFailureSample(**row) for row in payload.get("recent_failures", [])],
        )

    @app.get("/api/assistant/slo/evaluate", response_model=SLOEvaluationResponse)
    async def slo_evaluate(
        tenant_id: str | None = None,
        hours: int | None = None,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> SLOEvaluationResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        requested_tenant = (tenant_id or "").strip()
        if access["role"] != "admin":
            if requested_tenant and requested_tenant != access["tenant_id"]:
                raise HTTPException(status_code=403, detail="Cross-tenant SLO evaluation requires admin role.")
            requested_tenant = access["tenant_id"]

        trust_service = TrustService(app.state.runtime_store)
        payload = trust_service.evaluate_slo(
            tenant_id=requested_tenant or None,
            hours=int(hours or app.state.slo_window_hours),
            success_rate_target=float(app.state.slo_success_rate_target),
            p95_execution_ms_target=float(app.state.slo_p95_ms_target),
            warning_rate_target=float(app.state.slo_warning_rate_target),
            min_runs=int(app.state.slo_min_runs),
        )
        return SLOEvaluationResponse(
            generated_at=str(payload.get("generated_at") or ""),
            tenant_id=str(payload.get("tenant_id") or "all"),
            window_hours=int(payload.get("window_hours") or 0),
            runs=int(payload.get("runs") or 0),
            status=str(payload.get("status") or "unknown"),
            burn_rate=float(payload.get("burn_rate") or 0.0),
            success_rate=float(payload.get("success_rate") or 0.0),
            p95_execution_ms=float(payload.get("p95_execution_ms") or 0.0),
            warning_rate=float(payload.get("warning_rate") or 0.0),
            targets=dict(payload.get("targets") or {}),
            breaches=[SLOBreachMetric(**row) for row in payload.get("breaches", [])],
        )

    @app.get("/api/assistant/incidents", response_model=IncidentsResponse)
    async def list_incidents(
        tenant_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> IncidentsResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        requested_tenant = (tenant_id or "").strip()
        if access["role"] != "admin":
            if requested_tenant and requested_tenant != access["tenant_id"]:
                raise HTTPException(status_code=403, detail="Cross-tenant incident view requires admin role.")
            requested_tenant = access["tenant_id"]

        trust_service = TrustService(app.state.runtime_store)
        rows = trust_service.list_incidents(
            tenant_id=requested_tenant or None,
            limit=limit,
        )
        if status:
            rows = [row for row in rows if str(row.get("status") or "").lower() == str(status).lower()]
        return IncidentsResponse(incidents=[IncidentEvent(**row) for row in rows])

    @app.post("/api/assistant/incidents/ack", response_model=ConnectionActionResponse)
    async def acknowledge_incident(
        request: IncidentAcknowledgeRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectionActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        trust_service = TrustService(app.state.runtime_store)
        # Tenant boundary enforcement for non-admin roles.
        if access["role"] != "admin":
            matched = trust_service.list_incidents(
                tenant_id=access["tenant_id"],
                limit=500,
            )
            if not any(str(item.get("incident_id") or "") == request.incident_id for item in matched):
                raise HTTPException(status_code=403, detail="Cannot acknowledge incidents outside your tenant.")

        ok = trust_service.update_incident_status(
            incident_id=request.incident_id,
            status=request.status,
            note=request.note,
            acknowledged_by=access["user_id"],
        )
        if not ok:
            raise HTTPException(status_code=400, detail="Incident update failed.")
        return ConnectionActionResponse(success=True, message="Incident updated.")

    @app.get("/api/assistant/source-truth/check", response_model=SourceTruthResponse)
    async def source_truth_check(
        db_connection_id: str = "default",
        llm_mode: LLMMode = LLMMode.DETERMINISTIC,
        local_model: str | None = None,
        max_cases: int = 6,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> SourceTruthResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "analyst")
        team, db_path, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        if local_model:
            state = _get_local_models_state()
            installed = {opt.name.lower() for opt in state.options if opt.installed}
            if local_model.lower() in installed:
                _activate_local_models(local_model, local_model)
        runtime = _resolve_runtime(llm_mode)
        payload = run_source_truth_suite(
            team=team,
            db_path=db_path,
            runtime=runtime,
            max_cases=max_cases,
        )
        return SourceTruthResponse(
            db_connection_id=str(connection_entry.get("id") or db_connection_id),
            mode_requested=llm_mode.value,
            mode_actual=str(runtime.mode),
            provider=str(runtime.provider or ""),
            cases=int(payload.get("cases") or 0),
            evaluated_cases=int(payload.get("evaluated_cases") or 0),
            exact_matches=int(payload.get("exact_matches") or 0),
            accuracy_pct=float(payload.get("accuracy_pct") or 0.0),
            avg_latency_ms=float(payload.get("avg_latency_ms") or 0.0),
            parity_summary=dict(payload.get("parity_summary") or {}),
            runs=[SourceTruthCaseResult(**row) for row in payload.get("runs", [])],
        )

    @app.post("/api/assistant/query/async", response_model=AsyncQueryAccepted)
    async def query_async(
        request: AsyncQueryRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> AsyncQueryAccepted:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id,
            role=request.role,
            user_id=request.user_id,
            api_key_body=request.api_key,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        request.tenant_id = access["tenant_id"]
        request.role = access["role"]
        request.user_id = access["user_id"]

        global_load = app.state.runtime_store.count_async_jobs(statuses=("queued", "running"))
        if int(global_load.get("total", 0)) >= int(app.state.async_max_inflight):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Async query queue is currently at capacity. "
                    f"Try again shortly (max inflight={app.state.async_max_inflight})."
                ),
            )
        tenant_load = app.state.runtime_store.count_async_jobs(
            tenant_id=access["tenant_id"],
            statuses=("queued", "running"),
        )
        if int(tenant_load.get("total", 0)) >= int(app.state.async_max_inflight_per_tenant):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Tenant async queue limit reached for '{access['tenant_id']}'. "
                    f"Try again shortly (tenant max inflight={app.state.async_max_inflight_per_tenant})."
                ),
            )

        connection_id = (request.db_connection_id or "default").strip() or "default"
        session_id = (request.session_id or "default").strip()[:128] or "default"
        job_id = app.state.runtime_store.create_async_job(
            tenant_id=access["tenant_id"],
            connection_id=connection_id,
            session_id=session_id,
            request_payload=request.model_dump(),
        )
        payload = QueryRequest(**request.model_dump())

        def _run_job() -> None:
            started = time.perf_counter()
            try:
                app.state.runtime_store.update_async_job(job_id=job_id, status="running")
                response = _execute_query_request(
                    app,
                    payload,
                    tenant_id=access["tenant_id"],
                    role=access["role"],
                )
                app.state.runtime_store.update_async_job(
                    job_id=job_id,
                    status="completed",
                    response_payload=response.model_dump(),
                    runtime_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except Exception as exc:
                app.state.runtime_store.update_async_job(
                    job_id=job_id,
                    status="failed",
                    error_text=str(exc),
                    runtime_ms=round((time.perf_counter() - started) * 1000, 2),
                )

        app.state.async_executor.submit(_run_job)
        return AsyncQueryAccepted(
            success=True,
            message="Async query accepted.",
            job_id=job_id,
            status="queued",
            db_connection_id=connection_id,
            session_id=session_id,
            tenant_id=access["tenant_id"],
        )

    @app.get("/api/assistant/query/async/{job_id}", response_model=AsyncJobStatusResponse)
    async def query_async_status(
        job_id: str,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> AsyncJobStatusResponse:
        access = _resolve_access_context(
            app,
            tenant_id=x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        row = app.state.runtime_store.get_async_job(job_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"Unknown async job '{job_id}'.")
        if access["role"] != "admin" and str(row.get("tenant_id") or "") != access["tenant_id"]:
            raise HTTPException(status_code=403, detail="Cross-tenant async status access denied.")
        return AsyncJobStatusResponse(
            success=row.get("status") == "completed",
            job_id=row["job_id"],
            status=row["status"],
            db_connection_id=row.get("connection_id", "default"),
            session_id=row.get("session_id", "default"),
            tenant_id=row.get("tenant_id", "public"),
            runtime_ms=float(row.get("runtime_ms") or 0.0),
            response=row.get("response") or None,
            error=row.get("error") or None,
        )

    @app.post("/api/assistant/session/clear", response_model=ConnectionActionResponse)
    async def clear_session(
        request: SessionClearRequest,
        x_datada_api_key: str | None = Header(default=None),
        x_datada_tenant_id: str | None = Header(default=None),
        x_datada_role: str | None = Header(default=None),
        x_datada_user_id: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> ConnectionActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id or x_datada_tenant_id,
            role=x_datada_role,
            user_id=x_datada_user_id,
            api_key_body=None,
            api_key_header=x_datada_api_key,
            tenant_header=x_datada_tenant_id,
            role_header=x_datada_role,
            user_header=x_datada_user_id,
            authorization_header=authorization,
        )
        _require_min_role(access["role"], "viewer")
        _, connection_entry = _resolve_connection_db_for_query(app, request.db_connection_id)
        connection_id = str(connection_entry.get("id") or request.db_connection_id)
        scope = _tenant_session_scope(access["tenant_id"], connection_id, request.session_id)
        deleted = app.state.runtime_store.clear_session(scope)
        return ConnectionActionResponse(
            success=True,
            message=f"Cleared {deleted} turn(s) from session '{request.session_id}'.",
            connection=_connection_info_from_entry(
                connection_entry,
                is_default=connection_id == app.state.connection_registry.default_connection_id(),
            ),
        )

    @app.post("/ask", response_model=LegacyAskResponse)
    async def ask(request: LegacyAskRequest) -> LegacyAskResponse:
        res = await query(QueryRequest(goal=request.question, llm_mode=LLMMode.AUTO))
        return LegacyAskResponse(
            final_answer=res.answer_markdown,
            intent={"type": "agentic_team", "confidence": res.confidence_score},
            plan={"trace_id": res.trace_id, "runtime": res.runtime},
            queries=[res.sql] if res.sql else [],
            results=res.sample_rows,
            metadata={
                "confidence": res.confidence.value,
                "confidence_score": res.confidence_score,
                "definition_used": res.definition_used,
                "row_count": res.row_count,
            },
            warnings=[] if res.success else ["query returned degraded answer"],
            errors=[res.error] if res.error else [],
        )




def get_ui_html() -> str:
    ui_template = Path(__file__).with_name("ui.html")
    try:
        html = ui_template.read_text(encoding="utf-8")
        if "<html" in html.lower():
            return html
    except Exception:
        pass
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>dataDa</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0f0f0f;
      --surface: #161616;
      --surface-2: #1e1e1e;
      --surface-3: #282828;
      --gold: #c4a35a;
      --gold-dim: rgba(196,163,90,0.12);
      --gold-mid: rgba(196,163,90,0.25);
      --brick: #8b3a3a;
      --brick-dim: rgba(139,58,58,0.12);
      --text: #f0ece4;
      --text-muted: #8a8478;
      --text-dim: #5a5549;
      --success: #5a9e6f;
      --success-dim: rgba(90,158,111,0.12);
      --border: #262626;
      --mono: "SF Mono","Fira Code",Menlo,Consolas,monospace;
      --sans: -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
    }

    *{box-sizing:border-box;margin:0;padding:0}
    body{min-height:100vh;font-family:var(--sans);color:var(--text);background:var(--bg);-webkit-font-smoothing:antialiased}
    button{font-family:inherit;cursor:pointer;border:none;background:none;color:inherit}
    input,textarea,select{font-family:inherit;color:inherit;background:var(--surface-2);border:1px solid var(--border);border-radius:8px;outline:none}
    input:focus,textarea:focus,select:focus{border-color:var(--gold)}

    /* ---- layout ---- */
    .app{display:flex;flex-direction:column;height:100vh;max-width:920px;margin:0 auto;padding:0 20px}
    .topbar{display:flex;align-items:center;justify-content:space-between;padding:16px 0;flex-shrink:0}
    .logo{font-size:20px;font-weight:700;letter-spacing:-0.5px;color:var(--gold)}
    .logo span{color:var(--text-muted);font-weight:400;font-size:13px;margin-left:8px}
    .topbar-actions{display:flex;gap:8px;align-items:center}
    .icon-btn{width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:8px;transition:background .15s}
    .icon-btn:hover{background:var(--surface-2)}
    .icon-btn svg{width:18px;height:18px;stroke:var(--text-muted);fill:none;stroke-width:1.8}
    .status-dot{width:7px;height:7px;border-radius:50%;background:var(--success);display:inline-block;margin-right:6px}
    .status-dot.offline{background:var(--brick)}

    /* ---- thread ---- */
    .thread{flex:1;overflow-y:auto;padding:16px 0 24px;display:flex;flex-direction:column;gap:20px;scrollbar-width:thin;scrollbar-color:var(--surface-3) transparent;scroll-behavior:smooth}
    .thread::-webkit-scrollbar{width:6px}
    .thread::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:3px}

    /* ---- chat turn wrapper ---- */
    .turn{display:flex;flex-direction:column;gap:12px;animation:fadeIn .25s ease}
    @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

    /* ---- user message bubble ---- */
    .turn-user-bubble{align-self:flex-end;max-width:75%;padding:12px 18px;background:var(--gold-dim);border:1px solid var(--gold-mid);border-radius:18px 18px 4px 18px;font-size:14px;line-height:1.5;color:var(--text);word-wrap:break-word}

    /* ---- assistant response wrapper ---- */
    .turn-assistant{align-self:flex-start;width:100%}

    /* ---- empty state ---- */
    .empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:40px 0}
    .empty h2{font-size:22px;font-weight:600;color:var(--text)}
    .empty p{color:var(--text-muted);font-size:14px;max-width:400px;text-align:center;line-height:1.5}
    .pills{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:520px}
    .pill{padding:8px 16px;border:1px solid var(--gold-mid);border-radius:20px;font-size:13px;color:var(--gold);transition:all .15s;cursor:pointer}
    .pill:hover{background:var(--gold-dim);border-color:var(--gold)}

    /* ---- timestamp ---- */
    .turn-time{font-size:10px;color:var(--text-dim);text-align:right;padding-right:4px}

    /* ---- answer card ---- */
    .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
    .card-body{padding:20px 22px}
    .card-answer{font-size:15px;line-height:1.65;color:var(--text)}
    .card-answer p{margin-bottom:10px}
    .card-answer strong{color:var(--gold);font-weight:600}
    .card-answer code{font-family:var(--mono);font-size:12px;background:var(--surface-2);padding:2px 6px;border-radius:4px}
    .card-answer ul,.card-answer ol{padding-left:18px;margin-bottom:10px}
    .card-answer li{margin-bottom:4px}
    .card-answer h2,.card-answer h3{font-size:15px;font-weight:600;color:var(--gold);margin:14px 0 6px}

    /* ---- confidence badge ---- */
    .badge{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:600;letter-spacing:0.3px;text-transform:uppercase}
    .badge-high{background:var(--success-dim);color:var(--success)}
    .badge-medium{background:var(--gold-dim);color:var(--gold)}
    .badge-low{background:var(--brick-dim);color:var(--brick)}

    /* ---- card meta row ---- */
    .card-meta{display:flex;align-items:center;gap:12px;padding:12px 22px;border-top:1px solid var(--border);flex-wrap:wrap}
    .meta-chip{font-size:11px;color:var(--text-dim);font-family:var(--mono)}
    .meta-sep{width:1px;height:12px;background:var(--border)}

    /* ---- data table ---- */
    .table-wrap{overflow-x:auto;margin:14px 0 4px}
    table{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono)}
    thead th{text-align:left;padding:8px 12px;color:var(--text-muted);border-bottom:1px solid var(--border);font-weight:500;white-space:nowrap}
    tbody td{padding:7px 12px;border-bottom:1px solid var(--border);white-space:nowrap;color:var(--text)}
    tbody tr:last-child td{border-bottom:none}
    tbody tr:hover td{background:var(--surface-2)}

    /* ---- chart container ---- */
    .chart-wrap{position:relative;height:220px;margin:14px 0 4px}
    .chart-wrap canvas{border-radius:8px}

    /* ---- inspect panel (progressive disclosure) ---- */
    .inspect-toggle{display:flex;align-items:center;gap:6px;padding:10px 22px;border-top:1px solid var(--border);font-size:12px;color:var(--text-dim);cursor:pointer;transition:color .15s;user-select:none}
    .inspect-toggle:hover{color:var(--text-muted)}
    .inspect-toggle svg{width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:2;transition:transform .2s}
    .inspect-toggle.open svg{transform:rotate(90deg)}
    .inspect-content{display:none;padding:16px 22px;border-top:1px solid var(--border);font-size:12px;line-height:1.6}
    .inspect-content.open{display:block}

    .inspect-section{margin-bottom:16px}
    .inspect-section:last-child{margin-bottom:0}
    .inspect-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-dim);margin-bottom:6px}
    .inspect-sql{font-family:var(--mono);font-size:11px;background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:12px;white-space:pre-wrap;word-break:break-all;color:var(--text-muted);max-height:200px;overflow-y:auto}

    /* ---- agent trace ---- */
    .trace-list{display:flex;flex-direction:column;gap:6px}
    .trace-item{display:flex;align-items:center;gap:10px;padding:6px 10px;border-radius:6px;background:var(--surface-2)}
    .trace-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
    .trace-dot.ok{background:var(--success)}
    .trace-dot.warn{background:var(--gold)}
    .trace-dot.fail{background:var(--brick)}
    .trace-name{font-size:11px;font-weight:500;color:var(--text-muted);min-width:120px}
    .trace-summary{font-size:11px;color:var(--text-dim);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .trace-time{font-size:10px;color:var(--text-dim);font-family:var(--mono);white-space:nowrap}

    /* ---- audit checks ---- */
    .audit-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px}
    .audit-item{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:5px;font-size:11px}
    .audit-item.pass{background:var(--success-dim);color:var(--success)}
    .audit-item.warn{background:var(--gold-dim);color:var(--gold)}
    .audit-item.fail{background:var(--brick-dim);color:var(--brick)}
    .audit-icon{font-size:12px}

    /* ---- suggested questions ---- */
    .suggestions{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
    .suggestion{padding:6px 12px;border:1px solid var(--border);border-radius:16px;font-size:12px;color:var(--text-muted);cursor:pointer;transition:all .15s}
    .suggestion:hover{border-color:var(--gold-mid);color:var(--gold);background:var(--gold-dim)}

    /* ---- input area ---- */
    .input-area{flex-shrink:0;padding:16px 0 20px;border-top:1px solid var(--border)}
    .input-row{display:flex;gap:10px;align-items:flex-end}
    .input-row textarea{flex:1;resize:none;padding:12px 16px;font-size:14px;line-height:1.4;border-radius:12px;min-height:48px;max-height:150px;transition:border-color .15s}
    .run-btn{height:48px;padding:0 24px;border-radius:12px;background:var(--gold);color:var(--bg);font-size:14px;font-weight:600;letter-spacing:0.2px;transition:opacity .15s;flex-shrink:0}
    .run-btn:hover{opacity:0.88}
    .run-btn:disabled{opacity:0.4;cursor:not-allowed}

    /* ---- settings panel (slide-over) ---- */
    .settings-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:100;opacity:0;pointer-events:none;transition:opacity .2s}
    .settings-overlay.open{opacity:1;pointer-events:all}
    .settings-panel{position:fixed;top:0;right:0;bottom:0;width:340px;background:var(--surface);border-left:1px solid var(--border);z-index:101;transform:translateX(100%);transition:transform .25s ease;padding:24px;overflow-y:auto;display:flex;flex-direction:column;gap:20px}
    .settings-panel.open{transform:translateX(0)}
    .settings-panel h3{font-size:15px;font-weight:600;color:var(--gold)}
    .settings-panel label{display:block;font-size:12px;color:var(--text-muted);margin-bottom:6px}
    .settings-panel select,.settings-panel input[type="text"]{width:100%;padding:8px 12px;font-size:13px;border-radius:8px}
    .settings-panel .toggle-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0}
    .settings-panel .toggle-label{font-size:13px;color:var(--text)}
    .provider-status{font-size:12px}
    .provider-status .provider-title{font-size:11px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:8px}
    .provider-row{display:flex;align-items:center;gap:8px;padding:4px 0}
    .provider-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .provider-dot.available{background:#4caf50}
    .provider-dot.unavailable{background:#f44336}
    .provider-name{color:var(--text);font-size:12px}
    .provider-reason{color:var(--text-dim);font-size:10px;margin-left:auto}
    .settings-close{align-self:flex-end}
    .sep{height:1px;background:var(--border)}

    /* ---- toggle switch ---- */
    .toggle{position:relative;width:36px;height:20px;border-radius:10px;background:var(--surface-3);cursor:pointer;transition:background .2s;flex-shrink:0}
    .toggle.on{background:var(--gold)}
    .toggle::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:var(--text);transition:transform .2s}
    .toggle.on::after{transform:translateX(16px)}

    /* ---- spinner ---- */
    .spinner{width:20px;height:20px;border:2px solid var(--surface-3);border-top-color:var(--gold);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto}
    @keyframes spin{to{transform:rotate(360deg)}}
    .loading-card{display:flex;align-items:center;gap:12px;padding:20px 22px;font-size:13px;color:var(--text-muted)}

    /* ---- error card ---- */
    .error-card{background:var(--brick-dim);border:1px solid var(--brick);border-radius:12px;padding:16px 20px;font-size:13px;color:var(--brick)}

    /* ---- explain button in meta row ---- */
    .explain-btn{font-size:11px;font-weight:600;color:var(--gold);background:var(--gold-dim);border:1px solid var(--gold-mid);border-radius:6px;padding:3px 10px;cursor:pointer;transition:all .15s;letter-spacing:0.3px;text-transform:uppercase}
    .explain-btn:hover{background:var(--gold-mid);border-color:var(--gold)}

    /* ---- loading pulse ---- */
    .loading-dots{display:inline-flex;gap:4px;align-items:center;padding:4px 0}
    .loading-dots span{width:6px;height:6px;border-radius:50%;background:var(--text-dim);animation:pulse 1.2s ease infinite}
    .loading-dots span:nth-child(2){animation-delay:.2s}
    .loading-dots span:nth-child(3){animation-delay:.4s}
    @keyframes pulse{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

    /* ---- explain modal ---- */
    .modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:200;opacity:0;pointer-events:none;transition:opacity .2s;display:flex;align-items:center;justify-content:center}
    .modal-overlay.open{opacity:1;pointer-events:all}
    .modal{background:var(--surface);border:1px solid var(--border);border-radius:14px;width:90vw;max-width:720px;max-height:80vh;overflow-y:auto;padding:0;box-shadow:0 24px 60px rgba(0,0,0,0.5)}
    .modal-header{display:flex;align-items:center;justify-content:space-between;padding:18px 22px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--surface);z-index:1;border-radius:14px 14px 0 0}
    .modal-header h3{font-size:15px;font-weight:600;color:var(--gold);margin:0}
    .modal-body{padding:20px 22px;display:flex;flex-direction:column;gap:18px}

    /* ---- timeline in modal ---- */
    .timeline{display:flex;flex-direction:column;gap:0;position:relative}
    .timeline::before{content:'';position:absolute;left:11px;top:14px;bottom:14px;width:2px;background:var(--border)}
    .tl-item{display:flex;gap:12px;align-items:flex-start;padding:8px 0;position:relative}
    .tl-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:5px;z-index:1;border:2px solid var(--bg)}
    .tl-dot.ok{background:var(--success);border-color:var(--success)}
    .tl-dot.warn{background:var(--gold);border-color:var(--gold)}
    .tl-dot.fail{background:var(--brick);border-color:var(--brick)}
    .tl-body{flex:1;min-width:0}
    .tl-agent{font-size:12px;font-weight:600;color:var(--text)}
    .tl-summary{font-size:11px;color:var(--text-muted);margin-top:2px}
    .tl-meta{font-size:10px;color:var(--text-dim);font-family:var(--mono);margin-top:2px}
    .tl-reasoning{font-size:10px;color:var(--gold);font-family:var(--mono);margin-top:4px;padding:4px 8px;background:rgba(255,193,7,0.08);border-radius:4px}
    .tl-skills{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:5px}
    .tl-skills .flow-node{font-size:10px;padding:3px 8px;border-radius:999px}
    .flow-lane{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
    .flow-node{font-size:11px;color:var(--text);padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface-2)}
    .flow-arrow{font-size:12px;color:var(--gold)}
    details.raw-debug{background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:8px 10px}
    details.raw-debug > summary{cursor:pointer;color:var(--text-muted);font-size:11px;user-select:none}
    details.raw-debug pre{margin-top:8px;max-height:220px;overflow:auto;padding:10px;background:#121212;border:1px solid var(--border);border-radius:6px;font-size:11px;color:var(--text-muted)}

    /* ---- responsive ---- */
    @media(max-width:640px){
      .app{padding:0 12px}
      .card-body{padding:16px}
      .card-meta{padding:10px 16px}
      .inspect-toggle,.inspect-content{padding-left:16px;padding-right:16px}
      .settings-panel{width:100%}
    }
  </style>
</head>
<body>
  <div class="app">
    <!-- top bar -->
    <div class="topbar">
      <div class="logo">dataDa<span id="statusLine"></span></div>
      <div class="topbar-actions">
        <span id="healthDot" class="status-dot" title="Checking..."></span>
        <button class="icon-btn" id="settingsBtn" title="Settings">
          <svg viewBox="0 0 24 24"><path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/></svg>
        </button>
      </div>
    </div>

    <!-- conversation thread -->
    <div class="thread" id="thread"></div>

    <!-- input area -->
    <div class="input-area">
      <div class="input-row">
        <textarea id="goalInput" rows="1" placeholder="Ask your data anything..." autofocus></textarea>
        <button class="run-btn" id="runBtn" onclick="runQuery()">Run</button>
      </div>
    </div>
  </div>

  <!-- explain modal -->
  <div class="modal-overlay" id="explainOverlay" onclick="closeExplain()">
    <div class="modal" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3>Explain This Answer</h3>
        <button class="icon-btn" onclick="closeExplain()">
          <svg viewBox="0 0 24 24"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="modal-body" id="explainBody"></div>
    </div>
  </div>

  <!-- settings slide-over -->
  <div class="settings-overlay" id="settingsOverlay"></div>
  <div class="settings-panel" id="settingsPanel">
    <button class="icon-btn settings-close" id="settingsClose">
      <svg viewBox="0 0 24 24"><path d="M18 6L6 18M6 6l12 12"/></svg>
    </button>
    <h3>Settings</h3>
    <div>
      <label>Connection</label>
      <select id="connSelect"><option value="">default</option></select>
    </div>
    <div>
      <label>LLM Mode</label>
      <select id="modeSelect">
        <option value="auto">Auto</option>
        <option value="local">Local (Ollama)</option>
        <option value="openai">Cloud (OpenAI)</option>
        <option value="anthropic">Cloud (Anthropic)</option>
        <option value="deterministic">Deterministic</option>
      </select>
    </div>
    <div class="sep"></div>
    <div id="providerStatus" class="provider-status"></div>
    <div class="sep"></div>
    <div class="toggle-row">
      <span class="toggle-label">Storyteller mode</span>
      <div class="toggle" id="storytellerToggle" onclick="this.classList.toggle('on')"></div>
    </div>
    <div class="toggle-row">
      <span class="toggle-label">Auto-correction</span>
      <div class="toggle on" id="correctionToggle" onclick="this.classList.toggle('on')"></div>
    </div>
    <div class="toggle-row">
      <span class="toggle-label">Strict truth</span>
      <div class="toggle on" id="truthToggle" onclick="this.classList.toggle('on')"></div>
    </div>
    <div class="sep"></div>
    <div>
      <label>Max refinement rounds</label>
      <select id="refinementSelect">
        <option value="0">0</option><option value="1">1</option><option value="2" selected>2</option>
        <option value="3">3</option><option value="4">4</option>
      </select>
    </div>
    <div>
      <label>Candidate plans</label>
      <select id="candidateSelect">
        <option value="1">1</option><option value="3">3</option><option value="5" selected>5</option>
        <option value="8">8</option><option value="12">12</option>
      </select>
    </div>
  </div>

  <script>
    /* =========================================================
       dataDa — frontend runtime
       ========================================================= */

    const STORAGE_SESSION = 'datada_session_id';
    const STORAGE_THREAD = 'datada_thread';
    const STORAGE_CONN = 'datada_conn';

    const state = {
      sessionId: localStorage.getItem(STORAGE_SESSION) || crypto.randomUUID(),
      turns: JSON.parse(localStorage.getItem(STORAGE_THREAD) || '[]'),
      connectionId: localStorage.getItem(STORAGE_CONN) || ''
    };
    localStorage.setItem(STORAGE_SESSION, state.sessionId);

    /* ---- helpers ---- */
    const $ = id => document.getElementById(id);
    const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
    const fmt = n => n == null ? '—' : typeof n === 'number' ? n.toLocaleString() : String(n);

    function md(raw) {
      if (!raw) return '';
      return raw
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\\/li>)/gs, '<ul>$1</ul>')
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\\n{2,}/g, '</p><p>')
        .replace(/^(?!<[hulo])(.+)$/gm, '<p>$1</p>')
        .replace(/<ul>\\s*<ul>/g, '<ul>')
        .replace(/<\\/ul>\\s*<\\/ul>/g, '</ul>');
    }

    function confidenceBadge(score, label) {
      if (score >= 0.75) return `<span class="badge badge-high">High confidence</span>`;
      if (score >= 0.45) return `<span class="badge badge-medium">Medium confidence</span>`;
      return `<span class="badge badge-low">Low confidence</span>`;
    }

    function buildTable(columns, rows) {
      if (!columns || !rows || !rows.length) return '';
      const hdr = columns.map(c => `<th>${esc(c)}</th>`).join('');
      const body = rows.slice(0, 20).map(r =>
        '<tr>' + columns.map(c => `<td>${fmt(r[c])}</td>`).join('') + '</tr>'
      ).join('');
      return `<div class="table-wrap"><table><thead><tr>${hdr}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function detectChartType(columns, rows) {
      if (!columns || !rows || rows.length < 2 || columns.length < 2) return null;
      const numericCols = columns.filter(c => rows.some(r => typeof r[c] === 'number'));
      const textCols = columns.filter(c => !numericCols.includes(c));
      if (textCols.length >= 1 && numericCols.length >= 1) {
        const labelCol = textCols[0];
        const isTimeLike = /date|month|year|week|day|time|period/i.test(labelCol);
        return { type: isTimeLike ? 'line' : 'bar', labelCol, valueCols: numericCols };
      }
      return null;
    }

    let chartCounter = 0;
    function buildChart(columns, rows) {
      const spec = detectChartType(columns, rows);
      if (!spec) return '';
      const id = 'chart_' + (++chartCounter);
      const labels = rows.map(r => String(r[spec.labelCol] || ''));
      const datasets = spec.valueCols.slice(0, 3).map((col, i) => {
        const colors = ['rgba(196,163,90,0.8)', 'rgba(139,58,58,0.8)', 'rgba(90,158,111,0.8)'];
        const bgColors = ['rgba(196,163,90,0.15)', 'rgba(139,58,58,0.15)', 'rgba(90,158,111,0.15)'];
        return {
          label: col,
          data: rows.map(r => r[col]),
          borderColor: colors[i] || colors[0],
          backgroundColor: spec.type === 'line' ? bgColors[i] : colors[i],
          borderWidth: 2,
          fill: spec.type === 'line',
          tension: 0.3,
          borderRadius: spec.type === 'bar' ? 4 : 0
        };
      });
      setTimeout(() => {
        const el = document.getElementById(id);
        if (!el) return;
        new Chart(el, {
          type: spec.type,
          data: { labels, datasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: datasets.length > 1, labels: { color: '#8a8478', font: { size: 11 } } }
            },
            scales: {
              x: { ticks: { color: '#5a5549', font: { size: 10 } }, grid: { color: '#262626' } },
              y: { ticks: { color: '#5a5549', font: { size: 10 } }, grid: { color: '#262626' } }
            }
          }
        });
      }, 50);
      return `<div class="chart-wrap"><canvas id="${id}"></canvas></div>`;
    }

    function buildTrace(trace) {
      if (!trace || !trace.length) return '';
      return '<div class="trace-list">' + trace.map(t => {
        const st = (t.status || '').toLowerCase();
        const dot = st === 'failed' ? 'fail' : st === 'warning' ? 'warn' : 'ok';
        const ms = t.duration_ms != null ? Math.round(t.duration_ms) + 'ms' : '';
        const reason = t.reasoning ? esc(t.reasoning) : '';
        const titleAttr = reason ? ` title="${reason}"` : '';
        return `<div class="trace-item"${titleAttr}><span class="trace-dot ${dot}"></span><span class="trace-name">${esc(t.agent || t.role || '')}</span><span class="trace-summary">${esc(t.summary || '')}</span><span class="trace-time">${ms}</span></div>`;
      }).join('') + '</div>';
    }

    function buildAudit(checks) {
      if (!checks || !checks.length) return '';
      return '<div class="audit-grid">' + checks.map(c => {
        const cls = c.passed === false ? 'fail' : c.passed === true ? 'pass' : 'warn';
        const icon = c.passed === false ? '&#10007;' : c.passed === true ? '&#10003;' : '&#9888;';
        return `<div class="audit-item ${cls}"><span class="audit-icon">${icon}</span>${esc(c.check_name || c.name || '')}</div>`;
      }).join('') + '</div>';
    }

    /* ---- render ---- */
    let _shouldAutoScroll = true;

    function renderThread() {
      const el = $('thread');
      if (!state.turns.length) {
        el.innerHTML = `
          <div class="empty">
            <h2>What would you like to know?</h2>
            <p>Ask questions about your data in plain English. dataDa will analyze, verify, and explain.</p>
            <div class="pills">
              <div class="pill" onclick="askExample(this)">Total revenue last quarter</div>
              <div class="pill" onclick="askExample(this)">Compare bookings this month vs last</div>
              <div class="pill" onclick="askExample(this)">Top 10 customers by transaction volume</div>
              <div class="pill" onclick="askExample(this)">Show me monthly trends</div>
            </div>
          </div>`;
        return;
      }

      el.innerHTML = state.turns.map((turn, i) => {
        const ts = turn.timestamp ? `<div class="turn-time">${new Date(turn.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</div>` : '';
        let html = `<div class="turn">`;
        html += `${ts}<div class="turn-user-bubble">${esc(turn.goal)}</div>`;

        const r = turn.response;
        if (!r) {
          html += `<div class="turn-assistant"><div class="card"><div class="loading-card"><div class="loading-dots"><span></span><span></span><span></span></div>Analyzing your data...</div></div></div>`;
          html += `</div>`;
          return html;
        }
        if (!r.success && r.error) {
          html += `<div class="turn-assistant"><div class="error-card">${esc(r.error || r.answer_markdown || 'Query failed')}</div></div>`;
          html += `</div>`;
          return html;
        }

        const badge = confidenceBadge(r.confidence_score || 0);
        const answer = md(r.answer_markdown || '');
        const chart = buildChart(r.columns || [], r.sample_rows || []);
        const table = buildTable(r.columns || [], r.sample_rows || []);

        const suggestions = (r.suggested_questions || []).slice(0, 3).map(q =>
          `<div class="suggestion" onclick="askExample(this)">${esc(q)}</div>`
        ).join('');

        const hasSql = r.sql && r.sql.trim();
        const hasTrace = r.agent_trace && r.agent_trace.length;
        const hasChecks = r.sanity_checks && r.sanity_checks.length;
        const hasStats = r.stats_analysis && r.stats_analysis.summary;
        const hasExplain = hasSql || hasTrace || hasChecks || hasStats;

        const rowCount = r.row_count != null ? `<span class="meta-chip">${fmt(r.row_count)} rows returned</span>` : '';
        const explainBtn = hasExplain ? `<button class="explain-btn" onclick="openExplain(${i})">Explain Yourself</button>` : '';

        html += `<div class="turn-assistant"><div class="card">
          <div class="card-body">
            <div style="margin-bottom:12px">${badge}</div>
            <div class="card-answer">${answer}</div>
            ${chart}${table}
            ${suggestions ? '<div class="suggestions">' + suggestions + '</div>' : ''}
          </div>
          <div class="card-meta">${rowCount}${rowCount && explainBtn ? '<span class="meta-sep"></span>' : ''}${explainBtn}</div>
        </div></div>`;
        html += `</div>`;
        return html;
      }).join('');

      /* newest turn is rendered first; keep viewport pinned to top if user is near top */
      if (_shouldAutoScroll) {
        requestAnimationFrame(() => { el.scrollTop = 0; });
      }
    }

    /* detect if user moved away from top (older history) and avoid jump-to-top */
    document.addEventListener('DOMContentLoaded', () => {
      const el = $('thread');
      if (el) el.addEventListener('scroll', () => {
        const nearTop = el.scrollTop < 80;
        _shouldAutoScroll = nearTop;
      });
    });

    /* ---- explain modal ---- */
    function openExplain(turnIdx) {
      const turn = state.turns[turnIdx];
      if (!turn || !turn.response) return;
      const r = turn.response;
      const body = $('explainBody');
      let html = '';

      /* question */
      html += `<div><div class="inspect-label">Question</div><p style="color:var(--text);font-size:14px">${esc(turn.goal)}</p></div>`;

      /* agent decision flow */
      if (r.agent_trace && r.agent_trace.length) {
        const flowNames = r.agent_trace.map(t => (t.agent || t.role || '').trim()).filter(Boolean);
        const dedupFlow = flowNames.filter((name, idx) => idx === 0 || flowNames[idx - 1] !== name).slice(0, 14);
        if (dedupFlow.length) {
          html += `<div><div class="inspect-label">Decision Flow</div><div class="flow-lane">`;
          dedupFlow.forEach((name, idx) => {
            html += `<span class="flow-node">${esc(name)}</span>`;
            if (idx < dedupFlow.length - 1) html += `<span class="flow-arrow">&rarr;</span>`;
          });
          html += `</div></div>`;
        }

        html += `<div><div class="inspect-label">Agent Trace</div><div class="timeline">`;
        r.agent_trace.forEach(t => {
          const st = (t.status || '').toLowerCase();
          const dot = st === 'failed' ? 'fail' : st === 'warning' ? 'warn' : 'ok';
          const ms = t.duration_ms != null ? Math.round(t.duration_ms) + 'ms' : '';
          const reasonHtml = t.reasoning ? `<div class="tl-reasoning">${esc(t.reasoning)}</div>` : '';
          const skills = Array.isArray(t.selected_skills) ? t.selected_skills.filter(Boolean) : [];
          let skillsHtml = '';
          if (skills.length) {
            const chips = skills.map(s => `<span class="flow-node">${esc(s)}</span>`).join('');
            skillsHtml = `<div class="tl-skills"><span class="inspect-label" style="margin-right:6px">Skills</span>${chips}</div>`;
          }
          const contractBits = [t.skill_contract_file, t.skill_layer_file].filter(Boolean).join(' | ');
          const contractHtml = contractBits ? `<div class="tl-meta">contracts: ${esc(contractBits)}</div>` : '';
          const policyHtml = t.skill_policy_reason ? `<div class="tl-meta">policy: ${esc(t.skill_policy_reason)}</div>` : '';
          html += `<div class="tl-item"><div class="tl-dot ${dot}"></div><div class="tl-body"><div class="tl-agent">${esc(t.agent || t.role || '')}</div>${reasonHtml}${skillsHtml}<div class="tl-summary">${esc(t.summary || '')}</div>${contractHtml}${policyHtml}${ms ? '<div class="tl-meta">' + ms + '</div>' : ''}</div></div>`;
        });
        html += '</div></div>';
      }

      /* SQL */
      if (r.sql && r.sql.trim()) {
        html += `<div><div class="inspect-label">SQL Query</div><div class="inspect-sql">${esc(r.sql)}</div></div>`;
      }

      /* audit checks */
      if (r.sanity_checks && r.sanity_checks.length) {
        html += `<div><div class="inspect-label">Audit Checks</div>${buildAudit(r.sanity_checks)}</div>`;
      }

      /* stats analysis */
      if (r.stats_analysis && r.stats_analysis.summary) {
        const sa = r.stats_analysis;
        let sh = `<div><div class="inspect-label">Statistical Analysis</div>`;
        sh += `<p style="color:var(--text-muted);margin-bottom:8px;font-size:12px">${esc(sa.summary)}</p>`;
        if (sa.outliers && sa.outliers.length) {
          sh += '<div style="margin-bottom:6px">';
          sa.outliers.forEach(o => {
            sh += `<div class="audit-item warn" style="display:inline-flex;margin:2px"><span class="audit-icon">&#9888;</span>${esc(o.column)}: ${o.n_outliers} outliers (${o.pct_outliers}%)</div>`;
          });
          sh += '</div>';
        }
        if (sa.correlations && sa.correlations.length) {
          const notable = sa.correlations.filter(c => c.strength !== 'none' && c.strength !== 'weak');
          if (notable.length) {
            sh += '<div style="margin-bottom:6px">';
            notable.forEach(c => {
              const cls = c.strength === 'strong' ? 'pass' : 'warn';
              sh += `<div class="audit-item ${cls}" style="display:inline-flex;margin:2px">${esc(c.col_a)} &harr; ${esc(c.col_b)}: ${c.strength} (r=${c.pearson})</div>`;
            });
            sh += '</div>';
          }
        }
        if (sa.trends && sa.trends.length) {
          sh += '<div style="margin-bottom:6px">';
          sa.trends.forEach(t => {
            const icon = t.direction === 'up' ? '&uarr;' : t.direction === 'down' ? '&darr;' : '&rarr;';
            const cls = t.direction === 'up' ? 'pass' : t.direction === 'down' ? 'fail' : 'warn';
            sh += `<div class="audit-item ${cls}" style="display:inline-flex;margin:2px">${icon} ${esc(t.column)}: ${t.direction} (${t.pct_change_total > 0 ? '+' : ''}${t.pct_change_total}%, R&sup2;=${t.r_squared})</div>`;
          });
          sh += '</div>';
        }
        sh += '</div>';
        html += sh;
      }

      if (r.runtime || r.data_quality) {
        html += `<details class="raw-debug"><summary>Raw diagnostics (optional)</summary><pre>${esc(JSON.stringify({
          runtime: r.runtime || {},
          data_quality: r.data_quality || {}
        }, null, 2))}</pre></details>`;
      }

      /* confidence + meta footer */
      const confPct = Math.round((r.confidence_score || 0) * 100);
      html += `<div style="border-top:1px solid var(--border);padding-top:14px;display:flex;gap:16px;flex-wrap:wrap;align-items:center">`;
      html += `<span class="meta-chip" style="font-size:12px">Confidence: ${confPct}%</span>`;
      if (r.row_count != null) html += `<span class="meta-chip">${fmt(r.row_count)} rows</span>`;
      if (r.execution_time_ms != null) html += `<span class="meta-chip">${Math.round(r.execution_time_ms)}ms</span>`;
      if (r.runtime && r.runtime.mode) html += `<span class="meta-chip">Mode: ${esc(r.runtime.mode)}</span>`;
      html += `</div>`;

      body.innerHTML = html;
      $('explainOverlay').classList.add('open');
    }

    function closeExplain() {
      $('explainOverlay').classList.remove('open');
    }

    function askExample(el) {
      $('goalInput').value = el.textContent;
      runQuery();
    }

    function persistThread() {
      localStorage.setItem(STORAGE_THREAD, JSON.stringify(state.turns));
    }

    /* ---- query ---- */
    async function runQuery() {
      const goal = $('goalInput').value.trim();
      if (!goal) return;

      $('goalInput').value = '';
      autoResize($('goalInput'));
      $('runBtn').disabled = true;

      const turn = { goal, response: null, timestamp: Date.now() };
      state.turns.unshift(turn);
      _shouldAutoScroll = true;
      renderThread();
      persistThread();

      try {
        const body = {
          goal,
          db_connection_id: $('connSelect').value || state.connectionId || undefined,
          llm_mode: $('modeSelect').value,
          session_id: state.sessionId,
          storyteller_mode: $('storytellerToggle').classList.contains('on'),
          auto_correction: $('correctionToggle').classList.contains('on'),
          strict_truth: $('truthToggle').classList.contains('on'),
          max_refinement_rounds: parseInt($('refinementSelect').value),
          max_candidate_plans: parseInt($('candidateSelect').value),
          tenant_id: 'public',
          role: 'analyst'
        };

        const accepted = await fetch('/api/assistant/query/async', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const acceptedJson = await accepted.json();
        if (!accepted.ok) {
          const reason = acceptedJson.detail || acceptedJson.message || 'Failed to queue async query';
          throw new Error(reason);
        }
        const jobId = acceptedJson.job_id;
        if (!jobId) throw new Error('Async query was accepted without a job id.');

        const pollStarted = Date.now();
        const pollTimeoutMs = 180000;
        while (Date.now() - pollStarted < pollTimeoutMs) {
          await new Promise(resolve => setTimeout(resolve, 850));
          const statusResp = await fetch(`/api/assistant/query/async/${encodeURIComponent(jobId)}`);
          const statusJson = await statusResp.json();
          if (!statusResp.ok) {
            const reason = statusJson.detail || statusJson.message || 'Failed to fetch async job status';
            throw new Error(reason);
          }
          const status = String(statusJson.status || '').toLowerCase();
          if (status === 'completed') {
            turn.response = statusJson.response || { success: false, error: 'Async job completed without payload.' };
            break;
          }
          if (status === 'failed' || status === 'canceled') {
            throw new Error(statusJson.error || `Query ${status}.`);
          }
        }
        if (!turn.response) {
          throw new Error('Query timed out while waiting for completion.');
        }
      } catch (e) {
        turn.response = { success: false, error: e.message || 'Network error' };
      }

      _shouldAutoScroll = true;
      renderThread();
      persistThread();
      $('runBtn').disabled = false;
    }

    /* ---- textarea auto resize ---- */
    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    }

    /* ---- settings panel ---- */
    function openSettings() {
      $('settingsOverlay').classList.add('open');
      $('settingsPanel').classList.add('open');
    }
    function closeSettings() {
      $('settingsOverlay').classList.remove('open');
      $('settingsPanel').classList.remove('open');
    }

    /* ---- load connections ---- */
    async function loadConnections() {
      try {
        const r = await fetch('/api/assistant/connections');
        const d = await r.json();
        const sel = $('connSelect');
        sel.innerHTML = '<option value="">default</option>';
        (d.connections || []).forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.id;
          opt.textContent = c.id + (c.is_default ? ' (default)' : '');
          sel.appendChild(opt);
        });
        if (state.connectionId) sel.value = state.connectionId;
      } catch(e) {}
    }

    /* ---- health check ---- */
    async function checkHealth() {
      try {
        const r = await fetch('/api/assistant/health');
        const d = await r.json();
        const dot = $('healthDot');
        const line = $('statusLine');
        if (d.status === 'ok' || d.semantic_ready) {
          dot.classList.remove('offline');
          dot.title = 'Connected';
          line.textContent = ' ready';
        } else {
          dot.classList.add('offline');
          dot.title = 'Degraded';
          line.textContent = ' degraded';
        }
      } catch (e) {
        $('healthDot').classList.add('offline');
        $('healthDot').title = 'Offline';
        $('statusLine').textContent = ' offline';
      }
    }

    /* ---- GAP 40c: provider status ---- */
    async function loadProviders() {
      try {
        const r = await fetch('/api/assistant/providers');
        const d = await r.json();
        const el = $('providerStatus');
        if (!el) return;
        const checks = d.checks || {};
        const providerMap = {ollama: 'local', openai: 'openai', anthropic: 'anthropic'};
        let html = '<div class="provider-title">Provider Status</div>';
        for (const [key, label] of [['ollama', 'Ollama (Local)'], ['openai', 'OpenAI'], ['anthropic', 'Anthropic']]) {
          const check = checks[key] || {};
          const avail = check.available === true;
          const dotCls = avail ? 'available' : 'unavailable';
          const reason = avail ? '' : (check.reason || 'not configured');
          html += `<div class="provider-row"><span class="provider-dot ${dotCls}"></span><span class="provider-name">${label}</span>${reason ? '<span class="provider-reason">' + esc(reason) + '</span>' : ''}</div>`;
          // Disable unavailable options in modeSelect
          const sel = $('modeSelect');
          if (sel && providerMap[key]) {
            const opt = sel.querySelector('option[value="' + providerMap[key] + '"]');
            if (opt && !avail) {
              opt.disabled = true;
              opt.textContent += ' (unavailable)';
            }
          }
        }
        el.innerHTML = html;
      } catch(e) {}
    }

    /* ---- init ---- */
    function init() {
      $('settingsBtn').onclick = openSettings;
      $('settingsClose').onclick = closeSettings;
      $('settingsOverlay').onclick = closeSettings;
      $('connSelect').onchange = () => {
        state.connectionId = $('connSelect').value;
        localStorage.setItem(STORAGE_CONN, state.connectionId);
      };

      const ta = $('goalInput');
      ta.addEventListener('input', () => autoResize(ta));
      ta.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(); }
      });

      renderThread();
      loadConnections();
      loadProviders();
      checkHealth();
    }

    init();
  </script>
</body>
</html>
'''

app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("haikugraph.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
