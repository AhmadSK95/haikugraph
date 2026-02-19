"""FastAPI server for dataDa agentic POC."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import socket
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from haikugraph.api.connection_registry import ConnectionRegistry
from haikugraph.api.runtime_store import RuntimeStore
from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.llm.router import DEFAULT_MODELS
from haikugraph.poc import AgenticAnalyticsTeam, AutonomyConfig, RuntimeSelection, load_dotenv_file
from haikugraph.poc.source_truth import run_source_truth_suite


DEFAULT_DB_CANDIDATES = (
    Path("./data/haikugraph.db"),
    Path("./data/datada.duckdb"),
    Path("./data/haikugraph.duckdb"),
)


class LLMMode(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    OPENAI = "openai"
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
    llm_mode: LLMMode = Field(default=LLMMode.AUTO)
    local_model: str | None = Field(default=None)
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
    recent_failures: list[TrustFailureSample] = Field(default_factory=list)


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
        "6. Semantic Retrieval Agent - Maps query to semantic marts",
        "7. Planning Agent - Produces task graph and metric definitions",
        "8. Specialist Agents - Transactions, Customers, Revenue, Risk",
        "9. Query Engineer + Execution Agents - Compile and run SQL",
        "10. Audit Agent - Validates consistency, grounding, replay checks",
        "11. Autonomy Agent - Evaluates hypotheses with confidence decomposition + contradiction resolution",
        "12. Toolsmith Agent - Captures probe intelligence into staged/promoted reusable tools",
        "13. Narrative + Visualization Agents - Final insight and chart spec",
        "14. Trust Agent - Records reliability telemetry and drift indicators",
    ]
    guardrails: list[str] = [
        "Read-only SQL only",
        "Blocked destructive keywords",
        "Bounded result sizes",
        "Bounded autonomy controls (candidate/iteration caps)",
        "Tenant-aware session isolation",
        "Per-tenant query budgets",
        "Role-gated mutation endpoints (analyst/admin)",
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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


OLLAMA_MODEL_HINTS: dict[str, tuple[str, str]] = {
    "qwen2.5:14b-instruct": ("high", "best local reasoning quality"),
    "qwen2.5:7b-instruct": ("balanced", "default local reasoning"),
    "llama3.1:8b": ("balanced", "strong narrative responses"),
    "mistral:7b": ("balanced", "fast and stable local inference"),
    "llama3.2:latest": ("fast", "quick responses on laptop hardware"),
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
    installed_set = {m.lower() for m in installed}
    ordered_names: list[str] = []

    for model_name in OLLAMA_MODEL_HINTS:
        if model_name not in ordered_names:
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
    # Prefer instruction-tuned model for intent extraction.
    intent_choice = _pick_ollama_model(
        available_models,
        [
            "qwen2.5:14b-instruct",
            "qwen2.5:7b-instruct",
            "llama3.1:8b",
            "mistral:7b",
            "llama3.2:latest",
        ],
    )
    # Prefer fluent summarizer for narrative generation.
    narrator_choice = _pick_ollama_model(
        available_models,
        [
            "llama3.1:8b",
            "qwen2.5:7b-instruct",
            "mistral:7b",
            "llama3.2:latest",
        ],
    )

    if intent_choice:
        os.environ.setdefault("HG_OLLAMA_INTENT_MODEL", intent_choice)
    if narrator_choice:
        os.environ.setdefault("HG_OLLAMA_NARRATOR_MODEL", narrator_choice)

    return intent_choice, narrator_choice


def _activate_local_models(intent_model: str, narrator_model: str | None = None) -> None:
    clean_intent = intent_model.strip()
    clean_narrator = (narrator_model or intent_model).strip()
    if clean_intent:
        os.environ["HG_OLLAMA_INTENT_MODEL"] = clean_intent
    if clean_narrator:
        os.environ["HG_OLLAMA_NARRATOR_MODEL"] = clean_narrator


def _get_local_models_state() -> LocalModelsResponse:
    base_url = os.environ.get("HG_OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        installed = _fetch_ollama_models(base_url)
        if not installed:
            return LocalModelsResponse(
                available=False,
                base_url=base_url,
                options=_build_local_model_options([]),
                reason="Ollama reachable but no models installed.",
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


def _resolve_access_context(
    app: FastAPI,
    *,
    tenant_id: str | None,
    role: str | None,
    api_key_body: str | None,
    api_key_header: str | None,
) -> dict[str, str]:
    keys = app.state.api_key_policies
    require_key = bool(app.state.require_api_key)
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

    resolved_tenant = (
        (tenant_id or "").strip()
        or (key_policy or {}).get("tenant_id", "")
        or "public"
    )
    resolved_role = (
        (role or "").strip().lower()
        or (key_policy or {}).get("role", "")
        or "analyst"
    )
    if resolved_role not in {"viewer", "analyst", "admin"}:
        resolved_role = "analyst"
    return {"tenant_id": resolved_tenant, "role": resolved_role}


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


def _resolve_team_for_connection(
    app: FastAPI,
    connection_id: str | None,
) -> tuple[AgenticAnalyticsTeam, Path, dict[str, Any]]:
    requested = (connection_id or "default").strip() or "default"
    entry = app.state.connection_registry.resolve(requested)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown db_connection_id '{requested}'.")

    if not bool(entry.get("enabled", True)):
        raise HTTPException(
            status_code=400,
            detail=f"Connection '{entry.get('id')}' is disabled.",
        )

    kind = str(entry.get("kind") or "duckdb").lower()
    if kind != "duckdb":
        raise HTTPException(
            status_code=501,
            detail=(
                f"Connection kind '{kind}' is registered but not directly query-routable yet. "
                "Use mirrored ingestion into a DuckDB serving layer for bounded-autonomy runtime execution."
            ),
        )

    db_path = Path(str(entry.get("path") or "")).expanduser()
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found for connection '{entry.get('id')}' at {db_path}",
        )

    cid = str(entry["id"])
    with app.state.teams_lock:
        cached = app.state.teams.get(cid)
        if cached is not None:
            cached_path = Path(str(cached["db_path"]))
            if cached_path == db_path:
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

    return team, db_path, entry


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

    team, db_path, connection_entry = _resolve_team_for_connection(
        app,
        request.db_connection_id,
    )

    if request.local_model:
        state = _get_local_models_state()
        installed = {opt.name.lower() for opt in state.options if opt.installed}
        if request.local_model.lower() in installed:
            _activate_local_models(request.local_model, request.local_model)
        elif request.llm_mode in {LLMMode.LOCAL, LLMMode.AUTO}:
            raise HTTPException(
                status_code=400,
                detail=f"Local model '{request.local_model}' is not installed. Download it first.",
            )

    runtime = _resolve_runtime(request.llm_mode)
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

    history = app.state.runtime_store.load_session_turns(session_scope, limit=20)

    started = time.perf_counter()
    response = team.run(
        request.goal,
        runtime,
        conversation_context=history,
        storyteller_mode=request.storyteller_mode,
        autonomy=autonomy,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

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
        },
    )
    conversation_turns = len(history) + 1

    failed_checks = [c for c in (response.sanity_checks or []) if not bool(c.passed)]
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
        warning_count=len(failed_checks),
        metadata={
            "goal": request.goal,
            "trace_id": response.trace_id,
            "warning_terms": warning_terms if isinstance(warning_terms, list) else [],
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
    }
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


def _providers_snapshot() -> ProvidersResponse:
    raw_default = os.environ.get("HG_DEFAULT_LLM_MODE", LLMMode.AUTO.value).lower()
    default_mode = LLMMode(raw_default) if raw_default in {m.value for m in LLMMode} else LLMMode.AUTO

    checks = {
        "ollama": _ollama_check(),
        "openai": _openai_check(),
    }

    if checks["ollama"].available:
        recommended = LLMMode.LOCAL
    elif checks["openai"].available:
        recommended = LLMMode.OPENAI
    else:
        recommended = LLMMode.DETERMINISTIC

    return ProvidersResponse(default_mode=default_mode, recommended_mode=recommended, checks=checks)


def _resolve_runtime(mode: LLMMode) -> RuntimeSelection:
    providers = _providers_snapshot()
    local_intent_model = os.environ.get("HG_OLLAMA_INTENT_MODEL")
    local_narrator_model = os.environ.get("HG_OLLAMA_NARRATOR_MODEL")
    openai_intent_model = DEFAULT_MODELS.get("openai", {}).get("intent", "gpt-4o-mini")
    openai_narrator_model = DEFAULT_MODELS.get("openai", {}).get("narrator", "gpt-4o-mini")

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

    if mode == LLMMode.LOCAL:
        if providers.checks["ollama"].available:
            return RuntimeSelection(
                requested_mode=mode.value,
                mode=mode.value,
                use_llm=True,
                provider="ollama",
                reason="local ollama selected",
                intent_model=local_intent_model,
                narrator_model=local_narrator_model,
            )
        return RuntimeSelection(
            requested_mode=mode.value,
            mode=LLMMode.DETERMINISTIC.value,
            use_llm=False,
            provider=None,
            reason=f"ollama unavailable: {providers.checks['ollama'].reason}",
            intent_model=None,
            narrator_model=None,
        )

    if mode == LLMMode.OPENAI:
        if providers.checks["openai"].available:
            return RuntimeSelection(
                requested_mode=mode.value,
                mode=mode.value,
                use_llm=True,
                provider="openai",
                reason="openai selected",
                intent_model=openai_intent_model,
                narrator_model=openai_narrator_model,
            )
        return RuntimeSelection(
            requested_mode=mode.value,
            mode=LLMMode.DETERMINISTIC.value,
            use_llm=False,
            provider=None,
            reason=f"openai unavailable: {providers.checks['openai'].reason}",
            intent_model=None,
            narrator_model=None,
        )

    # auto mode
    if providers.checks["ollama"].available:
        return RuntimeSelection(
            requested_mode=mode.value,
            mode=LLMMode.LOCAL.value,
            use_llm=True,
            provider="ollama",
            reason="auto selected local ollama",
            intent_model=local_intent_model,
            narrator_model=local_narrator_model,
        )
    if providers.checks["openai"].available:
        return RuntimeSelection(
            requested_mode=mode.value,
            mode=LLMMode.OPENAI.value,
            use_llm=True,
            provider="openai",
            reason="auto selected openai",
            intent_model=openai_intent_model,
            narrator_model=openai_narrator_model,
        )

    return RuntimeSelection(
        requested_mode=mode.value,
        mode=LLMMode.DETERMINISTIC.value,
        use_llm=False,
        provider=None,
        reason="no llm provider available",
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

    initial_db_path = db_path or _get_db_path()
    registry_path = _get_connection_registry_path()
    app.state.connection_registry = ConnectionRegistry(registry_path, initial_db_path)
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

    app.state.db_path = Path(str(default_entry.get("path") or initial_db_path))
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

    @app.get("/api/assistant/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        default_entry = app.state.connection_registry.resolve("default")
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
                runtime_store_path=str(app.state.runtime_store.db_path),
            )

        db_path = Path(str(default_entry.get("path") or app.state.db_path))
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
        return HealthResponse(
            status="ok" if exists else "no_database",
            db_exists=exists,
            db_path=str(db_path),
            db_size_bytes=db_path.stat().st_size if exists else 0,
            semantic_ready=semantic_ready,
            default_connection_id=str(listed.get("default_connection_id") or "default"),
            available_connections=len(listed.get("connections", [])),
            active_connection_kind=str(default_entry.get("kind") or "duckdb"),
            runtime_store_path=str(app.state.runtime_store.db_path),
        )

    @app.get("/api/assistant/connections", response_model=ConnectionsResponse)
    async def connections() -> ConnectionsResponse:
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
    async def connectors() -> ConnectorsResponse:
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
                    query_routing_supported=False,
                    mirror_ingest_supported=True,
                    notes="Text-rich documents can be ingested into semantic evidence tables.",
                ),
            ]
        )

    @app.post("/api/assistant/connections/upsert", response_model=ConnectionActionResponse)
    async def upsert_connection(request: ConnectionUpsertRequest) -> ConnectionActionResponse:
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
    async def set_default_connection(request: ConnectionSetDefaultRequest) -> ConnectionActionResponse:
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
    async def test_connection(request: ConnectionTestRequest) -> ConnectionActionResponse:
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
        return _providers_snapshot()

    @app.get("/api/assistant/models/local", response_model=LocalModelsResponse)
    async def local_models() -> LocalModelsResponse:
        return _get_local_models_state()

    @app.post("/api/assistant/models/local/select", response_model=LocalModelActionResponse)
    async def select_local_model(request: LocalModelSelectRequest) -> LocalModelActionResponse:
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

    @app.post("/api/assistant/query", response_model=AssistantQueryResponse)
    async def query(
        request: QueryRequest,
        x_datada_api_key: str | None = Header(default=None),
    ) -> AssistantQueryResponse:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id,
            role=request.role,
            api_key_body=request.api_key,
            api_key_header=x_datada_api_key,
        )
        request.tenant_id = access["tenant_id"]
        request.role = access["role"]
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
    ) -> FeedbackResponse:
        access = _resolve_access_context(
            app,
            tenant_id=None,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "analyst")
        team, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        saved = team.record_feedback(
            trace_id=request.trace_id,
            session_id=request.session_id,
            goal=request.goal,
            issue=request.issue,
            suggested_fix=request.suggested_fix,
            severity=request.severity,
            keyword=request.keyword,
            target_table=request.target_table,
            target_metric=request.target_metric,
            target_dimensions=request.target_dimensions,
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
    ) -> CorrectionsResponse:
        team, _, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        rows = team.list_corrections(limit=limit, include_disabled=include_disabled)
        rules = [CorrectionRuleInfo(**row) for row in rows]
        return CorrectionsResponse(
            db_connection_id=str(connection_entry.get("id") or db_connection_id),
            rules=rules,
        )

    @app.post("/api/assistant/corrections/toggle", response_model=CorrectionToggleResponse)
    async def toggle_correction(
        request: CorrectionToggleRequest,
        x_datada_api_key: str | None = Header(default=None),
    ) -> CorrectionToggleResponse:
        access = _resolve_access_context(
            app,
            tenant_id=None,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "analyst")
        team, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        ok = team.set_correction_enabled(request.correction_id, request.enabled)
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
    ) -> CorrectionToggleResponse:
        access = _resolve_access_context(
            app,
            tenant_id=None,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "analyst")
        team, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        result = team.rollback_correction(request.correction_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rollback failed."))
        return CorrectionToggleResponse(
            success=True,
            message=str(result.get("message") or "Correction rollback applied."),
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            correction_id=request.correction_id,
            enabled=bool(result.get("enabled", True)),
        )

    @app.get("/api/assistant/toolsmith", response_model=ToolCandidatesResponse)
    async def list_toolsmith(
        db_connection_id: str = "default",
        status: str | None = None,
        limit: int = 120,
    ) -> ToolCandidatesResponse:
        team, _, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        rows = team.list_tool_candidates(status=status, limit=limit)
        tools = [ToolCandidateInfo(**row) for row in rows]
        return ToolCandidatesResponse(
            db_connection_id=str(connection_entry.get("id") or db_connection_id),
            tools=tools,
        )

    @app.post("/api/assistant/toolsmith/stage", response_model=ToolCandidateActionResponse)
    async def stage_toolsmith(
        request: ToolCandidateActionRequest,
        x_datada_api_key: str | None = Header(default=None),
    ) -> ToolCandidateActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=None,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "analyst")
        team, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        result = team.stage_tool_candidate(request.tool_id)
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
    ) -> ToolCandidateActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=None,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "admin")
        team, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        result = team.promote_tool_candidate(request.tool_id)
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
    ) -> ToolCandidateActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=None,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "admin")
        team, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        result = team.rollback_tool_candidate(request.tool_id, reason=request.reason)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Rollback failed."))
        return ToolCandidateActionResponse(
            success=True,
            message=str(result.get("message") or ""),
            db_connection_id=str(connection_entry.get("id") or request.db_connection_id),
            tool_id=request.tool_id,
            status="rolled_back",
        )

    @app.get("/api/assistant/trust/dashboard", response_model=TrustDashboardResponse)
    async def trust_dashboard(
        tenant_id: str | None = None,
        hours: int = 168,
    ) -> TrustDashboardResponse:
        payload = app.state.runtime_store.trust_dashboard(tenant_id=tenant_id, hours=hours)
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
            recent_failures=[TrustFailureSample(**row) for row in payload.get("recent_failures", [])],
        )

    @app.get("/api/assistant/source-truth/check", response_model=SourceTruthResponse)
    async def source_truth_check(
        db_connection_id: str = "default",
        llm_mode: LLMMode = LLMMode.DETERMINISTIC,
        local_model: str | None = None,
        max_cases: int = 6,
    ) -> SourceTruthResponse:
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
            runs=[SourceTruthCaseResult(**row) for row in payload.get("runs", [])],
        )

    @app.post("/api/assistant/query/async", response_model=AsyncQueryAccepted)
    async def query_async(
        request: AsyncQueryRequest,
        x_datada_api_key: str | None = Header(default=None),
    ) -> AsyncQueryAccepted:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id,
            role=request.role,
            api_key_body=request.api_key,
            api_key_header=x_datada_api_key,
        )
        request.tenant_id = access["tenant_id"]
        request.role = access["role"]

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
    async def query_async_status(job_id: str) -> AsyncJobStatusResponse:
        row = app.state.runtime_store.get_async_job(job_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"Unknown async job '{job_id}'.")
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
    ) -> ConnectionActionResponse:
        access = _resolve_access_context(
            app,
            tenant_id=request.tenant_id,
            role=None,
            api_key_body=None,
            api_key_header=x_datada_api_key,
        )
        _require_min_role(access["role"], "viewer")
        _, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
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
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>dataDa</title>
  <style>
    :root {
      --canvas: #f1f7f8;
      --panel: #ffffff;
      --ink: #092532;
      --subtle: #617784;
      --line: #d6e4e8;
      --brand: #0ca57d;
      --brand-dark: #0e5677;
      --ok-bg: #e8f8ef;
      --ok-ink: #217a4d;
      --warn-bg: #fff4e7;
      --warn-ink: #9f5d1d;
      --chip: #f7fbfd;
      --mono: "IBM Plex Mono", Menlo, Consolas, monospace;
      --sans: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      --soft-shadow: 0 10px 26px rgba(5, 44, 63, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, #d8ecff 0%, transparent 34%),
        radial-gradient(circle at 100% 0%, #c9f7eb 0%, transparent 32%),
        var(--canvas);
    }

    .app {
      max-width: 1300px;
      margin: 0 auto;
      padding: 18px 14px 28px;
    }

    .hero {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #fdfefe 100%);
      border-radius: 16px;
      box-shadow: var(--soft-shadow);
      padding: 16px;
      margin-bottom: 12px;
    }

    .hero-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }

    .title {
      margin: 0;
      font-size: clamp(1.7rem, 2.8vw, 2.4rem);
      color: var(--brand-dark);
      letter-spacing: -0.02em;
    }

    .subtitle {
      margin: 4px 0 0;
      color: var(--subtle);
      max-width: 780px;
    }

    .session-chip {
      background: #ecf8f3;
      color: #236845;
      border: 1px solid #badfcc;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.78rem;
      font-weight: 700;
    }

    .status-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px;
    }

    .status-card {
      border: 1px solid var(--line);
      border-radius: 11px;
      padding: 9px 10px;
      background: #fcffff;
    }

    .status-card label {
      display: block;
      margin-bottom: 4px;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--subtle);
    }

    .status-card .value {
      font-size: 0.92rem;
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(330px, 440px) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 14px;
      box-shadow: var(--soft-shadow);
      padding: 12px;
      margin-bottom: 10px;
    }

    .panel h3 {
      margin: 0 0 6px;
      font-size: 1.02rem;
      color: var(--brand-dark);
    }

    .hint {
      color: var(--subtle);
      font-size: 0.86rem;
    }

    textarea,
    select,
    button,
    input[type="checkbox"] {
      font-family: var(--sans);
    }

    textarea,
    select,
    button {
      border: 1px solid var(--line);
      border-radius: 11px;
    }

    textarea {
      width: 100%;
      min-height: 94px;
      resize: vertical;
      padding: 10px 11px;
      font-size: 0.95rem;
      color: var(--ink);
      background: #fbfefe;
    }

    .composer-actions {
      margin-top: 8px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .row {
      margin-top: 8px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      align-items: end;
    }

    select {
      width: 100%;
      padding: 9px;
      font-size: 0.9rem;
      background: #fff;
    }

    .btn {
      padding: 9px 12px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease;
      font-size: 0.86rem;
      background: #fff;
      color: var(--ink);
    }

    .btn:hover { transform: translateY(-1px); }
    .btn:disabled { opacity: 0.62; cursor: not-allowed; transform: none; }

    .btn.primary {
      background: linear-gradient(120deg, var(--brand), #2bbb95);
      border: none;
      color: #fff;
      box-shadow: 0 8px 20px rgba(12, 165, 125, 0.26);
    }

    .btn.ghost {
      color: var(--brand-dark);
      background: #fff;
    }

    .btn.warn {
      color: var(--warn-ink);
      border-color: #efd6b3;
      background: #fffdfa;
    }

    .examples,
    .model-catalog,
    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 8px;
    }

    .pill,
    .model-pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 5px 10px;
      background: var(--chip);
      color: #325463;
      font-size: 0.78rem;
      cursor: pointer;
    }

    .model-pill.install {
      border-style: dashed;
      color: var(--warn-ink);
      background: #fffaf5;
    }

    .model-pill.active {
      border-color: #95dcbc;
      color: var(--ok-ink);
      background: #ebfaf1;
      font-weight: 700;
    }

    .toggle-row {
      margin-top: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 11px;
      padding: 8px 10px;
      background: #f8fcfe;
      font-size: 0.85rem;
    }

    .toggle-row input { transform: scale(1.12); }

    .chat-shell {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      box-shadow: var(--soft-shadow);
      min-height: 480px;
      padding: 10px;
      display: flex;
      flex-direction: column;
    }

    .thread {
      display: grid;
      gap: 12px;
    }

    .empty-state {
      border: 1px dashed #bfd4dc;
      border-radius: 12px;
      padding: 18px;
      color: var(--subtle);
      background: #fbfeff;
    }

    .turn {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      overflow: hidden;
    }

    .turn-head {
      padding: 8px 10px;
      border-bottom: 1px solid #edf4f7;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      background: #fcfefe;
      font-size: 0.8rem;
      color: var(--subtle);
    }

    .turn-body { padding: 10px; }

    .bubble {
      border-radius: 10px;
      padding: 10px 11px;
      margin-bottom: 8px;
      line-height: 1.45;
      font-size: 0.92rem;
    }

    .bubble.user {
      background: #ecf5ff;
      border: 1px solid #c8def6;
      color: #17415f;
    }

    .bubble.assistant {
      background: #f9fffc;
      border: 1px solid #cae8d8;
      color: #184a39;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px;
      margin: 8px 0;
    }

    .kpi {
      border-radius: 10px;
      padding: 9px;
      border: 1px solid var(--line);
      background: #fafefe;
    }

    .kpi.label {
      background: linear-gradient(135deg, #e7fbf3, #f8fffd);
      border-color: #bce5d1;
    }

    .kpi .k {
      font-size: 0.72rem;
      color: var(--subtle);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .kpi .v {
      margin-top: 3px;
      font-size: 1.16rem;
      font-weight: 700;
      color: #124f39;
      overflow-wrap: anywhere;
    }

    .tag {
      display: inline-block;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 3px 8px;
      margin-right: 5px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .tag.ok { background: var(--ok-bg); color: var(--ok-ink); }
    .tag.warn { background: var(--warn-bg); color: var(--warn-ink); }

    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 9px;
      margin-top: 8px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }

    th, td {
      border-bottom: 1px solid #edf4f7;
      text-align: left;
      padding: 7px;
      white-space: nowrap;
    }

    th {
      background: #f7fcff;
      color: #1a516e;
      position: sticky;
      top: 0;
    }

    details {
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #fcfeff;
      padding: 6px 8px;
    }

    summary {
      cursor: pointer;
      font-weight: 700;
      color: #194a63;
      font-size: 0.84rem;
    }

    .mono {
      margin-top: 6px;
      font-family: var(--mono);
      font-size: 0.77rem;
      white-space: pre-wrap;
      line-height: 1.35;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      background: #f6fbfd;
      color: #143546;
    }

    .trace {
      display: grid;
      gap: 6px;
      margin-top: 6px;
    }

    .trace-step {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      background: #fff;
      font-size: 0.79rem;
    }

    .trace-step .head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: #1a516e;
      font-weight: 700;
      margin-bottom: 3px;
    }

    .architecture {
      margin-top: 8px;
      display: none;
    }

    .architecture.visible { display: block; }

    .arch-flow {
      display: grid;
      gap: 6px;
      margin-bottom: 8px;
    }

    .arch-step {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 7px;
      background: #fbfeff;
      font-size: 0.83rem;
    }

    .arch-agents {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 7px;
    }

    .arch-agent {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 8px;
      background: #fff;
    }

    .arch-agent h4 {
      margin: 0 0 4px;
      font-size: 0.9rem;
      color: var(--brand-dark);
    }

    .arch-agent .role {
      color: #2f657f;
      font-size: 0.76rem;
      margin-bottom: 5px;
    }

    /* Retro dark theme override */
    :root {
      --canvas: #0b1018;
      --panel: #111827;
      --ink: #e9f0ff;
      --subtle: #8aa0c2;
      --line: #24344c;
      --brand: #24d6a2;
      --brand-dark: #84d0ff;
      --ok-bg: #153327;
      --ok-ink: #9df5c9;
      --warn-bg: #3a2c19;
      --warn-ink: #ffcf8c;
      --chip: #1a2536;
      --soft-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    }

    body {
      color: var(--ink);
      background:
        radial-gradient(circle at 14% 4%, rgba(39, 84, 167, 0.35) 0%, transparent 33%),
        radial-gradient(circle at 88% 0%, rgba(16, 163, 126, 0.28) 0%, transparent 35%),
        linear-gradient(160deg, #090d14 0%, #111b29 46%, #0e1520 100%);
    }

    .hero,
    .panel,
    .chat-shell,
    .turn,
    .status-card,
    .arch-agent,
    .arch-step {
      background: linear-gradient(180deg, rgba(18, 28, 44, 0.96), rgba(14, 22, 34, 0.96));
      border-color: var(--line);
      box-shadow: var(--soft-shadow);
    }

    .subtitle,
    .hint,
    .status-card label,
    .turn-head {
      color: var(--subtle);
    }

    textarea,
    select {
      background: #0f1928;
      color: var(--ink);
      border-color: var(--line);
    }

    .btn,
    .pill,
    .model-pill {
      background: #152235;
      color: #b6d3f0;
      border-color: #2a3a56;
    }

    .btn.ghost {
      color: #99d7ff;
      background: #132033;
    }

    .btn.warn {
      color: var(--warn-ink);
      border-color: #6f5a34;
      background: #241c12;
    }

    .session-chip {
      background: #14312a;
      color: #9df5c9;
      border-color: #2a644f;
    }

    .toggle-row,
    .empty-state,
    .kpi,
    .kpi.label,
    details,
    .mono,
    .trace-step {
      background: #101a29;
      border-color: #2a3b58;
      color: #dceaff;
    }

    .kpi.label {
      background: linear-gradient(135deg, #173123, #11283a);
      border-color: #2f6f58;
    }

    .kpi .v {
      color: #9ff2ca;
    }

    .bubble.user {
      background: #182942;
      border-color: #2e4f7a;
      color: #cfe5ff;
    }

    .bubble.assistant {
      background: #122739;
      border-color: #1f4d63;
      color: #d6f7ff;
    }

    th {
      background: #152438;
      color: #a8d8ff;
    }

    td {
      color: #d3e1f4;
      border-bottom-color: #253750;
    }

    .table-wrap {
      border-color: #2a3b58;
      background: #101a29;
    }

    summary {
      color: #98d3ff;
    }

    .trace-bar {
      margin: 6px 0 4px;
      height: 6px;
      border-radius: 999px;
      background: #1a2a3f;
      overflow: hidden;
    }

    .trace-bar i {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #20d39f, #79c9ff);
    }

    .diag-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      margin: 8px 0;
    }

    .diag-card {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 8px;
      background: #122034;
    }

    .diag-card .k {
      font-size: 0.72rem;
      color: var(--subtle);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .diag-card .v {
      margin-top: 3px;
      font-size: 0.86rem;
      color: #def0ff;
      word-break: break-word;
    }

    .trace-wrap {
      max-height: 320px;
      overflow: auto;
      padding-right: 4px;
    }

    .md-block p {
      margin: 0 0 8px;
    }

    .md-block h4,
    .md-block h5 {
      margin: 0 0 8px;
      color: #9bd5ff;
      letter-spacing: 0.01em;
    }

    .md-block ul,
    .md-block ol {
      margin: 0 0 8px 18px;
      padding: 0;
    }

    .md-block li {
      margin-bottom: 4px;
    }

    .md-block code,
    .diag-card code {
      font-family: var(--mono);
      font-size: 0.78rem;
      color: #aee2ff;
      background: #0f1a2a;
      border: 1px solid #27374f;
      border-radius: 6px;
      padding: 1px 4px;
    }

    .diag-outcome {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 8px;
      margin: 8px 0;
      font-size: 0.84rem;
      line-height: 1.42;
    }

    .diag-outcome.ok {
      background: #12281f;
      border-color: #295743;
      color: #b8f7d5;
    }

    .diag-outcome.warn {
      background: #2a2116;
      border-color: #5f4a2d;
      color: #ffd8a3;
    }

    .trace-step.ok {
      border-color: #325f4e;
    }

    .trace-step.warn {
      border-color: #6d5532;
    }

    .trace-step .meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 0.74rem;
      color: var(--subtle);
      margin-top: 5px;
    }

    .correction-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 8px;
      max-height: 180px;
      overflow: auto;
      padding-right: 4px;
    }

    .correction-item {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 8px;
      background: #0e1524;
    }

    .correction-item .top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }

    .trust-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 8px;
      background: #0e1524;
      color: #dceaff;
      font-size: 0.78rem;
    }

    .trust-kpis {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-top: 6px;
    }

    .flow-wrap {
      margin: 8px 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #0e1524;
    }

    .flow-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 8px;
    }

    .flow-card {
      border: 1px solid #2c3d57;
      border-radius: 9px;
      padding: 7px;
      background: #101a2c;
      font-size: 0.76rem;
      color: #9fb7d4;
    }

    .flow-card strong {
      color: #d8ebff;
      display: block;
      margin-bottom: 3px;
    }

    @media (max-width: 1020px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .chat-shell {
        min-height: 420px;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1 class="title">dataDa</h1>
          <p class="subtitle">A conversational, agentic data analytics teammate. Ask broad questions, ask follow-ups, and the system keeps context in your active session.</p>
        </div>
        <div class="session-chip" id="sessionChip">session: initializing...</div>
      </div>
      <div class="status-grid">
        <div class="status-card"><label>Database</label><div class="value" id="dbPath">checking...</div></div>
        <div class="status-card"><label>Health</label><div class="value" id="healthState">checking...</div></div>
        <div class="status-card"><label>Semantic Layer</label><div class="value" id="semanticState">checking...</div></div>
        <div class="status-card"><label>Recommended Runtime</label><div class="value" id="recommendedMode">checking...</div></div>
      </div>
    </section>

    <section class="layout">
      <aside>
        <section class="panel">
          <h3>Ask dataDa</h3>
          <div class="hint">Try a business question, a vague exploration prompt, or a follow-up from the previous answer.</div>
          <textarea id="queryInput" placeholder="Example: Total MT103 transaction count split month wise and platform wise"></textarea>
          <div class="composer-actions">
            <button class="btn primary" id="runBtn" type="button">Run Query</button>
            <button class="btn ghost" id="newSessionBtn" type="button">New Session</button>
            <button class="btn warn" id="clearThreadBtn" type="button">Clear Thread</button>
          </div>

          <div class="row">
            <label>
              <div class="hint">LLM Mode</div>
              <select id="llmMode">
                <option value="auto">Auto</option>
                <option value="local">Local Ollama</option>
                <option value="openai">OpenAI</option>
                <option value="deterministic">Deterministic</option>
              </select>
            </label>
            <label>
              <div class="hint">Local Model</div>
              <select id="localModel">
                <option value="">Auto-select</option>
              </select>
            </label>
          </div>
          <div class="row">
            <label>
              <div class="hint">Data Connection</div>
              <select id="connectionSelect">
                <option value="default">default</option>
              </select>
            </label>
            <label>
              <div class="hint">Connections</div>
              <button class="btn ghost" id="refreshConnectionsBtn" type="button" style="margin-top:3px;">Refresh Connections</button>
            </label>
          </div>

          <div class="toggle-row">
            <label for="storyMode"><strong>Storyteller mode</strong> (friendlier narration)</label>
            <input type="checkbox" id="storyMode" />
          </div>

          <div class="composer-actions">
            <button class="btn ghost" id="refreshModelsBtn" type="button">Refresh Models</button>
            <button class="btn ghost" id="toggleArchBtn" type="button">View Agent Team Map</button>
          </div>
          <div class="composer-actions">
            <button class="btn ghost" id="refreshCorrectionsBtn" type="button">Review Corrections</button>
            <button class="btn ghost" id="refreshTrustBtn" type="button">Trust Dashboard</button>
            <button class="btn ghost" id="runTruthCheckBtn" type="button">Run Source Truth Check</button>
          </div>
          <div class="correction-list" id="correctionsList"></div>
          <div class="correction-list" id="trustPanel"></div>
          <div class="model-catalog" id="modelCatalog"></div>
          <div class="examples" id="examples"></div>
          <div class="hint" id="statusText" style="margin-top:9px;">Ready.</div>
        </section>

        <section class="panel architecture" id="architecturePanel">
          <h3>Agent Team Map</h3>
          <div class="hint">Each agent owns one responsibility. Together they act like a compact analytics + data engineering pod.</div>
          <div id="archContent" style="margin-top:8px;"></div>
        </section>
      </aside>

      <section class="chat-shell">
        <div class="thread" id="thread"></div>
      </section>
    </section>
  </main>

  <script>
    const EXAMPLES = [
      'What kind of data do I have?',
      'Total MT103 transactions count split by month wise and platform wise',
      'Top 5 platforms by total transaction amount in December 2025',
      'Compare this month vs last month transaction count',
      'Now show that by state',
      'Explain like I am new: what changed in bookings this year?'
    ];

    const STORAGE_SESSION_KEY = 'datada_session_id';
    const STORAGE_THREAD_KEY = 'datada_thread';
    const STORAGE_CONN_KEY = 'datada_connection_id';
    const STORAGE_TENANT_KEY = 'datada_tenant_id';

    const els = {
      queryInput: document.getElementById('queryInput'),
      runBtn: document.getElementById('runBtn'),
      newSessionBtn: document.getElementById('newSessionBtn'),
      clearThreadBtn: document.getElementById('clearThreadBtn'),
      modeSelect: document.getElementById('llmMode'),
      localModelSelect: document.getElementById('localModel'),
      connectionSelect: document.getElementById('connectionSelect'),
      refreshConnectionsBtn: document.getElementById('refreshConnectionsBtn'),
      refreshModelsBtn: document.getElementById('refreshModelsBtn'),
      refreshCorrectionsBtn: document.getElementById('refreshCorrectionsBtn'),
      refreshTrustBtn: document.getElementById('refreshTrustBtn'),
      runTruthCheckBtn: document.getElementById('runTruthCheckBtn'),
      modelCatalog: document.getElementById('modelCatalog'),
      correctionsList: document.getElementById('correctionsList'),
      trustPanel: document.getElementById('trustPanel'),
      examples: document.getElementById('examples'),
      statusText: document.getElementById('statusText'),
      thread: document.getElementById('thread'),
      storyMode: document.getElementById('storyMode'),
      sessionChip: document.getElementById('sessionChip'),
      architecturePanel: document.getElementById('architecturePanel'),
      toggleArchBtn: document.getElementById('toggleArchBtn'),
      archContent: document.getElementById('archContent'),
      dbPath: document.getElementById('dbPath'),
      healthState: document.getElementById('healthState'),
      semanticState: document.getElementById('semanticState'),
      recommendedMode: document.getElementById('recommendedMode')
    };

    const state = {
      sessionId: null,
      turns: [],
      architectureLoaded: false,
      connectionId: 'default',
      tenantId: 'public'
    };

    function esc(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function mdInline(text) {
      return esc(text)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    }

    function md(text) {
      if (!text) return '';
      const lines = String(text).replace(/\\r\\n/g, '\\n').split('\\n');
      const html = [];
      let listMode = null; // "ul" | "ol" | null

      const closeList = () => {
        if (listMode) {
          html.push(`</${listMode}>`);
          listMode = null;
        }
      };

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
          closeList();
          continue;
        }

        if (line.startsWith('### ')) {
          closeList();
          html.push(`<h5>${mdInline(line.slice(4))}</h5>`);
          continue;
        }
        if (line.startsWith('## ')) {
          closeList();
          html.push(`<h4>${mdInline(line.slice(3))}</h4>`);
          continue;
        }
        if (line.startsWith('- ') || line.startsWith('* ')) {
          if (listMode !== 'ul') {
            closeList();
            html.push('<ul>');
            listMode = 'ul';
          }
          html.push(`<li>${mdInline(line.slice(2))}</li>`);
          continue;
        }

        const ordered = line.match(/^\d+\.\s+(.*)$/);
        if (ordered) {
          if (listMode !== 'ol') {
            closeList();
            html.push('<ol>');
            listMode = 'ol';
          }
          html.push(`<li>${mdInline(ordered[1])}</li>`);
          continue;
        }

        closeList();
        html.push(`<p>${mdInline(line)}</p>`);
      }

      closeList();
      return html.join('');
    }

    function fmt(v) {
      if (typeof v === 'number') {
        return Number.isInteger(v)
          ? v.toLocaleString()
          : v.toLocaleString(undefined, { maximumFractionDigits: 4 });
      }
      if (v === null || v === undefined) return '';
      return String(v);
    }

    function fmtTimeFilter(timeFilter) {
      if (!timeFilter || typeof timeFilter !== 'object') return 'none';
      if (timeFilter.kind === 'month_year') {
        const month = Number(timeFilter.month || 0);
        const year = Number(timeFilter.year || 0);
        if (month >= 1 && month <= 12 && year > 0) {
          return `${year}-${String(month).padStart(2, '0')}`;
        }
      }
      if (timeFilter.kind === 'year_only' && timeFilter.year) return String(timeFilter.year);
      if (timeFilter.kind === 'relative' && timeFilter.value) return String(timeFilter.value).replace(/_/g, ' ');
      return esc(JSON.stringify(timeFilter));
    }

    function fmtFilters(valueFilters) {
      if (!Array.isArray(valueFilters) || !valueFilters.length) return 'none';
      return valueFilters
        .map((vf) => `${vf.column || '?'}=${vf.value || '?'}`)
        .join(', ');
    }

    function setStatus(msg) {
      els.statusText.textContent = msg;
    }

    function newSessionId() {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      return `sess-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    }

    function safeStorageGet(key) {
      try {
        return localStorage.getItem(key);
      } catch (err) {
        return null;
      }
    }

    function safeStorageSet(key, value) {
      try {
        localStorage.setItem(key, value);
      } catch (err) {
        // Ignore storage errors (private mode or blocked storage).
      }
    }

    function saveState() {
      safeStorageSet(STORAGE_SESSION_KEY, state.sessionId);
      safeStorageSet(STORAGE_THREAD_KEY, JSON.stringify(state.turns.slice(0, 30)));
      safeStorageSet(STORAGE_CONN_KEY, state.connectionId || 'default');
      safeStorageSet(STORAGE_TENANT_KEY, state.tenantId || 'public');
    }

    function loadState() {
      const existingSession = safeStorageGet(STORAGE_SESSION_KEY);
      state.sessionId = existingSession || newSessionId();
      try {
        const raw = safeStorageGet(STORAGE_THREAD_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        if (Array.isArray(parsed)) state.turns = parsed;
      } catch (err) {
        state.turns = [];
      }
      const savedConn = safeStorageGet(STORAGE_CONN_KEY);
      state.connectionId = savedConn || 'default';
      const savedTenant = safeStorageGet(STORAGE_TENANT_KEY);
      state.tenantId = (savedTenant || 'public').trim() || 'public';
      updateSessionChip();
    }

    function updateSessionChip() {
      const shortId = state.sessionId ? state.sessionId.slice(0, 12) : 'none';
      els.sessionChip.textContent = `session: ${shortId}`;
    }

    function resetSession(clearThread = true) {
      state.sessionId = newSessionId();
      if (clearThread) state.turns = [];
      saveState();
      renderThread();
      updateSessionChip();
      setStatus('Started a fresh session.');
    }

    function renderExamples() {
      els.examples.innerHTML = EXAMPLES
        .map((q) => `<button class="pill" type="button">${esc(q)}</button>`)
        .join('');
      Array.from(els.examples.querySelectorAll('button')).forEach((btn, idx) => {
        btn.addEventListener('click', () => {
          els.queryInput.value = EXAMPLES[idx];
          els.queryInput.focus();
        });
      });
    }

    function renderModelCatalog(modelState) {
      if (!modelState || !Array.isArray(modelState.options) || modelState.options.length === 0) {
        els.modelCatalog.innerHTML = '<span class="hint">No local model metadata.</span>';
        return;
      }

      const active = els.localModelSelect.value;
      els.modelCatalog.innerHTML = modelState.options.map((opt) => {
        const cls = [
          'model-pill',
          opt.installed ? '' : 'install',
          active === opt.name ? 'active' : ''
        ].join(' ').trim();
        const action = opt.installed ? 'Select' : 'Download';
        return `<button type="button" class="${cls}" data-model="${esc(opt.name)}" data-installed="${opt.installed ? '1' : '0'}">${esc(opt.name)} • ${esc(opt.tier)} • ${action}</button>`;
      }).join('');

      Array.from(els.modelCatalog.querySelectorAll('button[data-model]')).forEach((btn) => {
        btn.addEventListener('click', async () => {
          const model = btn.getAttribute('data-model');
          const installed = btn.getAttribute('data-installed') === '1';
          if (installed) await selectLocalModel(model);
          else await pullLocalModel(model);
        });
      });
    }

    async function loadLocalModels() {
      try {
        const stateResp = await fetch('/api/assistant/models/local').then((r) => r.json());
        const installed = (stateResp.options || []).filter((o) => o.installed);
        els.localModelSelect.innerHTML =
          '<option value="">Auto-select</option>' +
          installed.map((o) => `<option value="${esc(o.name)}">${esc(o.name)} (${esc(o.tier)})</option>`).join('');

        const active = stateResp.active_intent_model || '';
        if (active && installed.some((o) => o.name === active)) {
          els.localModelSelect.value = active;
        }
        renderModelCatalog(stateResp);
      } catch (err) {
        els.modelCatalog.innerHTML = `<span class="hint">Local model service unavailable: ${esc(err.message)}</span>`;
      }
    }

    async function loadConnections() {
      try {
        const data = await fetch('/api/assistant/connections').then((r) => r.json());
        const list = Array.isArray(data.connections) ? data.connections : [];
        if (!list.length) {
          els.connectionSelect.innerHTML = '<option value="default">default</option>';
          state.connectionId = 'default';
          saveState();
          return;
        }

        const defaultId = data.default_connection_id || 'default';
        els.connectionSelect.innerHTML = list.map((c) => {
          const label = `${c.id}${c.is_default ? ' (default)' : ''} • ${c.kind}${c.exists ? '' : ' • missing'}`;
          return `<option value="${esc(c.id)}">${esc(label)}</option>`;
        }).join('');

        const preferred = state.connectionId || defaultId;
        const hasPreferred = list.some((c) => c.id === preferred);
        state.connectionId = hasPreferred ? preferred : defaultId;
        els.connectionSelect.value = state.connectionId;
        saveState();
      } catch (err) {
        setStatus(`Connections unavailable: ${err.message}`);
      }
    }

    async function toggleCorrection(correctionId, enabled) {
      try {
        const response = await fetch('/api/assistant/corrections/toggle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            db_connection_id: state.connectionId || 'default',
            correction_id: correctionId,
            enabled
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.message || 'Toggle failed');
        setStatus(data.message || 'Correction updated.');
        await loadCorrections();
      } catch (err) {
        setStatus(`Correction update failed: ${err.message}`);
      }
    }

    async function loadCorrections() {
      try {
        const q = new URLSearchParams({
          db_connection_id: state.connectionId || 'default',
          include_disabled: 'true',
          limit: '60'
        });
        const data = await fetch(`/api/assistant/corrections?${q}`).then((r) => r.json());
        const rules = Array.isArray(data.rules) ? data.rules : [];
        if (!rules.length) {
          els.correctionsList.innerHTML = '<div class="hint">No learned corrections yet.</div>';
          return;
        }
        els.correctionsList.innerHTML = rules.map((rule) => {
          const dims = Array.isArray(rule.target_dimensions) && rule.target_dimensions.length
            ? rule.target_dimensions.join(', ')
            : 'none';
          return `
            <div class="correction-item">
              <div class="top">
                <strong>${esc(rule.keyword || 'keyword')}</strong>
                <button class="btn ${rule.enabled ? 'warn' : 'primary'}" data-correction-id="${esc(rule.correction_id)}" data-next-enabled="${rule.enabled ? '0' : '1'}" type="button" style="padding:4px 8px; font-size:0.72rem;">
                  ${rule.enabled ? 'Disable' : 'Enable'}
                </button>
              </div>
              <div class="hint">→ ${esc(rule.target_table)} • ${esc(rule.target_metric)} • dims: ${esc(dims)}</div>
              <div class="hint">weight=${esc(fmt(rule.weight || 0))} • ${rule.enabled ? 'enabled' : 'disabled'}</div>
            </div>
          `;
        }).join('');

        Array.from(els.correctionsList.querySelectorAll('button[data-correction-id]')).forEach((btn) => {
          btn.addEventListener('click', async () => {
            const correctionId = btn.getAttribute('data-correction-id');
            const nextEnabled = btn.getAttribute('data-next-enabled') === '1';
            if (!correctionId) return;
            await toggleCorrection(correctionId, nextEnabled);
          });
        });
      } catch (err) {
        els.correctionsList.innerHTML = `<div class="hint">Corrections unavailable: ${esc(err.message)}</div>`;
      }
    }

    function renderTrustDashboard(data) {
      const modes = Array.isArray(data.by_mode) ? data.by_mode : [];
      const modeRows = modes.length
        ? modes.map((m) => (
            `<div class="hint">${esc(m.mode)} • runs=${esc(fmt(m.runs))} • success=${esc(fmt(Math.round((m.success_rate || 0) * 100)))}% • avg=${esc(fmt(m.avg_execution_ms || 0))} ms</div>`
          )).join('')
        : '<div class="hint">No mode metrics yet.</div>';
      const failures = Array.isArray(data.recent_failures) ? data.recent_failures.slice(0, 4) : [];
      const failureRows = failures.length
        ? failures.map((f) => `<div class="hint">${esc(f.created_at)} • ${esc(f.llm_mode)} • ${esc((f.goal || '').slice(0, 90))}</div>`).join('')
        : '<div class="hint">No recent failed runs.</div>';
      els.trustPanel.innerHTML = `
        <div class="trust-card">
          <div><strong>Trust window:</strong> ${esc(fmt(data.window_hours || 0))}h • tenant=${esc(data.tenant_id || 'all')}</div>
          <div class="trust-kpis">
            <div class="hint">runs: <strong>${esc(fmt(data.runs || 0))}</strong></div>
            <div class="hint">success: <strong>${esc(fmt(Math.round((data.success_rate || 0) * 100)))}%</strong></div>
            <div class="hint">avg confidence: <strong>${esc(fmt(Math.round((data.avg_confidence || 0) * 100)))}%</strong></div>
            <div class="hint">p95 execution: <strong>${esc(fmt(data.p95_execution_ms || 0))} ms</strong></div>
          </div>
          <div style="margin-top:6px;"><strong>By mode</strong></div>
          ${modeRows}
          <div style="margin-top:6px;"><strong>Recent failures</strong></div>
          ${failureRows}
        </div>
      `;
    }

    async function loadTrustDashboard() {
      try {
        const q = new URLSearchParams({
          tenant_id: state.tenantId || 'public',
          hours: '168'
        });
        const data = await fetch(`/api/assistant/trust/dashboard?${q}`).then((r) => r.json());
        renderTrustDashboard(data);
      } catch (err) {
        els.trustPanel.innerHTML = `<div class="hint">Trust metrics unavailable: ${esc(err.message)}</div>`;
      }
    }

    async function runSourceTruthCheck() {
      setStatus('Running source-truth parity checks...');
      try {
        const q = new URLSearchParams({
          db_connection_id: state.connectionId || 'default',
          llm_mode: els.modeSelect.value || 'deterministic',
          max_cases: '6'
        });
        if (els.localModelSelect.value) q.set('local_model', els.localModelSelect.value);
        const response = await fetch(`/api/assistant/source-truth/check?${q}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Source-truth check failed');
        const rows = Array.isArray(data.runs) ? data.runs : [];
        const preview = rows.slice(0, 4).map((r) => (
          `<div class="hint">${esc(r.case_id)} • ${r.exact_match ? 'match' : 'mismatch'} • ${esc(fmt(r.latency_ms || 0))} ms</div>`
        )).join('');
        els.trustPanel.innerHTML = `
          <div class="trust-card">
            <div><strong>Source truth:</strong> accuracy=${esc(fmt(data.accuracy_pct || 0))}% • exact=${esc(fmt(data.exact_matches || 0))}/${esc(fmt(data.evaluated_cases || 0))} • mode=${esc(data.mode_actual || '')}</div>
            <div style="margin-top:6px;">${preview || '<span class="hint">No cases evaluated.</span>'}</div>
          </div>
        ` + els.trustPanel.innerHTML;
        setStatus('Source-truth check complete.');
      } catch (err) {
        setStatus(`Source-truth check failed: ${err.message}`);
      }
    }

    async function selectLocalModel(modelName) {
      setStatus(`Activating ${modelName}...`);
      try {
        const response = await fetch('/api/assistant/models/local/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelName, narrator_model: modelName })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.message || 'Model activation failed');
        els.localModelSelect.value = modelName;
        setStatus(data.message || `Activated ${modelName}`);
        await loadLocalModels();
      } catch (err) {
        setStatus(`Model activation failed: ${err.message}`);
      }
    }

    async function pullLocalModel(modelName) {
      setStatus(`Downloading ${modelName}. This may take a few minutes...`);
      try {
        const response = await fetch('/api/assistant/models/local/pull', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelName, activate_after_download: true })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.message || 'Model pull failed');
        setStatus(data.message || `${modelName} downloaded`);
        if (data.active_intent_model) {
          els.localModelSelect.value = data.active_intent_model;
        }
        await loadLocalModels();
      } catch (err) {
        setStatus(`Model download failed: ${err.message}`);
      }
    }

    function renderTable(columns, rows) {
      if (!Array.isArray(columns) || columns.length === 0 || !Array.isArray(rows) || rows.length === 0) {
        return '<div class="hint">No rows returned.</div>';
      }
      const head = `<tr>${columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr>`;
      const body = rows.map((row) => (
        `<tr>${columns.map((c) => `<td>${esc(fmt(row[c]))}</td>`).join('')}</tr>`
      )).join('');
      return `<div class="table-wrap"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
    }

    function renderTrace(trace) {
      if (!Array.isArray(trace) || trace.length === 0) return '<div class="hint">No trace.</div>';
      const maxDuration = Math.max(
        1,
        ...trace.map((step) => {
          const ms = Number(step && step.duration_ms);
          return Number.isFinite(ms) ? ms : 0;
        })
      );

      return `<div class="trace trace-wrap">${trace.map((step) => {
        const status = String(step.status || 'unknown').toLowerCase();
        const duration = Number(step.duration_ms || 0);
        const pct = Math.max(4, Math.round((Math.max(0, duration) / maxDuration) * 100));
        const cls = status === 'success' ? 'ok' : 'warn';
        return `
          <div class="trace-step ${cls}">
            <div class="head"><span>${esc(step.agent || 'agent')}</span><span>${esc(status)} • ${esc(fmt(duration))} ms</span></div>
            <div class="trace-bar"><i style="width:${pct}%"></i></div>
            <div>${esc(step.summary || '')}</div>
            <div class="meta"><span>${esc(step.role || '')}</span><span>${esc(step.time || '')}</span></div>
          </div>
        `;
      }).join('')}</div>`;
    }

    function renderBlackboardFlow(evidencePackets) {
      const packet = Array.isArray(evidencePackets)
        ? evidencePackets.find((p) => p && p.agent === 'Blackboard')
        : null;
      if (!packet || !Array.isArray(packet.artifacts) || !packet.artifacts.length) {
        return '<div class="hint">No blackboard artifacts captured.</div>';
      }
      const artifacts = packet.artifacts.slice(-12);
      const cards = artifacts.map((a) => {
        const consumers = Array.isArray(a.consumed_by) && a.consumed_by.length
          ? a.consumed_by.join(', ')
          : 'none';
        return `
          <div class="flow-card">
            <strong>${esc(a.producer || 'agent')} • ${esc(a.artifact_type || 'artifact')}</strong>
            <div>${esc(a.summary || '')}</div>
            <div class="hint" style="margin-top:4px;">consumed by: ${esc(consumers)}</div>
          </div>
        `;
      }).join('');
      const edgeLines = Array.isArray(packet.edges)
        ? packet.edges.slice(0, 24).map((e) => `${e.from} → ${e.to} (${e.artifact_type})`)
        : [];
      const edgeHtml = edgeLines.length
        ? `<div class="hint" style="margin-top:8px;"><strong>Flow edges:</strong> ${esc(edgeLines.join(' | '))}</div>`
        : '';
      return `<div class="flow-wrap"><div class="flow-grid">${cards}</div>${edgeHtml}</div>`;
    }

    function checksHtml(checks) {
      if (!Array.isArray(checks) || checks.length === 0) return '<span class="hint">No checks.</span>';
      return checks.map((c) => {
        const ok = !!c.passed;
        return `<span class="tag ${ok ? 'ok' : 'warn'}">${ok ? 'PASS' : 'WARN'} ${esc(c.check_name || 'check')}</span>`;
      }).join(' ');
    }

    function pickHeadlineValue(data) {
      if (!data || !Array.isArray(data.sample_rows) || data.sample_rows.length === 0) return null;
      const row0 = data.sample_rows[0] || {};
      if (Object.prototype.hasOwnProperty.call(row0, 'metric_value')) {
        return row0.metric_value;
      }
      const cols = Array.isArray(data.columns) ? data.columns : [];
      if (cols.length === 1) return row0[cols[0]];
      const numericCandidate = Object.entries(row0).find(([, v]) => typeof v === 'number');
      return numericCandidate ? numericCandidate[1] : null;
    }

    function responseCard(turn) {
      const data = turn.response;
      if (!data) return `<div class="bubble assistant">Thinking...</div>`;
      const runtime = data.runtime || {};
      const quality = data.data_quality || {};
      const grounding = quality.grounding || {};
      const confidencePct = Math.round((data.confidence_score || 0) * 100);
      const headline = pickHeadlineValue(data);
      const checks = Array.isArray(data.sanity_checks) ? data.sanity_checks : [];

      const suggestions = (data.suggested_questions || []).slice(0, 4);
      const suggestionHtml = suggestions.length
        ? `<div class="suggestions">${suggestions.map((q) => `<button class="pill suggest" type="button" data-q="${esc(q)}">${esc(q)}</button>`).join('')}</div>`
        : '';

      const termMisses = Array.isArray(grounding.goal_term_misses) ? grounding.goal_term_misses : [];
      const dims = Array.isArray(grounding.dimensions) ? grounding.dimensions : [];
      const dimsLabel = dims.length
        ? dims.map((d) => (d === '__month__' ? 'month' : d)).join(', ')
        : 'none';
      const filtersLabel = fmtFilters(grounding.value_filters || []);
      const conceptTagClass = termMisses.length ? 'warn' : 'ok';
      const replayText = grounding.replay_match === null || grounding.replay_match === undefined
        ? 'n/a'
        : (grounding.replay_match ? 'pass' : 'fail');
      const failedChecks = checks.filter((c) => !c.passed).map((c) => c.check_name || 'check');
      const outcomeMessages = [];
      if (termMisses.length) outcomeMessages.push(`Missing goal concepts: ${termMisses.join(', ')}`);
      if (grounding.replay_match === false) outcomeMessages.push('Replay check failed: deterministic replay did not match.');
      if (failedChecks.length) outcomeMessages.push(`Checks flagged: ${failedChecks.join(', ')}`);

      const diagOutcome = outcomeMessages.length
        ? `<div class="diag-outcome warn"><strong>Debug focus:</strong> ${esc(outcomeMessages.join(' | '))}</div>`
        : `<div class="diag-outcome ok"><strong>Validation:</strong> concept, replay, and sanity checks are aligned.</div>`;

      const diagHtml = `
        <div class="diag-grid">
          <div class="diag-card">
            <div class="k">Interpretation</div>
            <div class="v">intent=<code>${esc(grounding.intent || 'n/a')}</code> metric=<code>${esc(grounding.metric || 'n/a')}</code></div>
          </div>
          <div class="diag-card">
            <div class="k">Data Scope</div>
            <div class="v">table=<code>${esc(grounding.table || 'unknown')}</code> group_by=<code>${esc(dimsLabel)}</code></div>
          </div>
          <div class="diag-card">
            <div class="k">Time Window</div>
            <div class="v">${esc(fmtTimeFilter(grounding.time_filter))}</div>
          </div>
          <div class="diag-card">
            <div class="k">Applied Filters</div>
            <div class="v">${esc(filtersLabel)}</div>
          </div>
          <div class="diag-card">
            <div class="k">Validation</div>
            <div class="v">concept=<strong>${termMisses.length ? 'warn' : 'pass'}</strong>, replay=<strong>${esc(replayText)}</strong></div>
          </div>
          <div class="diag-card">
            <div class="k">Execution Path</div>
            <div class="v">mode=<code>${esc(runtime.mode || 'unknown')}</code> connection=<code>${esc(runtime.db_connection_id || 'default')}</code></div>
          </div>
        </div>
      `;
      const blackboardHtml = renderBlackboardFlow(data.evidence_packets || []);

      return `
        <div class="bubble assistant md-block">${md(data.answer_markdown || '')}</div>
        <div class="kpi-grid">
          <div class="kpi label">
            <div class="k">Key result</div>
            <div class="v">${headline === null || headline === undefined ? 'n/a' : esc(fmt(headline))}</div>
          </div>
          <div class="kpi">
            <div class="k">Confidence</div>
            <div class="v">${esc(data.confidence || 'unknown')} (${confidencePct}%)</div>
          </div>
          <div class="kpi">
            <div class="k">Execution</div>
            <div class="v">${esc(fmt(data.execution_time_ms || 0))} ms</div>
          </div>
        </div>
        <div style="margin-bottom:6px;">
          <span class="tag ${data.success ? 'ok' : 'warn'}">${data.success ? 'SUCCESS' : 'DEGRADED'}</span>
          <span class="tag ${data.success ? 'ok' : 'warn'}">${esc(runtime.mode || 'unknown')}</span>
          <span class="hint">rows: ${esc(fmt(data.row_count || 0))}</span>
        </div>
        <div style="margin-bottom:6px;">
          <span class="tag ${conceptTagClass}">Concept ${termMisses.length ? 'warn' : 'pass'}</span>
          <span class="tag ${grounding.replay_match === false ? 'warn' : 'ok'}">Replay ${esc(replayText)}</span>
        </div>
        ${diagHtml}
        ${diagOutcome}
        ${checksHtml(checks)}
        ${suggestionHtml}
        ${renderTable(data.columns || [], data.sample_rows || [])}
        <details>
          <summary>Technical details (SQL, trace, quality)</summary>
          <div class="mono">${esc(data.sql || 'No SQL generated')}</div>
          ${renderTrace(data.agent_trace || [])}
          ${blackboardHtml}
          <div class="mono">${esc(JSON.stringify(data.data_quality || {}, null, 2))}</div>
        </details>
      `;
    }

    function renderThread() {
      if (!state.turns.length) {
        els.thread.innerHTML = `
          <div class="empty-state">
            Ask a question to start. Try:
            <br/>• "What kind of data do I have?"
            <br/>• "Total MT103 transactions count split by month wise and platform wise"
            <br/>• Then ask a follow-up like "Now only for December 2025."
          </div>
        `;
        return;
      }

      els.thread.innerHTML = state.turns.map((turn, idx) => `
        <article class="turn">
          <div class="turn-head">
            <span>Turn ${state.turns.length - idx}</span>
            <span>${turn.response && turn.response.trace_id ? esc(turn.response.trace_id.slice(0, 12)) : 'pending'}</span>
          </div>
          <div class="turn-body">
            <div class="bubble user">${esc(turn.goal)}</div>
            ${responseCard(turn)}
          </div>
        </article>
      `).join('');

      Array.from(els.thread.querySelectorAll('button.suggest')).forEach((btn) => {
        btn.addEventListener('click', () => {
          const q = btn.getAttribute('data-q');
          els.queryInput.value = q || '';
          els.queryInput.focus();
        });
      });
    }

    async function loadArchitecture() {
      try {
        const data = await fetch('/api/assistant/architecture').then((r) => r.json());
        const flow = (data.pipeline_flow || []).map((line) => line.replace(/^\\d+\\.\\s*/, ''));
        const guardrails = (data.guardrails || []).map((g) => `<li>${esc(g)}</li>`).join('');
        const agents = (data.agents || []).map((a) => `
          <div class="arch-agent">
            <h4>${esc(a.name)}</h4>
            <div class="role">${esc(a.role)}</div>
            <div class="hint">${esc(a.description)}</div>
            <div class="hint" style="margin-top:4px;">In: ${esc((a.inputs || []).join(', '))}</div>
            <div class="hint">Out: ${esc((a.outputs || []).join(', '))}</div>
          </div>
        `).join('');

        els.archContent.innerHTML = `
          <div class="arch-flow">${flow.map((f) => `<div class="arch-step">${esc(f)}</div>`).join('')}</div>
          <div class="hint" style="margin-bottom:6px;"><strong>Guardrails:</strong></div>
          <ul style="margin: 0 0 8px 18px; color: #9eb7d6; font-size: 0.82rem;">${guardrails}</ul>
          <div class="arch-agents">${agents}</div>
        `;
        state.architectureLoaded = true;
      } catch (err) {
        els.archContent.innerHTML = `<div class="hint">Failed to load architecture: ${esc(err.message)}</div>`;
      }
    }

    async function initSystemStatus() {
      try {
        const [health, providers] = await Promise.all([
          fetch('/api/assistant/health').then((r) => r.json()),
          fetch('/api/assistant/providers').then((r) => r.json())
        ]);
        const defaultConn = health.default_connection_id || 'default';
        els.dbPath.textContent = `${health.db_path} (${defaultConn})`;
        els.healthState.textContent = health.status;
        els.semanticState.textContent = health.semantic_ready ? 'ready' : 'not ready';
        els.recommendedMode.textContent = providers.recommended_mode;
        els.modeSelect.value = providers.default_mode || 'auto';
      } catch (err) {
        els.healthState.textContent = 'unreachable';
        els.semanticState.textContent = 'unknown';
      }
    }

    async function runQuery() {
      const goal = els.queryInput.value.trim();
      if (!goal) {
        setStatus('Enter a question first.');
        return;
      }

      const turn = { goal, response: null };
      state.turns.unshift(turn);
      saveState();
      renderThread();

      els.runBtn.disabled = true;
      setStatus('Running agent team...');

      try {
        const response = await fetch('/api/assistant/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            goal,
            db_connection_id: state.connectionId || 'default',
            llm_mode: els.modeSelect.value,
            local_model: els.localModelSelect.value || null,
            session_id: state.sessionId,
            storyteller_mode: !!els.storyMode.checked,
            tenant_id: state.tenantId || 'public',
            role: 'analyst'
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.error || 'Query failed');
        }
        turn.response = data;
        if (data.runtime && data.runtime.session_id) {
          state.sessionId = data.runtime.session_id;
          updateSessionChip();
        }
        saveState();
        renderThread();
        setStatus('Done.');
      } catch (err) {
        turn.response = {
          success: false,
          answer_markdown: `**Request failed**\\n\\n${err.message}`,
          confidence: 'uncertain',
          confidence_score: 0,
          sanity_checks: [],
          columns: [],
          sample_rows: [],
          runtime: { mode: 'error' },
          execution_time_ms: 0,
          row_count: 0
        };
        saveState();
        renderThread();
        setStatus(`Request failed: ${err.message}`);
      } finally {
        els.runBtn.disabled = false;
      }
    }

    function wireEvents() {
      els.runBtn.addEventListener('click', runQuery);
      els.refreshModelsBtn.addEventListener('click', loadLocalModels);
      els.refreshCorrectionsBtn.addEventListener('click', loadCorrections);
      els.refreshTrustBtn.addEventListener('click', loadTrustDashboard);
      els.runTruthCheckBtn.addEventListener('click', runSourceTruthCheck);
      els.refreshConnectionsBtn.addEventListener('click', loadConnections);
      els.connectionSelect.addEventListener('change', () => {
        state.connectionId = els.connectionSelect.value || 'default';
        saveState();
        setStatus(`Using connection: ${state.connectionId}`);
        loadCorrections();
      });
      els.localModelSelect.addEventListener('change', async () => {
        const model = els.localModelSelect.value;
        if (!model) return;
        await selectLocalModel(model);
      });
      els.newSessionBtn.addEventListener('click', () => resetSession(true));
      els.clearThreadBtn.addEventListener('click', async () => {
        state.turns = [];
        saveState();
        renderThread();
        try {
          await fetch('/api/assistant/session/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              db_connection_id: state.connectionId || 'default',
              session_id: state.sessionId,
              tenant_id: state.tenantId || 'public'
            })
          });
        } catch (err) {
          // local clear already completed
        }
        setStatus('Thread cleared in this session.');
      });
      els.toggleArchBtn.addEventListener('click', async () => {
        const visible = els.architecturePanel.classList.contains('visible');
        if (visible) {
          els.architecturePanel.classList.remove('visible');
          els.toggleArchBtn.textContent = 'View Agent Team Map';
          return;
        }
        if (!state.architectureLoaded) await loadArchitecture();
        els.architecturePanel.classList.add('visible');
        els.toggleArchBtn.textContent = 'Hide Agent Team Map';
      });
      els.queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runQuery();
      });
    }

    async function init() {
      loadState();
      renderExamples();
      renderThread();
      wireEvents();
      await Promise.all([initSystemStatus(), loadLocalModels(), loadConnections()]);
      await Promise.all([loadCorrections(), loadTrustDashboard()]);
      setStatus('Ready.');
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
