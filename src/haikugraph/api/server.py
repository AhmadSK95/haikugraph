"""FastAPI server for dataDa agentic POC."""

from __future__ import annotations

import base64
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
from haikugraph.io.document_ingest import ingest_documents_to_duckdb
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


def _resolve_team_for_connection(
    app: FastAPI,
    connection_id: str | None,
) -> tuple[AgenticAnalyticsTeam, Path, dict[str, Any]]:
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
    if kind not in {"duckdb", "documents"}:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Connection kind '{kind}' is registered but not directly query-routable yet. "
                "Use mirrored ingestion into a DuckDB serving layer for bounded-autonomy runtime execution."
            ),
        )

    if kind == "documents":
        db_path = _resolve_documents_duckdb_path(app, entry)
    else:
        db_path = Path(str(entry.get("path") or "")).expanduser()
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found for connection '{entry.get('id')}' at {db_path}",
        )

    cid = str(entry["id"])
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
        tenant_id=tenant_id,
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
        team, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        saved = team.record_feedback(
            tenant_id=access["tenant_id"],
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
        team, _, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        rows = team.list_corrections(
            tenant_id=access["tenant_id"],
            limit=limit,
            include_disabled=include_disabled,
        )
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
        team, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        ok = team.set_correction_enabled(
            request.correction_id,
            request.enabled,
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
        team, _, connection_entry = _resolve_team_for_connection(
            app,
            request.db_connection_id,
        )
        result = team.rollback_correction(request.correction_id, tenant_id=access["tenant_id"])
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
        team, _, connection_entry = _resolve_team_for_connection(app, db_connection_id)
        rows = team.list_tool_candidates(
            tenant_id=access["tenant_id"],
            status=status,
            limit=limit,
        )
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
        team, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        result = team.stage_tool_candidate(request.tool_id, tenant_id=access["tenant_id"])
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
        team, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        result = team.promote_tool_candidate(request.tool_id, tenant_id=access["tenant_id"])
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
        team, _, connection_entry = _resolve_team_for_connection(app, request.db_connection_id)
        result = team.rollback_tool_candidate(
            request.tool_id,
            tenant_id=access["tenant_id"],
            reason=request.reason,
        )
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
        payload = app.state.runtime_store.trust_dashboard(tenant_id=requested_tenant or None, hours=hours)
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

        payload = app.state.runtime_store.evaluate_slo(
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

        rows = app.state.runtime_store.list_incidents(
            tenant_id=requested_tenant or None,
            status=status,
            limit=limit,
        )
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
        # Tenant boundary enforcement for non-admin roles.
        if access["role"] != "admin":
            matched = app.state.runtime_store.list_incidents(
                tenant_id=access["tenant_id"],
                status=None,
                limit=500,
            )
            if not any(str(item.get("incident_id") or "") == request.incident_id for item in matched):
                raise HTTPException(status_code=403, detail="Cannot acknowledge incidents outside your tenant.")

        result = app.state.runtime_store.update_incident_status(
            incident_id=request.incident_id,
            status=request.status,
            note=request.note,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=str(result.get("message") or "Incident update failed."))
        return ConnectionActionResponse(success=True, message=str(result.get("message") or "Incident updated."))

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
    .thread{flex:1;overflow-y:auto;padding:8px 0 24px;display:flex;flex-direction:column;gap:24px;scrollbar-width:thin;scrollbar-color:var(--surface-3) transparent}
    .thread::-webkit-scrollbar{width:6px}
    .thread::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:3px}

    /* ---- empty state ---- */
    .empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:40px 0}
    .empty h2{font-size:22px;font-weight:600;color:var(--text)}
    .empty p{color:var(--text-muted);font-size:14px;max-width:400px;text-align:center;line-height:1.5}
    .pills{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:520px}
    .pill{padding:8px 16px;border:1px solid var(--gold-mid);border-radius:20px;font-size:13px;color:var(--gold);transition:all .15s;cursor:pointer}
    .pill:hover{background:var(--gold-dim);border-color:var(--gold)}

    /* ---- user turn ---- */
    .turn-user{font-size:14px;color:var(--text-muted);padding-left:2px}
    .turn-user q{color:var(--text);font-style:normal;quotes:none}

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
        <option value="deterministic">Deterministic</option>
      </select>
    </div>
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
        return `<div class="trace-item"><span class="trace-dot ${dot}"></span><span class="trace-name">${esc(t.agent || t.role || '')}</span><span class="trace-summary">${esc(t.summary || '')}</span><span class="trace-time">${ms}</span></div>`;
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
        let html = `<div class="turn-user"><q>${esc(turn.goal)}</q></div>`;
        const r = turn.response;
        if (!r) {
          html += `<div class="card"><div class="loading-card"><div class="spinner"></div>Analyzing...</div></div>`;
          return html;
        }
        if (!r.success && r.error) {
          html += `<div class="error-card">${esc(r.error || r.answer_markdown || 'Query failed')}</div>`;
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
        const hasInspect = hasSql || hasTrace || hasChecks;

        let inspect = '';
        if (hasInspect) {
          inspect += `<div class="inspect-toggle" onclick="toggleInspect(this)"><svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"/></svg>Inspect</div>`;
          inspect += `<div class="inspect-content">`;
          if (hasSql) inspect += `<div class="inspect-section"><div class="inspect-label">SQL</div><div class="inspect-sql">${esc(r.sql)}</div></div>`;
          if (hasTrace) inspect += `<div class="inspect-section"><div class="inspect-label">Agent Trace</div>${buildTrace(r.agent_trace)}</div>`;
          if (hasChecks) inspect += `<div class="inspect-section"><div class="inspect-label">Audit Checks</div>${buildAudit(r.sanity_checks)}</div>`;
          inspect += `</div>`;
        }

        const rowCount = r.row_count != null ? `<span class="meta-chip">${fmt(r.row_count)} rows</span><span class="meta-sep"></span>` : '';
        const execTime = r.execution_time_ms != null ? `<span class="meta-chip">${Math.round(r.execution_time_ms)}ms</span>` : '';
        const mode = r.runtime && r.runtime.mode ? `<span class="meta-chip">${esc(r.runtime.mode)}</span><span class="meta-sep"></span>` : '';

        html += `<div class="card">
          <div class="card-body">
            <div style="margin-bottom:12px">${badge}</div>
            <div class="card-answer">${answer}</div>
            ${chart}${table}
            ${suggestions ? '<div class="suggestions">' + suggestions + '</div>' : ''}
          </div>
          <div class="card-meta">${mode}${rowCount}${execTime}</div>
          ${inspect}
        </div>`;
        return html;
      }).join('');
      el.scrollTop = el.scrollHeight;
    }

    function toggleInspect(el) {
      el.classList.toggle('open');
      const content = el.nextElementSibling;
      if (content) content.classList.toggle('open');
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

      const turn = { goal, response: null };
      state.turns.push(turn);
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

        const resp = await fetch('/api/assistant/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await resp.json();
        turn.response = data;
      } catch (e) {
        turn.response = { success: false, error: e.message || 'Network error' };
      }

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
