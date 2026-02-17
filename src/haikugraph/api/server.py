"""FastAPI server for dataDa agentic POC."""

from __future__ import annotations

import importlib.util
import os
import socket
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.llm.router import DEFAULT_MODELS
from haikugraph.poc import AgenticAnalyticsTeam, RuntimeSelection, load_dotenv_file


DEFAULT_DB_CANDIDATES = (
    Path("./data/datada.duckdb"),
    Path("./data/haikugraph.duckdb"),
    Path("./data/haikugraph.db"),
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
        "2. Intake Agent - Clarifies intent, metrics, filters, time scope",
        "3. Semantic Retrieval Agent - Maps query to semantic marts",
        "4. Planning Agent - Produces task graph and metric definitions",
        "5. Specialist Agents - Transactions, Customers, Revenue, Risk",
        "6. Query Engineer Agent - Compiles SQL plan",
        "7. Execution Agent - Runs SQL via safe executor",
        "8. Audit Agent - Validates consistency and confidence",
        "9. Governance Agent - Policy checks",
        "10. Narrative + Visualization Agents - Final insight and chart spec",
    ]
    guardrails: list[str] = [
        "Read-only SQL only",
        "Blocked destructive keywords",
        "Bounded result sizes",
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

    app.state.db_path = db_path or _get_db_path()
    app.state.team = AgenticAnalyticsTeam(app.state.db_path)
    app.state.sessions: dict[str, list[dict[str, Any]]] = {}
    app.state.sessions_lock = threading.RLock()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        app.state.team.close()

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def ui() -> str:
        return get_ui_html()

    @app.get("/api/assistant/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        db_path: Path = app.state.db_path
        exists = db_path.exists()
        semantic_ready = False
        if exists:
            try:
                app.state.team.semantic.prepare()
                semantic_ready = True
            except Exception:
                semantic_ready = False

        return HealthResponse(
            status="ok" if exists else "no_database",
            db_exists=exists,
            db_path=str(db_path),
            db_size_bytes=db_path.stat().st_size if exists else 0,
            semantic_ready=semantic_ready,
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
    async def query(request: QueryRequest) -> AssistantQueryResponse:
        db_path: Path = app.state.db_path
        if not db_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Database not found at {db_path}. Run ingestion first.",
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
        session_id = (request.session_id or "default").strip()[:128] or "default"

        with app.state.sessions_lock:
            history = list(app.state.sessions.get(session_id, []))

        response = app.state.team.run(
            request.goal,
            runtime,
            conversation_context=history,
            storyteller_mode=request.storyteller_mode,
        )

        turn = {
            "goal": request.goal,
            "user_goal": request.goal,
            "answer_markdown": response.answer_markdown,
            "success": response.success,
            "sql": response.sql,
            "confidence_score": response.confidence_score,
        }
        with app.state.sessions_lock:
            turns = app.state.sessions.setdefault(session_id, [])
            turns.append(turn)
            if len(turns) > 20:
                del turns[:-20]
            conversation_turns = len(turns)

        response.runtime = {
            **(response.runtime or {}),
            "session_id": session_id,
            "conversation_turns": conversation_turns,
        }
        return response

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

          <div class="toggle-row">
            <label for="storyMode"><strong>Storyteller mode</strong> (friendlier narration)</label>
            <input type="checkbox" id="storyMode" />
          </div>

          <div class="composer-actions">
            <button class="btn ghost" id="refreshModelsBtn" type="button">Refresh Models</button>
            <button class="btn ghost" id="toggleArchBtn" type="button">View Agent Team Map</button>
          </div>
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

    const els = {
      queryInput: document.getElementById('queryInput'),
      runBtn: document.getElementById('runBtn'),
      newSessionBtn: document.getElementById('newSessionBtn'),
      clearThreadBtn: document.getElementById('clearThreadBtn'),
      modeSelect: document.getElementById('llmMode'),
      localModelSelect: document.getElementById('localModel'),
      refreshModelsBtn: document.getElementById('refreshModelsBtn'),
      modelCatalog: document.getElementById('modelCatalog'),
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
      architectureLoaded: false
    };

    function esc(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function md(text) {
      if (!text) return '';
      const html = esc(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\\n/g, '<br/>');
      return html;
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

    function setStatus(msg) {
      els.statusText.textContent = msg;
    }

    function newSessionId() {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      return `sess-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    }

    function saveState() {
      localStorage.setItem(STORAGE_SESSION_KEY, state.sessionId);
      localStorage.setItem(STORAGE_THREAD_KEY, JSON.stringify(state.turns.slice(0, 30)));
    }

    function loadState() {
      const existingSession = localStorage.getItem(STORAGE_SESSION_KEY);
      state.sessionId = existingSession || newSessionId();
      try {
        const raw = localStorage.getItem(STORAGE_THREAD_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        if (Array.isArray(parsed)) state.turns = parsed;
      } catch (err) {
        state.turns = [];
      }
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
      return `<div class="trace">${trace.map((step) => `
        <div class="trace-step">
          <div class="head"><span>${esc(step.agent || 'agent')}</span><span>${esc(step.status || '')} • ${esc(step.duration_ms || 0)} ms</span></div>
          <div class="hint">${esc(step.role || '')}</div>
          <div>${esc(step.summary || '')}</div>
        </div>
      `).join('')}</div>`;
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

      const suggestions = (data.suggested_questions || []).slice(0, 4);
      const suggestionHtml = suggestions.length
        ? `<div class="suggestions">${suggestions.map((q) => `<button class="pill suggest" type="button" data-q="${esc(q)}">${esc(q)}</button>`).join('')}</div>`
        : '';
      const termMisses = Array.isArray(grounding.goal_term_misses) ? grounding.goal_term_misses : [];
      const groundingTagClass = termMisses.length ? 'warn' : 'ok';
      const replayText = grounding.replay_match === null || grounding.replay_match === undefined
        ? 'n/a'
        : (grounding.replay_match ? 'pass' : 'fail');
      const groundingHtml = `
        <div style="margin:8px 0; border:1px solid #dce9ee; border-radius:9px; padding:8px; background:#fbfeff;">
          <div class="hint"><strong>Grounding:</strong> table=<code>${esc(grounding.table || 'unknown')}</code> metric=<code>${esc(grounding.metric || 'unknown')}</code></div>
          <div style="margin-top:4px;">
            <span class="tag ${groundingTagClass}">Concept alignment ${termMisses.length ? 'warn' : 'pass'}</span>
            <span class="tag ${grounding.replay_match ? 'ok' : 'warn'}">Replay ${esc(replayText)}</span>
          </div>
          ${termMisses.length ? `<div class="hint" style="margin-top:4px;">Missing goal concepts: ${esc(termMisses.join(', '))}</div>` : ''}
        </div>
      `;

      return `
        <div class="bubble assistant">${md(data.answer_markdown || '')}</div>
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
        ${groundingHtml}
        ${checksHtml(data.sanity_checks || [])}
        ${suggestionHtml}
        ${renderTable(data.columns || [], data.sample_rows || [])}
        <details>
          <summary>Technical details (SQL, trace, quality)</summary>
          <div class="mono">${esc(data.sql || 'No SQL generated')}</div>
          ${renderTrace(data.agent_trace || [])}
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
          <ul style="margin: 0 0 8px 18px; color: #355969; font-size: 0.82rem;">${guardrails}</ul>
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
        els.dbPath.textContent = health.db_path;
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
            llm_mode: els.modeSelect.value,
            local_model: els.localModelSelect.value || null,
            session_id: state.sessionId,
            storyteller_mode: !!els.storyMode.checked
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
      els.localModelSelect.addEventListener('change', async () => {
        const model = els.localModelSelect.value;
        if (!model) return;
        await selectLocalModel(model);
      });
      els.newSessionBtn.addEventListener('click', () => resetSession(true));
      els.clearThreadBtn.addEventListener('click', () => {
        state.turns = [];
        saveState();
        renderThread();
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
      await Promise.all([initSystemStatus(), loadLocalModels()]);
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
