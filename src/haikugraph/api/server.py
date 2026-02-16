"""FastAPI server for dataDa agentic POC."""

from __future__ import annotations

import importlib.util
import os
import socket
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
        return app.state.team.run(request.goal, runtime)

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
      --ink: #0f2430;
      --muted: #617783;
      --panel: #ffffff;
      --canvas: #eef4f8;
      --accent: #0ea37f;
      --accent2: #0f3f5b;
      --line: #d7e3ea;
      --ok: #1c8a4d;
      --warn: #b15b13;
      --mono: "IBM Plex Mono", Menlo, Consolas, monospace;
      --sans: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(circle at 90% -5%, #d2f7ec 0%, transparent 35%),
        radial-gradient(circle at -10% 8%, #d9ecff 0%, transparent 40%),
        var(--canvas);
      color: var(--ink);
      min-height: 100vh;
    }

    .container {
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px 14px 40px;
    }

    .hero, .panel, .result-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 12px 28px rgba(6, 44, 58, 0.06);
    }

    .hero {
      padding: 22px;
      margin-bottom: 14px;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.7rem, 3vw, 2.4rem);
      color: var(--accent2);
      letter-spacing: -0.02em;
    }

    .sub {
      color: var(--muted);
      margin-top: 6px;
      max-width: 780px;
    }

    .status-grid {
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }

    .status-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fcfefe;
    }

    .status-card label {
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
      margin-bottom: 5px;
    }

    .status-card .value {
      font-weight: 700;
      font-size: 0.92rem;
      word-break: break-word;
    }

    .panel {
      padding: 16px;
      margin-bottom: 14px;
    }

    .query-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }

    textarea, select, button {
      border: 1px solid var(--line);
      border-radius: 12px;
      font-family: var(--sans);
    }

    textarea {
      min-height: 84px;
      width: 100%;
      padding: 12px;
      font-size: 0.96rem;
      resize: vertical;
      background: #fbfefe;
      color: var(--ink);
    }

    .controls {
      margin-top: 10px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      align-items: end;
    }

    select {
      width: 100%;
      padding: 10px;
      font-size: 0.9rem;
      background: #fff;
    }

    button {
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 700;
      transition: transform 120ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.65; cursor: not-allowed; transform: none; }

    .primary {
      border: none;
      background: linear-gradient(120deg, var(--accent), #2dbb90);
      color: #fff;
      box-shadow: 0 8px 18px rgba(15, 163, 127, 0.24);
    }

    .ghost {
      background: #fff;
      color: var(--accent2);
    }

    .examples {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .model-catalog {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 6px 10px;
      font-size: 0.82rem;
      color: var(--muted);
      cursor: pointer;
    }

    .model-pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 6px 10px;
      font-size: 0.79rem;
      color: var(--accent2);
      cursor: pointer;
    }

    .model-pill.install {
      color: var(--warn);
      border-style: dashed;
    }

    .model-pill.active {
      background: #e8f8ef;
      border-color: #92d7b8;
      color: var(--ok);
      font-weight: 700;
    }

    .result-panel {
      padding: 14px;
      display: none;
    }

    .result-panel.visible { display: block; }

    .card {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 10px;
      background: #fff;
    }

    .card h3 {
      margin: 0 0 8px;
      color: var(--accent2);
      font-size: 1rem;
    }

    .mono {
      font-family: var(--mono);
      font-size: 0.8rem;
      white-space: pre-wrap;
      background: #f6fbfe;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      margin: 0;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 8px;
    }

    .tag {
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-right: 5px;
    }

    .tag.ok { background: #e8f8ef; color: var(--ok); }
    .tag.warn { background: #fff2e4; color: var(--warn); }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
    }

    th, td {
      border-bottom: 1px solid #ecf2f6;
      text-align: left;
      padding: 7px 8px;
      white-space: nowrap;
    }

    th {
      background: #f8fcff;
      color: #21475f;
      position: sticky;
      top: 0;
    }

    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
    }

    .timeline {
      display: grid;
      gap: 6px;
    }

    .step {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      font-size: 0.83rem;
      background: #fcfeff;
    }

    .step .head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 5px;
    }

    .step .head strong { color: var(--accent2); }
    .hint { color: var(--muted); font-size: 0.88rem; }

    @media (max-width: 760px) {
      .query-row { grid-template-columns: 1fr; }
      .container { padding: 16px 10px 28px; }
    }
  </style>
</head>
<body>
  <main class="container">
    <section class="hero">
      <h1>dataDa</h1>
      <p class="sub">Agentic analytics team over your data. Each response is generated by a coordinated set of agents with evidence, SQL trace, and audit confidence.</p>
      <div class="status-grid">
        <div class="status-card"><label>Database</label><div class="value" id="dbPath">checking...</div></div>
        <div class="status-card"><label>Health</label><div class="value" id="healthState">checking...</div></div>
        <div class="status-card"><label>Semantic Layer</label><div class="value" id="semanticState">checking...</div></div>
        <div class="status-card"><label>Recommended Runtime</label><div class="value" id="recommendedMode">loading...</div></div>
      </div>
    </section>

    <section class="panel">
      <div class="query-row">
        <textarea id="queryInput" placeholder="Ask: Total transactions in December 2025 split by platform"></textarea>
        <button id="runBtn" class="primary">Run Query</button>
      </div>
      <div class="controls">
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
          <div class="hint">Local Model (try higher quality)</div>
          <select id="localModel">
            <option value="">Auto-select</option>
          </select>
        </label>
        <button id="archBtn" class="ghost" type="button">View Architecture</button>
      </div>
      <div class="controls">
        <button id="refreshModelsBtn" class="ghost" type="button">Refresh Local Models</button>
      </div>
      <div class="model-catalog" id="modelCatalog"></div>
      <div class="examples" id="examples"></div>
      <div class="hint" id="statusText" style="margin-top:10px;">Ready.</div>
    </section>

    <section class="result-panel" id="resultPanel"></section>
  </main>

  <script>
    const queryInput = document.getElementById('queryInput');
    const runBtn = document.getElementById('runBtn');
    const modeSelect = document.getElementById('llmMode');
    const localModelSelect = document.getElementById('localModel');
    const refreshModelsBtn = document.getElementById('refreshModelsBtn');
    const resultPanel = document.getElementById('resultPanel');
    const statusText = document.getElementById('statusText');
    let localModelState = null;

    const EXAMPLES = [
      'Total transactions in December 2025',
      'Top 5 platforms by transaction count in December 2025',
      'Total amount by month',
      'Refund rate by platform',
      'Quote volume by from_currency',
      'Compare this month vs last month transaction count'
    ];

    function esc(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function md(text) {
      if (!text) return '';
      return esc(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\\n/g, '<br/>');
    }

    function fmt(v) {
      if (typeof v === 'number') {
        return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 4 });
      }
      return String(v ?? '');
    }

    function setStatus(msg) { statusText.textContent = msg; }

    function renderModelCatalog(state) {
      const box = document.getElementById('modelCatalog');
      if (!state || !state.options || state.options.length === 0) {
        box.innerHTML = '<span class="hint">No local model metadata.</span>';
        return;
      }
      box.innerHTML = state.options.map(opt => {
        const cls = [
          'model-pill',
          opt.installed ? '' : 'install',
          localModelSelect.value === opt.name ? 'active' : ''
        ].join(' ').trim();
        const action = opt.installed ? 'Select' : 'Download';
        return `<button type="button" class="${cls}" data-model="${esc(opt.name)}" data-installed="${opt.installed ? '1' : '0'}">${esc(opt.name)} (${esc(opt.tier)}) • ${action}</button>`;
      }).join('');

      Array.from(box.querySelectorAll('button[data-model]')).forEach(btn => {
        btn.addEventListener('click', async () => {
          const model = btn.getAttribute('data-model');
          const installed = btn.getAttribute('data-installed') === '1';
          if (installed) {
            await selectLocalModel(model);
          } else {
            await pullLocalModel(model);
          }
        });
      });
    }

    async function loadLocalModels() {
      try {
        const state = await fetch('/api/assistant/models/local').then(r => r.json());
        localModelState = state;
        const installed = (state.options || []).filter(o => o.installed);
        localModelSelect.innerHTML =
          '<option value="">Auto-select</option>' +
          installed.map(o => `<option value="${esc(o.name)}">${esc(o.name)} (${esc(o.tier)})</option>`).join('');
        if (state.active_intent_model && installed.some(o => o.name === state.active_intent_model)) {
          localModelSelect.value = state.active_intent_model;
        }
        renderModelCatalog(state);
      } catch (err) {
        document.getElementById('modelCatalog').innerHTML =
          `<span class="hint">Local models unavailable: ${esc(err.message)}</span>`;
      }
    }

    async function selectLocalModel(modelName) {
      setStatus(`Activating local model ${modelName}...`);
      try {
        const response = await fetch('/api/assistant/models/local/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelName, narrator_model: modelName })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.message || 'Failed to select model');
        localModelSelect.value = modelName;
        setStatus(data.message || `Activated ${modelName}`);
        await loadLocalModels();
      } catch (err) {
        setStatus(`Model activation failed: ${err.message}`);
      }
    }

    async function pullLocalModel(modelName) {
      setStatus(`Downloading model ${modelName}. This can take several minutes...`);
      try {
        const response = await fetch('/api/assistant/models/local/pull', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelName, activate_after_download: true })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.message || 'Failed to pull model');
        if (data.active_intent_model) {
          localModelSelect.value = data.active_intent_model;
        }
        setStatus(data.message || `Model ${modelName} downloaded`);
        await loadLocalModels();
      } catch (err) {
        setStatus(`Model download failed: ${err.message}`);
      }
    }

    function renderExamples() {
      const box = document.getElementById('examples');
      box.innerHTML = EXAMPLES.map(q => `<button class="pill" type="button">${esc(q)}</button>`).join('');
      Array.from(box.querySelectorAll('button')).forEach((btn, i) => {
        btn.addEventListener('click', () => {
          queryInput.value = EXAMPLES[i];
          queryInput.focus();
        });
      });
    }

    function renderTable(columns, rows) {
      if (!columns || columns.length === 0 || !rows || rows.length === 0) {
        return '<div class="hint">No rows returned.</div>';
      }
      const head = `<tr>${columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr>`;
      const body = rows.map(r => `<tr>${columns.map(c => `<td>${esc(fmt(r[c]))}</td>`).join('')}</tr>`).join('');
      return `<div class="table-wrap"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
    }

    function renderTrace(trace) {
      if (!trace || trace.length === 0) return '<div class="hint">No agent trace.</div>';
      return `<div class="timeline">${trace.map(s => `
        <div class="step">
          <div class="head"><strong>${esc(s.agent)}</strong><span>${esc(s.status)} • ${esc(s.duration_ms)} ms</span></div>
          <div class="hint">${esc(s.role || '')}</div>
          <div>${esc(s.summary || '')}</div>
        </div>
      `).join('')}</div>`;
    }

    function renderChecks(checks) {
      if (!checks || checks.length === 0) return '<div class="hint">No checks.</div>';
      return checks.map(c => {
        const ok = c.passed;
        return `<span class="tag ${ok ? 'ok' : 'warn'}">${ok ? 'PASS' : 'WARN'} ${esc(c.check_name)}</span>`;
      }).join(' ');
    }

    function renderResponse(data) {
      const runtime = data.runtime || {};
      const modeTag = `<span class="tag ${data.success ? 'ok' : 'warn'}">${esc(runtime.mode || 'unknown')}</span>`;

      resultPanel.innerHTML = `
        <div class="card">
          <h3>Answer</h3>
          <div>${md(data.answer_markdown)}</div>
        </div>

        <div class="card grid-2">
          <div>
            <h3>Runtime</h3>
            <div>${modeTag} Provider: <strong>${esc(runtime.provider || 'none')}</strong></div>
            <div class="hint">${esc(runtime.reason || '')}</div>
            <div class="hint">Intent model: ${esc(runtime.intent_model || 'n/a')} | Narrator model: ${esc(runtime.narrator_model || 'n/a')}</div>
            <div style="margin-top:6px;">Trace: <code>${esc(data.trace_id)}</code></div>
          </div>
          <div>
            <h3>Confidence</h3>
            <div><strong>${esc(data.confidence)}</strong> (${Math.round((data.confidence_score || 0) * 100)}%)</div>
            <div class="hint">Rows: ${esc(data.row_count)} | SQL time: ${esc(data.execution_time_ms)} ms</div>
          </div>
        </div>

        <div class="card">
          <h3>Sanity Checks</h3>
          <div>${renderChecks(data.sanity_checks || [])}</div>
        </div>

        <div class="card">
          <h3>Result Preview</h3>
          ${renderTable(data.columns || [], data.sample_rows || [])}
        </div>

        <div class="card">
          <h3>SQL</h3>
          <pre class="mono">${esc(data.sql || 'No SQL')}</pre>
        </div>

        <div class="card">
          <h3>Agent Trace</h3>
          ${renderTrace(data.agent_trace || [])}
        </div>

        <div class="card grid-2">
          <div>
            <h3>Chart Spec</h3>
            <pre class="mono">${esc(JSON.stringify(data.chart_spec || {}, null, 2))}</pre>
          </div>
          <div>
            <h3>Data Quality</h3>
            <pre class="mono">${esc(JSON.stringify(data.data_quality || {}, null, 2))}</pre>
          </div>
        </div>
      `;
      resultPanel.classList.add('visible');
    }

    async function init() {
      try {
        const [h, p] = await Promise.all([
          fetch('/api/assistant/health').then(r => r.json()),
          fetch('/api/assistant/providers').then(r => r.json()),
        ]);
        document.getElementById('dbPath').textContent = h.db_path;
        document.getElementById('healthState').textContent = h.status;
        document.getElementById('semanticState').textContent = h.semantic_ready ? 'ready' : 'not ready';
        document.getElementById('recommendedMode').textContent = p.recommended_mode;
        modeSelect.value = p.default_mode || 'auto';
      } catch (err) {
        document.getElementById('healthState').textContent = 'unreachable';
      }
      await loadLocalModels();
    }

    async function runQuery() {
      const goal = queryInput.value.trim();
      if (!goal) {
        setStatus('Enter a question first.');
        return;
      }

      runBtn.disabled = true;
      setStatus('Running agent team...');
      resultPanel.classList.add('visible');
      resultPanel.innerHTML = '<div class="card"><div class="hint">Processing...</div></div>';

      try {
        const response = await fetch('/api/assistant/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            goal,
            llm_mode: modeSelect.value,
            local_model: localModelSelect.value || null
          })
        });
        const data = await response.json();
        renderResponse(data);
        setStatus('Done.');
      } catch (err) {
        setStatus(`Request failed: ${err.message}`);
        resultPanel.innerHTML = `<div class="card"><div class="hint">${esc(err.message)}</div></div>`;
      } finally {
        runBtn.disabled = false;
      }
    }

    async function showArchitecture() {
      try {
        const data = await fetch('/api/assistant/architecture').then(r => r.json());
        resultPanel.classList.add('visible');
        resultPanel.innerHTML = `
          <div class="card"><h3>${esc(data.system_name)}</h3><div class="hint">${esc(data.description)}</div></div>
          <div class="card"><h3>Pipeline</h3><pre class="mono">${esc((data.pipeline_flow || []).join('\\\\n'))}</pre></div>
          <div class="card"><h3>Guardrails</h3><pre class="mono">${esc((data.guardrails || []).join('\\\\n'))}</pre></div>
        `;
      } catch (err) {
        resultPanel.classList.add('visible');
        resultPanel.innerHTML = `<div class="card"><div class="hint">${esc(err.message)}</div></div>`;
      }
    }

    runBtn.addEventListener('click', runQuery);
    document.getElementById('archBtn').addEventListener('click', showArchitecture);
    refreshModelsBtn.addEventListener('click', loadLocalModels);
    localModelSelect.addEventListener('change', async () => {
      const value = localModelSelect.value;
      if (!value) return;
      await selectLocalModel(value);
    });
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runQuery();
    });

    renderExamples();
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
