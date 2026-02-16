"""Unified FastAPI server for the dataDa web product.

This module exposes:
- /                     -> interactive dataDa web interface
- /api/assistant/query  -> main NL query endpoint
- /api/assistant/health -> service + database status
- /api/assistant/providers -> runtime LLM provider status
- /api/assistant/architecture -> architecture metadata
"""

from __future__ import annotations

import importlib.util
import os
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from haikugraph.agents.contracts import (
    AssistantQueryResponse,
    ConfidenceLevel,
    SanityCheck,
)
from haikugraph.orchestrator.runtime import AnalystOrchestrator, OrchestratorConfig


DEFAULT_DB_CANDIDATES = (
    Path("./data/datada.duckdb"),
    Path("./data/haikugraph.duckdb"),
    Path("./data/haikugraph.db"),
)


class LLMMode(str, Enum):
    """Runtime mode for query execution."""

    AUTO = "auto"
    LOCAL = "local"
    OPENAI = "openai"
    DETERMINISTIC = "deterministic"


class ProviderCheck(BaseModel):
    """Health/capability check for an LLM provider."""

    available: bool
    reason: str


class ProvidersResponse(BaseModel):
    """Runtime LLM provider state for UI and diagnostics."""

    default_mode: LLMMode
    recommended_mode: LLMMode
    checks: dict[str, ProviderCheck]


class QueryRequest(BaseModel):
    """Request model for /api/assistant/query."""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query in natural language",
    )
    db_connection_id: str = Field(default="default", description="Database connection id")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Optional constraints")
    llm_mode: LLMMode = Field(default=LLMMode.AUTO, description="LLM runtime mode")


class LegacyAskRequest(BaseModel):
    """Legacy request model for /ask endpoint."""

    question: str = Field(..., min_length=1)


class LegacyAskResponse(BaseModel):
    """Legacy response model for /ask endpoint."""

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
    """Service health response."""

    status: str
    db_exists: bool
    db_path: str
    db_size_bytes: int = 0
    version: str = "1.0.0"


class AgentInfo(BaseModel):
    """Agent metadata."""

    name: str
    role: str
    description: str
    inputs: list[str]
    outputs: list[str]


class ArchitectureResponse(BaseModel):
    """Architecture metadata for UI display."""

    system_name: str = "dataDa Intelligent Data Assistant"
    version: str = "1.0.0"
    description: str = "Human-like analytics assistant over your data streams and tables"
    pipeline_flow: list[str] = [
        "1. IntakeAgent - Understand intent and constraints",
        "2. SchemaAgent - Build schema + semantic context",
        "3. QueryAgent - Generate and execute guarded SQL",
        "4. AuditAgent - Validate reliability and scope",
        "5. NarratorAgent - Produce user-facing insight narrative",
    ]
    guardrails: list[str] = [
        "Read-only SQL policy (SELECT/WITH/EXPLAIN only)",
        "Dangerous keyword blocking before execution",
        "LIMIT enforcement with bounded result sets",
        "Timeout and retry controls",
        "Prompt-injection sanitization in schema sampling",
        "Deterministic fallback mode when LLMs are unavailable",
    ]
    response_includes: list[str] = [
        "final insight summary",
        "confidence and sanity checks",
        "evidence and executed SQL",
        "sample rows and runtime metadata",
    ]
    agents: list[AgentInfo] = []


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ollama_check() -> ProviderCheck:
    base_url = os.environ.get("HG_OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return ProviderCheck(available=True, reason=f"reachable at {base_url}")
        return ProviderCheck(available=False, reason=f"unhealthy status {response.status_code}")
    except Exception as exc:  # pragma: no cover - network errors vary by env
        return ProviderCheck(available=False, reason=str(exc))


def _openai_check() -> ProviderCheck:
    if not _has_module("openai"):
        return ProviderCheck(available=False, reason="openai package not installed")

    key = os.environ.get("HG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ProviderCheck(available=False, reason="missing OPENAI_API_KEY")

    return ProviderCheck(available=True, reason="package + API key detected")


def _providers_snapshot() -> ProvidersResponse:
    raw_default_mode = os.environ.get("HG_DEFAULT_LLM_MODE", LLMMode.AUTO.value).lower()
    try:
        default_mode = LLMMode(raw_default_mode)
    except ValueError:
        default_mode = LLMMode.AUTO

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

    return ProvidersResponse(
        default_mode=default_mode,
        recommended_mode=recommended,
        checks=checks,
    )


def _resolve_runtime(mode: LLMMode) -> dict[str, Any]:
    providers = _providers_snapshot()

    if mode == LLMMode.DETERMINISTIC:
        return {
            "mode": mode.value,
            "use_llm": False,
            "provider": None,
            "reason": "Forced deterministic execution",
        }

    if mode == LLMMode.LOCAL:
        local_state = providers.checks["ollama"]
        if local_state.available:
            return {
                "mode": mode.value,
                "use_llm": True,
                "provider": "ollama",
                "reason": "Using local Ollama",
            }
        return {
            "mode": LLMMode.DETERMINISTIC.value,
            "use_llm": False,
            "provider": None,
            "reason": f"Local mode requested but Ollama unavailable: {local_state.reason}",
        }

    if mode == LLMMode.OPENAI:
        openai_state = providers.checks["openai"]
        if openai_state.available:
            return {
                "mode": mode.value,
                "use_llm": True,
                "provider": "openai",
                "reason": "Using OpenAI API",
            }
        return {
            "mode": LLMMode.DETERMINISTIC.value,
            "use_llm": False,
            "provider": None,
            "reason": f"OpenAI mode requested but unavailable: {openai_state.reason}",
        }

    if providers.checks["ollama"].available:
        return {
            "mode": LLMMode.LOCAL.value,
            "use_llm": True,
            "provider": "ollama",
            "reason": "Auto-selected local Ollama",
        }

    if providers.checks["openai"].available:
        return {
            "mode": LLMMode.OPENAI.value,
            "use_llm": True,
            "provider": "openai",
            "reason": "Auto-selected OpenAI",
        }

    return {
        "mode": LLMMode.DETERMINISTIC.value,
        "use_llm": False,
        "provider": None,
        "reason": "No LLM provider available; using deterministic path",
    }


def get_db_path() -> Path:
    """Resolve DB path from env override + known candidates."""
    env_path = os.environ.get("HG_DB_PATH")
    if env_path:
        return Path(env_path)

    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_DB_CANDIDATES[0]


def create_app(db_path: Path | None = None) -> FastAPI:
    """Create FastAPI application."""
    new_app = FastAPI(
        title="dataDa Assistant API",
        description="Intelligent conversational data assistant",
        version="1.0.0",
    )

    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    new_app.state.db_path = db_path or get_db_path()
    _register_routes(new_app)
    return new_app


def _register_routes(target_app: FastAPI) -> None:
    @target_app.get("/", response_class=HTMLResponse)
    async def serve_ui() -> str:
        return get_ui_html()

    @target_app.get("/api/assistant/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        db_path: Path = target_app.state.db_path
        exists = db_path.exists()
        return HealthResponse(
            status="ok" if exists else "no_database",
            db_exists=exists,
            db_path=str(db_path),
            db_size_bytes=db_path.stat().st_size if exists else 0,
        )

    @target_app.get("/api/assistant/providers", response_model=ProvidersResponse)
    async def providers_status() -> ProvidersResponse:
        return _providers_snapshot()

    @target_app.get("/api/assistant/architecture", response_model=ArchitectureResponse)
    async def architecture() -> ArchitectureResponse:
        agents = [
            AgentInfo(
                name="IntakeAgent",
                role="Goal parser",
                description="Converts natural language goals to structured intent",
                inputs=["goal", "constraints"],
                outputs=["intent", "metrics", "dimensions", "time window"],
            ),
            AgentInfo(
                name="SchemaAgent",
                role="Semantic schema profiler",
                description="Infers relevant tables, keys, joins, and semantic columns",
                inputs=["intent", "database schema"],
                outputs=["relevant tables", "join graph", "column recommendations"],
            ),
            AgentInfo(
                name="QueryAgent",
                role="SQL planner + executor",
                description="Builds safe SQL and executes with guardrails",
                inputs=["intent", "schema context"],
                outputs=["SQL", "result sample", "summary"],
            ),
            AgentInfo(
                name="AuditAgent",
                role="Result validator",
                description="Performs consistency and quality checks on query results",
                inputs=["query result", "schema context"],
                outputs=["audit checks", "refinement hints"],
            ),
            AgentInfo(
                name="NarratorAgent",
                role="Insight writer",
                description="Transforms raw results into user-facing insight",
                inputs=["query result", "audit result"],
                outputs=["answer", "confidence", "follow-up prompts"],
            ),
        ]
        return ArchitectureResponse(agents=agents)

    @target_app.post("/api/assistant/query", response_model=AssistantQueryResponse)
    async def query_assistant(request: QueryRequest) -> AssistantQueryResponse:
        db_path: Path = target_app.state.db_path

        if not db_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Database not found at {db_path}. Run ingestion before querying.",
            )

        runtime = _resolve_runtime(request.llm_mode)
        config = OrchestratorConfig(
            use_llm=runtime["use_llm"],
            llm_provider=runtime["provider"],
            max_refinement_loops=2,
        )

        try:
            with AnalystOrchestrator(db_path, config=config) as orchestrator:
                response = orchestrator.run(goal=request.goal, constraints=request.constraints)
            response.runtime = runtime
            return response
        except Exception as exc:
            return AssistantQueryResponse(
                success=False,
                answer_markdown=f"**Error processing request**\n\n{exc}",
                confidence=ConfidenceLevel.UNCERTAIN,
                confidence_score=0.0,
                definition_used=request.goal,
                evidence=[],
                sanity_checks=[
                    SanityCheck(check_name="api_error", passed=False, message=str(exc))
                ],
                trace_id="error",
                error=str(exc),
                suggested_questions=["Try rephrasing your question"],
                runtime=runtime,
            )

    @target_app.post("/ask", response_model=LegacyAskResponse)
    async def legacy_ask(request: LegacyAskRequest) -> LegacyAskResponse:
        response = await query_assistant(
            QueryRequest(goal=request.question, llm_mode=LLMMode.AUTO)
        )
        return LegacyAskResponse(
            final_answer=response.answer_markdown,
            intent={"type": "multi_agent", "confidence": response.confidence_score},
            plan={"trace_id": response.trace_id, "runtime": response.runtime},
            queries=[response.sql] if response.sql else [],
            results=response.sample_rows,
            metadata={
                "confidence": response.confidence.value,
                "confidence_score": response.confidence_score,
                "definition_used": response.definition_used,
                "row_count": response.row_count,
                "runtime": response.runtime,
            },
            errors=[response.error] if response.error else [],
        )


def get_ui_html() -> str:
    """Return inline HTML UI for dataDa."""
    return """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>dataDa</title>
    <style>
        :root {
            --ink: #172024;
            --muted: #5f707a;
            --canvas: #f3f6f4;
            --panel: #ffffff;
            --accent: #0f9f7c;
            --accent-2: #0b3b4b;
            --warn: #cc6f14;
            --ok: #238454;
            --border: #dce7e2;
            --mono: \"IBM Plex Mono\", \"SFMono-Regular\", Menlo, Consolas, monospace;
            --sans: \"Space Grotesk\", \"Avenir Next\", \"Segoe UI\", sans-serif;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: var(--sans);
            color: var(--ink);
            background:
                radial-gradient(circle at 85% -20%, #b7efe0 0%, transparent 40%),
                radial-gradient(circle at -10% 10%, #d6f0ff 0%, transparent 45%),
                var(--canvas);
            min-height: 100vh;
        }

        .shell {
            max-width: 1100px;
            margin: 0 auto;
            padding: 28px 16px 36px;
        }

        .hero {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 24px;
            background: linear-gradient(140deg, #ffffff 0%, #f4fffb 100%);
            box-shadow: 0 12px 30px rgba(9, 53, 67, 0.08);
            animation: reveal 420ms ease-out;
        }

        @keyframes reveal {
            from { transform: translateY(8px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .title {
            margin: 0;
            font-size: clamp(1.6rem, 3vw, 2.3rem);
            letter-spacing: -0.02em;
            color: var(--accent-2);
        }

        .subtitle {
            margin: 8px 0 0;
            color: var(--muted);
            max-width: 760px;
        }

        .meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 18px;
        }

        .chip {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 12px;
            background: #fff;
        }

        .chip label {
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 5px;
        }

        .chip .value {
            font-size: 0.93rem;
            color: var(--ink);
            font-weight: 600;
        }

        .panel {
            margin-top: 16px;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: var(--panel);
            box-shadow: 0 14px 32px rgba(5, 44, 56, 0.06);
            padding: 20px;
        }

        .controls {
            display: grid;
            gap: 12px;
            grid-template-columns: 1fr;
        }

        .query-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
        }

        .row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px;
        }

        textarea,
        select,
        button {
            border-radius: 12px;
            border: 1px solid var(--border);
            font-family: var(--sans);
        }

        textarea {
            width: 100%;
            min-height: 84px;
            resize: vertical;
            padding: 12px;
            font-size: 0.98rem;
            color: var(--ink);
            background: #fdfefe;
        }

        select {
            padding: 10px;
            font-size: 0.93rem;
            background: #fdfefe;
            color: var(--ink);
        }

        button {
            padding: 10px 16px;
            font-weight: 700;
            letter-spacing: 0.01em;
            cursor: pointer;
            transition: transform 120ms ease, box-shadow 120ms ease;
        }

        .primary {
            background: linear-gradient(120deg, var(--accent), #34b78d);
            color: #fff;
            border: none;
            box-shadow: 0 8px 18px rgba(20, 144, 111, 0.28);
        }

        .ghost {
            background: #fff;
            color: var(--accent-2);
        }

        button:hover { transform: translateY(-1px); }
        button:disabled { cursor: not-allowed; opacity: 0.65; transform: none; }

        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .example {
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 7px 12px;
            background: #fff;
            color: var(--muted);
            cursor: pointer;
            font-size: 0.84rem;
        }

        .example:hover {
            border-color: var(--accent);
            color: var(--accent-2);
        }

        .status {
            margin-top: 12px;
            font-size: 0.9rem;
            color: var(--muted);
        }

        .result {
            margin-top: 18px;
            display: none;
        }

        .result.visible { display: block; }

        .card {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
            background: #fff;
        }

        .card h3 {
            margin: 0 0 10px;
            font-size: 0.98rem;
            color: var(--accent-2);
        }

        .answer {
            line-height: 1.55;
        }

        .answer strong { color: #0a4f63; }

        .runtime {
            font-size: 0.85rem;
            color: var(--muted);
        }

        .pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .pill.ok { background: #e5f6ee; color: var(--ok); }
        .pill.warn { background: #fff2e2; color: var(--warn); }

        pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: var(--mono);
            font-size: 0.8rem;
            line-height: 1.5;
            background: #f6fbfa;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
        }

        .table-wrap {
            overflow: auto;
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
        }

        th, td {
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid #edf2ef;
            white-space: nowrap;
        }

        th {
            background: #f7fcfa;
            color: #234451;
            position: sticky;
            top: 0;
        }

        .checks {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .check {
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.78rem;
            border: 1px solid var(--border);
        }

        .check.pass { background: #eaf8f1; color: #1f7b4e; border-color: #b6e7cb; }
        .check.fail { background: #fff3ec; color: #ab4f10; border-color: #f5d3ba; }

        .followups {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .followup {
            background: #eef6ff;
            color: #1e4f79;
            border-radius: 999px;
            font-size: 0.8rem;
            padding: 6px 10px;
            border: 1px solid #d3e8ff;
        }

        @media (max-width: 760px) {
            .query-row { grid-template-columns: 1fr; }
            .shell { padding: 18px 10px 30px; }
            .hero, .panel { padding: 14px; border-radius: 14px; }
        }
    </style>
</head>
<body>
    <main class=\"shell\">
        <section class=\"hero\">
            <h1 class=\"title\">dataDa</h1>
            <p class=\"subtitle\">Intelligent, human-like analytics assistant on top of your data stream. Ask a business question and get insight, evidence, SQL, and result previews.</p>
            <div class=\"meta\">
                <div class=\"chip\">
                    <label>Database</label>
                    <div class=\"value\" id=\"dbPath\">checking...</div>
                </div>
                <div class=\"chip\">
                    <label>Health</label>
                    <div class=\"value\" id=\"healthState\">checking...</div>
                </div>
                <div class=\"chip\">
                    <label>Recommended Runtime</label>
                    <div class=\"value\" id=\"recommendedMode\">loading...</div>
                </div>
            </div>
        </section>

        <section class=\"panel\">
            <div class=\"controls\">
                <div class=\"query-row\">
                    <textarea id=\"queryInput\" placeholder=\"Ask anything about your data. Example: What is total transaction volume for December 2025 split by platform?\"></textarea>
                    <button class=\"primary\" id=\"askBtn\">Run Query</button>
                </div>
                <div class=\"row\">
                    <label>
                        <span style=\"display:block;font-size:12px;color:var(--muted);margin-bottom:4px;\">LLM Mode</span>
                        <select id=\"llmMode\">
                            <option value=\"auto\">Auto (recommended)</option>
                            <option value=\"local\">Local Ollama</option>
                            <option value=\"openai\">OpenAI API</option>
                            <option value=\"deterministic\">Deterministic (no LLM)</option>
                        </select>
                    </label>
                    <button class=\"ghost\" id=\"archBtn\" type=\"button\">View Architecture</button>
                </div>
            </div>
            <div class=\"examples\" id=\"examples\"></div>
            <div class=\"status\" id=\"statusText\">Ready.</div>
        </section>

        <section class=\"result\" id=\"result\"></section>
    </main>

    <script>
        const q = document.getElementById('queryInput');
        const askBtn = document.getElementById('askBtn');
        const llmMode = document.getElementById('llmMode');
        const result = document.getElementById('result');
        const statusText = document.getElementById('statusText');

        const EXAMPLES = [
            'How many customers do we have?',
            'What is total revenue from orders?',
            'Show top 5 customers by amount',
            'List transaction count for December 2025',
            'Compare this month vs last month volume',
        ];

        function escapeHtml(v) {
            return String(v ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;');
        }

        function markdownToHtml(text) {
            if (!text) return '';
            return escapeHtml(text)
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br />');
        }

        function setStatus(msg) {
            statusText.textContent = msg;
        }

        function renderExamples() {
            const box = document.getElementById('examples');
            box.innerHTML = EXAMPLES.map(item => `<button class=\"example\" type=\"button\">${escapeHtml(item)}</button>`).join('');
            box.querySelectorAll('button').forEach((el, idx) => {
                el.addEventListener('click', () => {
                    q.value = EXAMPLES[idx];
                    q.focus();
                });
            });
        }

        function renderRows(columns, rows) {
            if (!columns || columns.length === 0 || !rows || rows.length === 0) {
                return '<div class=\"runtime\">No rows returned.</div>';
            }

            const head = `<tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>`;
            const body = rows.map(r => `<tr>${columns.map(c => `<td>${escapeHtml(r[c])}</td>`).join('')}</tr>`).join('');
            return `<div class=\"table-wrap\"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
        }

        function renderResponse(data) {
            const runtime = data.runtime || {};
            const good = data.success ? 'ok' : 'warn';

            const checks = (data.sanity_checks || []).map(c => {
                const cls = c.passed ? 'pass' : 'fail';
                const prefix = c.passed ? 'PASS' : 'FAIL';
                return `<span class=\"check ${cls}\">${prefix}: ${escapeHtml(c.check_name)}</span>`;
            }).join('');

            const followups = (data.suggested_questions || [])
                .slice(0, 6)
                .map(x => `<span class=\"followup\">${escapeHtml(x)}</span>`)
                .join('');

            result.innerHTML = `
                <div class=\"card\">
                    <h3>Answer</h3>
                    <div class=\"answer\">${markdownToHtml(data.answer_markdown)}</div>
                </div>

                <div class=\"card\">
                    <h3>Execution Summary</h3>
                    <div class=\"runtime\">Trace: ${escapeHtml(data.trace_id)} | Confidence: ${escapeHtml(data.confidence)} (${Math.round((data.confidence_score || 0) * 100)}%)</div>
                    <div class=\"runtime\" style=\"margin-top:6px;\"><span class=\"pill ${good}\">${data.success ? 'success' : 'needs attention'}</span> Mode: <strong>${escapeHtml(runtime.mode || 'n/a')}</strong> | Provider: <strong>${escapeHtml(runtime.provider || 'none')}</strong> | ${escapeHtml(runtime.reason || '')}</div>
                    <div class=\"runtime\" style=\"margin-top:8px;\">Rows: ${data.row_count ?? '-'} | SQL time: ${data.execution_time_ms ? `${data.execution_time_ms.toFixed(1)} ms` : '-'}</div>
                </div>

                <div class=\"card\">
                    <h3>Result Preview</h3>
                    ${renderRows(data.columns || [], data.sample_rows || [])}
                </div>

                <div class=\"card\">
                    <h3>SQL</h3>
                    <pre>${escapeHtml(data.sql || 'No SQL generated')}</pre>
                </div>

                <div class=\"card\">
                    <h3>Sanity Checks</h3>
                    <div class=\"checks\">${checks || '<span class=\"runtime\">No checks</span>'}</div>
                </div>

                <div class=\"card\">
                    <h3>Suggested Follow-ups</h3>
                    <div class=\"followups\">${followups || '<span class=\"runtime\">No suggestions</span>'}</div>
                </div>
            `;

            result.classList.add('visible');
        }

        async function initState() {
            try {
                const [healthRes, providerRes] = await Promise.all([
                    fetch('/api/assistant/health'),
                    fetch('/api/assistant/providers'),
                ]);
                const health = await healthRes.json();
                const providers = await providerRes.json();

                document.getElementById('dbPath').textContent = health.db_path;
                document.getElementById('healthState').textContent = health.status === 'ok' ? 'ready' : 'database missing';
                document.getElementById('recommendedMode').textContent = providers.recommended_mode;
                llmMode.value = providers.default_mode || 'auto';
            } catch (err) {
                document.getElementById('healthState').textContent = 'unreachable';
            }
        }

        async function ask() {
            const goal = q.value.trim();
            if (!goal) {
                setStatus('Enter a question first.');
                return;
            }

            askBtn.disabled = true;
            setStatus('Running orchestrator...');
            result.classList.add('visible');
            result.innerHTML = '<div class=\"card\"><div class=\"runtime\">Working on your query...</div></div>';

            try {
                const response = await fetch('/api/assistant/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ goal, llm_mode: llmMode.value })
                });
                const data = await response.json();
                renderResponse(data);
                setStatus('Done.');
            } catch (err) {
                setStatus(`Request failed: ${err.message}`);
                result.innerHTML = `<div class=\"card\"><div class=\"runtime\">${escapeHtml(err.message)}</div></div>`;
            } finally {
                askBtn.disabled = false;
            }
        }

        async function viewArchitecture() {
            try {
                const response = await fetch('/api/assistant/architecture');
                const data = await response.json();
                result.classList.add('visible');
                result.innerHTML = `
                    <div class=\"card\">
                        <h3>${escapeHtml(data.system_name)} (${escapeHtml(data.version)})</h3>
                        <div class=\"runtime\">${escapeHtml(data.description)}</div>
                    </div>
                    <div class=\"card\">
                        <h3>Pipeline</h3>
                        <pre>${escapeHtml((data.pipeline_flow || []).join('\\n'))}</pre>
                    </div>
                    <div class=\"card\">
                        <h3>Guardrails</h3>
                        <pre>${escapeHtml((data.guardrails || []).join('\\n'))}</pre>
                    </div>
                `;
                setStatus('Architecture loaded.');
            } catch (err) {
                setStatus(`Architecture fetch failed: ${err.message}`);
            }
        }

        askBtn.addEventListener('click', ask);
        document.getElementById('archBtn').addEventListener('click', viewArchitecture);
        q.addEventListener('keydown', e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
        });

        renderExamples();
        initState();
    </script>
</body>
</html>
"""


app = create_app()


def main() -> None:
    """Run the dataDa API server with embedded web UI."""
    import uvicorn

    uvicorn.run("haikugraph.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
