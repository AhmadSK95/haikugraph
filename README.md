# dataDa

Status date: February 19, 2026

`dataDa` is an open-source, enterprise-focused analytics assistant that turns natural language questions into transparent, evidence-backed answers over your private data.

It is built as an agentic analytics runtime (not a plain SQL bot): agents decompose tasks, generate/query alternatives, audit results, self-correct when evidence is stronger, and keep memory of what worked.

## Why dataDa

Most AI chat tools are good at language but weak at accountable analytics on private enterprise data. dataDa is designed for:

- grounded answers (SQL + sample rows + checks)
- transparent execution (agent trace + confidence breakdown)
- bounded autonomy (self-correction without unsafe side effects)
- deployment flexibility (deterministic, local LLM, OpenAI, or auto)

## Product Scope

### Target users

- data and analytics teams
- operations and business teams that need conversational BI
- enterprises with privacy/governance constraints

### What this is

- a verifiable autonomous analytics engine
- a multi-agent data analyst teammate
- an open and inspectable runtime

### What this is not

- an unrestricted autonomous system that can silently mutate production systems
- a generic chatbot without lineage

## Current Architecture

```mermaid
flowchart LR
A["User Query"] --> B["Context Agent"]
B --> C["Memory Agent (recall)"]
C --> D["Intake Agent"]
D --> E["Semantic Retrieval Agent"]
E --> F["Planning Agent"]
F --> G["Specialist Agents"]
G --> H["Query Engineer Agent"]
H --> I["Execution Agent"]
I --> J["Audit Agent"]
J --> K["Autonomy Agent (candidate reconciliation + self-correction)"]
K --> L["Narrative + Visualization Agents"]
L --> M["Answer + SQL + Evidence + Confidence + Trace"]
K --> N["Memory Agent (write + learning)"]
N --> C
```

## Bounded Autonomy (Current Definition)

Autonomy in `dataDa` is intentionally split into two layers:

- cognitive autonomy: agents can decompose tasks, generate alternatives, self-check, self-correct, and learn
- operational bounds: policies limit unsafe side effects, not intelligence

Current bounded controls exposed in API:

- `autonomy_mode`
- `auto_correction`
- `strict_truth`
- `max_refinement_rounds`
- `max_candidate_plans`

## Implemented Capabilities (as of now)

- unified ingestion path for Excel -> DuckDB (`haikugraph ingest`)
- direct existing DB attach (`haikugraph use-db --db-path ...`)
- semantic marts for transactions, quotes, customers, bookings
- runtime mode selection: `deterministic`, `local`, `openai`, `auto`
- local model listing/selection/pull via Ollama APIs
- session continuity in UI/API
- confidence scoring + audit checks + replay consistency checks
- concept alignment warnings in technical details
- persistent autonomous memory store (sidecar DB)
- feedback endpoint that can register correction rules
- autonomous candidate-plan reconciliation and auto-switch to better-grounded plan
- deterministic failure narration with explicit subquestion-level error reporting
- robust comparison execution when one side returns NULL aggregates
- multi-agent blackboard with explicit producer/consumer artifact flow
- confidence decomposition per evaluated hypothesis with contradiction resolution metadata
- correction governance APIs + UI controls for one-click enable/disable rollback
- correction rollback support with policy-gated mutation endpoints
- toolsmith lifecycle APIs (candidate -> stage -> promote -> rollback)
- durable session store (tenant-aware session isolation persisted in runtime DuckDB)
- async query jobs + status polling endpoints for concurrency and long-running requests
- per-tenant query budgets with runtime transparency in response metadata
- trust dashboard API + UI panel for success/confidence/latency/drift visibility
- source-truth parity endpoint for canonical SQL comparison over active connection
- document ingestion command for text-heavy sources (`haikugraph ingest-docs`)
- connector capability registry for DuckDB/Postgres/Snowflake/BigQuery/Stream/Documents
- startup self-healing for stale default connections (including orphaned pytest/temp paths)
- memory sidecar initialization now auto-creates parent directories to prevent boot failures
- full automated test suite passing (`236 passed`, `15 skipped`)

## Quick Start

### 1. Environment

```bash
cd /Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Ingest data (Excel files)

```bash
haikugraph ingest --data-dir ./data --db-path ./data/haikugraph.db --force
```

### 3. Or point to an existing database

```bash
haikugraph use-db --db-path /path/to/existing.duckdb
```

### 4. Run web app

```bash
./run.sh
# UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## API Examples

### Query with autonomy controls

```bash
curl -s -X POST http://localhost:8000/api/assistant/query \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "What is the forex markup revenue for December 2025?",
    "llm_mode": "auto",
    "session_id": "demo-session-1",
    "autonomy_mode": "bounded",
    "auto_correction": true,
    "strict_truth": true,
    "max_refinement_rounds": 2,
    "max_candidate_plans": 6,
    "storyteller_mode": true
  }'
```

### Provide feedback and optionally teach a correction rule

```bash
curl -s -X POST http://localhost:8000/api/assistant/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "trace_id": "optional-trace-id",
    "session_id": "demo-session-1",
    "goal": "forex in december 2025",
    "issue": "Mapped to transactions when I expected quotes",
    "keyword": "forex",
    "target_table": "datada_mart_quotes",
    "target_metric": "forex_markup_revenue",
    "target_dimensions": ["__month__"]
  }'
```

## Data Stores Used by Runtime

- primary analytics DB: `HG_DB_PATH` or default `./data/haikugraph.db`
- autonomous memory DB: default `<primary_db_stem>_agent_memory.duckdb`
  - override with `HG_MEMORY_DB_PATH`
- connection registry DB map: `HG_CONNECTION_REGISTRY_PATH` or default `./data/connections.json`

## Connection Routing (New)

`db_connection_id` is now fully active in runtime.

- queries are routed to a registered connection
- per-connection team runtime is cached and reused
- sessions are scoped by `connection_id:session_id` to prevent cross-source context bleed
- UI now includes a connection selector + refresh action
- `create_app(db_path=...)` uses a sidecar registry by default, so test/one-off app instances do not overwrite the canonical `data/connections.json` (unless `HG_CONNECTION_REGISTRY_PATH` is explicitly set)

Connection APIs:

- `GET /api/assistant/connections`
- `POST /api/assistant/connections/upsert`
- `POST /api/assistant/connections/default`
- `POST /api/assistant/connections/test`

## Current Limitations

- connectors beyond Excel/DuckDB are not fully productized yet (DB/stream/document connector roadmap remains)
- not yet distributed multi-tenant execution fabric (currently single-node runtime)
- autonomous tool creation is still bounded to safe probe behavior, not full dynamic plugin lifecycle
- benchmark suites are strong but still narrower than open-world enterprise query distributions

## Progress Tracker

Overall program completion toward target vision: **79%**

### Epic-level tracker

| Epic | Status | Completion | Notes |
|---|---|---:|---|
| 1. Unified ingestion + direct DB attach | active | 92% | core done; multi-connection routing added |
| 2. Semantic intelligence reliability | active | 75% | marts + mappings + generic schema typing hardening + source-truth endpoint; ontology/versioning pending |
| 3. Agent autonomy core | active | 84% | memory + correction loop + blackboard + contradiction handling + toolsmith lifecycle APIs |
| 4. Truth and verification engine | active | 86% | audit/replay/concept checks + hypothesis decomposition + source-truth parity checks + full green suite pass |
| 5. Conversational UX and transparency | active | 82% | trace/details/story mode + blackboard flow graph + correction controls + trust panel |
| 6. Enterprise platform readiness | active | 66% | tenant-aware sessions + API key role gates + async jobs + budgets + trust telemetry |
| 7. Scale to billion-row enterprise workloads | active | 40% | async queue + connector capability model + DuckDB mirror architecture; warehouse pushdown pending |

### Detailed task list

#### A. Agent autonomy and learning

- [x] Persistent memory store for successful runs and outcomes
- [x] Correction rule registry and recall
- [x] Autonomous candidate-plan evaluation and switching
- [x] Feedback API to register correction rules
- [x] Multi-agent blackboard for explicit inter-agent negotiation
- [x] Autonomous toolsmith lifecycle (generate -> test -> stage -> promote)
- [x] Policy-gated self-updating procedural memory with rollback

#### B. Data platform and connectors

- [x] Unified Excel ingestion path
- [x] Existing DuckDB attach workflow
- [x] Connection registry (`connections.json`) and runtime routing via `db_connection_id`
- [x] Connection health/test/upsert/default APIs
- [x] UI connection selector and refresh control
- [x] Postgres connector (registry + DSN validation + mirror-ingest readiness)
- [x] Snowflake connector (registry + package validation + mirror-ingest readiness)
- [x] BigQuery connector (registry + package validation + mirror-ingest readiness)
- [x] Stream connector (Kafka/Kinesis URI registration + bounded snapshot readiness)
- [ ] Document connector (PDF/DOCX/text) with citation-grade retrieval
- [x] Document ingestion into semantic evidence table (`datada_documents`)

#### C. Truth, quality, and explainability

- [x] Execution success checks
- [x] Non-empty checks
- [x] Time-scope checks
- [x] Concept-alignment checks
- [x] Replay consistency checks
- [x] Confidence scoring tied to audit quality
- [x] Full regression suite rerun after autonomy + connection-routing changes
- [x] Multi-plan contradiction resolution with confidence decomposition per hypothesis
- [x] Cross-source truth checks (source-of-truth SQL/warehouse parity)

#### D. Product UX

- [x] Session continuity support
- [x] Technical details panel with SQL and trace
- [x] Storyteller mode support
- [x] Runtime choice (auto/local/openai/deterministic)
- [x] Rich visual diagnostics graph (agent-to-agent artifacts)
- [x] Guided correction UX (one-click apply/rollback suggestion)
- [x] Enterprise-grade dashboards for trust metrics and drift

#### E. Enterprise readiness

- [x] Logical multi-connection routing with deterministic default selection
- [ ] RBAC + SSO + tenant isolation
- [ ] Durable distributed session/memory backends
- [x] Async job orchestration and queueing
- [x] Cost controls and query budgets per tenant
- [ ] SLA/SLO observability and incident hooks

## What remains to reach full enterprise target

- SSO/OIDC integration and formal tenant RBAC policy store
- Distributed shared session/memory backend (Redis/Postgres) for multi-node horizontal scale
- Native pushdown connectors (Snowflake/BigQuery/Postgres) without DuckDB mirror step
- Citation-grade RAG retrieval over documents with source spans in final answers
- Incident hooks (PagerDuty/Slack/Webhook) and SLO burn-rate alerts

## Repo Documentation Policy

This repository intentionally uses **one canonical Markdown document**: this `README.md`.

All product, architecture, roadmap, and tracker updates should be maintained here to keep context centralized.
