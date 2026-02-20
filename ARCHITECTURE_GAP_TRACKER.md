# dataDa Agentic Analytics -- Architecture Gap Tracker

> **Created**: 2026-02-20
> **Status**: Gap analysis complete -- implementation not started
> **Primary file**: `src/haikugraph/poc/agentic_team.py` (~5460 lines)
> **Secondary files**: `src/haikugraph/api/server.py`, `src/haikugraph/llm/router.py`, `src/haikugraph/agents/contracts.py`

---

## Executive Summary

The system is marketed as a **multi-agent analytics team** but is architecturally a **deterministic linear pipeline with agent-shaped steps**. The gap between the current state and the vision of an intelligent, reasoning, collaborating agent team is structural — not fixable with keyword patches.

### Current Reality

```
User Question
  ↓
Keyword Matching (if/elif chains)
  ↓
Single Table Lookup (mutually exclusive domain)
  ↓
Template SQL (SELECT {metric} FROM {table} WHERE {filters})
  ↓
Fixed-Weight Scoring (hardcoded penalties)
  ↓
LLM Formats the Answer (only place LLM adds value)
```

### The Vision

```
User Question
  ↓
LLM Reasons About Intent (understands complex analytical questions)
  ↓
Multi-Domain Schema Analysis (identifies all relevant tables)
  ↓
Intelligent Query Planning (JOINs, CTEs, multi-step analysis)
  ↓
Agent Collaboration (specialists influence the plan, audit triggers re-planning)
  ↓
LLM Evaluates & Explains (reasoning about quality, not just formatting)
```

### Where LLM Is Used Today (4 call sites)

| Location | Line | Purpose | Impact |
|----------|------|---------|--------|
| `_intake_with_llm` | 3019 | Parse intent from goal | Optional, silently falls back to deterministic |
| `_chief_analyst` | 2239 | Generate mission brief | Output is metadata, doesn't drive execution |
| Unknown agent | 3933 | Likely semantic retrieval | Minor enrichment |
| `_narrative_agent` | 5179 | Format final answer | Cosmetic — the data is already decided |

**Diagnosis**: LLM is used for **parsing** (intake) and **formatting** (narrative). It is never used for **reasoning**, **planning**, **evaluation**, or **decision-making**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Gap Summary](#gap-summary)
3. [Dependency Graph](#dependency-graph)
4. [Phase Roadmap](#phase-roadmap)
5. [Gap 12: Silent LLM Fallback](#gap-12----silent-llm-fallback)
6. [Gap 13: Single-Domain Mutual Exclusion](#gap-13----single-domain-mutual-exclusion)
7. [Gap 14: Single-Table Query Engine](#gap-14----single-table-query-engine)
8. [Gap 15: Chief Analyst Is a Stub](#gap-15----chief-analyst-is-a-stub)
9. [Gap 16: Specialist Agents Are Discarded](#gap-16----specialist-agents-are-discarded)
10. [Gap 17: Blackboard Is Tracing-Only](#gap-17----blackboard-is-tracing-only)
11. [Gap 18: No LLM in Planning](#gap-18----no-llm-in-planning)
12. [Gap 19: No LLM in Audit/Scoring](#gap-19----no-llm-in-auditing)
13. [Gap 20: No Complex Analytical Patterns](#gap-20----no-complex-analytical-patterns)
14. [Gap 21: Deterministic Candidate Scoring](#gap-21----deterministic-candidate-scoring)
15. [Effort & Risk Matrix](#effort--risk-matrix)

---

## Architecture Overview

### Current Pipeline (Linear, Deterministic)

```
run() method (lines 882-1858) — FIXED sequential execution:

  ContextAgent          → resolves follow-up references           (line 893)
  DataEngineeringTeam   → builds semantic catalog                 (line 909)
  ChiefAnalystAgent     → returns {"mission": goal} (STUB)        (line 921)
  MemoryAgent           → recalls similar past queries             (line 937)
  IntakeAgent           → keyword→slot mapping + optional LLM      (line 967)
  ClarificationAgent    → keyword-counting ambiguity check         (line 985)
  GovernanceAgent       → policy blocklist check                   (line 1095)
  ─── Intent Branches ───
  │ data_overview        → catalog summary (early return)          (line 1136)
  │ schema_exploration   → column listing (early return)           (line 1271)
  │ document_qa          → document retrieval (early return)       (line 1341)
  ─── Metric Pipeline ───
  SemanticRetrievalAgent → maps domain → ONE table                 (line 1530)
  PlanningAgent          → template-fills query plan               (line 1536)
  SpecialistAgents (×4)  → generate warnings (DISCARDED)           (line 1556)
  QueryEngineerAgent     → f"SELECT {m} FROM {t} WHERE {w}"       (line 1610)
  ExecutionAgent         → runs SQL                                (line 1621)
  AuditAgent             → concept alignment scoring               (line 1632)
  ─── Feedback Loop ───
  Post-audit replan      → if score < 0.5, enrich & retry once     (line 1651)
  AutonomyAgent          → multi-round candidate refinement        (line 1693)
  ─── Output ───
  NarrativeAgent         → LLM formats the answer                  (line 1748)
  VisualizationAgent     → chart spec                              (line 1770)
  ChiefAnalystAgent      → finalize response                       (line 1786)
```

### What "Agent" Means Today

Every "agent" is a **method on the same class** called by `_run_agent()` (line 2079), which:
1. Measures timing
2. Calls the method
3. Logs to trace
4. Returns the result

No agent can: call another agent, modify execution flow, request clarification mid-pipeline, reason about its output, or access the blackboard.

---

## Gap Summary

| Gap | Title | Severity | Category |
|-----|-------|----------|----------|
| **12** | Silent LLM Fallback | CRITICAL | Transparency |
| **13** | Single-Domain Mutual Exclusion | CRITICAL | Architecture |
| **14** | Single-Table Query Engine | CRITICAL | Architecture |
| **15** | Chief Analyst Is a Stub | HIGH | Intelligence |
| **16** | Specialist Agents Are Discarded | HIGH | Intelligence |
| **17** | Blackboard Is Tracing-Only | HIGH | Communication |
| **18** | No LLM in Planning | HIGH | Intelligence |
| **19** | No LLM in Audit/Scoring | MEDIUM | Intelligence |
| **20** | No Complex Analytical Patterns | HIGH | Capability |
| **21** | Deterministic Candidate Scoring | MEDIUM | Intelligence |

---

## Dependency Graph

```
                    ┌──────────────┐
                    │   GAP 12     │  Silent LLM fallback (foundation)
                    │ transparency │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼──────┐   ┌────▼─────┐   ┌──────▼──────┐
    │  GAP 18    │   │ GAP 19   │   │   GAP 15    │
    │ LLM in     │   │ LLM in   │   │ Chief as    │
    │ planning   │   │ audit    │   │ orchestrator│
    └─────┬──────┘   └────┬─────┘   └──────┬──────┘
          │                │                │
          │           ┌────▼─────┐          │
          │           │ GAP 21   │          │
          │           │ learned  │          │
          │           │ scoring  │          │
          │           └──────────┘          │
          │                                 │
    ┌─────▼───────────────────────────┐     │
    │          GAP 13                 │◄────┘
    │  Multi-domain intent detection  │
    └─────┬───────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │          GAP 14                 │
    │  Multi-table query engine       │
    │  (JOINs, CTEs, subqueries)      │
    └─────┬───────────────────────────┘
          │
    ┌─────▼──────┐   ┌──────────┐   ┌──────────┐
    │  GAP 20    │   │ GAP 16   │   │ GAP 17   │
    │ complex    │   │ real     │   │ functional│
    │ analytics  │   │ speciali-│   │ blackboard│
    │ patterns   │   │ sts      │   │          │
    └────────────┘   └──────────┘   └──────────┘
```

**Dependency rules**:
- GAP 12 (transparency) is **foundation** — must be done first so all subsequent LLM integration has proper failure reporting
- GAP 18 (LLM planning) and GAP 15 (Chief orchestrator) **must precede** GAP 13 (multi-domain) — you need intelligent planning before multi-domain routing
- GAP 13 (multi-domain) **must precede** GAP 14 (multi-table) — you need to detect multi-domain intent before you can generate cross-table queries
- GAP 14 (multi-table) **must precede** GAP 20 (complex analytics) — complex patterns require JOIN capability
- GAP 16 (specialists), GAP 17 (blackboard), GAP 21 (scoring) are **parallel** — can be done in any order after their dependencies

---

## Phase Roadmap

### Phase A — Transparency & Trust
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 12 | Silent LLM Fallback | 1 day | NOT STARTED |

### Phase B — LLM-Powered Reasoning
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 18 | LLM in Planning | 3-4 days | NOT STARTED |
| 19 | LLM in Audit/Scoring | 2-3 days | NOT STARTED |
| 15 | Chief Analyst Orchestrator | 3-4 days | NOT STARTED |

### Phase C — Cross-Domain Intelligence
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 13 | Multi-Domain Intent Detection | 3-4 days | NOT STARTED |
| 14 | Multi-Table Query Engine | 5-7 days | NOT STARTED |

### Phase D — Agent Collaboration
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 16 | Real Specialist Agents | 3-4 days | NOT STARTED |
| 17 | Functional Blackboard | 3-4 days | NOT STARTED |
| 21 | LLM-Evaluated Scoring | 2-3 days | NOT STARTED |

### Phase E — Complex Analytics
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 20 | Complex Analytical Patterns | 5-7 days | NOT STARTED |

**Total estimated effort**: 30-40 days

---

## Gap Details

---

### GAP 12 — Silent LLM Fallback

**Severity**: CRITICAL
**Phase**: A (Transparency & Trust)
**Depends on**: None (foundation)

#### Problem

When a user selects Local LLM mode but Ollama is unavailable, the system silently falls back to deterministic mode. The user receives a keyword-matched result with no indication that their requested LLM was never used. The same applies when OpenAI API keys are invalid or the service is down.

The fallback information exists in `response.runtime.reason` but is never surfaced as a user-facing warning. The `_intake_with_llm` method catches **all** exceptions with a bare `except Exception: return None`, swallowing connection errors, timeouts, and model failures identically.

#### Failure Path

```
User requests: llm_mode="local"
  ↓
_resolve_runtime() (server.py:1329-1348)
  checks providers.checks["ollama"].available → False
  returns RuntimeSelection(mode="deterministic", use_llm=False, reason="ollama unavailable: ...")
  ↓
_intake_agent() (agentic_team.py:2845)
  `if runtime.use_llm and runtime.provider:` → False
  SKIPS LLM entirely, uses only deterministic parsing
  ↓
_intake_with_llm() (agentic_team.py:3026-3027) — even when LLM is attempted:
  `except Exception:` ← catches ALL errors
  `return None`       ← silent fallback, no logging, no warning
  ↓
User gets deterministic keyword-match result
  runtime.reason buried in JSON metadata — never shown
```

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `server.py` | 1329-1348 | `_resolve_runtime()` LOCAL mode — falls back to DETERMINISTIC silently |
| `server.py` | 1350-1369 | `_resolve_runtime()` OPENAI mode — same silent fallback |
| `server.py` | 1371-1401 | `_resolve_runtime()` AUTO mode — cascades LOCAL→OPENAI→DETERMINISTIC |
| `agentic_team.py` | 2845 | `_intake_agent()` — skips LLM when `use_llm=False` |
| `agentic_team.py` | 3026-3027 | `_intake_with_llm()` — bare `except Exception: return None` |
| `agentic_team.py` | 5604-5632 | `_runtime_payload()` — includes `reason` field but it's metadata only |

#### Implementation Tasks

1. **Surface LLM fallback as a pipeline warning** in `run()`:
   ```python
   if runtime.requested_mode != runtime.mode:
       self._pipeline_warnings.append(
           f"Requested LLM mode '{runtime.requested_mode}' is unavailable "
           f"({runtime.reason}). Fell back to '{runtime.mode}' mode."
       )
   ```

2. **Surface intake LLM failure as a warning** — replace bare exception in `_intake_with_llm`:
   ```python
   except Exception as exc:
       self._pipeline_warnings.append(
           f"LLM intake parsing failed ({type(exc).__name__}: {exc}). "
           f"Using deterministic parsing only."
       )
       return None
   ```

3. **Add `llm_effective` check after intake** — warn when LLM was requested but didn't contribute:
   ```python
   if runtime.use_llm and not parsed.get("_llm_intake_used"):
       self._pipeline_warnings.append(
           "LLM was available but did not contribute to query parsing. "
           "Result is based on deterministic keyword matching only."
       )
   ```

4. **Add narrative LLM failure warning** — same pattern for `_narrative_agent` failures.

5. **Frontend**: Render `response.warnings` prominently when they contain LLM fallback messages. Consider a distinct visual indicator (e.g., "Running in deterministic mode" badge).

#### Acceptance Criteria

- [ ] When local LLM is requested but unavailable, `response.warnings` contains a clear message
- [ ] When OpenAI is requested but unavailable, `response.warnings` contains a clear message
- [ ] When LLM intake parsing fails mid-call, `response.warnings` explains what happened
- [ ] `_intake_with_llm` logs the actual exception type and message (not swallowed)
- [ ] Frontend displays LLM mode mismatch prominently
- [ ] `response.runtime.requested_mode` vs `response.runtime.mode` mismatch is visually flagged

---

### GAP 13 — Single-Domain Mutual Exclusion

**Severity**: CRITICAL
**Phase**: C (Cross-Domain Intelligence)
**Depends on**: GAP 15 (Chief Analyst orchestrator), GAP 18 (LLM planning)

#### Problem

The intake layer selects exactly ONE domain per query via a mutually exclusive `elif` chain. The query *"How do individual and corporate customers differ in purchasing behavior?"* needs both `customers` (for type: INDIVIDUAL/CORPORATE) and `transactions` (for purchasing metrics). The system assigns `domain = "customers"` and never considers transactions.

This is the **single most limiting architectural constraint**. Every downstream component — retrieval, planning, query engine — operates on one domain/table because intake gives them one.

#### Data Flow (Current)

```
"How do individual and corporate customers differ in purchasing behavior?"
  ↓
_intake_deterministic (line 3312):
  domain = "transactions"          ← default
  elif "customer" in lower:        ← matched!
    domain = "customers"           ← FINAL — transactions domain lost
  ↓
_semantic_retrieval_agent (line 3960):
  table = DOMAIN_TO_MART["customers"] → "datada_dim_customers"
  metrics = {customer_count, payee_count, university_count}  ← NO transaction metrics
  ↓
_planning_agent (line 3976):
  plan = {"table": "datada_dim_customers", "metric": "customer_count"}
  ↓
_query_engine_agent (line 4782):
  SQL: SELECT COUNT(DISTINCT customer_key) FROM datada_dim_customers
  ↓
Result: "You have 2,457 customers" ← completely wrong answer
```

#### Data Flow (Required)

```
"How do individual and corporate customers differ in purchasing behavior?"
  ↓
Intent Analysis (LLM-powered):
  domains = ["customers", "transactions"]
  analysis_type = "comparative_behavior"
  group_by = customers.type (INDIVIDUAL vs CORPORATE)
  metrics_needed = [transaction_count, total_amount, avg_amount]
  ↓
Multi-Table Retrieval:
  tables = {
    "datada_dim_customers": {columns, metrics},
    "datada_mart_transactions": {columns, metrics},
  }
  join_path = customers.customer_id → transactions.customer_id
  ↓
Cross-Domain Plan:
  SQL: SELECT c.type,
              COUNT(DISTINCT t.transaction_key) AS tx_count,
              SUM(t.amount) AS total_spent,
              AVG(t.amount) AS avg_amount
       FROM datada_dim_customers c
       JOIN datada_mart_transactions t ON c.customer_id = t.customer_id
       GROUP BY c.type
  ↓
Result: "Individual customers: 1,842 transactions, $12.4M avg $6,732
         Corporate customers: 4,291 transactions, $29.0M avg $6,761"
```

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 3312-3346 | `_intake_deterministic` — mutually exclusive domain `elif` chain |
| `agentic_team.py` | 3319 | Return dict has single `"domain"` key |
| `agentic_team.py` | 3960-3974 | `_semantic_retrieval_agent` — maps single domain → single table |
| `agentic_team.py` | 3976-4057 | `_planning_agent` — uses single `retrieval["table"]` |
| `agentic_team.py` | 52-58 | `DOMAIN_TO_MART` — 1:1 mapping, no multi-domain support |

#### Implementation Tasks

1. **Add `domains` (plural) to intake output** alongside `domain`:
   ```python
   # In _intake_deterministic, after domain selection:
   secondary_domains = []
   if domain == "customers" and any(
       k in lower for k in ["transaction", "purchasing", "buying", "spending", "behavior", "payment"]
   ):
       secondary_domains.append("transactions")
   elif domain == "transactions" and any(
       k in lower for k in ["customer type", "individual", "corporate", "customer segment"]
   ):
       secondary_domains.append("customers")
   # ... similar for quotes/bookings cross-references

   return {
       "domain": domain,                  # primary (backward compat)
       "domains": [domain] + secondary_domains,  # all relevant domains
       ...
   }
   ```

2. **LLM-powered multi-domain detection** (requires GAP 18): When `runtime.use_llm`, ask the LLM to identify all relevant domains and the relationships between them.

3. **Modify `_semantic_retrieval_agent`** to return multiple tables:
   ```python
   def _semantic_retrieval_agent(self, intake, catalog):
       domains = intake.get("domains", [intake["domain"]])
       tables = {}
       for domain in domains:
           table = DOMAIN_TO_MART.get(domain)
           if table and table in catalog["marts"]:
               tables[domain] = {
                   "table": table,
                   "columns": catalog["marts"][table]["columns"],
                   "metrics": catalog["metrics_by_table"].get(table, {}),
                   ...
               }
       # Identify join paths from catalog
       join_paths = self._find_join_paths(tables, catalog)
       return {"tables": tables, "join_paths": join_paths, "primary_table": ...}
   ```

4. **Modify `_planning_agent`** to produce multi-table plans (requires GAP 14 for execution).

#### Acceptance Criteria

- [ ] Intake returns `domains` (list) alongside `domain` (string, backward compat)
- [ ] "customer purchasing behavior" detects both customers and transactions domains
- [ ] `_semantic_retrieval_agent` can return schemas for multiple tables
- [ ] Join paths between related tables are identified from catalog
- [ ] Existing single-domain queries are unaffected (regression check)

---

### GAP 14 — Single-Table Query Engine

**Severity**: CRITICAL
**Phase**: C (Cross-Domain Intelligence)
**Depends on**: GAP 13 (multi-domain detection)

#### Problem

`_query_engine_agent` (line 4782) generates SQL from a single `plan["table"]`. Every query is `SELECT ... FROM {one_table} WHERE ...`. There is no capability to generate JOINs, CTEs, subqueries, or multi-step analytical queries at runtime.

The only JOIN in the entire codebase is pre-materialized inside `_build_transactions_view` (line 338), which LEFT JOINs customers at view creation time. This adds `address_country` and `address_state` to the transactions view but does NOT make the full customer schema available.

#### Current Query Generation (line 4782-4832)

```python
def _query_engine_agent(self, plan, specialist_findings):
    table = plan["table"]                    # SINGLE table
    metric_expr = plan["metric_expr"]
    where_clause = self._build_where_clause(plan, for_comparison="current")

    if intent == "comparison":
        sql = f"SELECT 'current' AS period, {metric_expr} FROM {table} WHERE ..."
              f"UNION "
              f"SELECT 'comparison' AS period, {metric_expr} FROM {table} WHERE ..."
    elif intent == "grouped_metric":
        sql = f"SELECT {dims}, {metric_expr} FROM {table} WHERE ... GROUP BY ..."
    elif intent == "lookup":
        sql = f"SELECT * FROM {table} WHERE ... LIMIT ..."
    else:
        sql = f"SELECT {metric_expr} FROM {table} WHERE ..."
```

Every branch: single `FROM {table}`. No JOINs. No CTEs. No subqueries.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 4782-4832 | `_query_engine_agent` — all SQL templates use single table |
| `agentic_team.py` | 4833-4901 | `_build_where_clause` — filters for single table only |
| `agentic_team.py` | 3976-4057 | `_planning_agent` — produces `plan["table"]` (singular) |
| `agentic_team.py` | 4238-4375 | `_generate_candidate_plans` — each candidate has one table |

#### Implementation Tasks

1. **Extend plan schema** to support multi-table queries:
   ```python
   plan = {
       "tables": [
           {"name": "datada_dim_customers", "alias": "c", "role": "primary"},
           {"name": "datada_mart_transactions", "alias": "t", "role": "joined"},
       ],
       "joins": [
           {"from": "c.customer_id", "to": "t.customer_id", "type": "INNER"},
       ],
       "select": [
           {"expr": "c.type", "alias": "customer_type"},
           {"expr": "COUNT(DISTINCT t.transaction_key)", "alias": "tx_count"},
           {"expr": "SUM(t.amount)", "alias": "total_spent"},
       ],
       "group_by": ["c.type"],
       "where": [...],
   }
   ```

2. **Add JOIN generation** to `_query_engine_agent`:
   ```python
   if plan.get("joins"):
       from_clause = plan["tables"][0]["name"] + " " + plan["tables"][0]["alias"]
       for join in plan["joins"]:
           joined_table = next(t for t in plan["tables"] if t["role"] == "joined")
           from_clause += (
               f" {join['type']} JOIN {joined_table['name']} {joined_table['alias']}"
               f" ON {join['from']} = {join['to']}"
           )
       sql = f"SELECT {select_clause} FROM {from_clause} WHERE {where} GROUP BY {group_by}"
   ```

3. **Build join path catalog** — extend `_build_catalog` to store known FK relationships:
   ```python
   catalog["join_paths"] = {
       ("datada_dim_customers", "datada_mart_transactions"): {
           "from_col": "customer_id",
           "to_col": "customer_id",
           "type": "INNER",
           "cardinality": "1:N",
       },
       # ... more paths
   }
   ```

4. **Backward compatibility**: When `plan["tables"]` is absent, fall back to `plan["table"]` (single-table legacy path).

#### Acceptance Criteria

- [ ] Query engine can generate `SELECT ... FROM a JOIN b ON ... GROUP BY ...` SQL
- [ ] Plan schema supports multi-table with join specifications
- [ ] "Customer purchasing behavior" query produces a cross-table GROUP BY
- [ ] Join paths are stored in catalog and used by planner
- [ ] All existing single-table queries produce identical SQL (regression check)
- [ ] Generated JOINs are validated against catalog FK relationships

---

### GAP 15 — Chief Analyst Is a Stub

**Severity**: HIGH
**Phase**: B (LLM-Powered Reasoning)
**Depends on**: GAP 12 (transparent failures)

#### Problem

`_chief_analyst` (line 2158) is a 5-line method that returns `{"mission": goal}`. It does zero orchestration. The actual execution flow is hardcoded in `run()`. The Chief Analyst should be the decision-maker: what agents to invoke, in what order, when to loop back, when to stop.

#### Current Code (line 2158-2174)

```python
def _chief_analyst(self, goal, runtime, catalog):
    return {
        "mission": goal,
        "runtime_mode": runtime.mode,
        "specialists": ["transactions", "customers", "revenue", "risk"],
        "available_marts": list(catalog.get("marts", {}).keys()),
    }
```

This is called at line 921, its output (`mission`) is passed to IntakeAgent and... that's it. It never makes a decision.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 2158-2174 | `_chief_analyst` — stub returning static metadata |
| `agentic_team.py` | 921-929 | Call site in `run()` — output stored but barely used |
| `agentic_team.py` | 882-1858 | `run()` — hardcoded sequential pipeline does real orchestration |

#### Implementation Tasks

1. **Make Chief Analyst an LLM-powered decision maker**:
   ```python
   def _chief_analyst(self, goal, runtime, catalog):
       if not runtime.use_llm:
           return self._chief_analyst_deterministic(goal, catalog)

       prompt = f"""You are the Chief Analyst orchestrating a data analytics team.

       Question: {goal}

       Available data domains and their schemas:
       {self._format_catalog_for_llm(catalog)}

       Analyze this question and decide:
       1. Which domains are needed? (may be multiple)
       2. What type of analysis is this? (simple metric, comparison, behavioral, trend, exploration)
       3. Do we need cross-domain JOINs?
       4. What metrics and dimensions are relevant?
       5. Should we run specialists for domain-specific validation?

       Return JSON with: domains, analysis_type, requires_join, metrics, dimensions, specialists_needed
       """
       raw = call_llm([{"role": "system", "content": prompt}], role="planner", ...)
       return _extract_json_payload(raw)
   ```

2. **Use Chief Analyst output to drive execution** in `run()`:
   ```python
   mission = self._chief_analyst(goal, runtime, catalog)

   # Chief decides which domains, not intake keyword matching
   if mission.get("requires_join"):
       # Multi-domain path
       ...
   elif mission.get("analysis_type") == "exploration":
       # Schema exploration path
       ...
   else:
       # Standard single-domain path (current behavior)
       ...
   ```

3. **Add deterministic fallback** for when LLM is unavailable — preserves current behavior.

#### Acceptance Criteria

- [ ] Chief Analyst uses LLM to analyze the question when LLM is available
- [ ] Chief Analyst output drives execution path selection in `run()`
- [ ] Multi-domain questions are identified by Chief Analyst
- [ ] Deterministic fallback preserves current behavior when LLM unavailable
- [ ] Chief Analyst response includes reasoning for its decisions

---

### GAP 16 — Specialist Agents Are Discarded

**Severity**: HIGH
**Phase**: D (Agent Collaboration)
**Depends on**: GAP 17 (blackboard for communication)

#### Problem

Four specialist agents run (line 1556-1602) and produce domain-specific findings. Their output is passed to `_query_engine_agent` as `specialist_findings`. Inside that method, line 4789:

```python
_ = specialist_findings
```

The findings are **explicitly discarded**. Specialists generate warnings and suggestions that could improve query quality, but nothing uses them for SQL generation or plan modification.

#### Current Specialist Intelligence

Each specialist does useful validation (lines 4715-4773):
- **TransactionsSpecialist**: Validates metric availability, checks dimension existence, warns about missing boolean filters
- **CustomerSpecialist**: Validates customer dimensions, suggests geographic enrichment
- **RevenueSpecialist**: Validates revenue metrics exist
- **RiskSpecialist**: Checks MT103/refund filter logic

But their output feeds only `_pipeline_warnings` — cosmetic annotations on the response.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 4715-4733 | `_transactions_specialist` — validation + warnings |
| `agentic_team.py` | 4733-4750 | `_customer_specialist` — validation + suggestions |
| `agentic_team.py` | 4750-4760 | `_revenue_specialist` — validation |
| `agentic_team.py` | 4760-4773 | `_risk_specialist` — filter checks |
| `agentic_team.py` | 4789 | `_ = specialist_findings` — **discarded** |
| `agentic_team.py` | 1610 | Call site passes findings to query engine |

#### Implementation Tasks

1. **Specialists modify the plan, not just warn**:
   ```python
   def _transactions_specialist(self, plan, catalog):
       findings = {"notes": [], "warnings": [], "plan_modifications": []}

       # If goal mentions "purchasing behavior" but plan has no amount metric
       if "behavior" in plan.get("goal", "").lower():
           if plan["metric"] in {"transaction_count", "customer_count"}:
               findings["plan_modifications"].append({
                   "action": "add_metrics",
                   "metrics": ["total_amount", "avg_amount", "transaction_count"],
                   "reason": "Behavioral analysis requires multiple metrics",
               })

       # If boolean filter is implied but missing
       ...
       return findings
   ```

2. **Query engine uses specialist modifications**:
   ```python
   def _query_engine_agent(self, plan, specialist_findings):
       # Apply specialist plan modifications
       for findings in specialist_findings:
           for mod in findings.get("plan_modifications", []):
               if mod["action"] == "add_metrics":
                   plan["additional_metrics"] = mod["metrics"]
               elif mod["action"] == "add_join":
                   plan["joins"] = plan.get("joins", []) + [mod["join"]]
       # Generate SQL from enriched plan
       ...
   ```

3. **Specialists can request domain escalation** — if a customer specialist detects that transaction data is needed, it posts to the blackboard (GAP 17) requesting a re-plan with additional domains.

#### Acceptance Criteria

- [ ] `_ = specialist_findings` line is removed
- [ ] Specialist `plan_modifications` are applied before SQL generation
- [ ] Specialists can add metrics, suggest JOINs, and modify filters
- [ ] Specialist reasoning appears in `evidence_packets`
- [ ] Existing queries without specialist modifications are unaffected

---

### GAP 17 — Blackboard Is Tracing-Only

**Severity**: HIGH
**Phase**: D (Agent Collaboration)
**Depends on**: None (parallel with GAP 16)

#### Problem

`_blackboard_post` (line 4575) and `_blackboard_edges` (line 4601) create an immutable audit trail of which agent produced what artifact and who consumed it. But **no agent reads from the blackboard**. All data flows through function parameters in `run()`. The blackboard is observation-only.

#### Current Implementation (lines 4575-4616)

```python
def _blackboard_post(self, blackboard, *, producer, artifact_type, payload, consumed_by):
    blackboard.append({
        "producer": producer,
        "artifact_type": artifact_type,
        "payload": payload,
        "consumed_by": consumed_by,
        "timestamp": datetime.utcnow().isoformat(),
    })

def _blackboard_edges(self, blackboard):
    edges = []
    for entry in blackboard:
        for consumer in entry.get("consumed_by", []):
            edges.append({"from": entry["producer"], "to": consumer, ...})
    return edges
```

This is a **write-only log**. No method ever calls `blackboard_read()` or queries the blackboard for artifacts.

#### Implementation Tasks

1. **Add blackboard query capability**:
   ```python
   def _blackboard_query(self, blackboard, *, artifact_type=None, producer=None):
       """Agents can query the blackboard for artifacts from other agents."""
       return [
           entry for entry in blackboard
           if (artifact_type is None or entry["artifact_type"] == artifact_type)
           and (producer is None or entry["producer"] == producer)
       ]
   ```

2. **Specialists read from blackboard** instead of receiving data through parameters:
   ```python
   def _transactions_specialist(self, plan, catalog, blackboard):
       # Read what IntakeAgent found
       intake_artifacts = self._blackboard_query(blackboard, artifact_type="structured_goal")
       # Read what AuditAgent found in previous iteration
       audit_artifacts = self._blackboard_query(blackboard, artifact_type="audit_result")
       # Make decisions based on full context
       ...
   ```

3. **Agents post requests** to the blackboard (not just artifacts):
   ```python
   # Specialist posts a request for re-planning
   self._blackboard_post(blackboard,
       producer="CustomerSpecialist",
       artifact_type="replan_request",
       payload={"reason": "Need transaction data for behavior analysis",
                "add_domains": ["transactions"]},
       consumed_by=["ChiefAnalystAgent"],
   )
   ```

4. **Chief Analyst reads requests** and decides whether to act on them.

#### Acceptance Criteria

- [ ] `_blackboard_query` method exists and is used by at least 3 agents
- [ ] Specialists read from blackboard to access other agents' outputs
- [ ] Agents can post requests (not just artifacts) to the blackboard
- [ ] Chief Analyst (GAP 15) reads blackboard requests and makes decisions
- [ ] Blackboard serves as functional inter-agent communication, not just tracing

---

### GAP 18 — No LLM in Planning

**Severity**: HIGH
**Phase**: B (LLM-Powered Reasoning)
**Depends on**: GAP 12 (transparent failures)

#### Problem

`_planning_agent` (line 3976) is a template filler. It takes the intake result, looks up the metric expression from the catalog, validates dimensions against columns, and returns a plan dict. There is no reasoning about:
- Whether the selected metric actually answers the question
- Whether additional metrics would be more informative
- Whether the chosen table is the best source
- Whether a JOIN or multi-step approach is needed
- Whether the question requires analytical patterns beyond simple aggregation

#### Current Planning (line 3976-4057)

```python
def _planning_agent(self, intake, retrieval, catalog):
    metric_expr = retrieval["metrics"].get(intake["metric"])
    if metric_expr is None:
        fallback_metric, metric_expr = next(iter(retrieval["metrics"].items()))

    # Validate dimensions against available columns
    for dim in raw_dims:
        if dim in retrieval["columns"]:
            dimensions.append(dim)

    # Template-fill the plan
    return {
        "table": retrieval["table"],
        "metric": intake["metric"],
        "metric_expr": metric_expr,
        "dimensions": dimensions,
        "time_filter": intake.get("time_filter"),
        "value_filters": value_filters,
    }
```

No reasoning. No alternatives considered. No understanding of whether the plan answers the question.

#### Implementation Tasks

1. **Add LLM-powered plan evaluation**:
   ```python
   def _planning_agent(self, intake, retrieval, catalog, runtime):
       # Step 1: Build deterministic baseline plan (current behavior)
       base_plan = self._planning_agent_deterministic(intake, retrieval, catalog)

       if not runtime.use_llm:
           return base_plan

       # Step 2: Ask LLM to evaluate and improve the plan
       prompt = f"""You are a SQL planning agent. Evaluate this query plan:

       User question: {intake['goal']}
       Proposed plan: {json.dumps(base_plan)}
       Available tables and columns: {self._format_retrieval_for_llm(retrieval)}
       Available metrics: {json.dumps(retrieval['metrics'])}

       Does this plan answer the user's question? If not, suggest improvements:
       - Different metrics?
       - Additional dimensions?
       - Different table?
       - Need for JOINs?
       - Multi-step analysis?

       Return JSON with: approved (bool), improvements (list), revised_plan (dict if not approved)
       """
       evaluation = call_llm([...], role="planner", ...)

       # Step 3: Apply LLM improvements if any
       ...
   ```

2. **Pass `runtime` to `_planning_agent`** so it can decide whether to use LLM.

3. **LLM plan evaluation feeds back** to candidate generation in the autonomy loop.

#### Acceptance Criteria

- [ ] Planning agent uses LLM to evaluate plan quality when LLM is available
- [ ] LLM can suggest alternative metrics, dimensions, and tables
- [ ] Plan evaluation reasoning is logged in agent trace
- [ ] Deterministic fallback preserves current behavior
- [ ] Plan quality measurably improves for complex questions (audit score comparison)

---

### GAP 19 — No LLM in Audit/Scoring

**Severity**: MEDIUM
**Phase**: B (LLM-Powered Reasoning)
**Depends on**: GAP 12 (transparent failures)

#### Problem

`_audit_agent` (line 4915) validates query results using **hardcoded concept mappings** and **fixed-weight scoring**. It checks whether goal terms appear in the schema but cannot reason about whether the result *actually answers the question*. The scoring formula (lines 5090-5102) uses manually tuned penalties.

`_candidate_score_breakdown` (line 4480) uses the same fixed weights. No learning from outcomes.

#### Current Scoring (lines 5090-5102)

```python
score = 0.9  # baseline
if not success: score = 0.18
if row_count == 0: score -= 0.20
score -= 0.10 * failed_checks
score -= 0.10 * len(warnings)
if not concept_check_passed: score = min(score, 0.45)
score -= 0.08 * max(0.0, 1.0 - concept_coverage)
if not schema_grounded: score -= 0.08
```

These weights are arbitrary. `0.10` per warning? `0.08` per concept miss? These were hand-tuned for the demo queries.

#### Implementation Tasks

1. **LLM-powered result evaluation** in `_audit_agent`:
   ```python
   # After deterministic checks, ask LLM to evaluate
   if runtime.use_llm:
       prompt = f"""Does this SQL result answer the user's question?

       Question: {goal}
       SQL: {sql}
       Result: {result_summary}
       Schema used: {table} with columns {columns}

       Evaluate:
       1. Does the metric match what was asked? (score 0-1)
       2. Are relevant dimensions included? (score 0-1)
       3. Are there important aspects of the question not addressed? (list)
       4. What would make this answer better? (suggestions)
       """
       llm_eval = call_llm([...], role="planner", ...)
   ```

2. **Blend LLM evaluation with deterministic scoring** — LLM score as a weighted factor, not a replacement.

3. **Use LLM audit feedback** to drive smarter re-planning in the autonomy loop.

#### Acceptance Criteria

- [ ] Audit agent uses LLM to evaluate whether the result answers the question
- [ ] LLM evaluation contributes to the audit score
- [ ] LLM suggestions feed into candidate generation for re-planning
- [ ] Deterministic scoring still runs as baseline (LLM augments, doesn't replace)
- [ ] Complex questions get lower audit scores when answered with simple counts

---

### GAP 20 — No Complex Analytical Patterns

**Severity**: HIGH
**Phase**: E (Complex Analytics)
**Depends on**: GAP 14 (multi-table query engine)

#### Problem

The system can only generate four SQL patterns:
1. `SELECT {metric} FROM {table} WHERE {filters}` — scalar metric
2. `SELECT {dims}, {metric} FROM {table} GROUP BY {dims}` — grouped metric
3. `SELECT * FROM {table} LIMIT N` — lookup
4. `SELECT ... UNION SELECT ...` — comparison (same table, different WHERE)

Real analytical questions require:
- **Behavioral analysis**: JOINs + GROUP BY + multiple metrics
- **Cohort analysis**: Subqueries or CTEs for segment definition
- **Trend comparison**: Window functions (LAG, LEAD, moving averages)
- **Distribution analysis**: Percentiles, histograms, standard deviations
- **Correlation**: Cross-metric analysis within or across domains
- **Ranking**: ROW_NUMBER, RANK, DENSE_RANK with partitioning
- **Conditional aggregation**: Complex CASE WHEN logic beyond boolean flags

#### Example Queries That Cannot Be Answered Today

| Query | Why It Fails |
|-------|-------------|
| "How do individual and corporate customers differ in purchasing behavior?" | Needs JOIN + multi-metric GROUP BY |
| "Which platform has the highest transaction growth month over month?" | Needs window functions (LAG) |
| "What's the 90th percentile transaction amount by country?" | Needs PERCENTILE_CONT |
| "Show me customer segments by spending tier" | Needs CTE with CASE for tier bucketing |
| "Which customers have increasing refund rates?" | Needs window functions + trend detection |
| "Compare Q1 vs Q2 revenue by platform" | Needs flexible time comparison with grouping |

#### Implementation Tasks

1. **Add analytical pattern library** — a set of SQL templates for common analytical patterns:
   ```python
   ANALYTICAL_PATTERNS = {
       "behavioral_comparison": {
           "template": """
               SELECT {group_col}, {metric_exprs}
               FROM {primary_table} p
               JOIN {secondary_table} s ON {join_condition}
               GROUP BY {group_col}
           """,
           "requires": ["join", "multi_metric", "group_by"],
       },
       "trend_growth": {
           "template": """
               WITH monthly AS (
                   SELECT DATE_TRUNC('month', {time_col}) AS month,
                          {group_col}, {metric_expr} AS val
                   FROM {table} GROUP BY 1, 2
               )
               SELECT *, val - LAG(val) OVER (PARTITION BY {group_col} ORDER BY month) AS growth
               FROM monthly
           """,
           "requires": ["time_column", "window_function"],
       },
       # ... more patterns
   }
   ```

2. **LLM selects pattern** based on question analysis (requires GAP 18).

3. **Pattern-based query generation** as an alternative path in `_query_engine_agent`.

4. **Iterative — start with 3-4 patterns**, expand based on user demand.

#### Acceptance Criteria

- [ ] At least 5 analytical patterns are implemented (behavioral, trend, distribution, ranking, cohort)
- [ ] LLM or Chief Analyst selects the appropriate pattern
- [ ] "Customer purchasing behavior" uses the behavioral_comparison pattern
- [ ] "Month over month growth" uses the trend_growth pattern
- [ ] Pattern selection is logged in agent trace
- [ ] Fallback to simple patterns when complex patterns fail

---

### GAP 21 — Deterministic Candidate Scoring

**Severity**: MEDIUM
**Phase**: D (Agent Collaboration)
**Depends on**: GAP 19 (LLM audit)

#### Problem

`_candidate_score_breakdown` (line 4480) scores candidate plans with fixed weights:
```
score = audit_base (0-0.9)
    + execution_bonus (0.04)
    + non_empty_bonus (0.04)
    + dimension_bonus (0.02-0.03 per matched keyword)
    - goal_miss_penalty (0.08 per missed concept)
    - latency_penalty (0.03 if > 6000ms)
```

These weights are manually tuned. No learning from user feedback. No contextual adjustment. A plan that produces rows always scores higher than one that doesn't, even if the rows don't answer the question.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 4480-4529 | `_candidate_score_breakdown` — fixed-weight formula |
| `agentic_team.py` | 4059-4236 | `_autonomous_refinement_agent` — uses scores for plan selection |

#### Implementation Tasks

1. **Weight learning from feedback**: Store user feedback (thumbs up/down) associated with score breakdowns. Periodically adjust weights.

2. **LLM-based candidate comparison** (when LLM available): Instead of pure scoring, ask LLM to compare top-2 candidates and pick the better one:
   ```python
   prompt = f"""Compare these two query plans for answering: {goal}
   Plan A: {plan_a} → Score: {score_a}
   Plan B: {plan_b} → Score: {score_b}
   Which better answers the question and why?"""
   ```

3. **Context-sensitive penalties**: "No rows returned" should be penalized differently for a filter query (probably wrong filter) vs. an exploration query (legitimately empty).

#### Acceptance Criteria

- [ ] LLM evaluates top candidates when available
- [ ] Score weights are adjustable (config or learned)
- [ ] User feedback influences future scoring
- [ ] Context-sensitive penalties (not one-size-fits-all)

---

## Effort & Risk Matrix

| Gap | Phase | Effort | Risk | Mitigation |
|-----|-------|--------|------|------------|
| **12** | A | 1 day | LOW — additive warnings, no logic change | Test all LLM mode combinations (local/openai/auto/deterministic × available/unavailable) |
| **18** | B | 3-4 days | MEDIUM — LLM may produce invalid plans | Deterministic baseline as fallback; validate LLM plan against catalog before execution |
| **19** | B | 2-3 days | LOW — augments existing scoring, doesn't replace | Blend LLM score with deterministic score; cap LLM influence at 30% initially |
| **15** | B | 3-4 days | HIGH — changes execution flow in `run()` | Feature flag to toggle new orchestration; extensive testing of all intent paths |
| **13** | C | 3-4 days | HIGH — changes fundamental intake architecture | Keep `domain` (singular) for backward compat; add `domains` (plural) as new field |
| **14** | C | 5-7 days | HIGH — new SQL generation paths, risk of incorrect JOINs | Validate JOINs against catalog FK relationships; row-count sanity checks; fallback to single-table |
| **16** | D | 3-4 days | MEDIUM — specialists now influence query | Specialist modifications are suggestions, not mandates; planner validates before applying |
| **17** | D | 3-4 days | LOW — extending existing infrastructure | Backward-compatible; old blackboard usage still works |
| **21** | D | 2-3 days | LOW — augments existing scoring | Keep deterministic score as floor; LLM comparison as tiebreaker |
| **20** | E | 5-7 days | HIGH — complex SQL patterns may produce incorrect results | Start with 3 patterns; extensive testing with known-answer queries; DuckDB EXPLAIN validation |

**Total estimated effort**: 30-40 days

---

## Verification Plan

After each phase:

### Phase A (Transparency)
- [ ] Start server with Ollama stopped → query in local mode → response.warnings contains LLM fallback message
- [ ] Start server with invalid OpenAI key → query in openai mode → response.warnings contains fallback message
- [ ] Query in auto mode with no providers → response.warnings explains deterministic fallback
- [ ] All existing tests pass

### Phase B (LLM Reasoning)
- [ ] "How many transactions" with LLM → plan evaluation appears in trace
- [ ] "Customer purchasing behavior" with LLM → Chief Analyst identifies multi-domain need
- [ ] Audit score for complex questions improves with LLM evaluation
- [ ] Deterministic mode still works identically (no regression)

### Phase C (Cross-Domain)
- [ ] "Customer purchasing behavior" → intake returns `domains: ["customers", "transactions"]`
- [ ] Query produces `SELECT c.type, COUNT(*), SUM(amount) FROM customers JOIN transactions GROUP BY type`
- [ ] All existing single-domain queries produce identical results
- [ ] Join paths validated against catalog

### Phase D (Collaboration)
- [ ] Specialist findings appear in SQL generation decisions
- [ ] Blackboard shows agent-to-agent artifact exchange
- [ ] Candidate scoring uses LLM evaluation for top candidates
- [ ] 250 existing tests pass

### Phase E (Complex Analytics)
- [ ] "Month over month growth by platform" → window function SQL
- [ ] "Customer spending tiers" → CTE with CASE bucketing
- [ ] "90th percentile amount by country" → PERCENTILE_CONT SQL
- [ ] Pattern selection logged in trace

---

## Appendix: Honest Architecture Assessment

### What the system IS good at today:
1. **Single-domain simple metrics** — "how many transactions", "total quote value"
2. **Grouped metrics** — "transactions by platform and month"
3. **Value filtering** — "mt103 transactions", "refunds in December"
4. **Bounded autonomy** — multi-round refinement improves results for simple queries
5. **Memory learning** — stores successful patterns for reuse
6. **Data overview / schema exploration** — describes available data well

### What the system CANNOT do today:
1. Answer questions requiring multiple domains
2. Generate JOINs at query time
3. Reason about whether a plan answers the question
4. Use LLM intelligence for planning, auditing, or scoring
5. Allow agents to communicate or influence each other
6. Generate complex SQL (CTEs, window functions, subqueries)
7. Tell the user when LLM failed and deterministic kicked in
8. Understand analytical concepts like "behavior", "trends", "correlation"

### The path from here to there:
1. **Phase A** (1 day): Be honest with the user — tell them when LLM fails
2. **Phase B** (8-11 days): Put LLM where it matters — planning, auditing, orchestration
3. **Phase C** (8-11 days): Break the single-table constraint — multi-domain + JOINs
4. **Phase D** (8-11 days): Make agents real — specialists that influence, blackboard that communicates
5. **Phase E** (5-7 days): Go beyond SELECT COUNT — analytical patterns for real questions
