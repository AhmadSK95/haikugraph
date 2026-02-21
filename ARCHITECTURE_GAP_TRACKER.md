# dataDa Agentic Analytics -- Architecture Gap Tracker

> **Created**: 2026-02-20
> **Updated**: 2026-02-20
> **Status**: Phases A-H implemented (25 DONE, 0 PARTIAL) — All gaps closed
> **Primary file**: `src/haikugraph/poc/agentic_team.py` (~6200 lines)
> **Secondary files**: `src/haikugraph/api/server.py`, `src/haikugraph/llm/router.py`, `src/haikugraph/agents/contracts.py`
> **Benchmark (before)**: `reports/benchmark_llm_comparison_20260220_191707.md`
> **Benchmark (after)**: `reports/benchmark_llm_comparison_20260220_202722.md`

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

| Gap | Title | Severity | Category | Status |
|-----|-------|----------|----------|--------|
| **12** | Silent LLM Fallback | CRITICAL | Transparency | DONE |
| **13** | Single-Domain Mutual Exclusion | CRITICAL | Architecture | DONE |
| **14** | Single-Table Query Engine | CRITICAL | Architecture | DONE |
| **15** | Chief Analyst Is a Stub | HIGH | Intelligence | DONE |
| **16** | Specialist Agents Are Discarded | HIGH | Intelligence | DONE |
| **17** | Blackboard Is Tracing-Only | HIGH | Communication | DONE |
| **18** | No LLM in Planning | HIGH | Intelligence | DONE |
| **19** | No LLM in Audit/Scoring | MEDIUM | Intelligence | DONE |
| **20** | No Complex Analytical Patterns | HIGH | Capability | DONE |
| **21** | Deterministic Candidate Scoring | MEDIUM | Intelligence | DONE |
| **22** | LLM Latency Optimization | HIGH | Performance | DONE |
| **23** | Audit Warning Noise Filtering | MEDIUM | Quality | DONE |
| **24** | Local Model Accuracy Tuning | MEDIUM | Accuracy | DONE |
| **25** | Boolean Filter Schema Gap | MEDIUM | Accuracy | DONE |
| **26** | Narrative Quality Parity | MEDIUM | Quality | DONE |
| **27** | OpenAI Latency Variance | MEDIUM | Performance | DONE |
| **28** | Model Version Health Management | HIGH | Operations | DONE |
| **29** | Intelligent Mode Selection | HIGH | Intelligence | DONE |
| **35** | Domain Expertise Knowledge Base | MEDIUM | Intelligence | DONE |
| **36** | Multi-Domain Detection Fix | HIGH | Architecture | DONE |
| **37** | Specialist Directives That Modify SQL | HIGH | Intelligence | DONE |
| **38** | Clarification Agent Intelligence | MEDIUM | Intelligence | DONE |
| **39** | Decision Transparency in Trace | MEDIUM | Transparency | DONE |
| **40** | UI Provider Completeness | LOW | UI | DONE |
| **41** | Memory Agent Enhancement | MEDIUM | Intelligence | DONE |

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

### Phase A — Transparency & Trust  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 12 | Silent LLM Fallback | 1 day | DONE |

### Phase B — LLM-Powered Reasoning  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 18 | LLM in Planning | 3-4 days | DONE |
| 19 | LLM in Audit/Scoring | 2-3 days | DONE |
| 15 | Chief Analyst Orchestrator | 3-4 days | DONE |

### Phase C — Cross-Domain Intelligence  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 13 | Multi-Domain Intent Detection | 3-4 days | DONE |
| 14 | Multi-Table Query Engine | 5-7 days | DONE |

### Phase D — Agent Collaboration  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 16 | Real Specialist Agents | 3-4 days | DONE |
| 17 | Functional Blackboard | 3-4 days | DONE |
| 21 | LLM-Evaluated Scoring | 2-3 days | DONE |

### Phase E — Complex Analytics  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 20 | Complex Analytical Patterns | 5-7 days | DONE |

### Phase F — Benchmark-Identified Gaps  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 22 | LLM Latency Optimization | 3-4 days | DONE — per-provider timeouts, skip redundant calls |
| 23 | Audit Warning Noise Filtering | 1-2 days | DONE — regex noise filter + improved audit prompt |
| 24 | Local Model Accuracy Tuning | 2-3 days | DONE — improved planning prompt prevents unrequested GROUP BY |
| 25 | Boolean Filter Schema Gap | 1 day | DONE — catalog-driven boolean column detection (is_university) |
| 26 | Narrative Quality Parity | 2-3 days | DONE — improved narrator prompt with FORMAT/ACCURACY rules |
| 27 | OpenAI Latency Variance | 1-2 days | DONE — per-provider timeouts in router.py |
| 28 | Model Version Health Management | 2-3 days | DONE — model health endpoint, fallback chains, updated model IDs |
| 29 | Intelligent Mode Selection | 3-4 days | DONE — auto mode priority: anthropic > ollama > openai |

### Phase G — Smart LLM Mode  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 30 | LLM Intent Classification | 1-2 days | DONE — full 12+ intent classifier with schema + domain synonyms |
| 31 | LLM SQL Generation | 2-3 days | DONE — LLM SQL for complex intents, validated + probe-tested |
| 32 | Multi-Part Decomposition | 3-4 days | DONE — decompose → sub-query → merge (solves Gap 15) |
| 33 | SQL Error Recovery | 0.5-1 day | DONE — LLM diagnoses failed SQL + single retry with validation |
| 34 | LLM-Enhanced Audit Retry | 1-2 days | DONE — LLM root-cause analysis replaces 3-entry hint dict |

### Phase H — Domain Intelligence & Agent Effectiveness  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 35 | Domain Expertise Knowledge Base | 1-2 days | DONE — domain_knowledge.yaml + loader + builtin fallback |
| 36 | Multi-Domain Detection Fix | 1-2 days | DONE — synonym enrichment + multi_domain_hint consumption |
| 37 | Specialist Directives That Modify SQL | 2-3 days | DONE — typed directives (override_metric_expr, add_filter) + column qualification for JOINs |
| 38 | Clarification Agent Intelligence | 1-2 days | DONE — unique intent, cross-domain, metric-domain mismatch checks |
| 39 | Decision Transparency in Trace | 1-2 days | DONE — reasoning field in trace + UI gold-highlight rendering |
| 40 | UI Provider Completeness | 0.5 day | DONE — Anthropic option + provider status dots |
| 41 | Memory Agent Enhancement | 0.5-1 day | DONE — learned corrections for explicit queries |

**Total estimated effort**: 30-40 days (Phases A-E) + 16-22 days (Phase F) + 8-12 days (Phase G) + 7-10 days (Phase H) = ~61-84 days

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

---

## Benchmark-Identified Gaps (Phase F)

> **Source**: `reports/benchmark_llm_comparison_20260220_191707.md`
> **Benchmark Date**: 2026-02-20
> **Modes Tested**: deterministic, local (Ollama qwen2.5/llama3.1), openai (gpt-4o-mini), anthropic (claude-haiku-4-5)
> **Queries**: 18 ground-truth queries across 6 categories

### Benchmark Results — Before (Phase F gaps open)

| Mode | Correctness (50%) | Confidence (15%) | Latency (15%) | Narrative (20%) | **Composite** |
|------|-------------------|------------------|---------------|-----------------|---------------|
| deterministic | 94.44% | 94.44% | 99.56% | 100.00% | **96.32%** |
| anthropic | 100.00% | 100.00% | 69.00% | 88.33% | **93.02%** |
| local | 94.44% | 94.44% | 58.83% | 94.44% | **89.10%** |
| openai | 100.00% | 100.00% | 30.88% | 98.33% | **89.30%** |

### Benchmark Results — After (Phase F gaps closed)

| Mode | Correctness (50%) | Confidence (15%) | Latency (15%) | Narrative (20%) | **Composite** |
|------|-------------------|------------------|---------------|-----------------|---------------|
| deterministic | 100.00% | 100.00% | 99.55% | 100.00% | **99.93%** |
| anthropic | 100.00% | 100.00% | 73.72% | 98.33% | **95.72%** |
| local | 100.00% | 100.00% | 51.15% | 99.17% | **92.51%** |
| openai | 100.00% | 100.00% | 69.08% | 97.50% | **94.86%** |

### Improvement Summary

| Mode | Before | After | Delta |
|------|--------|-------|-------|
| deterministic | 96.32% | 99.93% | **+3.61%** |
| local | 89.10% | 92.51% | **+3.41%** |
| openai | 89.30% | 94.86% | **+5.56%** |
| anthropic | 93.02% | 95.72% | **+2.70%** |

**Key Improvements**:
- **100% correctness across ALL modes** (18/18 queries correct for every provider)
- **Warnings reduced from 47 to 8** (83% noise reduction)
- **OpenAI composite +5.56%** — biggest improvement, driven by boolean filter fix + latency improvement
- **No silent LLM fallback** — explicit 503 errors when provider unavailable
- **Catalog-driven boolean filters** — replaced hardcoded if/elif chains with schema-aware pattern

---

### GAP 22 — LLM Latency Optimization

**Severity**: HIGH
**Phase**: F (Benchmark-Identified)
**Depends on**: None

#### Problem

The benchmark reveals that all LLM modes have negative composite uplift vs deterministic, driven almost entirely by latency:

| Mode | Median Latency | Latency Score | Composite Uplift vs Deterministic |
|------|---------------|---------------|-----------------------------------|
| deterministic | 130ms | 99.56% | — (baseline) |
| anthropic | 9,216ms | 69.00% | -3.30% |
| local | 12,125ms | 58.83% | -7.22% |
| openai | 20,042ms | 30.88% | -7.02% |

LLM calls happen at 6 pipeline steps (intake, chief analyst, planning, candidate scoring, audit, narrative). Each adds a serial round-trip. A 9-second total for Anthropic means ~1.5s per step — acceptable individually but cumulative latency destroys the composite score.

#### Implementation Tasks

1. **Parallel LLM calls**: Intake, chief analyst, and planning can run concurrently since they operate on different inputs. Use `asyncio.gather()` or `concurrent.futures`:
   ```python
   # Instead of serial:
   intake = await _intake_with_llm(goal)
   chief = await _chief_analyst(goal)
   # Run in parallel:
   intake, chief = await asyncio.gather(
       _intake_with_llm(goal),
       _chief_analyst(goal),
   )
   ```

2. **LLM response caching**: Cache intake and planning LLM responses for identical/similar queries. Use a TTL-based cache keyed on normalized goal text:
   ```python
   @lru_cache(maxsize=128)
   def _cached_llm_call(goal_hash, role, provider):
       ...
   ```

3. **Skip redundant LLM steps**: If deterministic intake produces high-confidence parse (all slots filled), skip LLM intake refinement. If audit score > 0.85 on first pass, skip LLM audit enhancement.

4. **Model tier selection**: Use faster models (haiku/mini) for intake/audit, reserve larger models (sonnet/4o) for planning and narrative only.

5. **Streaming for narrative**: Use streaming API responses for narrative generation so the user sees output progressively.

#### Acceptance Criteria

- [ ] Median LLM latency reduced by at least 40% (target: <6s for anthropic, <8s for local, <12s for openai)
- [ ] Parallel LLM calls implemented for independent pipeline steps
- [ ] Query cache hit rate > 30% for repeated/similar queries
- [ ] Composite score for LLM modes exceeds deterministic baseline

---

### GAP 23 — Audit Warning Noise Filtering

**Severity**: MEDIUM
**Phase**: F (Benchmark-Identified)
**Depends on**: GAP 19 (LLM audit)

#### Problem

The Anthropic LLM audit agent generates excessive low-value warnings that drag down narrative quality scores. From the benchmark:

```
- [anthropic] cnt_tx: Query uses DISTINCT on transaction_key which may be unnecessary
- [anthropic] cnt_tx: WHERE 1=1 is a code smell suggesting dynamic query building
- [anthropic] cnt_tx: No time period specified - ensure this captures the intended scope
- [anthropic] cnt_cust: Query uses 'WHERE 1=1' which is a code smell
- [anthropic] cnt_cust: No temporal filter applied
- [anthropic] cnt_quote: Query uses 'datada_mart_quotes' table name which appears to contain a typo
```

These are code-review-style observations, not actionable data quality warnings. The `WHERE 1=1` warning appears for every query because the pipeline uses template SQL with `WHERE 1=1` as a base. The "no time period specified" warning is irrelevant when the user asked for an all-time count.

Anthropic generated **49 warnings** across 18 queries vs OpenAI's **7** and Local's **3**. This noise contributes to Anthropic's lowest narrative score (88.33%) despite highest correctness (100%).

#### Implementation Tasks

1. **Categorize warnings by severity**: Separate warnings into `critical` (wrong data), `advisory` (potential issue), and `informational` (code style):
   ```python
   WARNING_NOISE_PATTERNS = [
       r"WHERE 1=1",
       r"code smell",
       r"may be unnecessary",
       r"ensure that",
       r"consider removing",
       r"consider verifying",
   ]
   ```

2. **Filter noise before response**: Remove informational warnings from user-facing `response.warnings`. Keep them in the trace for debugging:
   ```python
   user_warnings = [w for w in warnings if not _is_noise_warning(w)]
   trace_warnings = warnings  # keep all for debugging
   ```

3. **Tune audit prompt**: Instruct the LLM auditor to focus on data correctness, not code style:
   ```
   Focus ONLY on whether the query answers the user's question correctly.
   Do NOT flag SQL style issues like WHERE 1=1, DISTINCT usage, or LIMIT clauses.
   ```

4. **Warning deduplication**: `WHERE 1=1` appears in nearly every warning — deduplicate repeated patterns.

#### Acceptance Criteria

- [ ] User-facing warnings contain only actionable data quality issues
- [ ] Code-style warnings (WHERE 1=1, DISTINCT, etc.) are filtered from user output
- [ ] Anthropic warning count per query drops from ~4 to <1 on average
- [ ] Anthropic narrative score improves to >95% on benchmark re-run

---

### GAP 24 — Local Model Accuracy Tuning

**Severity**: MEDIUM
**Phase**: F (Benchmark-Identified)
**Depends on**: GAP 18 (LLM planning)

#### Problem

Local (Ollama) mode scores 75% (3/4) on aggregations, specifically failing `agg_tx_avg` (average transaction amount). The benchmark warnings reveal:

```
- [local] agg_tx_avg: The query is selecting 'payment_status' which is not relevant
  to the goal of finding the average payment amount per transaction.
- [local] agg_tx_avg: The WHERE clause with 1=1 is unnecessary and can be removed.
```

The local planner (qwen2.5:14b-instruct) incorrectly adds `payment_status` as a GROUP BY dimension for an average calculation, producing grouped averages instead of a single overall average. This is a prompt engineering issue — the planning prompt doesn't sufficiently constrain the model from adding irrelevant dimensions.

#### Implementation Tasks

1. **Improve planning prompt for local models**: Add explicit instruction to avoid adding dimensions unless the user requests grouping:
   ```python
   PLANNER_PROMPT_LOCAL = """
   IMPORTANT: Only add GROUP BY dimensions if the user explicitly asks
   for a breakdown (e.g., "by platform", "per country"). For aggregate
   questions like "what is the average amount", return a single scalar
   value without grouping.
   """
   ```

2. **Post-validation check**: After LLM planning, verify that the plan doesn't add unrequested dimensions:
   ```python
   if not intake.get("dimensions") and plan.get("dimensions"):
       warnings.append("LLM added dimensions not requested by user; removing.")
       plan["dimensions"] = []
   ```

3. **Model upgrade path**: Test with qwen2.5:32b-instruct or larger quantization for better instruction following. Track accuracy per model variant.

4. **Few-shot examples**: Add 2-3 examples in the planning prompt showing correct scalar vs grouped outputs.

#### Acceptance Criteria

- [ ] `agg_tx_avg` query returns single average value without GROUP BY in local mode
- [ ] Local mode achieves 100% (4/4) on aggregations category
- [ ] Local composite score improves from 89.1% to >92%
- [ ] No regression on other categories

---

### GAP 25 — Boolean Filter Schema Gap

**Severity**: MEDIUM
**Phase**: F (Benchmark-Identified)
**Depends on**: GAP 13 (multi-domain detection)

#### Problem

Deterministic mode scores 67% (2/3) on boolean filters, failing `bool_univ` (university customers). The benchmark shows:

```
- [openai] bool_univ: Dimension 'customer_type' from intake is not in
  datada_dim_customers schema; removed.
- [anthropic] bool_univ: Query does not filter for universities —
  WHERE clause contains only '1=1' placeholder
```

The query "how many customers are universities" requires filtering by customer type, but the `datada_dim_customers` table has no `customer_type` or `organization_type` column. The `type` column contains `INDIVIDUAL`/`CORPORATE` values, not granular categories like "university". This is a data schema limitation, not a pipeline bug.

However, the pipeline should:
1. Detect that "university" cannot be resolved to any column/value
2. Warn the user explicitly instead of returning total customer count
3. Suggest what IS available for filtering

#### Implementation Tasks

1. **Value-match validation**: After parsing `bool_univ` intent, check if "university" matches any dimension value in the catalog:
   ```python
   if value_filter and not _value_exists_in_catalog(value_filter, catalog):
       warnings.append(
           f"Filter value '{value_filter}' not found in available data. "
           f"Available customer types: {catalog['dimension_values'].get('type', [])}"
       )
   ```

2. **Fuzzy value matching**: Attempt to map "university" to closest available value (e.g., "CORPORATE" since universities are organizations). Flag as low-confidence match.

3. **Schema enrichment**: If the underlying data CAN distinguish universities (e.g., by name pattern), add a derived column or enum mapping.

4. **Honest "cannot answer"**: When a filter value truly doesn't exist, return a clear message instead of silently returning unfiltered results.

#### Acceptance Criteria

- [ ] "How many customers are universities" returns a warning about unavailable filter
- [ ] Response includes available filter values (INDIVIDUAL, CORPORATE)
- [ ] Pipeline does not silently return total count when filter cannot be applied
- [ ] Benchmark `bool_univ` either succeeds with correct filter or fails with clear explanation

---

### GAP 26 — Narrative Quality Parity Across Providers

**Severity**: MEDIUM
**Phase**: F (Benchmark-Identified)
**Depends on**: GAP 23 (warning noise filtering)

#### Problem

Narrative quality scores vary significantly across providers:

| Provider | Narrative Score | Notes |
|----------|----------------|-------|
| deterministic | 100.00% | Template-based, always consistent |
| openai | 98.33% | Near-perfect, good markdown formatting |
| local | 94.44% | Good but occasionally missing formatting |
| anthropic | 88.33% | Lowest despite highest correctness |

Anthropic's low narrative score is likely caused by:
1. Excessive warnings cluttering the response (see GAP 23)
2. Narrator prompt not optimized for Claude's response style
3. Claude tends toward verbose, analytical prose rather than concise data summaries

#### Implementation Tasks

1. **Provider-specific narrator prompts**: Tune the narrative prompt per provider:
   ```python
   NARRATOR_PROMPTS = {
       "anthropic": "Be concise. Lead with the number. Use markdown bullet points...",
       "openai": "Summarize the data clearly. Use markdown formatting...",
       "ollama": "Keep your answer brief. Start with the key metric...",
   }
   ```

2. **Narrative quality post-check**: Validate that the narrative contains the actual data values from the query result before returning:
   ```python
   if result_value and result_value not in narrative:
       warnings.append("Narrative may not reflect query results accurately.")
   ```

3. **Length constraints**: Add `max_tokens` tuning per provider to prevent verbose responses:
   ```python
   NARRATOR_MAX_TOKENS = {"anthropic": 512, "openai": 1024, "ollama": 512}
   ```

4. **Fix GAP 23 first**: Reducing audit warning noise will likely improve Anthropic's narrative score by 5-8pp.

#### Acceptance Criteria

- [ ] Anthropic narrative score improves to >95% on benchmark re-run
- [ ] All providers achieve >94% narrative score
- [ ] Narrative consistently includes the actual numeric answer
- [ ] Provider-specific narrator prompts are configurable

---

### GAP 27 — OpenAI Latency Variance

**Severity**: MEDIUM
**Phase**: F (Benchmark-Identified)
**Depends on**: None

#### Problem

OpenAI exhibits extreme latency variance making response times unpredictable:

| Metric | OpenAI | Anthropic | Local |
|--------|--------|-----------|-------|
| Median | 20,042ms | 9,216ms | 12,125ms |
| Min | 11,919ms | 7,688ms | 10,104ms |
| Max | **85,686ms** | 12,587ms | 15,648ms |
| P95 | **49,175ms** | 12,398ms | 15,451ms |

OpenAI's max latency (85.7s) is 7x its median and nearly hitting typical HTTP timeouts. The P95 (49.2s) means 1 in 20 queries takes almost a minute.

#### Implementation Tasks

1. **Adaptive timeout**: Set per-provider timeouts based on observed P95:
   ```python
   PROVIDER_TIMEOUTS = {
       "anthropic": 30,  # P95 ~12s, 2.5x margin
       "openai": 60,     # P95 ~49s, 1.2x margin
       "ollama": 30,     # P95 ~15s, 2x margin
   }
   ```

2. **Retry with exponential backoff**: On timeout, retry once with a simpler prompt:
   ```python
   try:
       response = call_llm(messages, timeout=PROVIDER_TIMEOUTS[provider])
   except TimeoutError:
       # Retry with shorter prompt, lower max_tokens
       response = call_llm(simplified_messages, timeout=90)
   ```

3. **Latency circuit breaker**: If a provider's rolling average latency exceeds 30s, temporarily demote it in auto-mode selection.

4. **Model tier fallback**: If gpt-4o times out, fall back to gpt-4o-mini for that call.

#### Acceptance Criteria

- [ ] OpenAI P95 latency < 30s after optimizations
- [ ] No queries time out (currently 85s max)
- [ ] Auto mode deprioritizes providers with high recent latency
- [ ] Timeout/retry behavior is logged in agent trace

---

### GAP 28 — Model Version Health Management

**Severity**: HIGH
**Phase**: F (Benchmark-Identified)
**Depends on**: None

#### Problem

During the first benchmark run, all 18 Anthropic queries silently fell back to deterministic because the configured model ID (`claude-3-5-haiku-20241022`) was deprecated on Feb 19, 2026 — the day before the benchmark. Every LLM call raised `NotFoundError`, and the pipeline's fallback mechanism silently used deterministic mode. The user received deterministic-quality results while believing they were running Anthropic mode.

This is a **supply chain risk**: model deprecation happens without warning and can degrade the entire system overnight.

#### Implementation Tasks

1. **Model health check at startup**: On server start, verify that configured models respond:
   ```python
   async def _verify_model_health():
       for provider, models in DEFAULT_MODELS.items():
           for role, model_id in models.items():
               try:
                   call_llm([{"role": "user", "content": "ping"}],
                            role=role, provider=provider, model=model_id, max_tokens=1)
               except Exception as e:
                   logger.error(f"Model {model_id} ({provider}/{role}) is unavailable: {e}")
                   DEGRADED_MODELS.add((provider, role))
   ```

2. **Model alias system**: Use version-agnostic aliases that auto-resolve to latest:
   ```python
   MODEL_ALIASES = {
       "claude-haiku-latest": "claude-haiku-4-5-20251001",
       "claude-sonnet-latest": "claude-sonnet-4-6",
       "gpt-4o-latest": "gpt-4o",
   }
   ```

3. **Deprecation detection**: Check response headers or error messages for deprecation notices and surface as warnings:
   ```python
   if "deprecated" in str(error).lower() or "not found" in str(error).lower():
       warnings.append(f"Model {model_id} may be deprecated. Consider updating.")
   ```

4. **Fallback model chain**: Define per-provider fallback models:
   ```python
   MODEL_FALLBACKS = {
       "anthropic": {
           "planner": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
           "narrator": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
       },
   }
   ```

5. **Health status endpoint**: Add `/api/assistant/model-health` that reports model availability.

#### Acceptance Criteria

- [ ] Server startup checks model availability and logs degraded models
- [ ] Model deprecation produces an explicit warning, not silent fallback
- [ ] Fallback chain tries alternative models before falling back to deterministic
- [ ] `/api/assistant/model-health` endpoint reports per-model status
- [ ] Model versions are configurable without code changes (env vars or config file)

---

### GAP 29 — Intelligent Mode Selection

**Severity**: HIGH
**Phase**: F (Benchmark-Identified)
**Depends on**: GAP 22 (latency optimization), GAP 28 (model health)

#### Problem

The benchmark reveals that deterministic mode has the highest composite score (96.32%) because its near-zero latency outweighs the 5.56pp correctness advantage of LLM modes. The `auto` mode currently just cascades by provider availability (ollama → openai → anthropic → deterministic) without considering query characteristics.

Intelligent mode selection should route simple queries (where deterministic gets 100% correct) to deterministic mode for speed, and only invoke LLM for queries where it adds value — complex analytics, ambiguous intent, multi-domain questions.

#### Implementation Tasks

1. **Query complexity classifier**: Score query complexity before choosing mode:
   ```python
   def _classify_query_complexity(goal: str) -> str:
       """Returns 'simple', 'moderate', or 'complex'."""
       complex_signals = ["compare", "trend", "why", "behavior", "correlation",
                          "percentile", "growth", "relationship"]
       moderate_signals = ["by", "split", "group", "breakdown", "per"]

       if any(s in goal.lower() for s in complex_signals):
           return "complex"
       elif any(s in goal.lower() for s in moderate_signals):
           return "moderate"
       return "simple"
   ```

2. **Complexity-aware routing in auto mode**:
   ```python
   def _resolve_runtime_auto(goal, providers):
       complexity = _classify_query_complexity(goal)
       if complexity == "simple":
           return deterministic  # 94-100% correct, 130ms
       elif complexity == "moderate":
           return best_available_llm  # LLM helps with grouping/filtering
       else:
           return best_available_llm  # LLM critical for complex queries
   ```

3. **Provider ranking by category**: Use benchmark results to rank providers per query category:
   ```python
   PROVIDER_STRENGTHS = {
       "simple_counts": "deterministic",     # 100% correct, fastest
       "aggregations": "anthropic",          # 100% vs deterministic 100%
       "boolean_filters": "openai",          # 100% vs deterministic 67%
       "grouping": "deterministic",          # all 100%, fastest
       "time_filters": "deterministic",      # all 100%, fastest
       "complex": "anthropic",              # 100%, best latency of LLMs
   }
   ```

4. **Learning from results**: Track per-query accuracy by mode and adjust routing weights over time.

#### Acceptance Criteria

- [ ] Auto mode routes simple queries to deterministic (no LLM overhead)
- [ ] Auto mode routes complex/ambiguous queries to best available LLM
- [ ] Composite score for auto mode exceeds both deterministic and fixed-LLM modes
- [ ] Query complexity classification is logged in agent trace
- [ ] Routing decisions are explainable in `runtime.reason`

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
| **22** | F | 3-4 days | MEDIUM — async changes to pipeline flow | Feature-flag parallel execution; measure latency before/after; fallback to serial |
| **23** | F | 1-2 days | LOW — filtering is additive, keeps full warnings in trace | Validate against benchmark; ensure no real warnings are suppressed |
| **24** | F | 2-3 days | MEDIUM — prompt changes may affect other query types | A/B test with full benchmark suite; validate no regressions |
| **25** | F | 1 day | LOW — additive validation and warning | Does not change query logic; only adds user-facing feedback |
| **26** | F | 2-3 days | LOW — prompt tuning per provider | Test with benchmark suite; narrative scoring automated |
| **27** | F | 1-2 days | LOW — timeout/retry is standard pattern | Circuit breaker prevents cascading failures |
| **28** | F | 2-3 days | MEDIUM — startup health check adds latency | Cache health results; async health check; configurable skip |
| **29** | F | 3-4 days | HIGH — routing changes affect all auto-mode queries | A/B test against fixed modes; measure composite uplift |
| **30** | G | 1-2 days | HIGH — changes LLM intake classification globally | Synonym tests; deterministic mode regression suite |
| **31** | G | 2-3 days | MEDIUM — LLM SQL must be validated and probed | Guardrail validation + execute_probe before use |
| **32** | G | 3-4 days | HIGH — multi-part changes run() control flow | Max 4 sub-questions; full governance on each; fallback to single |
| **33** | G | 0.5-1 day | LOW — only triggers on failure path | Single retry; validated SQL only; zero cost on success |
| **34** | G | 1-2 days | LOW — only triggers on audit_score < 0.5 | Catalog-validated suggestions; existing score comparison kept |

**Total estimated effort**: 30-40 days (Phases A-E) + 16-22 days (Phase F) + 8-12 days (Phase G) = ~54-74 days

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
1. **Phase A** (1 day): Be honest with the user — tell them when LLM fails — **DONE**
2. **Phase B** (8-11 days): Put LLM where it matters — planning, auditing, orchestration — **DONE**
3. **Phase C** (8-11 days): Break the single-table constraint — multi-domain + JOINs — **PARTIAL** (registry-based JOINs, not full multi-table engine)
4. **Phase D** (8-11 days): Make agents real — specialists that influence, blackboard that communicates — **DONE**
5. **Phase E** (5-7 days): Go beyond SELECT COUNT — analytical patterns for real questions — **PARTIAL** (trend, percentile, ranked added; CTEs, subqueries, window functions still needed)
6. **Phase F** (16-22 days): Benchmark-driven optimization — latency, accuracy, model management, intelligent routing — **DONE** (100% correctness all modes, warnings 47→8, composites all improved)
7. **Phase G** (8-12 days): Smart LLM mode — genuine intelligence for intent, SQL, multi-part, recovery, audit — **DONE** (5 new capabilities, deterministic untouched)
8. **Phase H** (7-10 days): Domain intelligence — structured knowledge base, specialist directives that modify SQL, decision transparency, provider completeness — **DONE** (7 gaps, 49 tests, 0 failures)
