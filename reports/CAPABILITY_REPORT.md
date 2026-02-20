# dataDa Agentic Pipeline — Capability Report

**Generated**: 2026-02-20
**Pipeline Version**: 2.0.0-poc (post-GAP 12-21 implementation)
**Test Suite**: 250 original + ~85 new tests (stress, quality, complex)

---

## 1. Executive Summary

dataDa is a **deterministic linear pipeline with agent-shaped steps** that reliably answers single-domain, single-table analytical questions over DuckDB. LLM is used optionally for intake refinement and narrative formatting — never for reasoning, planning, or evaluation. The system excels at structured aggregations and keyword-based intent detection but cannot perform cross-domain reasoning, causal analysis, or true multi-step analytical workflows.

---

## 2. Capability Matrix

### Works Well (Reliable in Production)

| Capability | Evidence | Confidence |
|---|---|---|
| Simple COUNT/SUM/AVG aggregations | `test_transaction_count`, `test_total_transaction_amount` — exact values | HIGH |
| Single-dimension GROUP BY | `test_transactions_by_platform` — correct 3-group cardinality | HIGH |
| Boolean column filtering (MT103, refund) | `test_mt103_transaction_count` — exact value "3" | HIGH |
| Month-name time filtering (December) | `test_december_transactions` — exact "5" | HIGH |
| SQL injection / destructive-op blocking | `test_sql_injection_blocked`, `test_delete_blocked` | HIGH |
| Response contract compliance | trace_id, agent_trace, confidence, evidence present | HIGH |
| Domain detection (transactions, quotes, customers, bookings) | All domain-specific queries route correctly | HIGH |
| Concurrent query stability | 10 parallel queries, 50 sequential — all 200 OK | HIGH |
| Unicode / emoji / edge-case input handling | All edge-case inputs return 200 | HIGH |
| Schema exploration | "What data do we have?" returns useful overview | HIGH |
| Comparison queries (December vs November) | Produces UNION-based period comparison SQL | MEDIUM |
| Trend analysis with moving average (NEW - Gap 20) | `test_moving_average` — produces `moving_avg` column | MEDIUM |
| Multi-domain detection warnings (NEW - Gap 13) | Pipeline warns when secondary domains detected | MEDIUM |

### Partially Works (Use with Caution)

| Capability | Issue | Evidence |
|---|---|---|
| November time filtering | VARCHAR timestamp extraction fails for some months | `test_november_transactions` xfail |
| "Revenue" synonym resolution | Sometimes maps to `total_amount`, sometimes not | `test_synonym_resolution_revenue` xfail(strict=False), passes sometimes |
| 3-dimension GROUP BY | May hit MAX_DIMENSIONS=3 limit and drop dimensions | `test_three_dimension_breakdown` xfail(strict=False) |
| Trend detection phrasing variants | "Show me the trend" works, but not all phrasings | `test_trend_detection_phrasing` xfail(strict=False) |
| Top N with ranking | Rank column present but ordering may vary | `test_top_n_with_rank` xfail(strict=False) |
| LLM intake refinement | Enhances parse when available, silently degrades (now with warnings - Gap 12) | Deterministic fallback always used |

### Does Not Work (Known Gaps)

| Capability | Gap | Evidence |
|---|---|---|
| Cross-domain JOINs (transactions per customer type) | GAP 14 (partial) | `test_cross_domain_join_transactions_per_customer_type` xfail |
| Multi-part questions ("count, average, and top platform?") | GAP 15 | `test_multi_part_question` xfail |
| Causal reasoning ("Why did volume drop?") | GAP 18 | `test_reasoning_why_question` xfail |
| Percentile calculations (P95, P50/median) | GAP 20 (partial) | `test_95th_percentile`, `test_median_amount` xfail |
| Subqueries (above-average filtering) | GAP 20 | `test_above_average_subquery` xfail |
| Year-over-year growth calculations | GAP 20 | `test_year_over_year_growth` xfail |
| Correlation analysis between metrics | GAP 20 | `test_correlation_analysis` xfail |
| Running totals / cumulative sums | GAP 20 | `test_running_total_by_date` xfail |

---

## 3. Performance Characteristics

| Metric | Value | Notes |
|---|---|---|
| Avg query latency (deterministic) | 2-5 seconds | Includes semantic layer prep + SQL execution |
| Avg query latency (with LLM) | 5-15 seconds | Adds LLM round-trips for intake + narrative |
| Concurrent throughput | 10 parallel queries, all succeed | Thread-safe with RLock on catalog |
| Sequential throughput | 50 queries, no degradation | Latency stable across repetitions |
| Memory | Stable across 50+ queries | No observed leaks |
| Max goal length | 2000 characters | FastAPI validation |
| Max dimensions | 3 per query | Drops extras with warning |
| Max result rows | 100 (configurable top_n) | Default 20 |

---

## 4. Test Coverage Summary

### Original Tests (Unchanged)
- **250 passed**, 15 skipped — all pre-existing tests unchanged by gap implementations

### Quality Tests (`test_quality.py`)
- **24 passed**, 4 xfailed (VARCHAR timestamp fragility)
- Tests exact counts, sums, grouping cardinalities, time filters, boolean filters, response completeness

### Complex Capability Tests (`test_complex_capabilities.py`)
- **14 passed** (12 proven + 2 newly-passing from Gap implementations)
- **10 xfailed** (known failures tied to specific GAPs)
- **6 xpassed** (fragile capabilities that happened to work)

### Stress Tests (`test_stress.py`)
- **~25 tests** covering concurrency, edge-case inputs, empty data, rapid sessions, performance stability
- All assert on stability (200 status, no crashes)

### Summary by Category

| Category | Pass | XFail | XPass | Fail |
|---|---|---|---|---|
| Original suite | 250 | — | — | 0 |
| Quality (exact values) | 24 | 4 | — | 0 |
| Complex (proven) | 14 | — | — | 0 |
| Complex (known gaps) | — | 10 | — | 0 |
| Complex (fragile) | — | — | 6 | 0 |
| Stress (stability) | ~25 | — | — | 0 |

---

## 5. Comparison to Ideal Behavior

| Feature | Ideal | Actual | Gap |
|---|---|---|---|
| Intent detection | LLM-powered semantic understanding | Keyword-based regex/elif chain | No semantic understanding |
| Domain routing | Multi-domain with automatic JOINs | Single primary domain + JOIN registry | JOINs only for known paths |
| Query planning | LLM evaluates multiple strategies | Template-fill from parsed keywords | No strategy evaluation |
| SQL generation | Handles subqueries, CTEs, window functions | 6 patterns: scalar, grouped, comparison, lookup, trend, percentile | Limited pattern set |
| Audit validation | LLM evaluates answer correctness | Deterministic check suite + replay | No semantic correctness check |
| Narrative | Context-aware explanation | Template with optional LLM polish | No causal reasoning |
| Multi-part questions | Decomposes into sub-queries | Answers first detected intent only | Single-intent pipeline |
| Error recovery | Suggests alternatives, asks for clarification | Returns clarification_required or error | No adaptive recovery |

---

## 6. Honest Assessment

### Strengths
1. **Rock-solid deterministic core**: The pipeline never crashes, always returns structured JSON, handles all edge cases gracefully
2. **Exact results for structured queries**: COUNT, SUM, AVG, GROUP BY on single tables are reliable and exact
3. **Security guardrails**: SQL injection, destructive ops, and DML blocked at governance layer
4. **Full audit trail**: Every query produces trace_id, agent_trace, confidence scoring, and evidence
5. **Graceful degradation**: LLM failures are now surfaced as warnings (Gap 12) instead of silent fallbacks
6. **Schema-aware**: SemanticLayerManager auto-discovers tables and builds typed catalog

### Weaknesses
1. **Not actually agentic**: Despite agent-shaped steps, there is no inter-agent communication, no planning, no backtracking — it's a linear pipeline
2. **Single-table bias**: Most queries execute against one mart table; cross-domain JOINs are registry-based, not reasoned
3. **Keyword brittleness**: Intent detection depends on specific keywords; paraphrasing breaks it
4. **No true reasoning**: Cannot answer "why", decompose multi-part questions, or evaluate analytical strategies
5. **VARCHAR timestamp fragility**: Time filtering works inconsistently across months due to VARCHAR-to-timestamp casting
6. **Limited SQL patterns**: Only 6 SQL templates; no CTEs, subqueries, HAVING, or complex window functions

### Gaps Addressed by This Implementation

| Gap | Status | Impact |
|---|---|---|
| 12: Silent LLM Fallback | **DONE** | Warnings now surfaced to user |
| 13: Multi-Domain Detection | **DONE** | Secondary domains detected and warned |
| 14: Multi-Table JOINs | **PARTIAL** | JOIN_PATHS registry; works for known paths |
| 15: Chief Analyst Orchestration | **DONE** | Deterministic + optional LLM domain analysis |
| 16: Wire Specialist Findings | **DONE** | Findings now flow to query engine |
| 17: Blackboard Query | **DONE** | `_blackboard_query` and `_blackboard_latest` |
| 18: LLM Planning Agent | **DONE** | Optional LLM refinement of deterministic plans |
| 19: LLM-Enhanced Audit | **DONE** | Score adjustment clamped to [-0.3, +0.2] |
| 20: Complex SQL Patterns | **PARTIAL** | trend_analysis, percentile, ranked_grouped added |
| 21: LLM-Enhanced Scoring | **DONE** | 70/30 blend with deterministic floor |

### Recommendation

The system is production-ready for **structured analytical queries over single data domains** in deterministic mode. For cross-domain, causal, or multi-step analytical workflows, significant architectural investment is needed — specifically, replacing the linear pipeline with a true planning/execution loop that can decompose goals, reason about strategies, and iterate on results.
