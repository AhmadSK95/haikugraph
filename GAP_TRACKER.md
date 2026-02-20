# dataDa Agentic Analytics -- Gap Tracker

> **Created**: 2026-02-20
> **Status**: Phases 1, 2 & 3 implemented -- 250 tests passing
> **Primary file**: `src/haikugraph/poc/agentic_team.py` (~5150 lines)
> **Secondary file**: `src/haikugraph/agents/contracts.py` (441 lines)

---

## Table of Contents

1. [Symptom Summary](#symptom-summary)
2. [Dependency Graph](#dependency-graph)
3. [Phase Roadmap](#phase-roadmap)
4. [Gap Details (1-11)](#gap-details)
5. [Effort & Risk Matrix](#effort--risk-matrix)

---

## Symptom Summary

Three test queries expose the same cluster of defects:

| Query | Expected | Actual | Root Gaps |
|-------|----------|--------|-----------|
| "transactions split by platform and month" | 2-dim breakdown | Correct | -- |
| "transactions split by platform and month and region" | 3-dim breakdown incl. region | **Identical to query 1** -- region silently dropped, 3rd dim truncated | 1, 2, 4, 8 |
| "transaction trends by specific regions" | Region-wise time series | **Single total count** -- "trend" forces `__month__`, region unrecognized | 1, 3, 4, 8 |

**Core diagnosis**: Intelligence lives in hardcoded rules (`dim_candidates`, `_enforce_intake_consistency`) rather than in the agents' ability to reason about the schema. The agents are executors of a deterministic pipeline, not yet an autonomous reasoning team.

---

## Dependency Graph

```
                        +-----------+
                        |  GAP  8   |  warnings field (foundation)
                        | contracts |
                        +-----+-----+
                              |
              all other gaps benefit
              |               |              |
        +-----v-----+  +-----v-----+  +-----v------+
        |  GAP  2   |  |  GAP  3   |  |  GAP  4    |
        | dim cap   |  | trend kw  |  | cross-JOIN |
        +-----+-----+  +-----------+  +-----+------+
              |                              |
              |         +--------------------+
              |         |
        +-----v---------v--+
        |      GAP  1      |
        | region mapping   |
        +------------------+

        +-----+-----+
        |  GAP  5   |  dynamic dim discovery
        +-----+-----+
              |
      +-------+-------+
      |               |
+-----v-----+  +-----v------+
|  GAP  6   |  |  GAP  10   |
| autonomy  |  | LLM overrides|
| dims      |  +-------------+
+-----------+

        +-----+-----+       +-----+-----+       +-----+-----+
        |  GAP  7   |       |  GAP  9   |       |  GAP  11  |
        | specialists|       | lateral   |       | contrib   |
        | (indep.)  |       | comms     |       | map       |
        +-----------+       +-----------+       +-----------+
```

**Explicit dependency rules**:
- GAP 4 (cross-table JOIN) **must complete before** GAP 1 (region mapping)
- GAP 5 (dynamic discovery) **must complete before** GAP 6 and GAP 10
- GAP 8 (warnings) **should complete first** -- all other gaps emit warnings through it
- GAP 2 (dim cap) **should be done alongside** GAP 1
- GAP 3 (trend keyword) is **independent**
- GAP 7, 9, 11 are **independent** of each other

---

## Phase Roadmap

### Phase 1 -- Foundation  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 8 | Silent-drop warnings field | 1 day | DONE |

### Phase 2 -- Core Fixes  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 4 | Cross-table JOIN capability | 2 days | DONE |
| 2 | Dimension cap removal | 0.5 day | DONE |
| 3 | Trend keyword decoupling | 0.5 day | DONE |

### Phase 3 -- Dimension Intelligence  DONE
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 1 | Region / geography mapping | 1.5 days | DONE (completed in Phase 2) |
| 5 | Dynamic dimension discovery | 2 days | DONE |
| 10 | LLM override guard-rails | 1 day | DONE |

### Phase 4 -- Advanced
| Gap | Title | Est. Effort | Status |
|-----|-------|-------------|--------|
| 6 | Autonomy dimension variants | 1.5 days | DONE |
| 7 | Real specialist agents | 2 days | DONE |
| 9 | Lateral agent communication | 3-5 days | DONE (post-audit feedback loop) |
| 11 | Agent contribution map | 2 days | DONE |

---

## Gap Details

---

### GAP 1 -- "region" Missing from Dimension Vocabulary

**Severity**: CRITICAL
**Phase**: 3 (Dimension Intelligence)
**Depends on**: GAP 4 (cross-table JOIN), GAP 2 (dimension cap)

#### Problem

User queries containing "region", "country", or "city" are silently ignored. The `dim_candidates` dictionary only maps 6 keywords for transactions: `platform`, `state`, `status`, `flow`, `customer`, `month`. When a user asks "split by region", the keyword loop never matches, so the dimension is silently dropped. Queries 1 and 2 from the symptom table produce identical SQL.

Geographic data (`address_country`, `address_state`, `address_city`) exists only in `datada_dim_customers`, not in the transactions mart. Even adding the keyword to `dim_candidates` would fail without a JOIN path (GAP 4).

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 2919-2951 | `dim_candidates` dict -- no `"region"`, `"country"`, `"city"` keys for `"transactions"` |
| `agentic_team.py` | 3064-3135 | `_enforce_intake_consistency` -- lines 3124-3128 hard-code only `"platform"` and `"state"` as recognized transaction dimensions |
| `agentic_team.py` | 241-325 | `_build_transactions_view` -- no geographic columns in view |
| `agentic_team.py` | 601-606 | `dim_map["datada_mart_transactions"]` -- only `["platform_name", "state", "payment_status", "txn_flow"]` |

#### Current Code (dim_candidates for transactions)

```python
# agentic_team.py:2919
"transactions": {
    "platform": "platform_name",
    "state": "state",
    "status": "payment_status",
    "flow": "txn_flow",
    "customer": "customer_id",
    "month": "__month__",
},
```

#### Implementation Tasks

1. **After GAP 4 is done** (transactions view now includes geographic cols via JOIN):
   - Add to `dim_candidates["transactions"]` (line ~2919):
     ```python
     "region": "address_country",
     "country": "address_country",
     "city": "address_city",
     ```
   - Add to `dim_map["datada_mart_transactions"]` (line ~601):
     ```python
     "datada_mart_transactions": [
         "platform_name", "state", "payment_status", "txn_flow",
         "address_country", "address_state",
     ],
     ```

2. **Update `_enforce_intake_consistency`** (lines 3124-3128):
   - Add recognition for `"region"`, `"country"`, `"city"` keywords alongside `"platform"` and `"state"`:
     ```python
     if "region" in lower or "country" in lower:
         if dim_signal and "address_country" not in dims:
             dims.append("address_country")
     if "city" in lower:
         if dim_signal and "address_city" not in dims:
             dims.append("address_city")
     ```

3. **Emit warning** (via GAP 8 infrastructure) when a dimension keyword has no mapping.

#### Acceptance Criteria

- [ ] Query "transactions split by platform and month and region" returns 3-column GROUP BY including `address_country`
- [ ] Query "transaction trends by specific regions" returns region-wise time-series, not a single total
- [ ] `dim_candidates["transactions"]` includes `"region"`, `"country"`, `"city"` keys
- [ ] `_enforce_intake_consistency` recognizes region/country/city for transactions domain
- [ ] No silent drops -- if region still can't resolve, a warning is surfaced

---

### GAP 2 -- Hard 2-Dimension Cap

**Severity**: HIGH
**Phase**: 2 (Core Fixes)
**Depends on**: GAP 8 (for warning when cap is hit)

#### Problem

The expression `dimensions = dimensions[:2]` is enforced at **6 separate locations**. A 3-dimension query like "by platform, month, and region" silently drops the 3rd dimension with no warning.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 2978 | `_intake_deterministic`: `dimensions = dimensions[:2]` |
| `agentic_team.py` | 3129 | `_enforce_intake_consistency`: `dims = dims[:2]` |
| `agentic_team.py` | 3509 | `_planning_agent`: `dimensions = dimensions[:2]` |
| `agentic_team.py` | 3868 | `_generate_candidate_plans`: `merged_dims = (existing_dims + enrich_dims)[:2]` |
| `agentic_team.py` | 3926 | `_build_plan_variant`: `plan["dimensions"] = valid_dims[:2]` |
| `agentic_team.py` | 4849 | `_sanitize_intake_payload`: `cleaned["dimensions"] = dims[:2]` |
| `agentic_team.py` | 3174 | `_apply_memory_hints`: `merged["dimensions"] = past_dims[:2]` |

#### Implementation Tasks

1. **Define a configurable constant** near the top of the file:
   ```python
   MAX_DIMENSIONS = 3  # was 2
   ```

2. **Replace all 7 occurrences** of `[:2]` dimension slicing with `[:MAX_DIMENSIONS]`:
   - Line 2978: `dimensions = dimensions[:MAX_DIMENSIONS]`
   - Line 3129: `dims = dims[:MAX_DIMENSIONS]`
   - Line 3174: `merged["dimensions"] = past_dims[:MAX_DIMENSIONS]`
   - Line 3509: `dimensions = dimensions[:MAX_DIMENSIONS]`
   - Line 3868: `merged_dims = (existing_dims + enrich_dims)[:MAX_DIMENSIONS]`
   - Line 3926: `plan["dimensions"] = valid_dims[:MAX_DIMENSIONS]`
   - Line 4849: `cleaned["dimensions"] = dims[:MAX_DIMENSIONS]`

3. **Update `_query_engine_agent`** (lines 4247-4267) to handle 3+ dimensions in SELECT/GROUP BY generation. Currently iterates `dimensions` list, so should work if cap is raised, but verify:
   - Each dim adds a SELECT part and a GROUP BY part
   - `__month__` handled specially as `DATE_TRUNC('month', {time_col}) AS month_bucket`

4. **Add warning** when cap is hit: "Requested {n} dimensions; limited to {MAX_DIMENSIONS}."

#### Acceptance Criteria

- [ ] 3-dimension queries produce correct 3-column GROUP BY SQL
- [ ] `MAX_DIMENSIONS` constant exists and is referenced by all 7 cap locations
- [ ] When cap is exceeded, a warning is emitted (not silent)
- [ ] Existing 1- and 2-dimension queries are unaffected (regression check)

---

### GAP 3 -- "trend" Keyword Forces `__month__` Grouping

**Severity**: HIGH
**Phase**: 2 (Core Fixes)
**Depends on**: None (independent)

#### Problem

At line 2958, the regex `r"\b(by month|monthly|month[\s-]?wise|trend)\b"` treats "trend" identically to "monthly". This forces `__month__` as a dimension regardless of what the user actually asked. "transaction trends by region" becomes a month-only query because "trend" triggers `__month__` and "region" is unrecognized.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 2958-2960 | `has_month_signal` regex includes `trend` |
| `agentic_team.py` | 2963-2964 | `__month__` appended when `has_month_signal` is True |
| `agentic_team.py` | 3122 | `_enforce_intake_consistency` repeats the pattern |
| `agentic_team.py` | 3861 | `_generate_candidate_plans` repeats the pattern |
| `agentic_team.py` | 3967 | `_candidate_score_breakdown` awards bonus for `trend` + `__month__` |

#### Current Code

```python
# agentic_team.py:2958
has_month_signal = bool(
    re.search(r"\b(by month|monthly|month[\s-]?wise|trend)\b", lower)
    or ("month" in lower and dim_signal)
    or ("split my month" in lower)
)
```

#### Implementation Tasks

1. **Separate "trend" from month detection** in `_intake_deterministic` (line 2958):
   ```python
   has_month_signal = bool(
       re.search(r"\b(by month|monthly|month[\s-]?wise)\b", lower)
       or ("month" in lower and dim_signal)
       or ("split my month" in lower)
   )
   has_trend_signal = bool(re.search(r"\btrend\b", lower))
   ```

2. **Add `__month__` for trend only when no other dimension is explicit** (after line 2964):
   ```python
   if has_month_signal:
       if "__month__" not in dimensions:
           dimensions.append("__month__")
   elif has_trend_signal and not dimensions:
       # "trend" implies time-series, but only if user didn't specify another dim
       dimensions.append("__month__")
   ```

3. **Update `_enforce_intake_consistency`** (line 3122) with the same logic.

4. **Update `_generate_candidate_plans`** (line 3861): remove `trend` from the month regex.

5. **Update `_candidate_score_breakdown`** (line 3967): don't award month-alignment bonus purely because "trend" matched:
   ```python
   # Only award bonus if user explicitly asked for monthly, not just "trend"
   if re.search(r"\b(by month|month wise|monthly)\b", lower) and "__month__" in dims:
       dimension_bonus += 0.03
   ```

#### Acceptance Criteria

- [ ] "transaction trends by region" does NOT force `__month__` when region is the explicit dimension
- [ ] "monthly transaction trends" still produces month-based grouping
- [ ] "show me trends" (no explicit dimension) defaults to `__month__`
- [ ] Candidate scoring doesn't inflate scores for trend+month alignment

---

### GAP 4 -- No Cross-Table JOIN Capability

**Severity**: HIGH
**Phase**: 2 (Core Fixes)
**Depends on**: GAP 8 (for warnings)

#### Problem

Every query runs against a single mart view. The `_query_engine_agent` builds SQL from a single `plan["table"]`. Geographic data (`address_country`, `address_state`) lives in `datada_dim_customers` but cannot be accessed when querying `datada_mart_transactions`. There is no JOIN mechanism.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 241-325 | `_build_transactions_view` -- no JOIN to customers |
| `agentic_team.py` | 4225-4273 | `_query_engine_agent` -- `FROM {table}` uses single table |
| `agentic_team.py` | 3464-3484 | `_semantic_retrieval_agent` -- maps domain to single table |
| `agentic_team.py` | 601-606 | `dim_map` -- transactions has no geographic dims |

#### Current Code (view builder, abbreviated)

```python
# agentic_team.py:241-325 (_build_transactions_view)
# Joins raw_transactions with itself but NOT with customers
# Columns: transaction_key, transaction_id, customer_id, payee_id,
#   platform_name, state, txn_flow, payment_status, ...
# No address_country, address_state, address_city
```

#### Implementation Tasks

**Recommended approach: Option A -- enrich mart views at build time** (simpler, no query-engine changes).

1. **Modify `_build_transactions_view`** (lines 241-325) to LEFT JOIN `datada_dim_customers`:
   - After the existing FROM/JOIN block, add:
     ```sql
     LEFT JOIN datada_dim_customers c
       ON t.customer_id = c.customer_key
     ```
   - Add to SELECT list:
     ```sql
     c.address_country,
     c.address_state,
     c.address_city
     ```
   - Ensure the JOIN uses the correct key columns (`customer_id` from transactions, `customer_key` from customers). Verify column names in `_build_customers_view` (lines 406-463).

2. **Update `dim_map`** (line 601-606) to include geographic columns:
   ```python
   "datada_mart_transactions": [
       "platform_name", "state", "payment_status", "txn_flow",
       "address_country", "address_state",
   ],
   ```

3. **Rebuild catalog** -- the `_build_catalog` method (line 523) iterates `dim_map` and fetches top-15 values. After adding the geographic dims, the catalog will auto-discover `address_country` and `address_state` values.

4. **Verify `_query_engine_agent`** (lines 4225-4273) handles the new columns correctly -- since they're now in the view, no changes should be needed to query generation.

5. **Alternative: Option B** (add a JOIN planner to `_query_engine_agent`):
   - More flexible but significantly more complex
   - Would require `_semantic_retrieval_agent` to return multiple tables
   - Would require `_planning_agent` to produce JOIN specifications
   - Defer to Phase 4+ if cross-mart queries beyond customer geography are needed

#### Acceptance Criteria

- [ ] `datada_mart_transactions` view includes `address_country`, `address_state`, `address_city` columns
- [ ] `catalog["dimension_values"]` includes geographic dimension values for transactions
- [ ] `SELECT * FROM datada_mart_transactions LIMIT 5` shows geographic columns populated
- [ ] Existing transaction queries are unaffected (LEFT JOIN means no row loss)
- [ ] JOIN key relationship is correct (no duplicated rows from 1:N)

---

### GAP 5 -- No Dynamic Dimension Discovery

**Severity**: MEDIUM
**Phase**: 3 (Dimension Intelligence)
**Depends on**: GAP 4 (geographic columns in catalog)

#### Problem

The `dim_candidates` dict (line 2919) is a static keyword-to-column mapping. If the schema changes (new columns added), the system cannot adapt. The catalog already stores `catalog["dimension_values"]` and `catalog["marts"][table]["columns"]` but this information is only used for value-filter matching (e.g., matching "B2C-APP" to `platform_name`), not for discovering which dimensions are available for GROUP BY.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 2919-2951 | `dim_candidates` -- static, manually maintained |
| `agentic_team.py` | 601-626 | `dim_map` + catalog building -- stores dimension columns and values but not used for intake |
| `agentic_team.py` | 596-599 | `catalog["marts"][mart]` -- stores `row_count` and `columns` per mart |
| `agentic_team.py` | 2953-2978 | Dimension keyword matching loop in `_intake_deterministic` |

#### Implementation Tasks

1. **Add a dynamic fallback after the static `dim_candidates` loop** (after line ~2978 in `_intake_deterministic`):
   ```python
   # After static dim_candidates matching, try dynamic discovery
   if catalog and unmatched_dim_keywords:
       table = domain_table.get(domain, "")
       available_cols = set(catalog.get("marts", {}).get(table, {}).get("columns", []))
       dim_value_cols = set(catalog.get("dimension_values", {}).keys())
       for keyword in unmatched_dim_keywords:
           # Substring match: "country" -> "address_country"
           for col in available_cols & dim_value_cols:
               if keyword in col or col in keyword:
                   dimensions.append(col)
                   break
   ```

2. **Track unmatched dimension keywords**: During the keyword loop (lines 2953-2978), collect keywords that matched `dim_signal` but had no entry in `dim_candidates`:
   ```python
   unmatched_dim_keywords = []
   for kw, col in dim_candidates.get(domain, {}).items():
       if kw in lower:
           dimensions.append(col)
   # After loop, find dimension-like words not yet matched
   ```

3. **Expose catalog to `_intake_deterministic`**: Verify that the catalog is accessible. Currently built in `SemanticLayerManager.prepare()` and returned to `run()`. Pass it to `_intake_deterministic` or store as `self._catalog`.

4. **Emit warning for fuzzy matches**: "Dimension 'country' dynamically resolved to column 'address_country'."

#### Acceptance Criteria

- [ ] A dimension keyword not in `dim_candidates` but matching a catalog column is auto-resolved
- [ ] Fuzzy/substring matching works: "country" -> `address_country`
- [ ] A warning is emitted for dynamically resolved dimensions
- [ ] Static `dim_candidates` matches still take priority (no regression)
- [ ] Performance: catalog lookup adds < 1ms to intake

---

### GAP 6 -- Autonomy Agent Cannot Fix Dimension Errors

**Severity**: MEDIUM
**Phase**: 4 (Advanced)
**Depends on**: GAP 5 (dynamic dim discovery)

#### Problem

The `_generate_candidate_plans` method (lines 3742-3889) generates alternative plans by switching tables and metrics, but **never by varying dimensions**. The `_build_plan_variant` method (line 3891) accepts a `dimensions` parameter, but no caller passes dimension variants. If the base plan has wrong dimensions, all candidates inherit the same dimensions.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 3742-3889 | `_generate_candidate_plans` -- only varies table and metric |
| `agentic_team.py` | 3891-3938 | `_build_plan_variant` -- accepts `dimensions` param, but callers don't use it |
| `agentic_team.py` | 3949-3996 | `_candidate_score_breakdown` -- no dimension-quality scoring |
| `agentic_team.py` | 4356-4567 | `_audit_agent` -- `goal_term_misses` may contain dimension names |

#### Current Code

```python
# agentic_team.py:3891
def _build_plan_variant(self, base_plan, catalog, *, table, metric,
                        dimensions=None, time_filter=..., reason):
    # dimensions parameter exists but callers only pass table + metric
```

#### Implementation Tasks

1. **Add dimension variant generation** in `_generate_candidate_plans` (after line ~3839):
   ```python
   # If audit detected goal_term_misses that look like dimensions,
   # generate candidates with those dimensions added
   if audit and audit.get("goal_term_misses"):
       for missed_term in audit["goal_term_misses"]:
           # Use dynamic discovery (GAP 5) to resolve term to column
           resolved_col = self._resolve_dim_dynamically(missed_term, catalog, table)
           if resolved_col:
               new_dims = base_plan.get("dimensions", []) + [resolved_col]
               candidates.append(self._build_plan_variant(
                   base_plan, catalog,
                   table=table, metric=metric,
                   dimensions=new_dims,
                   reason=f"Add missed dimension: {missed_term} -> {resolved_col}",
               ))
   ```

2. **Add dimension-quality scoring** in `_candidate_score_breakdown` (line ~3967):
   ```python
   # Penalize candidates that don't cover user-requested dimensions
   requested_dim_terms = [t for t in goal_term_misses if is_dimension_like(t)]
   covered = sum(1 for t in requested_dim_terms if any(t in d for d in dims))
   if requested_dim_terms:
       dimension_coverage = covered / len(requested_dim_terms)
       dimension_bonus += 0.05 * dimension_coverage
   ```

3. **Wire audit `goal_term_misses`** into candidate generation: Ensure the audit result dict is passed to `_generate_candidate_plans`.

#### Acceptance Criteria

- [ ] When audit detects "region" as a goal_term_miss, a candidate plan with `address_country` is generated
- [ ] Dimension variants are scored and can win over the base plan
- [ ] `_build_plan_variant` is called with `dimensions` parameter by at least one caller
- [ ] Existing table/metric variation still works (regression check)

---

### GAP 7 -- Specialist Agents Are Stubs

**Severity**: MEDIUM
**Phase**: 4 (Advanced)
**Depends on**: GAP 5 (beneficial), GAP 8 (for warnings)

#### Problem

The four specialist agents (`_transactions_specialist`, `_customer_specialist`, `_revenue_specialist`, `_risk_specialist`) return 1-2 line static notes. The `specialist_findings` variable is explicitly discarded at line 4230:
```python
del specialist_findings  # reserved for richer planner integration.
```

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 4182-4188 | `_transactions_specialist` -- stub |
| `agentic_team.py` | 4190-4196 | `_customer_specialist` -- stub |
| `agentic_team.py` | 4198-4206 | `_revenue_specialist` -- stub |
| `agentic_team.py` | 4208-4216 | `_risk_specialist` -- stub |
| `agentic_team.py` | 4230 | `del specialist_findings` -- findings discarded |

#### Implementation Tasks

1. **Implement dimension validation** in each specialist (start with `_transactions_specialist`):
   ```python
   def _transactions_specialist(self, plan, catalog):
       findings = {"notes": [], "warnings": [], "dimension_suggestions": []}
       # Check if requested dimensions exist in schema
       for dim in plan.get("dimensions", []):
           if dim != "__month__" and dim not in catalog["marts"][plan["table"]]["columns"]:
               findings["warnings"].append(
                   f"Dimension '{dim}' not found in {plan['table']}"
               )
               # Suggest alternatives from catalog
               ...
       return findings
   ```

2. **Remove `del specialist_findings`** (line 4230) and integrate findings:
   - Pass `specialist_findings` to `_query_engine_agent` for SQL enrichment
   - Append `specialist_findings["warnings"]` to the response warnings list (GAP 8)
   - Use `specialist_findings["dimension_suggestions"]` in planning

3. **Implement metric validation**: Each specialist checks that the chosen metric makes sense for the domain.

4. **Implement cross-domain hints**: `_customer_specialist` can suggest JOINs when customer attributes are needed.

#### Acceptance Criteria

- [ ] `del specialist_findings` is removed
- [ ] Each specialist validates dimensions and metrics against the catalog
- [ ] Specialist warnings appear in the final response warnings
- [ ] Specialist suggestions influence SQL generation or planning
- [ ] Existing queries without specialist-relevant issues are unaffected

---

### GAP 8 -- No Silent-Drop Warnings

**Severity**: MEDIUM
**Phase**: 1 (Foundation)
**Depends on**: None (this is the foundation)

#### Problem

When `_planning_agent` drops a dimension (line 3507 -- dim not in `retrieval["columns"]`), or when `_intake_deterministic` cannot match a dimension keyword, the user receives no feedback. The dimension is silently removed from the query plan. The `AssistantQueryResponse` model has no `warnings` field.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `contracts.py` | 376-435 | `AssistantQueryResponse` -- no `warnings` field |
| `agentic_team.py` | 3500-3510 | `_planning_agent` -- silent dim drop |
| `agentic_team.py` | 2953-2978 | `_intake_deterministic` -- unrecognized keywords ignored |
| `agentic_team.py` | 4356-4567 | `_audit_agent` -- has internal `warnings: list[str] = []` (line 4363) but these don't surface to response |

#### Current Code (planning_agent dim validation)

```python
# agentic_team.py:3500-3510
for dim in raw_dims:
    if not isinstance(dim, str):
        continue
    if dim == "__month__":
        dimensions.append(dim)
        continue
    if dim in retrieval["columns"]:
        dimensions.append(dim)
    # ELSE: silently dropped -- no warning
```

#### Implementation Tasks

1. **Add `warnings` field to `AssistantQueryResponse`** in `contracts.py` (after line ~431):
   ```python
   warnings: list[str] = Field(
       default_factory=list,
       description="User-visible warnings about query interpretation.",
   )
   ```

2. **Initialize warnings list** in `run()` method (early in the pipeline):
   ```python
   warnings: list[str] = []
   ```

3. **Emit warnings in `_planning_agent`** (line 3507):
   ```python
   if dim in retrieval["columns"]:
       dimensions.append(dim)
   else:
       warnings.append(
           f"Dimension '{dim}' was requested but is not available in {retrieval['table']}. It was removed from the query."
       )
   ```

4. **Emit warnings in `_intake_deterministic`** when a dimension-like keyword is detected but has no mapping:
   ```python
   if dim_signal and not dimensions:
       warnings.append(
           "A grouping was requested but no recognized dimensions were found."
       )
   ```

5. **Collect audit warnings** -- the `_audit_agent` already builds a `warnings` list (line 4363). Merge it:
   ```python
   warnings.extend(audit_result.get("warnings", []))
   ```

6. **Include warnings in response** (in every `AssistantQueryResponse(...)` constructor call):
   ```python
   warnings=warnings,
   ```
   There are multiple constructors: lines ~1030, ~1097, ~1181, ~1351, ~1762.

7. **Surface in UI**: The frontend should render `response.warnings` (separate task, not tracked here).

#### Acceptance Criteria

- [ ] `AssistantQueryResponse` has a `warnings: list[str]` field
- [ ] Dropped dimensions produce a warning message
- [ ] Unrecognized dimension keywords produce a warning message
- [ ] Audit-agent warnings are propagated to the response
- [ ] All `AssistantQueryResponse(...)` constructors pass `warnings`
- [ ] Existing responses without warnings return `warnings: []`

---

### GAP 9 -- Sequential Pipeline, No Lateral Communication

**Severity**: MEDIUM
**Phase**: 4 (Advanced)
**Depends on**: None strictly, but all other gaps should be done first

#### Problem

The `run()` method executes all agents in a fixed sequence. No agent can ask another a question, request re-planning, or skip irrelevant steps. This corresponds to **Epic 11** ("Deliberative Agent Architecture") at 0% complete.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 863-1832 | `run()` -- fixed sequential pipeline |
| `agentic_team.py` | 1834-1866 | `_run_agent` -- no loop-back capability |
| `agentic_team.py` | 1868-1884 | `_chief_analyst` -- returns static metadata, no orchestration |
| `agentic_team.py` | 4042-4083 | `_blackboard_post` / `_blackboard_edges` -- exists but no lateral messaging |

#### Implementation Tasks (intermediate step -- not full Epic 11)

1. **Add a post-audit feedback loop** in `run()` (after audit, before narrative):
   ```python
   # After audit_result is computed:
   if audit_result["score"] < 0.5 and audit_result.get("goal_term_misses"):
       # Re-invoke planning with audit feedback
       enriched_intake = self._enrich_intake_from_audit(intake, audit_result)
       plan = self._planning_agent(enriched_intake, retrieval, catalog)
       sql = self._query_engine_agent(plan)
       execution = self._execution_agent(sql)
       audit_result = self._audit_agent(plan, execution, goal)
       # Track that a re-plan happened
       warnings.append("Query was automatically refined based on audit feedback.")
   ```

2. **Implement `_enrich_intake_from_audit`**: Extract `goal_term_misses` and `concept_misses` from audit, map them to dimensions/metrics, and merge into intake.

3. **Guard against infinite loops**: Allow at most 1 re-plan cycle. The AutonomyAgent (lines 3564-3740) already has `max_probe_queries` as a precedent for bounded iteration.

4. **Longer-term**: Promote `_chief_analyst` from static metadata to an orchestrator that decides agent execution order based on query type.

#### Acceptance Criteria

- [ ] When audit score < 0.5 and goal_term_misses exist, a re-plan cycle occurs
- [ ] Re-plan cycle is bounded to 1 iteration (no infinite loops)
- [ ] Re-plan is visible in agent_trace
- [ ] A warning is emitted when re-planning occurs
- [ ] Queries that pass audit on first try are unaffected

---

### GAP 10 -- Deterministic Overrides Kill LLM Intelligence

**Severity**: MEDIUM
**Phase**: 3 (Dimension Intelligence)
**Depends on**: GAP 5 (dynamic dim discovery for validation list)

#### Problem

The `_enforce_intake_consistency` method (lines 3064-3135) forcibly overrides LLM-parsed dimensions. Even if the LLM correctly identifies "region" as a requested dimension, the enforce function only allows "platform" and "state" for transactions (lines 3124-3128), effectively deleting any LLM-discovered dimensions.

The function is a **replacer** when it should be a **validator/enricher**.

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 3064-3135 | `_enforce_intake_consistency` -- hard-coded overrides |
| `agentic_team.py` | 3115-3131 | Dimension processing: only appends known dims, replaces LLM output |
| `agentic_team.py` | 3124-3128 | Only "platform" and "state" recognized for transactions |
| `agentic_team.py` | 2543-2572 | `_intake_agent` -- merges deterministic + LLM, then enforce overrides LLM |

#### Current Code

```python
# agentic_team.py:3124-3128
if out.get("domain") == "transactions":
    if "platform" in lower and dim_signal and "platform_name" not in dims:
        dims.append("platform_name")
    if "state" in lower and dim_signal and "state" not in dims:
        dims.append("state")
```

#### Implementation Tasks

1. **Change `_enforce_intake_consistency` to PRESERVE LLM dimensions** that are valid:
   ```python
   # Instead of building dims from scratch with only known keywords,
   # start with LLM-provided dims and validate against catalog
   llm_dims = out.get("dimensions", [])
   valid_dims = []
   table = domain_table.get(out.get("domain", ""), "")
   available_cols = set(catalog.get("marts", {}).get(table, {}).get("columns", []))

   for dim in llm_dims:
       if dim == "__month__" or dim in available_cols:
           valid_dims.append(dim)
       else:
           warnings.append(f"LLM-suggested dimension '{dim}' not in schema; dropped.")
   ```

2. **THEN add missing known dimensions** (the current behavior, but additive):
   ```python
   # Add dimensions that should be present based on keywords
   if "platform" in lower and dim_signal and "platform_name" not in valid_dims:
       valid_dims.append("platform_name")
   # ... same for other keywords
   ```

3. **Pass catalog to `_enforce_intake_consistency`**: Currently the function may not have access to the catalog. Add it as a parameter.

4. **Emit warnings** for overridden dimensions.

#### Acceptance Criteria

- [ ] LLM-parsed dimensions that exist in the catalog are preserved
- [ ] LLM-parsed dimensions that DON'T exist in the catalog are dropped with a warning
- [ ] Deterministic rules ADD missing dimensions, don't REPLACE LLM output
- [ ] Hard-coded "platform"/"state" logic becomes additive, not exclusive
- [ ] `_enforce_intake_consistency` receives catalog as a parameter

---

### GAP 11 -- No Agent Contribution Map or Failure Transparency

**Severity**: LOW
**Phase**: 4 (Advanced)
**Depends on**: GAP 8 (warnings infrastructure)

#### Problem

The trace shows agent execution order and timing but not what each agent actually contributed to the final answer. Corresponds to **Epic 13** at 0% complete.

- No `contribution` field in trace entries
- No `dropped_items` tracking
- Confidence shows raw percentage, not categorical level with reasoning
- Answer text doesn't link back to producing SQL

#### Affected Code

| File | Lines | What |
|------|-------|------|
| `agentic_team.py` | 1834-1866 | `_run_agent` -- trace entries lack `contribution` field |
| `agentic_team.py` | 1846-1854 | Trace append: only `agent`, `role`, `status`, `duration_ms`, `summary` |
| `contracts.py` | 376-435 | `AssistantQueryResponse` -- no contribution map field |

#### Current Trace Entry Format

```python
# agentic_team.py:1846-1854
trace.append({
    "agent": agent,
    "role": role,
    "status": "success",
    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
    "summary": _compact(out),
})
```

#### Implementation Tasks

1. **Extend trace entries** with `contribution` and `dropped_items`:
   ```python
   trace.append({
       "agent": agent,
       "role": role,
       "status": "success",
       "duration_ms": round((time.perf_counter() - start) * 1000, 2),
       "summary": _compact(out),
       "contribution": contribution,   # what this agent added/changed
       "dropped_items": dropped_items,  # what this agent removed
   })
   ```

2. **Each agent produces a contribution summary**: Modify agent methods to return `(result, contribution_meta)` tuples, or have `_run_agent` diff the blackboard state before/after each agent.

3. **Add `contribution_map` to `AssistantQueryResponse`** in `contracts.py`:
   ```python
   contribution_map: list[dict[str, Any]] = Field(
       default_factory=list,
       description="Per-agent contribution summary.",
   )
   ```

4. **Add categorical confidence with reasoning**:
   ```python
   confidence_reasoning: str = Field(
       default="",
       description="Explanation for the confidence level.",
   )
   ```

5. **Link evidence to SQL**: In `evidence_packets`, include the `sql_reference` field (already defined in `EvidenceItem` model at line 283 in `contracts.py`).

#### Acceptance Criteria

- [ ] Each trace entry includes a `contribution` string
- [ ] Dropped dimensions/metrics appear in `dropped_items`
- [ ] `AssistantQueryResponse` includes `contribution_map`
- [ ] Confidence level includes reasoning text
- [ ] Evidence items link to producing SQL

---

## Effort & Risk Matrix

| Gap | Phase | Effort | Risk | Mitigation |
|-----|-------|--------|------|------------|
| **8** | 1 | 1 day | LOW -- additive field, no logic change | Feature-flag warnings display in UI |
| **2** | 2 | 0.5 day | LOW -- mechanical replacement | Run full test suite after each `[:2]` change |
| **3** | 2 | 0.5 day | MEDIUM -- regex changes can have edge-case effects | Add unit tests for "trend" vs "monthly" vs "trends by X" |
| **4** | 2 | 2 days | MEDIUM -- LEFT JOIN may change row counts or performance | Verify row count before/after; add index on `customer_key`; run existing tests |
| **1** | 3 | 1.5 days | LOW (after GAP 4) -- additive keyword mapping | Test all 3 symptom queries end-to-end |
| **5** | 3 | 2 days | MEDIUM -- fuzzy matching can produce false positives | Static `dim_candidates` takes priority; dynamic is fallback only; log all dynamic resolutions |
| **10** | 3 | 1 day | HIGH -- changing override logic can destabilize intake | A/B test: run both old and new logic, compare outputs; fall back to old if divergence > threshold |
| **6** | 4 | 1.5 days | MEDIUM -- new candidate generation path | Bound dimension variants to 3 per audit cycle |
| **7** | 4 | 2 days | LOW -- currently stubs, so any implementation is additive | Start with one specialist (transactions); expand after validation |
| **9** | 4 | 3-5 days | HIGH -- architectural change to execution flow | Bound re-plan to 1 cycle; guard with feature flag; measure latency impact |
| **11** | 4 | 2 days | LOW -- observability enhancement, no logic change | Ship incrementally: contribution first, then confidence reasoning |

**Total estimated effort**: 15-17 days

---

## Verification Checklist

- [x] All 11 gaps documented with problem statements
- [x] Every gap has file paths and line numbers
- [x] Every gap has concrete code-level implementation tasks
- [x] Every gap has acceptance criteria
- [x] Dependencies are documented and ordering is correct (GAP 4 before GAP 1, GAP 5 before GAP 6/10)
- [x] GAP 8 (foundation) is Phase 1
- [x] Effort estimates and risk mitigations included
- [x] Dependency graph is consistent with phase ordering
