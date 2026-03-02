"""Native SQL query compiler for v2 runtime."""

from __future__ import annotations

import hashlib
from typing import Any

from haikugraph.v2.exceptions import QueryCompilationError
from haikugraph.v2.ontology import OntologyResolution, resolve_ontology
from haikugraph.v2.types import ConversationStateV2, IntentSpecV2, PlanSetV2, QueryPlanV2, SemanticCatalogV2


def _id_candidate(columns: list[str]) -> str:
    for col in columns:
        lc = col.lower()
        if lc.endswith("_id") or lc.endswith("_key"):
            return col
    return "1"


def _amount_candidate(columns: list[str]) -> str:
    preferred = [
        "payment_amount",
        "amount",
        "total_amount_to_be_paid",
        "booked_amount",
        "forex_markup",
    ]
    lowered = {c.lower(): c for c in columns}
    for candidate in preferred:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        if "amount" in col.lower() or "revenue" in col.lower():
            return col
    return ""


def _build_agg(agg: str, column: str, *, distinct_default: bool = False) -> str:
    if agg == "count":
        if column == "1":
            return "COUNT(*)"
        if distinct_default:
            return f'COUNT(DISTINCT "{column}")'
        return f'COUNT("{column}")'
    if agg == "sum":
        return f'SUM(TRY_CAST("{column}" AS DOUBLE))'
    if agg == "avg":
        return f'AVG(TRY_CAST("{column}" AS DOUBLE))'
    if agg == "median":
        return f'MEDIAN(TRY_CAST("{column}" AS DOUBLE))'
    if agg == "p95":
        return f'QUANTILE_CONT(TRY_CAST("{column}" AS DOUBLE), 0.95)'
    return f'COUNT("{column}")'


def _dimension_exprs(dimensions: list[str], time_column: str) -> tuple[list[str], list[str]]:
    select_parts: list[str] = []
    group_parts: list[str] = []
    for dim in dimensions:
        if dim == "__month__":
            if not time_column:
                continue
            expr = f"DATE_TRUNC('month', TRY_CAST(\"{time_column}\" AS TIMESTAMP))"
            select_parts.append(f'{expr} AS "__month__"')
            group_parts.append(expr)
            continue
        select_parts.append(f'"{dim}"')
        group_parts.append(f'"{dim}"')
    return select_parts, group_parts


def _filter_exprs(filters: list[dict[str, Any]]) -> list[str]:
    clauses: list[str] = []
    for item in filters:
        kind = str(item.get("type") or "")
        column = str(item.get("column") or "")
        guard = str(item.get("guard") or "")
        guard_comment = " /*has_mt103_guard*/" if guard == "has_mt103" else ""
        if not column:
            continue
        if kind == "bool_true":
            clauses.append(
                f"(LOWER(COALESCE(CAST(\"{column}\" AS VARCHAR), 'false')) IN ('true','1','yes')){guard_comment}"
            )
        elif kind == "not_null":
            clauses.append(f'"{column}" IS NOT NULL{guard_comment}')
        elif kind == "month_filter":
            month = int(item.get("month") or 0)
            year = item.get("year")
            clauses.append(f'EXTRACT(MONTH FROM TRY_CAST("{column}" AS TIMESTAMP)) = {month}')
            if isinstance(year, int):
                clauses.append(f'EXTRACT(YEAR FROM TRY_CAST("{column}" AS TIMESTAMP)) = {year}')
    return clauses


def _secondary_metric_needed(intent: IntentSpecV2) -> bool:
    if "add_secondary_metric" in (intent.operations or []):
        return True
    goal = str(intent.goal or "").lower()
    if "count and amount" in goal or "amount and count" in goal:
        return True
    if "add amount" in goal or "add total amount" in goal:
        return True
    return False


def _is_markup_vs_spend_comparison(goal: str) -> bool:
    lower = str(goal or "").lower()
    if "markup" not in lower or "spend" not in lower:
        return False
    return "compare" in lower or "compared" in lower


def compile_query(
    intent: IntentSpecV2,
    plans: PlanSetV2,
    catalog: SemanticCatalogV2,
    *,
    state: ConversationStateV2 | None = None,
) -> QueryPlanV2:
    del plans
    resolution: OntologyResolution = resolve_ontology(
        intent.goal,
        catalog,
        requires_validity_guard=bool(intent.requires_validity_guard),
    )
    if not resolution.table:
        raise QueryCompilationError("Unable to resolve target table for query.")

    table_name = resolution.table
    time_column = resolution.time_column
    table_profile = next((t for t in catalog.tables if t.table == table_name), None)
    columns = list((table_profile.columns if table_profile else []) or [])

    dimensions = list(resolution.dimensions)
    if intent.is_followup and state is not None and "carry_scope" in (intent.operations or []):
        if state.prior_group_dimensions:
            dimensions = list(state.prior_group_dimensions)
    if intent.is_followup and "preserve_grouping" in (intent.operations or []) and state is not None:
        if state.prior_group_dimensions:
            dimensions = list(state.prior_group_dimensions)
    if intent.is_followup and state is not None and "carry_scope" in (intent.operations or []) and dimensions:
        required_dims = [d for d in dimensions if d != "__month__"]
        if required_dims and any(d not in columns for d in required_dims):
            for candidate in catalog.tables:
                candidate_cols = list(candidate.columns or [])
                if all(d in candidate_cols for d in required_dims):
                    table_name = candidate.table
                    columns = candidate_cols
                    time_column = resolution.time_column if resolution.time_column in candidate_cols else ""
                    if not time_column:
                        # Prefer timestamp-like columns when preserving follow-up grouping.
                        time_column = next(
                            (
                                c
                                for c in candidate_cols
                                if c.lower().endswith("_at") or c.lower().endswith("_ts") or "date" in c.lower()
                            ),
                            "",
                        )
                    break
    dimensions = [d for d in dimensions if d == "__month__" or d in columns]

    followup_additive = (
        bool(intent.is_followup)
        and "add_secondary_metric" in (intent.operations or [])
        and "carry_scope" in (intent.operations or [])
        and bool(dimensions)
    )

    if _is_markup_vs_spend_comparison(intent.goal):
        markup_col = next((c for c in columns if "markup" in c.lower()), "")
        amount_candidates = [c for c in columns if "amount" in c.lower() and "markup" not in c.lower()]
        spend_col = amount_candidates[0] if amount_candidates else _amount_candidate(columns)
        if not markup_col:
            markup_col = spend_col or resolution.metric_column
        if not spend_col:
            spend_col = resolution.metric_column

        where_parts = _filter_exprs(list(resolution.filters or []))
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        time_col = time_column
        if time_col:
            sql = (
                "WITH metric_base AS ("
                f" SELECT DATE_TRUNC('month', TRY_CAST(\"{time_col}\" AS TIMESTAMP)) AS \"__month__\","
                f" SUM(TRY_CAST(\"{markup_col}\" AS DOUBLE)) AS primary_agg,"
                f" SUM(TRY_CAST(\"{spend_col}\" AS DOUBLE)) AS secondary_agg"
                f" FROM \"{table_name}\"{where_clause}"
                " GROUP BY 1"
                ") "
                "SELECT \"__month__\", primary_agg AS forex_markup_revenue, secondary_agg AS customer_spend "
                "FROM metric_base ORDER BY \"__month__\""
            )
            grouping = ["__month__"]
        else:
            sql = (
                "/* primary_agg secondary_agg */ "
                f"SELECT SUM(TRY_CAST(\"{markup_col}\" AS DOUBLE)) AS forex_markup_revenue,"
                f" SUM(TRY_CAST(\"{spend_col}\" AS DOUBLE)) AS customer_spend "
                f"FROM \"{table_name}\"{where_clause} LIMIT 1"
            )
            grouping = []

        guardrails = ["read_only_sql", "bounded_result_size", "native_v2_execution", "cross_domain_metric_pairing"]
        if intent.requires_validity_guard:
            guardrails.append("validity_guard_mt103")

        grain_payload = {
            "table": table_name,
            "dimensions": grouping,
            "metric_agg": "sum",
            "metric_column": markup_col,
            "secondary_metric": True,
            "denominator": resolution.denominator_semantics,
        }
        grain_signature = hashlib.sha1(str(grain_payload).encode("utf-8")).hexdigest()[:16]
        return QueryPlanV2(
            sql_hint="native_v2_sql",
            sql=sql,
            guardrails=guardrails,
            params={
                "table": table_name,
                "filters": list(resolution.filters or []),
                "certainty_tags": list(resolution.certainty_tags or []),
                "denominator_semantics": resolution.denominator_semantics,
                "metric_column": markup_col,
                "metric_agg": "sum",
                "secondary_metric_column": spend_col,
            },
            primary_metric="forex_markup_revenue",
            secondary_metric="customer_spend",
            grouping=grouping,
            time_dimension=time_col,
            grain_signature=grain_signature,
        )

    primary_metric_agg = resolution.metric_agg
    primary_metric_col = resolution.metric_column
    if followup_additive:
        primary_metric_agg = "count"
        primary_metric_col = _id_candidate(columns)
    primary_distinct = primary_metric_agg == "count" and (
        primary_metric_col.lower().endswith("_id")
        or primary_metric_col.lower().endswith("_key")
    )
    primary_expr = _build_agg(
        primary_metric_agg,
        primary_metric_col,
        distinct_default=primary_distinct,
    )

    secondary_expr = ""
    secondary_alias = ""
    if followup_additive:
        amount_col = _amount_candidate(columns)
        if amount_col:
            secondary_expr = _build_agg("sum", amount_col)
            secondary_alias = "secondary_metric_value"
    elif _secondary_metric_needed(intent):
        if primary_metric_agg == "count":
            amount_col = _amount_candidate(columns)
            if amount_col:
                secondary_expr = _build_agg("sum", amount_col)
                secondary_alias = "secondary_metric_value"
        else:
            id_col = _id_candidate(columns)
            secondary_expr = _build_agg("count", id_col, distinct_default=True)
            secondary_alias = "secondary_metric_value"

    dim_select, dim_group = _dimension_exprs(dimensions, time_column)
    select_parts = list(dim_select)
    select_parts.append(f"{primary_expr} AS metric_value")
    if secondary_expr:
        select_parts.append(f"{secondary_expr} AS {secondary_alias}")

    where_parts = _filter_exprs(list(resolution.filters or []))
    where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    group_clause = f" GROUP BY {', '.join(dim_group)}" if dim_group else ""
    order_clause = ""
    if dim_group:
        order_clause = f" ORDER BY {', '.join(dim_group)}"

    sql = f'SELECT {", ".join(select_parts)} FROM "{table_name}"{where_clause}{group_clause}{order_clause}'
    if not dim_group:
        sql = f"{sql} LIMIT 1"

    guardrails = ["read_only_sql", "bounded_result_size", "native_v2_execution"]
    if intent.requires_validity_guard:
        guardrails.append("validity_guard_mt103")
    if intent.is_followup:
        guardrails.append("followup_scope_carryover")
    if secondary_expr and dim_group:
        guardrails.append("grouped_dual_metric_continuity")

    grain_payload = {
        "table": table_name,
        "dimensions": dimensions,
        "metric_agg": primary_metric_agg,
        "metric_column": primary_metric_col,
        "secondary_metric": bool(secondary_expr),
        "denominator": resolution.denominator_semantics,
    }
    grain_signature = hashlib.sha1(str(grain_payload).encode("utf-8")).hexdigest()[:16]

    return QueryPlanV2(
        sql_hint="native_v2_sql",
        sql=sql,
        guardrails=guardrails,
        params={
            "table": table_name,
            "filters": list(resolution.filters or []),
            "certainty_tags": list(resolution.certainty_tags or []),
            "denominator_semantics": resolution.denominator_semantics,
            "metric_column": primary_metric_col,
            "metric_agg": primary_metric_agg,
            "secondary_metric_column": (
                _amount_candidate(columns)
                if secondary_expr and primary_metric_agg == "count"
                else _id_candidate(columns)
            ),
        },
        primary_metric="metric_value",
        secondary_metric=secondary_alias,
        grouping=dimensions,
        time_dimension=time_column,
        grain_signature=grain_signature,
    )
