"""Dataset-agnostic ontology resolver used by v2 planner and compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from haikugraph.v2.types import SemanticCatalogV2


@dataclass(frozen=True)
class OntologyResolution:
    table: str
    metric_agg: str
    metric_column: str
    metric_alias: str
    dimensions: list[str] = field(default_factory=list)
    time_column: str = ""
    filters: list[dict[str, Any]] = field(default_factory=list)
    denominator_semantics: str = ""
    certainty_tags: list[str] = field(default_factory=list)


def _table_columns(catalog: SemanticCatalogV2, table: str) -> list[str]:
    for item in catalog.tables:
        if item.table == table:
            return list(item.columns or [])
    return []


def _first_table(catalog: SemanticCatalogV2) -> str:
    return str((catalog.tables[0].table if catalog.tables else "") or "")


def _score_table(goal_lower: str, table_name: str, columns: list[str]) -> float:
    score = 0.0
    tn = table_name.lower()
    if "transaction" in goal_lower and ("transaction" in tn or "payment" in tn):
        score += 3.0
    if "customer" in goal_lower and "customer" in tn:
        score += 3.0
    if "quote" in goal_lower and "quote" in tn:
        score += 3.0
    if "booking" in goal_lower and ("booking" in tn or "deal" in tn):
        score += 3.0
    if "refund" in goal_lower and "refund" in tn:
        score += 2.0
    if "markup" in goal_lower and ("quote" in tn or "markup" in tn):
        score += 2.0
    if "policy" in goal_lower and "document" in tn:
        score += 3.0
    for col in columns:
        lc = col.lower()
        if "amount" in goal_lower and "amount" in lc:
            score += 0.6
        if "markup" in goal_lower and "markup" in lc:
            score += 1.4
        if "refund" in goal_lower and "refund" in lc:
            score += 1.1
        if "country" in goal_lower and "country" in lc:
            score += 1.0
        if "university" in goal_lower and "university" in lc:
            score += 1.0
        if "state" in goal_lower and "state" in lc:
            score += 0.4
        if "platform" in goal_lower and "platform" in lc:
            score += 0.4
        if "country" in goal_lower and "country" in lc:
            score += 0.4
    if ("university" in goal_lower or "universities" in goal_lower) and ("country" in goal_lower or "united kingdom" in goal_lower):
        colset = {c.lower() for c in columns}
        if "is_university" in colset and ("address_country" in colset or "country" in colset):
            score += 4.0
    if "united kingdom" in goal_lower and any(c.lower() in {"address_country", "country"} for c in columns):
        score += 2.0
    if tn.startswith("datada_agent_"):
        score -= 1.5
    return score


def choose_table(goal: str, catalog: SemanticCatalogV2) -> str:
    goal_lower = str(goal or "").lower()
    if not catalog.tables:
        return ""
    scored: list[tuple[float, str]] = []
    for table in catalog.tables:
        scored.append((_score_table(goal_lower, table.table, list(table.columns or [])), table.table))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_score, top_table = scored[0]
    if top_score <= 0.0:
        return _first_table(catalog)
    return top_table


def _pick_metric(goal_lower: str, columns: list[str]) -> tuple[str, str, str]:
    numeric_preference = [
        "payment_amount",
        "amount",
        "total_amount_to_be_paid",
        "booked_amount",
        "forex_markup",
        "value",
        "rate",
    ]
    id_candidates = [c for c in columns if c.lower().endswith("_id") or c.lower().endswith("_key")]
    amount_candidates = [c for c in columns if "amount" in c.lower() or "revenue" in c.lower()]
    markup_candidates = [c for c in columns if "markup" in c.lower()]

    agg = "count"
    metric_col = id_candidates[0] if id_candidates else (columns[0] if columns else "1")
    alias = "metric_value"

    if any(k in goal_lower for k in ("total", "sum", "spend", "revenue", "amount")):
        agg = "sum"
        if "markup" in goal_lower and markup_candidates:
            metric_col = markup_candidates[0]
        elif amount_candidates:
            priority_map = {name: idx for idx, name in enumerate(numeric_preference)}
            amount_candidates.sort(key=lambda c: priority_map.get(c.lower(), 999))
            metric_col = amount_candidates[0]
        alias = "metric_value"
    elif any(k in goal_lower for k in ("average", "avg", "mean")):
        agg = "avg"
        if amount_candidates:
            metric_col = amount_candidates[0]
        alias = "metric_value"
    elif any(k in goal_lower for k in ("median", "p50")) and amount_candidates:
        agg = "median"
        metric_col = amount_candidates[0]
        alias = "metric_value"
    elif any(k in goal_lower for k in ("95th", "p95", "percentile")) and amount_candidates:
        agg = "p95"
        metric_col = amount_candidates[0]
        alias = "metric_value"
    return agg, metric_col, alias


def _pick_dimensions(goal_lower: str, columns: list[str]) -> list[str]:
    dimensions: list[str] = []
    dimension_hints = {
        "platform": ["platform_name", "platform"],
        "state": ["state"],
        "country": ["address_country", "country"],
        "universit": ["is_university"],
        "type": ["type", "deal_type"],
        "status": ["status", "payment_status"],
        "month": ["__month__"],
    }
    for hint, candidates in dimension_hints.items():
        if hint not in goal_lower and f"by {hint}" not in goal_lower and f"per {hint}" not in goal_lower:
            continue
        for candidate in candidates:
            if candidate == "__month__":
                if any("created_at" == c.lower() or c.lower().endswith("_at") for c in columns):
                    dimensions.append("__month__")
                    break
                continue
            if candidate in columns:
                dimensions.append(candidate)
                break
    if not dimensions:
        if "split by" in goal_lower or "by " in goal_lower:
            for col in columns:
                lc = col.lower()
                if any(tok in lc for tok in ("platform", "state", "country", "type", "status")):
                    dimensions.append(col)
                    break
    return dimensions


def _pick_time_column(columns: list[str]) -> str:
    preferred = [
        "created_at",
        "created_ts",
        "updated_at",
        "payment_created_at",
    ]
    lowered = {c.lower(): c for c in columns}
    for candidate in preferred:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        if col.lower().endswith("_at") or col.lower().endswith("_ts") or "date" in col.lower():
            return col
    return ""


def _extract_month(goal_lower: str) -> int | None:
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    for name, value in month_map.items():
        if name in goal_lower:
            return value
    return None


def resolve_ontology(goal: str, catalog: SemanticCatalogV2, *, requires_validity_guard: bool) -> OntologyResolution:
    """Resolve table/metrics/dimensions in a dataset-agnostic way."""
    goal_lower = str(goal or "").lower()
    table = choose_table(goal, catalog)
    columns = _table_columns(catalog, table)
    agg, metric_col, alias = _pick_metric(goal_lower, columns)
    dimensions = _pick_dimensions(goal_lower, columns)
    time_col = _pick_time_column(columns)
    filters: list[dict[str, Any]] = []
    certainty_tags: list[str] = []
    denominator = ""

    if "per customer" in goal_lower:
        denominator = "per_customer"
        certainty_tags.append("inferred_denominator")
    elif "per transaction" in goal_lower or "per txn" in goal_lower:
        denominator = "per_transaction"
        certainty_tags.append("inferred_denominator")
    elif "per quote" in goal_lower:
        denominator = "per_quote"
        certainty_tags.append("inferred_denominator")

    if requires_validity_guard:
        if "has_mt103" in columns:
            filters.append({"type": "bool_true", "column": "has_mt103", "guard": "has_mt103"})
            certainty_tags.append("rule_applied")
        elif "mt103_created_at" in columns:
            filters.append({"type": "not_null", "column": "mt103_created_at", "guard": "has_mt103"})
            certainty_tags.append("rule_applied_inferred")
        else:
            certainty_tags.append("rule_requested_unavailable")

    if "refund" in goal_lower:
        if "refund_refund_id" in columns:
            filters.append({"type": "not_null", "column": "refund_refund_id"})
    if "university" in goal_lower:
        if "is_university" in columns:
            filters.append({"type": "bool_true", "column": "is_university"})
    month = _extract_month(goal_lower)
    year_match = re.search(r"\b(20\d{2})\b", goal_lower)
    if month is not None and time_col:
        year = int(year_match.group(1)) if year_match else None
        filters.append(
            {
                "type": "month_filter",
                "column": time_col,
                "month": month,
                "year": year,
            }
        )

    return OntologyResolution(
        table=table,
        metric_agg=agg,
        metric_column=metric_col,
        metric_alias=alias,
        dimensions=dimensions,
        time_column=time_col,
        filters=filters,
        denominator_semantics=denominator,
        certainty_tags=sorted(set(certainty_tags)),
    )
