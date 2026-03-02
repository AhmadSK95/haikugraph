"""Intent parser for v2 using operation-oriented follow-up interpretation."""

from __future__ import annotations

import re

from haikugraph.v2.types import ConversationStateV2, IntentSpecV2


def _metrics_from_text(lower: str) -> list[str]:
    metrics: list[str] = []
    if any(k in lower for k in ("count", "how many", "number of")):
        metrics.append("count")
    if any(k in lower for k in ("amount", "revenue", "spend", "sum", "total")):
        metrics.append("amount")
    if any(k in lower for k in ("average", "avg", "mean")):
        metrics.append("average")
    if any(k in lower for k in ("median", "p50")):
        metrics.append("median")
    if any(k in lower for k in ("95th", "p95", "percentile")):
        metrics.append("p95")
    if "markup" in lower:
        metrics.append("markup")
    return sorted(set(metrics))


def _dimensions_from_text(lower: str) -> list[str]:
    dims: list[str] = []
    mapping = {
        "platform": "platform_name",
        "state": "state",
        "country": "address_country",
        "universit": "is_university",
        "type": "type",
        "deal type": "deal_type",
        "status": "status",
        "month": "__month__",
    }
    for token, dim in mapping.items():
        if f"by {token}" in lower or f"per {token}" in lower or f"split by {token}" in lower:
            dims.append(dim)
    if "split by month and platform" in lower or "by month and platform" in lower:
        dims.extend(["__month__", "platform_name"])
    if "universities vs non-universities" in lower or "university" in lower:
        dims.append("is_university")
    return sorted(set(dims))


def _time_scope_from_text(lower: str) -> str:
    if "this month" in lower:
        return "this_month"
    if "last month" in lower:
        return "last_month"
    if "december" in lower:
        return "december"
    if "november" in lower:
        return "november"
    return ""


def parse_intent(goal: str, state: ConversationStateV2) -> IntentSpecV2:
    lower = str(goal or "").lower().strip()
    followup_markers = [
        "same slice",
        "same scope",
        "keep that",
        "keep same",
        "now",
        "also",
        "add",
        "switch metric",
        "what about",
    ]
    is_followup = bool(state.turn_index > 1 and any(m in lower for m in followup_markers))
    operations: list[str] = []
    if is_followup:
        operations.append("carry_scope")
        if str(state.prior_slice_signature or "").strip():
            operations.append("carry_slice_signature")

    if "switch metric" in lower or "instead" in lower:
        operations.append("switch_metric")
    if re.search(r"\b(add|include|also|plus)\b.*\b(total\s+)?(amount|value|revenue)\b", lower):
        operations.append("add_secondary_metric")
    if re.search(r"\b(add|include|also|plus)\b.*\b(count|volume)\b", lower):
        operations.append("add_secondary_metric")
    if "same grouped output" in lower or "same grouping" in lower:
        operations.append("preserve_grouping")
    if "keep same slice" in lower:
        operations.extend(["preserve_grouping", "carry_scope"])
    if "decision memo" in lower or "memo format" in lower:
        operations.append("decision_memo")
    if any(k in lower for k in ("what if", "scenario", "if we")):
        operations.append("scenario_analysis")
    if re.search(r"\bvalid transaction(s)?\b", lower) or "mt103" in lower:
        operations.append("enforce_validity_guard")
    if re.search(r"\b(spend|amount|revenue|value)\b.*\btransaction", lower):
        operations.append("enforce_validity_guard")

    requires_validity_guard = "enforce_validity_guard" in operations
    strategy = "comparison" if any(k in lower for k in ["compare", "versus", " vs "]) else "metric"
    requested_metrics = _metrics_from_text(lower)
    requested_dimensions = _dimensions_from_text(lower)
    time_scope = _time_scope_from_text(lower)

    denominator = ""
    if "per customer" in lower:
        denominator = "per_customer"
    elif "per transaction" in lower or "per txn" in lower:
        denominator = "per_transaction"
    elif "per quote" in lower:
        denominator = "per_quote"

    output_mode = "decision_memo" if "decision_memo" in operations else "answer"
    return IntentSpecV2(
        goal=goal,
        is_followup=is_followup,
        operations=sorted(set(operations)),
        requires_validity_guard=requires_validity_guard,
        strategy=strategy,
        requested_metrics=requested_metrics,
        requested_dimensions=requested_dimensions,
        requested_time_scope=time_scope,
        denominator_semantics=denominator,
        output_mode=output_mode,
    )
