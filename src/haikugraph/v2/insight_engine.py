"""Insight helpers for v2 response enrichment."""

from __future__ import annotations

import re
from typing import Any

from haikugraph.v2.contradiction_engine import detect_contradictions
from haikugraph.v2.decision_memo import build_decision_memo
from haikugraph.v2.recommendation_engine import build_recommendations
from haikugraph.v2.types import (
    ContradictionReportV2,
    DecisionMemoV2,
    ExecutionResultV2,
    InsightReportV2,
    IntentSpecV2,
    ProvenanceEntryV2,
    QualityReportV2,
    QueryPlanV2,
)


def _assumptions(intent: IntentSpecV2, goal: str) -> list[str]:
    lower = str(goal or "").lower()
    assumptions: list[str] = []
    if any(k in lower for k in ["if we", "what if", "estimate", "likely impact", "scenario"]):
        assumptions.append("Scenario estimate uses observed historical patterns, not causal proof.")
        assumptions.append("No external market variables are used unless present in the dataset.")
    if intent.requires_validity_guard:
        assumptions.append("Transaction validity interpreted using MT103 policy.")
    if re.search(r"\bforecast|predict|projection\b", lower):
        assumptions.append("Forecast-like interpretation is bounded by available historical data only.")
    if intent.denominator_semantics:
        assumptions.append(f"Denominator semantics applied: {intent.denominator_semantics}.")
    return assumptions


def _provenance(query_plan: QueryPlanV2) -> list[ProvenanceEntryV2]:
    items: list[ProvenanceEntryV2] = []
    table = str((query_plan.params or {}).get("table") or "")
    if query_plan.primary_metric:
        items.append(
            ProvenanceEntryV2(
                field=query_plan.primary_metric,
                source_table=table,
                source_column=str((query_plan.params or {}).get("metric_column") or ""),
                expression="primary_aggregation",
                confidence=0.82,
                note="Derived from ontology-selected primary metric.",
            )
        )
    if query_plan.secondary_metric:
        items.append(
            ProvenanceEntryV2(
                field=query_plan.secondary_metric,
                source_table=table,
                source_column=str((query_plan.params or {}).get("secondary_metric_column") or ""),
                expression="secondary_aggregation",
                confidence=0.78,
                note="Derived from follow-up/additive metric intent.",
            )
        )
    return items


def _summary_markdown(
    *,
    execution: ExecutionResultV2,
    quality: QualityReportV2,
    recommendations_count: int,
) -> str:
    if not execution.success:
        return "Execution failed. Please review filters, grain, and policy constraints."
    if execution.row_count <= 0:
        return "No rows matched the current slice. Consider broadening time or filter constraints."
    return (
        f"Query returned **{execution.row_count}** rows. "
        f"Truth score: **{quality.truth_score:.2f}**. "
        f"Generated **{recommendations_count}** recommended action(s)."
    )


def build_insight(
    *,
    intent: IntentSpecV2,
    goal: str,
    query_plan: QueryPlanV2,
    execution: ExecutionResultV2,
    quality: QualityReportV2,
) -> InsightReportV2:
    assumptions = _assumptions(intent, goal)
    contradiction: ContradictionReportV2 = detect_contradictions(goal, intent)
    recommendations = build_recommendations(
        goal=goal,
        quality_flags=list(quality.quality_flags or []),
        row_count=int(execution.row_count or 0),
        has_grouping=bool(query_plan.grouping),
    )
    certainty_tags = sorted(set((quality.certainty_tags or []) + list((query_plan.params or {}).get("certainty_tags") or [])))
    summary_markdown = _summary_markdown(
        execution=execution,
        quality=quality,
        recommendations_count=len(recommendations),
    )
    memo: DecisionMemoV2 | None = None
    if intent.output_mode == "decision_memo":
        memo = build_decision_memo(
            goal=goal,
            summary=summary_markdown,
            assumptions=assumptions,
            quality_flags=list(quality.quality_flags or []),
            recommendations=recommendations,
            certainty_tags=certainty_tags,
        )
    return InsightReportV2(
        assumptions=assumptions,
        certainty_tags=certainty_tags,
        provenance=_provenance(query_plan),
        recommendations=recommendations,
        contradiction=contradiction,
        decision_memo=memo,
        summary_markdown=summary_markdown,
    )
