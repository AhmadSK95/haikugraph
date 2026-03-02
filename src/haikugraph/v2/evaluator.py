"""Quality and truth-score evaluator for v2."""

from __future__ import annotations

from haikugraph.v2.types import ExecutionResultV2, IntentSpecV2, QualityReportV2, QueryPlanV2, SemanticCatalogV2


def _quality_flags(
    execution: ExecutionResultV2,
    intent: IntentSpecV2,
    catalog: SemanticCatalogV2,
    query_plan: QueryPlanV2,
    *,
    provider_degraded: bool,
) -> list[str]:
    flags: list[str] = []
    sql_lower = str(query_plan.sql or "").lower()

    if provider_degraded:
        flags.append("provider_degraded")
    if intent.is_followup and "carry_scope" not in (intent.operations or []):
        flags.append("continuity_risk")
    if intent.requires_validity_guard and "has_mt103" in sql_lower:
        flags.append("semantic_guard_applied")
    elif intent.requires_validity_guard and "mt103_created_at" in sql_lower:
        flags.append("semantic_guard_applied")
    elif intent.requires_validity_guard:
        flags.append("semantic_guard_missing")
    if int(catalog.quality_summary.get("high_risk_join_edges") or 0) > 0:
        flags.append("join_fragility")
    if not execution.success:
        flags.append("execution_failed")
    if execution.row_count == 0 and execution.success:
        flags.append("empty_result")
    if query_plan.secondary_metric and not query_plan.grouping:
        flags.append("dual_metric_scalar")
    return sorted(set(flags))


def _certainty_tags(
    intent: IntentSpecV2,
    catalog: SemanticCatalogV2,
    execution: ExecutionResultV2,
    flags: list[str],
) -> list[str]:
    tags: list[str] = []
    if execution.success and execution.row_count > 0:
        tags.append("evidence_backed")
    if "join_fragility" in flags:
        tags.append("join_fragility")
    if int(catalog.quality_summary.get("sparse_table_count") or 0) > 0:
        tags.append("sparse_data_risk")
    if intent.requires_validity_guard and "semantic_guard_applied" in flags:
        tags.append("rule_applied")
    if intent.requires_validity_guard and "semantic_guard_missing" in flags:
        tags.append("rule_missing")
    if intent.is_followup and "continuity_risk" not in flags:
        tags.append("continuity_preserved")
    return sorted(set(tags))


def evaluate_quality(
    *,
    execution: ExecutionResultV2,
    intent: IntentSpecV2,
    catalog: SemanticCatalogV2,
    query_plan: QueryPlanV2,
    provider_effective: str,
    fallback_used: dict[str, object],
) -> QualityReportV2:
    degraded = bool((fallback_used or {}).get("used"))
    flags = _quality_flags(
        execution,
        intent,
        catalog,
        query_plan,
        provider_degraded=degraded,
    )
    certainty_tags = _certainty_tags(intent, catalog, execution, flags)

    score = 100.0
    if not execution.success:
        score -= 65.0
    if execution.row_count == 0:
        score -= 12.0
    if "join_fragility" in flags:
        score -= 9.0
    if "semantic_guard_missing" in flags:
        score -= 25.0
    if "continuity_risk" in flags:
        score -= 18.0
    if degraded:
        score -= 8.0
    score = max(0.0, min(100.0, score))

    return QualityReportV2(
        truth_score=round(score, 2),
        quality_flags=flags,
        provider_effective=str(provider_effective or "deterministic"),
        fallback_used=dict(fallback_used or {}),
        certainty_tags=certainty_tags,
        grain_signature=str(query_plan.grain_signature or ""),
        denominator_semantics=str(intent.denominator_semantics or ""),
    )
