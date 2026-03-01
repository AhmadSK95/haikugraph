"""Quality and truth-score evaluator for v2."""

from __future__ import annotations

from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.v2.types import IntentSpecV2, QualityReportV2, SemanticCatalogV2


def _check_pass_ratio(response: AssistantQueryResponse) -> float:
    checks = list(response.sanity_checks or [])
    if not checks:
        return 1.0 if response.success else 0.0
    passed = sum(1 for c in checks if bool(c.passed))
    return passed / max(1, len(checks))


def _quality_flags(
    response: AssistantQueryResponse,
    intent: IntentSpecV2,
    catalog: SemanticCatalogV2,
) -> list[str]:
    flags: list[str] = []
    runtime = dict(response.runtime or {})
    if bool(runtime.get("llm_degraded")):
        flags.append("provider_degraded")
    if intent.is_followup and not str(response.slice_signature or ""):
        flags.append("continuity_risk")
    if intent.requires_validity_guard and "has_mt103" in str(response.sql or "").lower():
        flags.append("semantic_guard_applied")
    if intent.requires_validity_guard and "has_mt103" not in str(response.sql or "").lower():
        flags.append("semantic_guard_missing")
    if int(catalog.quality_summary.get("high_risk_join_edges") or 0) > 0:
        flags.append("join_fragility")
    if any(str(w).startswith("[SkillContract]") for w in (response.warnings or [])):
        flags.append("contract_noise")
    return sorted(set(flags))


def evaluate_quality(
    response: AssistantQueryResponse,
    intent: IntentSpecV2,
    catalog: SemanticCatalogV2,
) -> QualityReportV2:
    pass_ratio = _check_pass_ratio(response)
    warning_penalty = min(0.35, 0.03 * len(response.warnings or []))
    score = 100.0 * pass_ratio
    score -= (1.0 - float(response.confidence_score or 0.0)) * 12.0
    score -= warning_penalty * 100.0
    if not response.success:
        score = min(score, 35.0)
    flags = _quality_flags(response, intent, catalog)
    runtime = dict(response.runtime or {})
    fallback_used = {
        "used": bool(runtime.get("llm_degraded")),
        "reason": str(runtime.get("llm_degraded_reason") or ""),
    }
    provider_effective = str(runtime.get("provider") or "")
    return QualityReportV2(
        truth_score=round(max(0.0, min(100.0, score)), 2),
        quality_flags=flags,
        provider_effective=provider_effective,
        fallback_used=fallback_used,
    )

