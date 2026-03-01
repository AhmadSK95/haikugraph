"""Plan candidate generation for v2."""

from __future__ import annotations

from haikugraph.v2.types import IntentSpecV2, PlanCandidateV2, PlanSetV2, SemanticCatalogV2


def build_plan_set(intent: IntentSpecV2, catalog: SemanticCatalogV2) -> PlanSetV2:
    risks: list[str] = []
    high_risk = int(catalog.quality_summary.get("high_risk_join_edges") or 0)
    if high_risk > 0:
        risks.append("join_fragility")
    if intent.requires_validity_guard:
        risks.append("validity_guard_required")
    base_coverage = 100.0
    if "join_fragility" in risks:
        base_coverage -= 18.0
    if intent.is_followup:
        base_coverage -= 4.0
    primary = PlanCandidateV2(
        candidate_id="c1_primary",
        strategy=intent.strategy,
        objective_coverage_pct=max(0.0, min(100.0, base_coverage)),
        risk_flags=sorted(set(risks)),
    )
    fallback = PlanCandidateV2(
        candidate_id="c2_fallback",
        strategy="metric",
        objective_coverage_pct=max(0.0, min(100.0, base_coverage - 8.0)),
        risk_flags=sorted(set(risks + ["fallback"])),
    )
    return PlanSetV2(selected_id=primary.candidate_id, candidates=[primary, fallback])

