"""Recommendation generation for analyst-grade output."""

from __future__ import annotations

from haikugraph.v2.types import RecommendationOptionV2


def build_recommendations(
    *,
    goal: str,
    quality_flags: list[str],
    row_count: int,
    has_grouping: bool,
) -> list[RecommendationOptionV2]:
    lower = str(goal or "").lower()
    recs: list[RecommendationOptionV2] = []

    if "join_fragility" in quality_flags:
        recs.append(
            RecommendationOptionV2(
                action="Validate key coverage before cross-domain decisions",
                expected_impact="Reduces false conclusions from weak joins",
                risk="low",
                effort="low",
                rationale="Join coverage indicates fragile linkage across tables.",
            )
        )
    if row_count <= 0:
        recs.append(
            RecommendationOptionV2(
                action="Relax filters or widen time scope",
                expected_impact="Increase result coverage for analysis",
                risk="low",
                effort="low",
                rationale="Current slice produced no records.",
            )
        )
    if "funnel" in lower:
        recs.append(
            RecommendationOptionV2(
                action="Track stage-by-stage conversion drop-off",
                expected_impact="Highlights highest-loss stage for intervention",
                risk="medium",
                effort="medium",
                rationale="Funnel asks are decision-oriented and need stage attribution.",
            )
        )
    if has_grouping:
        recs.append(
            RecommendationOptionV2(
                action="Investigate top and bottom segments separately",
                expected_impact="Surfaces actionable drivers by segment",
                risk="low",
                effort="medium",
                rationale="Grouped output supports targeted interventions.",
            )
        )
    if not recs:
        recs.append(
            RecommendationOptionV2(
                action="Validate trend with adjacent period comparison",
                expected_impact="Improves confidence in observed movement",
                risk="low",
                effort="low",
                rationale="Baseline recommendation when no stronger trigger is present.",
            )
        )
    return recs[:3]

