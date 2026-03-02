"""Decision memo formatter for analyst-team handoff mode."""

from __future__ import annotations

from haikugraph.v2.types import DecisionMemoV2, RecommendationOptionV2


def build_decision_memo(
    *,
    goal: str,
    summary: str,
    assumptions: list[str],
    quality_flags: list[str],
    recommendations: list[RecommendationOptionV2],
    certainty_tags: list[str],
) -> DecisionMemoV2:
    risks: list[str] = []
    for flag in quality_flags:
        if flag == "join_fragility":
            risks.append("Cross-table join coverage is fragile.")
        elif flag == "semantic_guard_missing":
            risks.append("A required semantic validity guard was not applied.")
        elif flag == "provider_degraded":
            risks.append("Provider degradation affected this run.")
        elif flag == "continuity_risk":
            risks.append("Follow-up continuity may be incomplete.")
    if not risks:
        risks.append("No material execution risks detected.")

    drivers = [
        f"Goal: {goal}",
        f"Observed quality flags: {', '.join(quality_flags) if quality_flags else 'none'}",
    ]
    return DecisionMemoV2(
        title="Decision memo",
        summary=summary,
        drivers=drivers,
        risks=risks,
        recommendations=recommendations,
        assumptions=assumptions,
        certainty_tags=certainty_tags,
    )

