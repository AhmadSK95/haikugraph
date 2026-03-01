"""Insight helpers for v2 response enrichment."""

from __future__ import annotations

import re

from haikugraph.v2.types import InsightReportV2, IntentSpecV2


def build_insight(intent: IntentSpecV2, goal: str) -> InsightReportV2:
    lower = str(goal or "").lower()
    assumptions: list[str] = []
    if any(k in lower for k in ["if we", "what if", "estimate", "likely impact", "scenario"]):
        assumptions.append("Scenario estimate relies on historical correlation, not causal certainty.")
        assumptions.append("No external market variables are included unless explicitly present in dataset.")
    if intent.requires_validity_guard:
        assumptions.append("Transaction validity interpreted with MT103 guard policy.")
    if re.search(r"\bforecast|predict|projection\b", lower):
        assumptions.append("Forecast intent requires explicit forecast capability and is constrained by policy.")
    return InsightReportV2(assumptions=assumptions)

