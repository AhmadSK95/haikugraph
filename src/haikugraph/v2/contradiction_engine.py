"""Contradiction detection for clarification-first behavior."""

from __future__ import annotations

from haikugraph.v2.types import ContradictionReportV2, IntentSpecV2


def detect_contradictions(goal: str, intent: IntentSpecV2 | None) -> ContradictionReportV2:
    lower = str(goal or "").lower()
    conflicts: list[str] = []

    if "same slice" in lower and "all data" in lower:
        conflicts.append("Requested both 'same slice' and 'all data'.")
    if "top" in lower and "bottom" in lower and "single" in lower:
        conflicts.append("Requested both top and bottom while asking for single result.")
    if "per customer" in lower and "per transaction" in lower:
        conflicts.append("Mixed denominator semantics: per customer and per transaction.")
    if intent is not None and intent.strategy == "comparison" and "no comparison" in lower:
        conflicts.append("Intent indicates comparison while prompt negates comparison.")

    if not conflicts:
        return ContradictionReportV2(detected=False, conflicts=[], clarification_prompt="", severity="none")
    return ContradictionReportV2(
        detected=True,
        conflicts=conflicts,
        clarification_prompt=(
            "I found conflicting constraints in your request. "
            "Should I prioritize the first instruction or return both variants?"
        ),
        severity="medium",
    )

