"""Candidate contradiction resolution utilities."""

from __future__ import annotations

from typing import Any


def _metric_family(metric_name: str) -> str:
    lower = str(metric_name or "").strip().lower()
    if "count" in lower:
        return "count"
    if any(tok in lower for tok in ("amount", "revenue", "spend", "value", "charge", "markup")):
        return "amount"
    if any(tok in lower for tok in ("rate", "ratio", "pct", "percent", "conversion")):
        return "rate"
    return "other"


def resolve_candidate_contradictions(candidate_evals: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect and summarize near-tie conflicts between candidate plans."""
    if len(candidate_evals) < 2:
        return {
            "detected": False,
            "needs_clarification": False,
            "reason": "single_candidate",
            "score_gap": 1.0,
            "conflict_signals": [],
        }

    ranked = sorted(candidate_evals, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    best = ranked[0]
    runner = ranked[1]
    score_gap = float(best.get("score", 0.0)) - float(runner.get("score", 0.0))
    low_margin = score_gap <= 0.08

    best_metric = str(best.get("metric") or "")
    runner_metric = str(runner.get("metric") or "")
    best_table = str(best.get("table") or "")
    runner_table = str(runner.get("table") or "")

    signals: list[str] = []
    if best_table and runner_table and best_table != runner_table:
        signals.append("table_conflict")
    if best_metric and runner_metric and best_metric != runner_metric:
        signals.append("metric_conflict")
    if _metric_family(best_metric) != _metric_family(runner_metric):
        signals.append("metric_family_conflict")

    best_misses = list(best.get("goal_term_misses") or [])
    runner_misses = list(runner.get("goal_term_misses") or [])
    if best_misses != runner_misses:
        signals.append("goal_term_conflict")

    best_failures = list(best.get("objective_failures") or [])
    runner_failures = list(runner.get("objective_failures") or [])
    if best_failures != runner_failures:
        signals.append("objective_failure_conflict")

    best_rows = int(best.get("row_count") or 0)
    runner_rows = int(runner.get("row_count") or 0)
    if abs(best_rows - runner_rows) >= max(2, int(0.4 * max(best_rows, runner_rows, 1))):
        signals.append("row_shape_conflict")

    detected = bool(low_margin and signals)
    severity = "none"
    if detected:
        if score_gap <= 0.03 and "metric_family_conflict" in signals:
            severity = "high"
        elif score_gap <= 0.05:
            severity = "medium"
        else:
            severity = "low"

    needs_clarification = bool(
        detected
        and (
            severity in {"high", "medium"}
            or "metric_family_conflict" in signals
            or ("table_conflict" in signals and "goal_term_conflict" in signals)
        )
    )
    clarification_prompt = ""
    if needs_clarification:
        clarification_prompt = (
            "I found two plausible interpretations with similar confidence. "
            f"Should I focus on `{best_metric or 'candidate A'}` from `{best_table or 'table A'}` "
            f"or `{runner_metric or 'candidate B'}` from `{runner_table or 'table B'}`?"
        )

    return {
        "detected": detected,
        "needs_clarification": needs_clarification,
        "reason": "near_tie_conflict" if detected else "winner_clear_or_no_conflict",
        "severity": severity,
        "score_gap": round(score_gap, 4),
        "conflict_signals": signals,
        "winner": {
            "candidate": best.get("candidate"),
            "table": best_table,
            "metric": best_metric,
            "score": best.get("score"),
            "metric_family": _metric_family(best_metric),
        },
        "runner_up": {
            "candidate": runner.get("candidate"),
            "table": runner_table,
            "metric": runner_metric,
            "score": runner.get("score"),
            "metric_family": _metric_family(runner_metric),
        },
        "clarification_prompt": clarification_prompt,
    }

