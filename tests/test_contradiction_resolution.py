from __future__ import annotations

from haikugraph.poc.contradiction_resolution import resolve_candidate_contradictions


def test_contradiction_resolution_single_candidate():
    result = resolve_candidate_contradictions(
        [{"candidate": "base_plan", "metric": "transaction_count", "table": "datada_mart_transactions", "score": 0.91}]
    )
    assert result.get("detected") is False
    assert result.get("needs_clarification") is False
    assert result.get("reason") == "single_candidate"


def test_contradiction_resolution_near_tie_metric_family_conflict_requests_clarification():
    result = resolve_candidate_contradictions(
        [
            {
                "candidate": "count_variant",
                "metric": "transaction_count",
                "table": "datada_mart_transactions",
                "score": 0.83,
                "goal_term_misses": ["amount"],
                "objective_failures": ["metric_alignment"],
                "row_count": 4,
            },
            {
                "candidate": "amount_variant",
                "metric": "total_amount",
                "table": "datada_mart_transactions",
                "score": 0.81,
                "goal_term_misses": [],
                "objective_failures": [],
                "row_count": 4,
            },
        ]
    )
    assert result.get("detected") is True
    assert result.get("needs_clarification") is True
    assert "metric_family_conflict" in (result.get("conflict_signals") or [])
    assert "should i focus on" in str(result.get("clarification_prompt") or "").lower()


def test_contradiction_resolution_clear_winner_no_clarification():
    result = resolve_candidate_contradictions(
        [
            {
                "candidate": "best",
                "metric": "transaction_count",
                "table": "datada_mart_transactions",
                "score": 0.92,
                "goal_term_misses": [],
                "objective_failures": [],
                "row_count": 10,
            },
            {
                "candidate": "runner",
                "metric": "transaction_count",
                "table": "datada_mart_transactions",
                "score": 0.70,
                "goal_term_misses": [],
                "objective_failures": [],
                "row_count": 9,
            },
        ]
    )
    assert result.get("detected") is False
    assert result.get("needs_clarification") is False
    assert result.get("reason") == "winner_clear_or_no_conflict"
