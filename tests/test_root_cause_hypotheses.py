from __future__ import annotations

from haikugraph.poc.root_cause import build_root_cause_hypotheses


def test_root_cause_hypotheses_rank_drivers_with_evidence() -> None:
    payload = build_root_cause_hypotheses(
        goal="What is the root cause of volume changes?",
        plan={"intent": "grouped_metric", "metric": "transaction_count"},
        execution={
            "sample_rows": [
                {"platform_name": "WEB", "metric_value": 120},
                {"platform_name": "APP", "metric_value": 80},
            ]
        },
        audit={"warnings": [], "grounding": {}},
    )
    assert payload["enabled"] is True
    ranked = payload.get("ranked_drivers") or []
    assert len(ranked) >= 1
    first = ranked[0]
    assert first["rank"] == 1
    assert "driver" in first and str(first["driver"])
    assert "evidence_score" in first
    assert "caveat" in first and str(first["caveat"])
