from __future__ import annotations

from haikugraph.poc.blackboard import (
    append_blackboard_artifact,
    blackboard_edges,
    latest_blackboard,
    query_blackboard,
)


def _compact(value: object, max_len: int) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def test_blackboard_blocks_invalid_critical_handoff() -> None:
    blackboard: list[dict[str, object]] = []
    entry = append_blackboard_artifact(
        blackboard,
        producer="PlanningAgent",
        artifact_type="query_plan",
        payload={"sql": "SELECT 1"},
        consumed_by=["ExecutionAgent"],
        compact=_compact,
    )
    handoff = entry.get("handoff_contract")
    assert isinstance(handoff, dict)
    assert handoff.get("valid") is False
    assert handoff.get("blocked") is True
    assert entry.get("consumed_by") == []
    assert "handoff_missing_table" in list(handoff.get("reason_codes") or [])


def test_blackboard_preserves_non_critical_consumers_with_reason_codes() -> None:
    blackboard: list[dict[str, object]] = []
    append_blackboard_artifact(
        blackboard,
        producer="AutonomyAgent",
        artifact_type="refinement_decision",
        payload={"correction_applied": False},
        consumed_by=["NarrationAgent"],
        compact=_compact,
    )
    edges = blackboard_edges(blackboard)
    assert len(edges) == 1
    assert edges[0]["to"] == "NarrationAgent"
    assert "handoff_missing_objective_coverage" in edges[0]["reason_codes"]


def test_blackboard_query_and_latest_helpers() -> None:
    blackboard: list[dict[str, object]] = []
    append_blackboard_artifact(
        blackboard,
        producer="AgentA",
        artifact_type="business_answer",
        payload={"answer_markdown": "first"},
        consumed_by=[],
        compact=_compact,
    )
    append_blackboard_artifact(
        blackboard,
        producer="AgentB",
        artifact_type="business_answer",
        payload={"answer_markdown": "second"},
        consumed_by=[],
        compact=_compact,
    )
    by_producer = query_blackboard(blackboard, producer="AgentA")
    assert len(by_producer) == 1
    latest = latest_blackboard(blackboard, artifact_type="business_answer")
    assert latest is not None
    payload_preview = latest.get("payload_preview")
    assert isinstance(payload_preview, dict)
    assert payload_preview.get("answer_markdown") == "second"
