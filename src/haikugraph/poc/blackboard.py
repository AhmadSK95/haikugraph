"""Blackboard artifact helpers used by the agentic team orchestration layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from haikugraph.poc.decision_spine import (
    is_critical_handoff_artifact,
    validate_handoff_contract,
)


def append_blackboard_artifact(
    blackboard: list[dict[str, Any]],
    *,
    producer: str,
    artifact_type: str,
    payload: Any,
    consumed_by: list[str] | None,
    compact: Callable[[Any, int], Any],
) -> dict[str, Any]:
    """Append an artifact entry with contract metadata to the blackboard."""
    handoff = validate_handoff_contract(artifact_type, payload)
    blocked = bool(is_critical_handoff_artifact(artifact_type) and not handoff.valid)
    entry: dict[str, Any] = {
        "artifact_id": f"bb_{len(blackboard) + 1:03d}",
        "time": datetime.utcnow().isoformat() + "Z",
        "producer": producer,
        "artifact_type": artifact_type,
        "consumed_by": [] if blocked else list(consumed_by or []),
        "summary": compact(payload, 220),
        "handoff_contract": {
            "valid": handoff.valid,
            "required_fields": handoff.required_fields,
            "missing_fields": handoff.missing_fields,
            "reason_codes": handoff.reason_codes,
            "severity": handoff.severity,
            "blocked": blocked,
        },
    }
    if isinstance(payload, dict):
        preview: dict[str, Any] = {}
        for key in list(payload)[:8]:
            preview[str(key)] = compact(payload[key], 140)
        entry["payload_preview"] = preview
    else:
        entry["payload_preview"] = compact(payload, 180)
    blackboard.append(entry)
    return entry


def blackboard_edges(blackboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build producer->consumer edges with handoff contract reason codes."""
    edges: list[dict[str, Any]] = []
    for artifact in blackboard:
        producer = str(artifact.get("producer") or "")
        artifact_id = str(artifact.get("artifact_id") or "")
        artifact_type = str(artifact.get("artifact_type") or "")
        handoff = artifact.get("handoff_contract") if isinstance(artifact, dict) else {}
        reason_codes: list[str] = []
        if isinstance(handoff, dict):
            reason_codes = [str(code) for code in (handoff.get("reason_codes") or []) if str(code)]
        for consumer in artifact.get("consumed_by", []):
            edges.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "from": producer,
                    "to": str(consumer),
                    "reason_codes": reason_codes,
                }
            )
    return edges


def query_blackboard(
    blackboard: list[dict[str, Any]],
    *,
    producer: str | None = None,
    artifact_type: str | None = None,
) -> list[dict[str, Any]]:
    """Query blackboard entries by producer and/or artifact type."""
    return [
        entry
        for entry in blackboard
        if (not producer or entry.get("producer") == producer)
        and (not artifact_type or entry.get("artifact_type") == artifact_type)
    ]


def latest_blackboard(
    blackboard: list[dict[str, Any]],
    *,
    artifact_type: str,
) -> dict[str, Any] | None:
    """Return the latest artifact of a given type."""
    matches = query_blackboard(blackboard, artifact_type=artifact_type)
    return matches[-1] if matches else None
