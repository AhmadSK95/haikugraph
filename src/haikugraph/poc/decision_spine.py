"""Shared decision-spine utilities for the agentic orchestrator.

These helpers intentionally stay pure and side-effect free so that
the orchestration runtime can reuse them across execution paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeSnapshot:
    requested_mode: str
    mode: str
    use_llm: bool
    provider: str | None
    reason: str


@dataclass(frozen=True)
class HandoffContractResult:
    valid: bool
    artifact_type: str
    required_fields: list[str]
    missing_fields: list[str]
    reason_codes: list[str]
    severity: str


_HANDOFF_CONTRACTS: dict[str, list[str]] = {
    "structured_goal": ["intent", "domain"],
    "domain_mapping": ["domain", "metrics", "columns"],
    "execution_plan": ["table", "metric", "intent"],
    "query_plan": ["sql", "table"],
    "execution_result": ["success", "row_count"],
    "audit_report": ["score"],
    "refinement_decision": ["correction_applied", "objective_coverage"],
    "business_answer": ["answer_markdown"],
    "visualization_spec": ["type"],
}

_CRITICAL_ARTIFACTS = {"structured_goal", "execution_plan", "query_plan", "execution_result", "audit_report"}


def grouped_signal(goal_text: str) -> bool:
    lower = str(goal_text or "").lower()
    return bool(
        re.search(
            r"\b(split|breakdown|group(?:ed)?|by month|month[\s-]?wise|platform[\s-]?wise)\b",
            lower,
        )
        or re.search(
            r"\bper\s+(platform|country|region|state|currency|deal\s+type|month|flow|status|customer\s+type|customer_type|type)\b",
            lower,
        )
    )


def metric_family(metric_name: str) -> str:
    low = str(metric_name or "").strip().lower()
    if not low:
        return "unknown"
    if low.endswith("_count") or "count" in low:
        return "count"
    if low.endswith("_rate") or low.startswith("avg_") or "ratio" in low or "average" in low:
        return "rate"
    if any(tok in low for tok in ("amount", "revenue", "spend", "value", "charges", "markup")):
        return "amount"
    return "other"


def validate_handoff_contract(artifact_type: str, payload: Any) -> HandoffContractResult:
    artifact = str(artifact_type or "").strip()
    required = list(_HANDOFF_CONTRACTS.get(artifact, []))
    if not required:
        return HandoffContractResult(
            valid=True,
            artifact_type=artifact,
            required_fields=[],
            missing_fields=[],
            reason_codes=[],
            severity="info",
        )
    data = payload if isinstance(payload, dict) else {}
    if artifact == "execution_plan":
        plan_intent = str(data.get("intent") or "").strip().lower()
        if plan_intent in {"schema_exploration", "data_overview", "document_qa"}:
            required = ["intent"]
    missing: list[str] = []
    for key in required:
        value = data.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
        elif isinstance(value, list) and not value:
            missing.append(key)
        elif isinstance(value, dict) and not value:
            missing.append(key)
    reason_codes = [f"handoff_missing_{name}" for name in missing]
    severity = "error" if (artifact in _CRITICAL_ARTIFACTS and missing) else ("warn" if missing else "info")
    return HandoffContractResult(
        valid=not missing,
        artifact_type=artifact,
        required_fields=required,
        missing_fields=missing,
        reason_codes=reason_codes,
        severity=severity,
    )


def is_critical_handoff_artifact(artifact_type: str) -> bool:
    return str(artifact_type or "").strip() in _CRITICAL_ARTIFACTS


def deterministic_runtime_snapshot(
    *,
    requested_mode: str,
    mode: str,
    provider: str | None,
) -> RuntimeSnapshot:
    if not requested_mode:
        requested_mode = "deterministic"
    if not mode:
        mode = requested_mode
    return RuntimeSnapshot(
        requested_mode=requested_mode,
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason=f"contract_guard_from_{mode}",
    )
