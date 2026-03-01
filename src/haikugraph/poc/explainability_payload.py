"""Helpers for building dual-level explainability payloads."""

from __future__ import annotations

import re
from typing import Any


def build_dual_level_explainability_payload(
    *,
    goal: str,
    answer_markdown: str,
    confidence_score: float,
    intake: dict[str, Any],
    contract: dict[str, Any],
    contract_validation: dict[str, Any],
    decision_flow: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    sql: str,
    runtime_payload: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    contribution_map: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build dual-level explainability payload for business and technical users."""

    def _flatten_markdown(text: str) -> str:
        cleaned = re.sub(r"[*_`>#-]+", " ", str(text or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    metric = str(contract.get("metric") or intake.get("metric") or "metric")
    domain = str(contract.get("domain") or intake.get("domain") or "domain")
    table = str(contract.get("table") or "dataset")
    dimensions = [str(dim) for dim in (contract.get("dimensions") or intake.get("dimensions") or []) if str(dim)]
    time_scope = str(contract.get("time_scope") or intake.get("time_filter") or "all available data")
    validation_ok = bool(contract_validation.get("valid"))
    warning_list = list(warnings or [])

    row_count = 0
    execution_step = next(
        (step for step in decision_flow if str(step.get("step")) == "execution"),
        {},
    )
    if isinstance(execution_step, dict):
        details = execution_step.get("details")
        if isinstance(details, dict):
            row_count = int(details.get("row_count") or 0)

    business_steps = [
        (
            f"I interpreted your question as a `{intake.get('intent', 'metric')}` analysis "
            f"for the `{domain}` domain."
        ),
        (
            f"I bound the analysis to metric `{metric}` on `{table}` "
            f"with time scope `{time_scope}`."
        ),
        (
            "I verified that the generated SQL stayed inside the analysis contract: "
            f"{'passed' if validation_ok else 'failed with fixes/warnings'}."
        ),
        (
            f"I executed the query and returned {row_count} row(s), then scored confidence at "
            f"{round(float(confidence_score) * 100, 1)}%."
        ),
    ]
    if dimensions:
        business_steps.insert(
            2,
            f"I grouped results by: {', '.join(dimensions)}.",
        )
    if warning_list:
        business_steps.append(
            f"Warnings observed: {len(warning_list)} (see details below)."
        )

    answer_summary = _flatten_markdown(answer_markdown)[:320]

    return {
        "business_view": {
            "question": goal,
            "answer_summary": answer_summary,
            "focus": {
                "intent": intake.get("intent"),
                "domain": domain,
                "metric": metric,
                "table": table,
                "dimensions": dimensions,
                "time_scope": time_scope,
            },
            "quality": {
                "confidence_score": round(float(confidence_score), 4),
                "contract_valid": validation_ok,
                "warning_count": len(warning_list),
            },
            "plain_steps": business_steps,
            "warnings": warning_list,
        },
        "technical_view": {
            "sql": sql,
            "decision_flow": decision_flow,
            "agent_trace": trace,
            "contract_spec": contract,
            "contract_validation": contract_validation,
            "runtime": runtime_payload or {},
            "data_quality": data_quality or {},
            "contribution_map": contribution_map or [],
        },
    }
