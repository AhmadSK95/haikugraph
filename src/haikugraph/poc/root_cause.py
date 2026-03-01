"""Root-cause hypothesis ranking helpers."""

from __future__ import annotations

from typing import Any


def _asks_for_root_cause(goal: str) -> bool:
    lower = str(goal or "").lower()
    tokens = [
        "root cause",
        "what caused",
        "why",
        "driver",
        "drivers",
        "reason for",
        "drop",
        "spike",
        "decline",
    ]
    return any(tok in lower for tok in tokens)


def _driver_label(row: dict[str, Any]) -> str:
    dims = [k for k in row.keys() if k not in {"metric_value", "secondary_metric_value"}]
    if not dims:
        return "overall"
    values = [f"{key}={row.get(key)}" for key in dims[:2]]
    return " | ".join(values)


def build_root_cause_hypotheses(
    *,
    goal: str,
    plan: dict[str, Any],
    execution: dict[str, Any],
    audit: dict[str, Any],
    max_drivers: int = 5,
) -> dict[str, Any]:
    rows = list(execution.get("sample_rows") or [])
    should_rank = _asks_for_root_cause(goal) or str(plan.get("intent") or "") in {
        "diagnostic",
        "grouped_metric",
        "comparison",
    }

    if not should_rank:
        return {
            "enabled": False,
            "reason": "goal_not_diagnostic",
            "ranked_drivers": [],
        }

    numeric_rows: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        try:
            value = float(row.get("metric_value"))
        except Exception:
            continue
        numeric_rows.append((row, value))

    if not numeric_rows:
        return {
            "enabled": True,
            "reason": "no_rankable_rows",
            "ranked_drivers": [],
            "needs_more_data": True,
        }

    total_abs = sum(abs(val) for _, val in numeric_rows) or 1.0
    ranked = sorted(numeric_rows, key=lambda item: abs(item[1]), reverse=True)[: max(1, int(max_drivers))]

    out: list[dict[str, Any]] = []
    for idx, (row, value) in enumerate(ranked, start=1):
        share = abs(value) / total_abs
        evidence_score = round(min(0.99, 0.45 + (share * 0.9)), 3)
        label = _driver_label(row)
        caveat = "Association from aggregated rows; validate with finer-grain cuts before action."
        if label == "overall":
            caveat = "No segmentation columns detected; this is directional only."
        out.append(
            {
                "rank": idx,
                "driver": label,
                "metric_value": value,
                "evidence_score": evidence_score,
                "evidence": f"Observed contribution share {share * 100:.1f}% in returned result set.",
                "caveat": caveat,
            }
        )

    warnings = [str(w) for w in (audit.get("warnings") or []) if str(w).strip()]
    misses = [str(m) for m in ((audit.get("grounding") or {}).get("goal_term_misses") or []) if str(m).strip()]

    return {
        "enabled": True,
        "reason": "ranked",
        "ranked_drivers": out,
        "warning_count": len(warnings),
        "grounding_misses": misses,
    }


def render_root_cause_markdown(payload: dict[str, Any]) -> str:
    rows = payload.get("ranked_drivers") if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return ""
    lines = ["**Root-Cause Hypotheses**"]
    for item in rows[:5]:
        lines.append(
            "- "
            f"#{int(item.get('rank') or 0)} {item.get('driver')}: "
            f"evidence score **{float(item.get('evidence_score') or 0.0):.2f}**. "
            f"{item.get('evidence')} Caveat: {item.get('caveat')}"
        )
    return "\n".join(lines)
