"""Action recommendation helpers (action + impact + risk)."""

from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_action_recommendations(
    *,
    goal: str,
    plan: dict[str, Any],
    execution: dict[str, Any],
    audit: dict[str, Any],
    advanced_packs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic, evidence-backed recommendations."""
    rows = execution.get("sample_rows") or []
    if not isinstance(rows, list):
        rows = []

    recs: list[dict[str, Any]] = []

    if str(plan.get("intent") or "") == "grouped_metric" and rows:
        values = []
        labels = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = _to_float(row.get("metric_value"))
            if val is None:
                continue
            values.append(float(val))
            dim_keys = [k for k in row.keys() if k not in {"metric_value", "secondary_metric_value"}]
            label = " | ".join(str(row.get(k)) for k in dim_keys if row.get(k) is not None) or "overall"
            labels.append(label)
        if values:
            total = sum(v for v in values if v >= 0)
            peak = max(values)
            peak_idx = values.index(peak)
            peak_share = (peak / total * 100.0) if total > 0 else 0.0
            if peak_share >= 45.0:
                recs.append(
                    {
                        "action": f"Reduce concentration on segment '{labels[peak_idx]}'.",
                        "expected_impact": "Improves resilience by reducing dependency on a single segment.",
                        "risk": "May reduce short-term performance in top segment while diversification ramps.",
                        "evidence": f"Top segment share is {peak_share:.1f}% of shown metric total.",
                    }
                )

    warnings = [str(w).strip() for w in (audit.get("warnings") or []) if str(w).strip()]
    if warnings:
        recs.append(
            {
                "action": "Resolve highest-priority data warning before operationalizing this insight.",
                "expected_impact": "Reduces risk of decisioning on incomplete or unstable slices.",
                "risk": "Delays action while validation is performed.",
                "evidence": warnings[0],
            }
        )

    packs = (advanced_packs or {}).get("packs", {}) if isinstance(advanced_packs, dict) else {}
    outlier_pack = packs.get("outlier", {}) if isinstance(packs, dict) else {}
    if outlier_pack.get("status") == "ok" and int(outlier_pack.get("outlier_count") or 0) > 0:
        recs.append(
            {
                "action": "Investigate outlier rows before setting KPI targets.",
                "expected_impact": "Prevents anomalous points from distorting planning baselines.",
                "risk": "Can over-correct if outliers represent valid business events.",
                "evidence": f"Detected {int(outlier_pack.get('outlier_count') or 0)} outlier points.",
            }
        )

    variance_pack = packs.get("variance", {}) if isinstance(packs, dict) else {}
    if variance_pack.get("status") == "ok":
        delta = _to_float(variance_pack.get("delta"))
        if delta is not None and delta < 0:
            recs.append(
                {
                    "action": "Run targeted mitigation on declining KPI segments.",
                    "expected_impact": "Can recover the observed decline in the next reporting cycle.",
                    "risk": "Misdirected interventions if decline is seasonal rather than structural.",
                    "evidence": f"Variance delta is {delta:.2f} vs selected baseline.",
                }
            )

    # Keep recommendations concise and high-signal.
    unique: list[dict[str, Any]] = []
    seen = set()
    for rec in recs:
        key = str(rec.get("action") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique[:3]


def render_recommendations_markdown(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return ""
    lines = ["**Recommended Actions**"]
    for rec in recommendations[:3]:
        action = str(rec.get("action") or "").strip()
        impact = str(rec.get("expected_impact") or "").strip()
        risk = str(rec.get("risk") or "").strip()
        evidence = str(rec.get("evidence") or "").strip()
        if not action:
            continue
        lines.append(
            f"- **Action:** {action} "
            f"**Impact:** {impact or 'n/a'} "
            f"**Risk:** {risk or 'n/a'}"
            + (f" **Evidence:** {evidence}" if evidence else "")
        )
    return "\n".join(lines).strip()

