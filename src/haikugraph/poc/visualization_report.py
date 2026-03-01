"""Visualization/report composition helpers."""

from __future__ import annotations

from typing import Any


def build_visualization_spec(plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    """Build chart spec with richer report panels while preserving legacy fields."""
    if not execution.get("success") or int(execution.get("row_count") or 0) <= 0:
        return {"type": "none", "reason": "no_data"}

    cols = list(execution.get("columns") or [])
    intent = str(plan.get("intent") or "")
    dims = [d for d in (plan.get("dimensions") or [plan.get("dimension")]) if d]
    lower_goal = str(plan.get("goal") or "").lower()

    if intent == "comparison" and set(cols) >= {"period", "metric_value"}:
        return {
            "type": "bar",
            "x": "period",
            "y": "metric_value",
            "title": "Current vs Comparison",
            "report": {
                "layout": "stack",
                "panels": [
                    {"type": "bar", "x": "period", "y": "metric_value", "title": "Period comparison"},
                    {"type": "kpi_delta", "left": "current", "right": "comparison", "title": "Delta summary"},
                ],
            },
        }

    if any(tok in lower_goal for tok in ("funnel", "conversion", "drop-off", "dropoff")):
        stage_col = next((c for c in cols if str(c).lower() in {"payment_status", "status", "stage"}), None)
        value_col = "metric_value" if "metric_value" in cols else (cols[-1] if cols else "value")
        return {
            "type": "funnel",
            "x": stage_col or "stage",
            "y": value_col,
            "title": "Funnel conversion",
            "report": {
                "layout": "stack",
                "panels": [
                    {"type": "funnel", "x": stage_col or "stage", "y": value_col, "title": "Stage conversion"},
                    {"type": "table", "columns": cols, "title": "Stage detail"},
                ],
            },
        }

    if any(tok in lower_goal for tok in ("cohort", "retention")):
        return {
            "type": "heatmap",
            "x": "activity_month",
            "y": "cohort_month",
            "value": "active_entities",
            "title": "Cohort retention grid",
            "report": {
                "layout": "stack",
                "panels": [
                    {"type": "heatmap", "x": "activity_month", "y": "cohort_month", "value": "active_entities"},
                    {"type": "table", "columns": cols, "title": "Retention data"},
                ],
            },
        }

    if intent == "grouped_metric" and len(cols) >= 2:
        if "secondary_metric_value" in cols:
            return {
                "type": "table",
                "columns": cols,
                "title": (
                    f"{plan.get('metric', 'metric')} and "
                    f"{plan.get('secondary_metric', 'secondary_metric')} by group"
                ),
                "report": {
                    "layout": "grid",
                    "panels": [
                        {
                            "type": "table",
                            "columns": cols,
                            "title": "Dual-metric table",
                        },
                        {
                            "type": "bar",
                            "x": cols[0],
                            "y": "metric_value",
                            "title": f"{plan.get('metric', 'metric')} by group",
                        },
                    ],
                },
            }
        chart_type = "line" if "__month__" in dims else "bar"
        x_col = cols[0]
        y_col = cols[-1]
        series_col = cols[1] if len(cols) >= 3 else None
        return {
            "type": chart_type,
            "x": x_col,
            "y": y_col,
            "series": series_col,
            "title": f"{plan.get('metric', 'metric')} by {', '.join([d for d in dims if d]) or x_col}",
            "report": {
                "layout": "stack",
                "panels": [
                    {"type": chart_type, "x": x_col, "y": y_col, "series": series_col},
                    {"type": "table", "columns": cols, "title": "Underlying grouped rows"},
                ],
            },
        }

    if len(cols) == 1 and int(execution.get("row_count") or 0) == 1:
        return {
            "type": "indicator",
            "metric": cols[0],
            "title": str(plan.get("metric") or cols[0]),
            "report": {
                "layout": "single",
                "panels": [{"type": "indicator", "metric": cols[0], "title": str(plan.get("metric") or cols[0])}],
            },
        }

    return {
        "type": "table",
        "columns": cols,
        "report": {
            "layout": "single",
            "panels": [{"type": "table", "columns": cols, "title": "Result table"}],
        },
    }

