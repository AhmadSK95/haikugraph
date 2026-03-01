from __future__ import annotations

from haikugraph.poc.visualization_report import build_visualization_spec


def test_visualization_report_for_comparison_includes_panels():
    plan = {"intent": "comparison", "goal": "compare this month vs last month"}
    execution = {
        "success": True,
        "row_count": 2,
        "columns": ["period", "metric_value"],
        "sample_rows": [{"period": "current", "metric_value": 10}, {"period": "comparison", "metric_value": 8}],
    }
    spec = build_visualization_spec(plan, execution)
    assert spec.get("type") == "bar"
    report = spec.get("report") or {}
    assert isinstance(report.get("panels"), list)
    assert len(report.get("panels")) >= 2


def test_visualization_report_for_dual_metric_grouped_has_grid_report():
    plan = {
        "intent": "grouped_metric",
        "metric": "transaction_count",
        "secondary_metric": "total_amount",
        "dimensions": ["platform_name"],
        "goal": "count and amount by platform",
    }
    execution = {
        "success": True,
        "row_count": 2,
        "columns": ["platform_name", "metric_value", "secondary_metric_value"],
        "sample_rows": [
            {"platform_name": "web", "metric_value": 20, "secondary_metric_value": 2000.0},
            {"platform_name": "app", "metric_value": 15, "secondary_metric_value": 1300.0},
        ],
    }
    spec = build_visualization_spec(plan, execution)
    assert spec.get("type") == "table"
    report = spec.get("report") or {}
    assert report.get("layout") == "grid"


def test_visualization_report_detects_funnel_goal():
    plan = {"intent": "grouped_metric", "goal": "show funnel conversion by stage"}
    execution = {
        "success": True,
        "row_count": 3,
        "columns": ["payment_status", "metric_value"],
        "sample_rows": [
            {"payment_status": "initiated", "metric_value": 100},
            {"payment_status": "processed", "metric_value": 60},
            {"payment_status": "completed", "metric_value": 40},
        ],
    }
    spec = build_visualization_spec(plan, execution)
    assert spec.get("type") == "funnel"
    assert "report" in spec

