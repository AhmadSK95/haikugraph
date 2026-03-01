from __future__ import annotations

import pandas as pd

from haikugraph.analytics.advanced_packs import run_advanced_packs


def test_advanced_packs_include_outlier_variance_scenario_forecast_policy_block():
    df = pd.DataFrame(
        {
            "metric_value": [10.0, 11.0, 12.0, 100.0],
            "payment_status": ["created", "processed", "processed", "completed"],
        }
    )
    result = run_advanced_packs(
        df,
        goal_text="find outliers and variance",
        forecast_enabled=False,
    )
    packs = result.get("packs", {})
    assert packs.get("outlier", {}).get("status") == "ok"
    assert int(packs.get("outlier", {}).get("outlier_count", 0)) >= 1
    assert (packs.get("outlier", {}).get("policy") or {}).get("z_threshold") == 1.5
    assert (packs.get("outlier", {}).get("alert") or {}).get("level") in {"low", "medium", "high", "none"}
    assert packs.get("variance", {}).get("status") == "ok"
    assert packs.get("variance", {}).get("baseline_mode") == "first"
    assert packs.get("scenario", {}).get("status") == "ok"
    assert int(packs.get("scenario", {}).get("assumption_set_size", 0)) >= 4
    assert packs.get("forecast", {}).get("status") == "blocked_by_policy"


def test_advanced_packs_forecast_enabled_returns_projection():
    df = pd.DataFrame({"metric_value": [8.0, 10.0, 12.0, 14.0]})
    result = run_advanced_packs(
        df,
        goal_text="forecast trend",
        forecast_enabled=True,
    )
    projection = result.get("packs", {}).get("forecast", {})
    assert projection.get("status") == "ok"
    assert projection.get("horizon_steps") == 3
    assert len(projection.get("projection", [])) == 3
    assert len(projection.get("confidence_intervals", [])) == 3
    calibration = projection.get("calibration", {})
    assert calibration.get("sample_points", 0) >= 1
    assert "mae" in calibration


def test_advanced_packs_include_funnel_when_goal_mentions_funnel():
    df = pd.DataFrame(
        {
            "metric_value": [1, 1, 1, 1, 1],
            "payment_status": ["initiated", "initiated", "validated", "completed", "completed"],
        }
    )
    result = run_advanced_packs(
        df,
        goal_text="show funnel conversion",
        forecast_enabled=False,
    )
    funnel = result.get("packs", {}).get("funnel", {})
    assert funnel.get("status") == "ok"
    assert str(funnel.get("template_id") or "").startswith("funnel.")
    templates = funnel.get("sql_templates") or []
    assert isinstance(templates, list)
    assert len(templates) >= 1
    assert "select" in str(templates[0].get("sql") or "").lower()
    assert len(funnel.get("stages", [])) >= 2


def test_advanced_packs_include_cohort_when_goal_mentions_retention():
    df = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c2", "c3"],
            "created_ts": [
                "2025-01-01",
                "2025-02-01",
                "2025-01-05",
                "2025-03-01",
                "2025-01-07",
            ],
            "metric_value": [1, 1, 1, 1, 1],
        }
    )
    result = run_advanced_packs(
        df,
        goal_text="cohort retention analysis",
        forecast_enabled=False,
    )
    cohort = result.get("packs", {}).get("cohort_retention", {})
    assert cohort.get("status") == "ok"
    assert len(cohort.get("retention", [])) >= 1
    grid = cohort.get("cohort_grid", {})
    assert isinstance(grid.get("rows"), list)


def test_advanced_packs_accept_variance_and_scenario_controls():
    df = pd.DataFrame({"metric_value": [10.0, 20.0, 40.0, 50.0]})
    result = run_advanced_packs(
        df,
        goal_text="variance and scenario planning",
        forecast_enabled=True,
        options={
            "variance_baseline_mode": "mean",
            "scenario_assumptions": [
                {"name": "tight_downside", "change_pct": -0.05, "note": "minor shock"},
                {"name": "aggressive_upside", "change_pct": 0.25, "note": "strong growth"},
            ],
            "forecast_horizon_steps": 2,
            "outlier_z_threshold": 1.2,
            "outlier_alert_pct_threshold": 10.0,
        },
    )
    packs = result.get("packs", {})
    variance = packs.get("variance", {})
    scenario = packs.get("scenario", {})
    forecast = packs.get("forecast", {})
    controls = result.get("pack_controls", {})
    assert variance.get("baseline_mode") == "mean"
    assert scenario.get("assumption_set_size") == 2
    assert len(scenario.get("scenarios", [])) == 2
    assert forecast.get("horizon_steps") == 2
    assert controls.get("outlier_z_threshold") == 1.2
