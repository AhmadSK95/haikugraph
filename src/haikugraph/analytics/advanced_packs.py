"""Deterministic advanced analytics packs with policy and calibration metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for name in df.columns:
        try:
            series = pd.to_numeric(df[name], errors="coerce")
        except Exception:
            continue
        if series.notna().sum() >= 2:
            cols.append(str(name))
    return cols


def _pick_metric_column(df: pd.DataFrame) -> str | None:
    preferred = ("metric_value", "value", "amount", "count", "total")
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for token in preferred:
        if token in lower_map:
            return lower_map[token]
    numeric = _numeric_columns(df)
    return numeric[0] if numeric else None


def _outlier_pack(
    df: pd.DataFrame,
    metric_col: str,
    *,
    z_threshold: float,
    alert_pct_threshold: float,
) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if len(series) < 3:
        return {"status": "insufficient_data", "metric": metric_col, "outliers": []}
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return {"status": "no_variance", "metric": metric_col, "outliers": []}
    z = (series - mean) / std
    outlier_idx = z.abs() >= float(z_threshold)
    outliers = []
    for idx in series.index[outlier_idx]:
        outliers.append(
            {
                "row_index": int(idx),
                "value": float(series.loc[idx]),
                "z_score": round(float(z.loc[idx]), 3),
            }
        )
    outlier_count = len(outliers)
    pct_outliers = round((outlier_count / max(1, len(series))) * 100.0, 2)
    if pct_outliers >= max(1.0, alert_pct_threshold * 1.5):
        alert_level = "high"
    elif pct_outliers >= alert_pct_threshold:
        alert_level = "medium"
    elif outlier_count > 0:
        alert_level = "low"
    else:
        alert_level = "none"
    return {
        "status": "ok",
        "metric": metric_col,
        "outlier_count": outlier_count,
        "pct_outliers": pct_outliers,
        "policy": {
            "z_threshold": float(z_threshold),
            "alert_pct_threshold": float(alert_pct_threshold),
        },
        "alert": {
            "triggered": alert_level in {"medium", "high"},
            "level": alert_level,
            "reason": (
                "outlier share exceeded configured threshold"
                if alert_level in {"medium", "high"}
                else "below configured alert threshold"
            ),
        },
        "outliers": outliers[:12],
    }


def _select_variance_baseline(series: pd.Series, baseline_mode: str) -> tuple[float | None, str]:
    mode = str(baseline_mode or "first").strip().lower()
    if series.empty:
        return None, mode
    if mode == "mean":
        return float(series.mean()), mode
    if mode == "median":
        return float(series.median()), mode
    if mode == "previous_period":
        if len(series) >= 2:
            return float(series.iloc[-2]), mode
        return float(series.iloc[0]), mode
    if mode == "min":
        return float(series.min()), mode
    if mode == "max":
        return float(series.max()), mode
    # Default: first observed point.
    return float(series.iloc[0]), "first"


def _variance_pack(df: pd.DataFrame, metric_col: str, *, baseline_mode: str) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if len(series) < 2:
        return {"status": "insufficient_data", "metric": metric_col}
    baseline, baseline_mode_used = _select_variance_baseline(series, baseline_mode)
    actual = float(series.iloc[-1])
    if baseline is None:
        return {"status": "insufficient_data", "metric": metric_col}
    delta = actual - baseline
    pct = (delta / abs(baseline) * 100.0) if abs(baseline) > 1e-9 else None
    return {
        "status": "ok",
        "metric": metric_col,
        "baseline": baseline,
        "actual": actual,
        "delta": delta,
        "delta_pct": round(pct, 3) if pct is not None else None,
        "baseline_mode": baseline_mode_used,
    }


def _scenario_pack(
    df: pd.DataFrame,
    metric_col: str,
    *,
    assumptions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if series.empty:
        return {"status": "insufficient_data", "metric": metric_col}
    base = float(series.iloc[-1])

    prepared: list[dict[str, Any]] = []
    if isinstance(assumptions, list):
        for item in assumptions:
            if not isinstance(item, dict):
                continue
            change = _as_float(item.get("change_pct"))
            if change is None:
                continue
            prepared.append(
                {
                    "name": str(item.get("name") or f"assumption_{len(prepared) + 1}"),
                    "change_pct": float(change),
                    "note": str(item.get("note") or ""),
                }
            )
    if not prepared:
        prepared = [
            {"name": "stress_down_20", "change_pct": -0.2, "note": "stress downside"},
            {"name": "stress_down_10", "change_pct": -0.1, "note": "mild downside"},
            {"name": "upside_10", "change_pct": 0.1, "note": "mild upside"},
            {"name": "upside_20", "change_pct": 0.2, "note": "aggressive upside"},
        ]

    scenarios = []
    for item in prepared:
        pct = float(item.get("change_pct") or 0.0)
        scenarios.append(
            {
                "name": str(item.get("name") or ""),
                "change_pct": pct,
                "projected_value": round(base * (1.0 + pct), 6),
                "note": str(item.get("note") or ""),
            }
        )
    return {
        "status": "ok",
        "metric": metric_col,
        "base_value": base,
        "assumption_set_size": len(prepared),
        "scenarios": scenarios,
    }


def _rolling_ma_projection(series: pd.Series, horizon_steps: int) -> list[float]:
    values = [float(v) for v in series.tolist()]
    if not values:
        return []
    projection: list[float] = []
    window = min(3, len(values))
    history = list(values)
    for _ in range(max(1, horizon_steps)):
        pred = float(sum(history[-window:]) / max(1, window))
        projection.append(round(pred, 6))
        history.append(pred)
    return projection


def _forecast_pack(
    df: pd.DataFrame,
    metric_col: str,
    *,
    enabled: bool,
    horizon_steps: int,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "blocked_by_policy", "reason": "forecast pack disabled by policy gate"}
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if len(series) < 4:
        return {"status": "insufficient_data", "metric": metric_col}

    projection = _rolling_ma_projection(series, max(1, int(horizon_steps)))

    # Backtest calibration on one-step rolling-mean forecasts.
    errors: list[float] = []
    ape: list[float] = []
    vals = [float(v) for v in series.tolist()]
    for idx in range(3, len(vals)):
        pred = sum(vals[max(0, idx - 3):idx]) / min(3, idx)
        actual = vals[idx]
        err = abs(actual - pred)
        errors.append(err)
        if abs(actual) > 1e-9:
            ape.append((err / abs(actual)) * 100.0)
    mae = round(sum(errors) / max(1, len(errors)), 6)
    mape = round(sum(ape) / max(1, len(ape)), 3) if ape else None
    resid_std = float(pd.Series(errors).std(ddof=0)) if errors else 0.0
    ci_half = 1.96 * resid_std
    confidence_intervals = [
        {
            "step": i + 1,
            "lower": round(float(pred - ci_half), 6),
            "upper": round(float(pred + ci_half), 6),
        }
        for i, pred in enumerate(projection)
    ]
    return {
        "status": "ok",
        "metric": metric_col,
        "method": "rolling_mean_3",
        "horizon_steps": len(projection),
        "projection": projection,
        "confidence_intervals": confidence_intervals,
        "calibration": {
            "mae": mae,
            "mape_pct": mape,
            "sample_points": len(errors),
            "quality": (
                "good"
                if (mape is not None and mape <= 12.0)
                else ("moderate" if mape is not None and mape <= 20.0 else "uncertain")
            ),
        },
    }


def _funnel_pack(df: pd.DataFrame, *, goal_text: str) -> dict[str, Any]:
    status_col = next((c for c in df.columns if str(c).lower() in {"payment_status", "status", "stage"}), None)
    if status_col is None:
        return {"status": "not_available", "reason": "no stage/status column in result set"}
    counts = df.groupby(status_col, dropna=False).size().sort_values(ascending=False)
    if counts.empty:
        return {"status": "insufficient_data", "stages": []}
    top = float(counts.iloc[0])
    stages = [
        {
            "stage": str(idx),
            "count": int(val),
            "conversion_vs_top_pct": round((float(val) / top) * 100.0, 2) if top else None,
        }
        for idx, val in counts.items()
    ]
    lower_goal = str(goal_text or "").lower()
    if "payment" in lower_goal or "transaction" in lower_goal:
        template_id = "funnel.transactions.payment_status_v1"
        sql_templates = [
            {
                "id": "txn_stage_counts",
                "sql": (
                    "SELECT payment_status AS stage, COUNT(*) AS stage_count "
                    "FROM datada_mart_transactions GROUP BY 1 ORDER BY 2 DESC"
                ),
            },
            {
                "id": "txn_stage_conversion",
                "sql": (
                    "WITH stage_counts AS ("
                    "SELECT payment_status AS stage, COUNT(*) AS stage_count "
                    "FROM datada_mart_transactions GROUP BY 1"
                    ") SELECT stage, stage_count, "
                    "ROUND(stage_count * 100.0 / NULLIF(MAX(stage_count) OVER (), 0), 2) AS conversion_vs_top_pct "
                    "FROM stage_counts ORDER BY stage_count DESC"
                ),
            },
        ]
    elif "quote" in lower_goal:
        template_id = "funnel.quotes.status_v1"
        sql_templates = [
            {
                "id": "quote_stage_counts",
                "sql": (
                    "SELECT quote_status AS stage, COUNT(*) AS stage_count "
                    "FROM datada_mart_quotes GROUP BY 1 ORDER BY 2 DESC"
                ),
            }
        ]
    else:
        template_id = "funnel.generic.status_v1"
        sql_templates = [
            {
                "id": "generic_stage_counts",
                "sql": "SELECT status AS stage, COUNT(*) AS stage_count FROM <table> GROUP BY 1 ORDER BY 2 DESC",
            }
        ]
    return {
        "status": "ok",
        "template_id": template_id,
        "sql_template_hints": [
            "COUNT(*) grouped by status/stage",
            "conversion ratio vs first stage",
            "drop-off between adjacent stages",
        ],
        "sql_templates": sql_templates,
        "stages": stages,
    }


def _cohort_pack(df: pd.DataFrame) -> dict[str, Any]:
    cohort_col = next((c for c in df.columns if "customer_id" in str(c).lower() or "cohort" in str(c).lower()), None)
    time_col = next((c for c in df.columns if any(t in str(c).lower() for t in ("month", "_ts", "_at", "date"))), None)
    if cohort_col is None or time_col is None:
        return {"status": "not_available", "reason": "cohort/time columns not present in result set"}
    frame = df[[cohort_col, time_col]].dropna().copy()
    if frame.empty:
        return {"status": "insufficient_data", "retention": []}
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col])
    if frame.empty:
        return {"status": "insufficient_data", "retention": []}
    frame["cohort_month"] = frame.groupby(cohort_col)[time_col].transform("min").dt.to_period("M").astype(str)
    frame["activity_month"] = frame[time_col].dt.to_period("M").astype(str)
    cohort_counts = (
        frame.groupby(["cohort_month", "activity_month"])[cohort_col]
        .nunique()
        .reset_index(name="active_entities")
    )
    cohort_size = (
        frame.groupby("cohort_month")[cohort_col]
        .nunique()
        .reset_index(name="cohort_size")
    )
    merged = cohort_counts.merge(cohort_size, on="cohort_month", how="left")
    merged["retention_pct"] = (
        (merged["active_entities"] / merged["cohort_size"].clip(lower=1)) * 100.0
    ).round(2)

    pivot = merged.pivot_table(
        index="cohort_month",
        columns="activity_month",
        values="retention_pct",
        aggfunc="max",
    ).fillna(0.0)
    grid_rows: list[dict[str, Any]] = []
    for cohort_month, row in pivot.iterrows():
        entry = {"cohort_month": str(cohort_month)}
        for col_name, val in row.items():
            entry[str(col_name)] = float(val)
        grid_rows.append(entry)

    return {
        "status": "ok",
        "retention": merged.to_dict(orient="records")[:80],
        "cohort_grid": {
            "x_axis": [str(c) for c in list(pivot.columns)],
            "rows": grid_rows[:24],
        },
    }


def run_advanced_packs(
    df: pd.DataFrame,
    *,
    goal_text: str,
    forecast_enabled: bool = False,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run advanced analytics packs on query results with explicit controls."""
    if df is None or df.empty:
        return {"status": "empty_result", "packs": {}}

    options = dict(options or {})
    outlier_z_threshold = float(options.get("outlier_z_threshold", 1.5))
    outlier_alert_pct_threshold = float(options.get("outlier_alert_pct_threshold", 5.0))
    variance_baseline_mode = str(options.get("variance_baseline_mode", "first"))
    scenario_assumptions = options.get("scenario_assumptions")
    forecast_horizon_steps = int(options.get("forecast_horizon_steps", 3))

    metric_col = _pick_metric_column(df)
    packs: dict[str, Any] = {}
    if metric_col:
        packs["outlier"] = _outlier_pack(
            df,
            metric_col,
            z_threshold=outlier_z_threshold,
            alert_pct_threshold=outlier_alert_pct_threshold,
        )
        packs["variance"] = _variance_pack(
            df,
            metric_col,
            baseline_mode=variance_baseline_mode,
        )
        packs["scenario"] = _scenario_pack(
            df,
            metric_col,
            assumptions=scenario_assumptions if isinstance(scenario_assumptions, list) else None,
        )
        packs["forecast"] = _forecast_pack(
            df,
            metric_col,
            enabled=bool(forecast_enabled),
            horizon_steps=forecast_horizon_steps,
        )

    lower_goal = str(goal_text or "").lower()
    if any(tok in lower_goal for tok in ("funnel", "conversion", "drop-off", "dropoff")):
        packs["funnel"] = _funnel_pack(df, goal_text=goal_text)
    if any(tok in lower_goal for tok in ("cohort", "retention")):
        packs["cohort_retention"] = _cohort_pack(df)

    return {
        "status": "ok",
        "metric_column": metric_col,
        "pack_controls": {
            "outlier_z_threshold": outlier_z_threshold,
            "outlier_alert_pct_threshold": outlier_alert_pct_threshold,
            "variance_baseline_mode": variance_baseline_mode,
            "forecast_horizon_steps": forecast_horizon_steps,
            "forecast_enabled": bool(forecast_enabled),
        },
        "packs": packs,
    }
