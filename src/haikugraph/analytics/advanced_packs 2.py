"""Deterministic advanced analytics packs with explicit policy controls."""

from __future__ import annotations

from typing import Any

import pandas as pd


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


def _outlier_pack(df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if len(series) < 3:
        return {"status": "insufficient_data", "metric": metric_col, "outliers": []}
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return {"status": "no_variance", "metric": metric_col, "outliers": []}
    z = (series - mean) / std
    outlier_idx = z.abs() >= 1.5
    outliers = []
    for idx in series.index[outlier_idx]:
        outliers.append(
            {
                "row_index": int(idx),
                "value": float(series.loc[idx]),
                "z_score": round(float(z.loc[idx]), 3),
            }
        )
    return {"status": "ok", "metric": metric_col, "outlier_count": len(outliers), "outliers": outliers[:12]}


def _variance_pack(df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if len(series) < 2:
        return {"status": "insufficient_data", "metric": metric_col}
    first = float(series.iloc[0])
    last = float(series.iloc[-1])
    delta = last - first
    pct = (delta / abs(first) * 100.0) if abs(first) > 1e-9 else None
    return {
        "status": "ok",
        "metric": metric_col,
        "baseline": first,
        "actual": last,
        "delta": delta,
        "delta_pct": round(pct, 3) if pct is not None else None,
    }


def _scenario_pack(df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if series.empty:
        return {"status": "insufficient_data", "metric": metric_col}
    base = float(series.iloc[-1])
    scenarios = []
    for pct in (-0.2, -0.1, 0.1, 0.2):
        scenarios.append(
            {
                "change_pct": pct,
                "projected_value": round(base * (1.0 + pct), 6),
            }
        )
    return {"status": "ok", "metric": metric_col, "base_value": base, "scenarios": scenarios}


def _forecast_pack(df: pd.DataFrame, metric_col: str, *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "blocked_by_policy", "reason": "forecast pack disabled by policy gate"}
    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if len(series) < 3:
        return {"status": "insufficient_data", "metric": metric_col}
    window = min(3, len(series))
    rolling_mean = float(series.iloc[-window:].mean())
    projection = [round(rolling_mean, 6) for _ in range(3)]
    return {
        "status": "ok",
        "metric": metric_col,
        "method": f"moving_average_{window}",
        "horizon_steps": 3,
        "projection": projection,
    }


def _funnel_pack(df: pd.DataFrame) -> dict[str, Any]:
    status_col = next((c for c in df.columns if str(c).lower() in {"payment_status", "status", "stage"}), None)
    if status_col is None:
        return {"status": "not_available", "reason": "no stage/status column in result set"}
    counts = (
        df.groupby(status_col, dropna=False)
        .size()
        .sort_values(ascending=False)
    )
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
    return {"status": "ok", "stages": stages}


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
    return {
        "status": "ok",
        "retention": cohort_counts.to_dict(orient="records")[:60],
    }


def run_advanced_packs(
    df: pd.DataFrame,
    *,
    goal_text: str,
    forecast_enabled: bool = False,
) -> dict[str, Any]:
    """Run advanced analytics packs on query results."""
    if df is None or df.empty:
        return {"status": "empty_result", "packs": {}}

    metric_col = _pick_metric_column(df)
    packs: dict[str, Any] = {}
    if metric_col:
        packs["outlier"] = _outlier_pack(df, metric_col)
        packs["variance"] = _variance_pack(df, metric_col)
        packs["scenario"] = _scenario_pack(df, metric_col)
        packs["forecast"] = _forecast_pack(df, metric_col, enabled=bool(forecast_enabled))

    lower_goal = str(goal_text or "").lower()
    if any(tok in lower_goal for tok in ("funnel", "conversion", "drop-off", "dropoff")):
        packs["funnel"] = _funnel_pack(df)
    if any(tok in lower_goal for tok in ("cohort", "retention")):
        packs["cohort_retention"] = _cohort_pack(df)

    return {"status": "ok", "metric_column": metric_col, "packs": packs}
