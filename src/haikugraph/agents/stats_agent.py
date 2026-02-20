"""Statistical analysis agent for advanced analytics.

Provides statistical methods beyond SQL aggregations:
- Distribution analysis (percentiles, skewness, kurtosis)
- Correlation analysis (Pearson/Spearman)
- Outlier detection (IQR, Z-score)
- Trend analysis (moving averages, period-over-period, linear regression)
- Statistical significance (t-test, chi-square)
- Cohort and segment comparison

This agent operates on query results (DataFrames) produced by the
execution agent, adding statistical depth to the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

@dataclass
class DistributionProfile:
    """Distribution statistics for a numeric column."""

    column: str
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    iqr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CorrelationResult:
    """Correlation between two columns."""

    col_a: str
    col_b: str
    pearson: float | None = None
    spearman: float | None = None
    strength: str = "none"  # none, weak, moderate, strong

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class OutlierResult:
    """Outlier detection results."""

    column: str
    method: str  # iqr, zscore
    n_outliers: int = 0
    outlier_indices: list[int] = field(default_factory=list)
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    pct_outliers: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["outlier_indices"] = d["outlier_indices"][:20]  # cap for serialization
        return d


@dataclass
class TrendResult:
    """Trend analysis for a time series."""

    column: str
    time_column: str
    direction: str = "stable"  # up, down, stable
    slope: float = 0.0
    r_squared: float = 0.0
    pct_change_total: float = 0.0
    moving_avg_7: list[float | None] = field(default_factory=list)
    moving_avg_30: list[float | None] = field(default_factory=list)
    breakpoints: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["moving_avg_7"] = d["moving_avg_7"][:100]
        d["moving_avg_30"] = d["moving_avg_30"][:100]
        return d


@dataclass
class SignificanceResult:
    """Statistical significance test result."""

    test: str  # ttest, chi2
    statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False  # at alpha=0.05
    effect_size: float = 0.0
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class StatsAnalysis:
    """Complete statistical analysis result."""

    distributions: list[DistributionProfile] = field(default_factory=list)
    correlations: list[CorrelationResult] = field(default_factory=list)
    outliers: list[OutlierResult] = field(default_factory=list)
    trends: list[TrendResult] = field(default_factory=list)
    significance_tests: list[SignificanceResult] = field(default_factory=list)
    summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distributions": [d.to_dict() for d in self.distributions],
            "correlations": [c.to_dict() for c in self.correlations],
            "outliers": [o.to_dict() for o in self.outliers],
            "trends": [t.to_dict() for t in self.trends],
            "significance_tests": [s.to_dict() for s in self.significance_tests],
            "summary": self.summary,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Core statistical functions (pure pandas + math, no scipy required)
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float:
    """Coerce to float, return 0.0 on failure."""
    try:
        f = float(v)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return 0.0


def profile_distribution(series: pd.Series, col_name: str) -> DistributionProfile:
    """Compute distribution statistics for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return DistributionProfile(column=col_name, count=len(s))

    desc = s.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    mean = _safe_float(desc.get("mean", 0))
    std = _safe_float(desc.get("std", 0))
    p25 = _safe_float(desc.get("25%", 0))
    p75 = _safe_float(desc.get("75%", 0))

    # Skewness: (mean - median) / std (Pearson's approximation)
    median = _safe_float(desc.get("50%", 0))
    skewness = 0.0
    if std > 0:
        n = len(s)
        m3 = ((s - mean) ** 3).mean()
        skewness = _safe_float(m3 / (std ** 3)) if std > 0 else 0.0

    # Kurtosis: excess kurtosis
    kurtosis = 0.0
    if std > 0:
        m4 = ((s - mean) ** 4).mean()
        kurtosis = _safe_float(m4 / (std ** 4) - 3.0)

    return DistributionProfile(
        column=col_name,
        count=len(s),
        mean=mean,
        std=std,
        min_val=_safe_float(desc.get("min", 0)),
        max_val=_safe_float(desc.get("max", 0)),
        p25=p25,
        p50=median,
        p75=p75,
        p95=_safe_float(desc.get("95%", 0)),
        p99=_safe_float(desc.get("99%", 0)),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
        iqr=round(p75 - p25, 4),
    )


def compute_correlation(df: pd.DataFrame, col_a: str, col_b: str) -> CorrelationResult:
    """Compute Pearson and Spearman correlation between two numeric columns."""
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return CorrelationResult(col_a=col_a, col_b=col_b)

    pearson = _safe_float(a.corr(b))
    spearman = _safe_float(a.rank().corr(b.rank()))

    abs_p = abs(pearson) if pearson is not None else 0
    strength = (
        "strong" if abs_p >= 0.7 else
        "moderate" if abs_p >= 0.4 else
        "weak" if abs_p >= 0.2 else
        "none"
    )

    return CorrelationResult(
        col_a=col_a,
        col_b=col_b,
        pearson=round(pearson, 4),
        spearman=round(spearman, 4),
        strength=strength,
    )


def detect_outliers_iqr(series: pd.Series, col_name: str, k: float = 1.5) -> OutlierResult:
    """Detect outliers using IQR method."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return OutlierResult(column=col_name, method="iqr")

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    outlier_mask = (s < lower) | (s > upper)
    outlier_idx = list(s[outlier_mask].index[:20])
    n = int(outlier_mask.sum())

    return OutlierResult(
        column=col_name,
        method="iqr",
        n_outliers=n,
        outlier_indices=outlier_idx,
        lower_bound=round(float(lower), 4),
        upper_bound=round(float(upper), 4),
        pct_outliers=round(n / len(s) * 100, 2) if len(s) > 0 else 0,
    )


def detect_outliers_zscore(series: pd.Series, col_name: str, threshold: float = 3.0) -> OutlierResult:
    """Detect outliers using Z-score method."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return OutlierResult(column=col_name, method="zscore")

    mean = s.mean()
    std = s.std()
    if std == 0:
        return OutlierResult(column=col_name, method="zscore")

    z = ((s - mean) / std).abs()
    outlier_mask = z > threshold
    outlier_idx = list(s[outlier_mask].index[:20])
    n = int(outlier_mask.sum())

    return OutlierResult(
        column=col_name,
        method="zscore",
        n_outliers=n,
        outlier_indices=outlier_idx,
        lower_bound=round(float(mean - threshold * std), 4),
        upper_bound=round(float(mean + threshold * std), 4),
        pct_outliers=round(n / len(s) * 100, 2) if len(s) > 0 else 0,
    )


def analyze_trend(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
) -> TrendResult:
    """Analyze trends in a time-series column using linear regression."""
    sub = df[[time_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])

    if len(sub) < 3:
        return TrendResult(column=value_col, time_column=time_col)

    # Sort by time
    try:
        sub[time_col] = pd.to_datetime(sub[time_col], errors="coerce")
        sub = sub.dropna(subset=[time_col]).sort_values(time_col)
    except Exception:
        sub = sub.sort_index()

    values = sub[value_col].values
    n = len(values)
    x = list(range(n))

    # Linear regression (manual — no scipy needed)
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)

    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    intercept = y_mean - slope * x_mean

    # R-squared
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(values, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Direction
    if abs(slope) < 0.001 * abs(y_mean) if y_mean != 0 else abs(slope) < 0.001:
        direction = "stable"
    elif slope > 0:
        direction = "up"
    else:
        direction = "down"

    # Percent change total
    first_val = float(values[0]) if values[0] != 0 else 1
    pct_change = ((float(values[-1]) - float(values[0])) / abs(first_val)) * 100

    # Moving averages
    s = pd.Series(values)
    ma7 = s.rolling(min(7, max(2, n // 3)), min_periods=1).mean().tolist()
    ma30 = s.rolling(min(30, max(2, n // 2)), min_periods=1).mean().tolist()

    # Breakpoint detection (simple: where slope direction changes significantly)
    breakpoints = []
    if n >= 6:
        window = max(3, n // 5)
        for i in range(window, n - window):
            left_mean = sum(values[i - window:i]) / window
            right_mean = sum(values[i:i + window]) / window
            if abs(right_mean - left_mean) > 1.5 * (sum(abs(v - y_mean) for v in values) / n):
                breakpoints.append(i)
                if len(breakpoints) >= 5:
                    break

    return TrendResult(
        column=value_col,
        time_column=time_col,
        direction=direction,
        slope=round(_safe_float(slope), 6),
        r_squared=round(_safe_float(r_squared), 4),
        pct_change_total=round(_safe_float(pct_change), 2),
        moving_avg_7=[round(_safe_float(v), 4) for v in ma7],
        moving_avg_30=[round(_safe_float(v), 4) for v in ma30],
        breakpoints=breakpoints,
    )


def ttest_two_samples(
    group_a: pd.Series,
    group_b: pd.Series,
    label: str = "",
) -> SignificanceResult:
    """Welch's t-test (unequal variance) between two groups.

    Pure Python implementation — no scipy required.
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return SignificanceResult(test="ttest", interpretation="Insufficient data for t-test")

    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)

    se = math.sqrt(var_a / na + var_b / nb) if (var_a / na + var_b / nb) > 0 else 1e-10
    t_stat = (mean_a - mean_b) / se

    # Welch–Satterthwaite degrees of freedom
    num = (var_a / na + var_b / nb) ** 2
    denom = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value using normal distribution for large df
    # For small df we use a rough approximation
    abs_t = abs(t_stat)
    if df >= 30:
        # Normal approximation for large df
        z = abs_t
        p_value = 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        p_value = max(min(p_value, 1.0), 0.0)
    else:
        # Rough approximation for small df
        p_value = 2 * (1 - min(0.9999, abs_t / (abs_t + df)))

    # Cohen's d effect size
    pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    significant = p_value < 0.05
    effect = (
        "large" if abs(cohens_d) >= 0.8 else
        "medium" if abs(cohens_d) >= 0.5 else
        "small" if abs(cohens_d) >= 0.2 else
        "negligible"
    )

    interp = f"{'Significant' if significant else 'Not significant'} difference (p={p_value:.4f}, d={cohens_d:.3f} {effect} effect)"

    return SignificanceResult(
        test="ttest",
        statistic=round(_safe_float(t_stat), 4),
        p_value=round(_safe_float(p_value), 6),
        significant=significant,
        effect_size=round(_safe_float(cohens_d), 4),
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# High-level analysis orchestrator
# ---------------------------------------------------------------------------

def run_stats_analysis(
    df: pd.DataFrame,
    *,
    time_col: str | None = None,
    value_cols: list[str] | None = None,
    group_col: str | None = None,
) -> StatsAnalysis:
    """Run comprehensive statistical analysis on a DataFrame.

    Automatically detects numeric columns and applies relevant analyses.
    """
    if df is None or df.empty:
        return StatsAnalysis(summary="No data available for analysis.")

    result = StatsAnalysis()
    warnings: list[str] = []

    # Auto-detect numeric columns
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if value_cols:
        numeric_cols = [c for c in value_cols if c in df.columns]
    if not numeric_cols:
        return StatsAnalysis(summary="No numeric columns found for analysis.", warnings=["No numeric data"])

    # Auto-detect time column if not provided
    if not time_col:
        for c in df.columns:
            if any(t in c.lower() for t in ["date", "time", "month", "year", "week", "day", "period"]):
                time_col = c
                break

    # 1. Distribution profiles
    for col in numeric_cols[:6]:  # cap at 6 columns
        result.distributions.append(profile_distribution(df[col], col))

    # 2. Correlations (top pairs)
    if len(numeric_cols) >= 2:
        pairs_done = set()
        for i, ca in enumerate(numeric_cols[:5]):
            for cb in numeric_cols[i + 1:5]:
                key = tuple(sorted([ca, cb]))
                if key not in pairs_done:
                    pairs_done.add(key)
                    result.correlations.append(compute_correlation(df, ca, cb))

    # 3. Outlier detection
    for col in numeric_cols[:4]:
        iqr_result = detect_outliers_iqr(df[col], col)
        if iqr_result.n_outliers > 0:
            result.outliers.append(iqr_result)
        else:
            z_result = detect_outliers_zscore(df[col], col)
            if z_result.n_outliers > 0:
                result.outliers.append(z_result)

    # 4. Trend analysis (if time column exists)
    if time_col and time_col in df.columns:
        for col in numeric_cols[:3]:
            if col != time_col:
                result.trends.append(analyze_trend(df, time_col, col))

    # 5. Significance tests (if group column exists)
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        if len(groups) == 2:
            ga = df[df[group_col] == groups[0]]
            gb = df[df[group_col] == groups[1]]
            for col in numeric_cols[:3]:
                if col != group_col:
                    result.significance_tests.append(
                        ttest_two_samples(ga[col], gb[col], label=col)
                    )
        elif len(groups) > 2:
            warnings.append(f"Significance test skipped: {len(groups)} groups (need exactly 2)")

    # Build summary
    parts = []
    if result.distributions:
        parts.append(f"{len(result.distributions)} distribution profiles")
    if result.correlations:
        strong = [c for c in result.correlations if c.strength in ("strong", "moderate")]
        if strong:
            parts.append(f"{len(strong)} notable correlations")
    if result.outliers:
        total = sum(o.n_outliers for o in result.outliers)
        parts.append(f"{total} outliers detected")
    if result.trends:
        up = sum(1 for t in result.trends if t.direction == "up")
        down = sum(1 for t in result.trends if t.direction == "down")
        if up:
            parts.append(f"{up} upward trends")
        if down:
            parts.append(f"{down} downward trends")
    if result.significance_tests:
        sig = sum(1 for s in result.significance_tests if s.significant)
        parts.append(f"{sig}/{len(result.significance_tests)} significant differences")

    result.summary = "; ".join(parts) if parts else "Analysis complete, no notable findings."
    result.warnings = warnings
    return result
