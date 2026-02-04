"""Join coverage probing with DuckDB."""

import re

import duckdb


def safe_ident(name: str) -> str:
    """Validate SQL identifier is safe to use."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(
            f"Invalid SQL identifier: {name!r}. "
            "Must start with letter/underscore and contain only alphanumerics/underscores."
        )
    return name


def safe_table_ident(name: str) -> str:
    """Validate table identifier (allows dot-qualified names like schema.table)."""
    # Split by dots and validate each segment
    segments = name.split(".")
    for segment in segments:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", segment):
            raise ValueError(
                f"Invalid table identifier segment: {segment!r} in {name!r}. "
                "Each segment must start with letter/underscore and contain "
                "only alphanumerics/underscores."
            )
    return name


def probe_relation(
    conn: duckdb.DuckDBPyConnection,
    left_table: str,
    right_table: str,
    join_col: str,
    sample_limit: int = 200000,
) -> dict:
    """
    Probe a potential join relationship with bidirectional metrics and sampling.

    Args:
        conn: DuckDB connection
        left_table: Left table name
        right_table: Right table name
        join_col: Join column name
        sample_limit: Max rows for sampling distinct keys

    Returns:
        Dictionary with bidirectional probe metrics
    """
    # Validate identifiers
    safe_table_ident(left_table)
    safe_table_ident(right_table)
    safe_ident(join_col)

    col_safe = f'"{join_col}"'

    # Row counts
    left_rows = conn.execute(f"SELECT COUNT(*) FROM {left_table}").fetchone()[0]
    right_rows = conn.execute(f"SELECT COUNT(*) FROM {right_table}").fetchone()[0]

    # Distinct keys and null percentage
    left_stats = conn.execute(
        f"""
        SELECT
            approx_count_distinct({col_safe}) as distinct_keys,
            (COUNT(*) - COUNT({col_safe})) * 100.0 / NULLIF(COUNT(*), 0) as null_pct
        FROM {left_table}
        """
    ).fetchone()
    left_distinct_keys = left_stats[0]
    left_null_pct = round(left_stats[1], 2)

    right_stats = conn.execute(
        f"""
        SELECT
            approx_count_distinct({col_safe}) as distinct_keys,
            (COUNT(*) - COUNT({col_safe})) * 100.0 / NULLIF(COUNT(*), 0) as null_pct
        FROM {right_table}
        """
    ).fetchone()
    right_distinct_keys = right_stats[0]
    right_null_pct = round(right_stats[1], 2)

    # Duplication ratios
    left_dup_ratio = round(left_rows / left_distinct_keys, 2) if left_distinct_keys > 0 else 0.0
    right_dup_ratio = round(right_rows / right_distinct_keys, 2) if right_distinct_keys > 0 else 0.0

    # Use sampling for large tables
    use_left_sample = left_rows > sample_limit
    use_right_sample = right_rows > sample_limit

    # Key coverage: left to right
    try:
        if use_left_sample:
            # Sample distinct keys from left
            lk_query = f"""
                SELECT DISTINCT {col_safe} as key
                FROM {left_table} USING SAMPLE {sample_limit} ROWS
                WHERE {col_safe} IS NOT NULL
            """
        else:
            lk_query = f"""
                SELECT DISTINCT {col_safe} as key
                FROM {left_table}
                WHERE {col_safe} IS NOT NULL
            """

        coverage_l2r = conn.execute(
            f"""
            WITH lk AS ({lk_query}),
            rk AS (
                SELECT DISTINCT {col_safe} as key
                FROM {right_table}
                WHERE {col_safe} IS NOT NULL
            )
            SELECT
                COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM lk), 0) as coverage
            FROM lk
            INNER JOIN rk ON lk.key = rk.key
            """
        ).fetchone()
        key_coverage_l2r = round(coverage_l2r[0], 2) if coverage_l2r[0] else 0.0
    except Exception:
        key_coverage_l2r = 0.0

    # Key coverage: right to left
    try:
        if use_right_sample:
            # Sample distinct keys from right
            rk_query = f"""
                SELECT DISTINCT {col_safe} as key
                FROM {right_table} USING SAMPLE {sample_limit} ROWS
                WHERE {col_safe} IS NOT NULL
            """
        else:
            rk_query = f"""
                SELECT DISTINCT {col_safe} as key
                FROM {right_table}
                WHERE {col_safe} IS NOT NULL
            """

        coverage_r2l = conn.execute(
            f"""
            WITH rk AS ({rk_query}),
            lk AS (
                SELECT DISTINCT {col_safe} as key
                FROM {left_table}
                WHERE {col_safe} IS NOT NULL
            )
            SELECT
                COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM rk), 0) as coverage
            FROM rk
            INNER JOIN lk ON rk.key = lk.key
            """
        ).fetchone()
        key_coverage_r2l = round(coverage_r2l[0], 2) if coverage_r2l[0] else 0.0
    except Exception:
        key_coverage_r2l = 0.0

    # Row joinable percentage - use SEMI JOIN for efficiency
    try:
        # Left rows that join
        left_joinable = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {left_table} l
            SEMI JOIN {right_table} r ON l.{col_safe} = r.{col_safe}
            WHERE l.{col_safe} IS NOT NULL
            """
        ).fetchone()[0]
        row_joinable_pct_left = round((left_joinable * 100.0) / left_rows, 2)
    except Exception:
        row_joinable_pct_left = 0.0

    try:
        # Right rows that join
        right_joinable = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {right_table} r
            SEMI JOIN {left_table} l ON r.{col_safe} = l.{col_safe}
            WHERE r.{col_safe} IS NOT NULL
            """
        ).fetchone()[0]
        row_joinable_pct_right = round((right_joinable * 100.0) / right_rows, 2)
    except Exception:
        row_joinable_pct_right = 0.0

    # Orphan key percentages
    orphan_key_pct_left = round(100.0 - key_coverage_l2r, 2)
    orphan_key_pct_right = round(100.0 - key_coverage_r2l, 2)

    # Estimate cardinality with duplication ratios
    cardinality = estimate_cardinality(
        left_rows=left_rows,
        right_rows=right_rows,
        left_distinct=left_distinct_keys,
        right_distinct=right_distinct_keys,
        left_dup_ratio=left_dup_ratio,
        right_dup_ratio=right_dup_ratio,
        key_coverage_l2r=key_coverage_l2r,
        key_coverage_r2l=key_coverage_r2l,
        row_joinable_pct_left=row_joinable_pct_left,
    )

    return {
        "left_rows": left_rows,
        "right_rows": right_rows,
        "left_distinct_keys": left_distinct_keys,
        "right_distinct_keys": right_distinct_keys,
        "left_null_pct": left_null_pct,
        "right_null_pct": right_null_pct,
        "left_dup_ratio": left_dup_ratio,
        "right_dup_ratio": right_dup_ratio,
        "left_sampled": use_left_sample,
        "right_sampled": use_right_sample,
        "key_coverage_distinct_left_to_right": key_coverage_l2r,
        "key_coverage_distinct_right_to_left": key_coverage_r2l,
        "row_joinable_pct_left": row_joinable_pct_left,
        "row_joinable_pct_right": row_joinable_pct_right,
        "orphan_key_pct_left": orphan_key_pct_left,
        "orphan_key_pct_right": orphan_key_pct_right,
        "estimated_cardinality": cardinality,
    }


def estimate_cardinality(
    left_rows: int,
    right_rows: int,
    left_distinct: int,
    right_distinct: int,
    left_dup_ratio: float,
    right_dup_ratio: float,
    key_coverage_l2r: float,
    key_coverage_r2l: float,
    row_joinable_pct_left: float,
) -> str:
    """
    Estimate join cardinality using duplication ratios and bidirectional coverage.

    Returns:
        One of: "many_to_one", "one_to_one", "many_to_many", "unknown"
    """
    # Many-to-one: right side is dimension, left has duplicates
    if (
        right_dup_ratio <= 1.2  # Right side mostly unique
        and left_dup_ratio > 2  # Left has many duplicates
        and left_rows > 2 * right_rows  # Left has many more rows
        and key_coverage_l2r >= 70  # Good coverage left to right
    ):
        return "many_to_one"

    # One-to-one: both sides mostly unique with high coverage
    if (
        left_dup_ratio <= 1.2
        and right_dup_ratio <= 1.2
        and key_coverage_l2r >= 80
        and key_coverage_r2l >= 80
    ):
        return "one_to_one"

    # Many-to-many: both sides have duplicates and good joinability
    if left_dup_ratio > 2 and right_dup_ratio > 2 and row_joinable_pct_left >= 50:
        return "many_to_many"

    return "unknown"


def update_confidence_from_probe(base_confidence: float, probe: dict) -> float:
    """Update confidence score based on bidirectional probe results."""
    confidence = base_confidence

    # Bidirectional key coverage - interpret based on cardinality
    key_cov_l2r = probe.get("key_coverage_distinct_left_to_right", 0)
    key_cov_r2l = probe.get("key_coverage_distinct_right_to_left", 0)
    cardinality = probe.get("estimated_cardinality", "unknown")

    # Choose appropriate coverage metric based on relationship type
    if cardinality == "many_to_one":
        # For many-to-one, left-to-right coverage is most important (FK->PK)
        effective_coverage = key_cov_l2r
    elif cardinality == "one_to_one":
        # For one-to-one, both sides should have high coverage
        effective_coverage = min(key_cov_l2r, key_cov_r2l)
    else:
        # For many-to-many and unknown, use average
        effective_coverage = (key_cov_l2r + key_cov_r2l) / 2

    # Adjust confidence based on effective coverage
    if effective_coverage >= 90:
        confidence += 0.10
    elif effective_coverage >= 70:
        confidence += 0.05
    elif effective_coverage < 30:
        confidence -= 0.25

    # Penalize if both sides have very low coverage
    if key_cov_l2r < 30 and key_cov_r2l < 30:
        confidence -= 0.10

    # Adjust based on row joinability (primarily left, with right as secondary)
    row_joinable_left = probe.get("row_joinable_pct_left", 0)
    row_joinable_right = probe.get("row_joinable_pct_right", 0)

    if row_joinable_left < 30:
        confidence -= 0.20
    elif row_joinable_left >= 80 and row_joinable_right >= 80:
        confidence += 0.05

    # Penalize high null percentages
    if probe["left_null_pct"] > 50 or probe["right_null_pct"] > 50:
        confidence -= 0.05

    # Clamp to valid range
    return max(0.05, min(0.98, confidence))
