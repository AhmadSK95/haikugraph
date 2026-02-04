"""DuckDB table profiling utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


def profile_column(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column_name: str,
    column_type: str,
    row_count: int,
    sample_rows: int,
    top_k: int,
) -> dict[str, Any]:
    """
    Profile a single column.

    Args:
        conn: DuckDB connection
        table_name: Name of table
        column_name: Name of column
        column_type: DuckDB type
        row_count: Total rows in table
        sample_rows: Max rows for expensive operations
        top_k: Top values to return for categorical columns

    Returns:
        Dictionary with column profile
    """
    profile: dict[str, Any] = {
        "name": column_name,
        "duckdb_type": column_type,
    }

    # Escape column name for SQL (handle special chars)
    col_safe = f'"{column_name}"'

    try:
        # Basic stats: null count, distinct count
        use_approx = row_count > sample_rows

        if use_approx:
            stats_query = f"""
                SELECT
                    COUNT(*) - COUNT({col_safe}) as null_count,
                    approx_count_distinct({col_safe}) as distinct_count
                FROM {table_name}
            """
        else:
            stats_query = f"""
                SELECT
                    COUNT(*) - COUNT({col_safe}) as null_count,
                    COUNT(DISTINCT {col_safe}) as distinct_count
                FROM {table_name}
            """

        result = conn.execute(stats_query).fetchone()
        null_count = result[0]
        distinct_count = result[1]

        profile["null_count"] = null_count
        profile["null_pct"] = round(null_count / row_count * 100, 2) if row_count > 0 else 0
        profile["distinct_count"] = distinct_count

        # Sample values (non-null)
        sample_query = f"""
            SELECT DISTINCT {col_safe}
            FROM {table_name}
            WHERE {col_safe} IS NOT NULL
            LIMIT 5
        """
        sample_results = conn.execute(sample_query).fetchall()
        profile["sample_values"] = [str(row[0]) for row in sample_results]

        # Type-specific profiling
        lower_type = column_type.lower()

        # Numeric types
        if any(t in lower_type for t in ["int", "float", "double", "decimal", "numeric", "real"]):
            numeric_query = f"""
                SELECT
                    MIN({col_safe}) as min_val,
                    MAX({col_safe}) as max_val,
                    AVG({col_safe}) as mean_val
                FROM {table_name}
                WHERE {col_safe} IS NOT NULL
            """
            num_result = conn.execute(numeric_query).fetchone()
            if num_result[0] is not None:
                profile["min"] = float(num_result[0])
                profile["max"] = float(num_result[1])
                profile["mean"] = round(float(num_result[2]), 4) if num_result[2] else None

        # Date/Timestamp types
        elif any(t in lower_type for t in ["date", "timestamp"]):
            date_query = f"""
                SELECT
                    MIN({col_safe}) as min_date,
                    MAX({col_safe}) as max_date
                FROM {table_name}
                WHERE {col_safe} IS NOT NULL
            """
            date_result = conn.execute(date_query).fetchone()
            if date_result[0] is not None:
                profile["min_date"] = str(date_result[0])
                profile["max_date"] = str(date_result[1])

        # String/categorical - show top values if low cardinality
        if "varchar" in lower_type or "char" in lower_type or "text" in lower_type:
            # Only compute top values if reasonably low cardinality
            if distinct_count <= 1000:
                top_query = f"""
                    SELECT
                        {col_safe} as value,
                        COUNT(*) as count
                    FROM {table_name}
                    WHERE {col_safe} IS NOT NULL
                    GROUP BY {col_safe}
                    ORDER BY count DESC
                    LIMIT {top_k}
                """
                top_results = conn.execute(top_query).fetchall()
                profile["top_values"] = [
                    {
                        "value": str(row[0]),
                        "count": int(row[1]),
                        "pct": round(row[1] / row_count * 100, 2),
                    }
                    for row in top_results
                ]

    except Exception as e:
        profile["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return profile


def profile_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    sample_rows: int,
    top_k: int,
) -> dict[str, Any]:
    """
    Profile a single table.

    Args:
        conn: DuckDB connection
        table_name: Name of table to profile
        sample_rows: Max rows for expensive operations
        top_k: Top values to return

    Returns:
        Dictionary with table profile
    """
    profile: dict[str, Any] = {"table_name": table_name}

    try:
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        profile["row_count"] = row_count

        # Get column information
        columns_query = f"DESCRIBE {table_name}"
        columns_info = conn.execute(columns_query).fetchall()

        # Profile each column
        columns = []
        for col_info in columns_info:
            col_name = col_info[0]
            col_type = col_info[1]

            col_profile = profile_column(
                conn=conn,
                table_name=table_name,
                column_name=col_name,
                column_type=col_type,
                row_count=row_count,
                sample_rows=sample_rows,
                top_k=top_k,
            )
            columns.append(col_profile)

        profile["columns"] = columns

        # Get table sample (first 5 rows)
        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
        sample_result = conn.execute(sample_query).fetchdf()
        profile["table_sample"] = sample_result.to_dict("records")

    except Exception as e:
        profile["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return profile


def profile_database(
    db_path: Path,
    out_path: Path,
    sample_rows: int = 20000,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Profile all tables in DuckDB database and write to JSON.

    Args:
        db_path: Path to DuckDB database
        out_path: Path to output JSON file
        sample_rows: Max rows for expensive scans
        top_k: Top values for categorical columns

    Returns:
        Profile results dictionary
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)

    # Get all tables
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [table[0] for table in tables]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path.absolute()),
        "tables": {},
    }

    for table_name in table_names:
        table_profile = profile_table(
            conn=conn,
            table_name=table_name,
            sample_rows=sample_rows,
            top_k=top_k,
        )
        results["tables"][table_name] = table_profile

    conn.close()

    # Write to JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def format_profile_summary(results: dict[str, Any], out_path: Path) -> str:
    """Format profile results as a summary."""
    lines = []

    lines.append("✅ Profiling complete\n")
    lines.append(f"Database: {results['db_path']}")
    lines.append(f"Tables profiled: {len(results['tables'])}")

    if results["tables"]:
        lines.append("\nTables:")
        for table_name, table_info in results["tables"].items():
            row_count = table_info.get("row_count", 0)
            col_count = len(table_info.get("columns", []))
            lines.append(f"  • {table_name:20s} ({row_count:,} rows × {col_count} columns)")

    lines.append(f"\nProfile written to: {out_path.absolute()}")

    return "\n".join(lines)
