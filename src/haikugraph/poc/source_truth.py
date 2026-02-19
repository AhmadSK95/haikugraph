"""Source-truth verification harness for runtime parity checks."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from haikugraph.poc.agentic_team import RuntimeSelection


@dataclass(frozen=True)
class SourceTruthCase:
    case_id: str
    question: str
    expected_sql: str
    note: str


DEFAULT_CASES: list[SourceTruthCase] = [
    SourceTruthCase(
        case_id="tx_count",
        question="How many transactions are there?",
        expected_sql=(
            "SELECT COUNT(DISTINCT transaction_key) AS metric_value "
            "FROM datada_mart_transactions WHERE 1=1"
        ),
        note="Baseline transaction cardinality.",
    ),
    SourceTruthCase(
        case_id="tx_total_amount_dec_2025",
        question="What is the total transaction amount in December 2025?",
        expected_sql=(
            "SELECT SUM(amount) AS metric_value "
            "FROM datada_mart_transactions WHERE event_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM event_ts)=2025 AND EXTRACT(MONTH FROM event_ts)=12"
        ),
        note="Month-scoped transaction amount.",
    ),
    SourceTruthCase(
        case_id="mt103_month_platform",
        question="Total MT103 transactions count split by month wise and platform wise",
        expected_sql=(
            "SELECT DATE_TRUNC('month', event_ts) AS month_bucket, platform_name AS dimension, "
            "SUM(CASE WHEN has_mt103 THEN 1 ELSE 0 END) AS metric_value "
            "FROM datada_mart_transactions WHERE event_ts IS NOT NULL "
            "GROUP BY 1,2 ORDER BY 3 DESC NULLS LAST, 1 ASC, 2 ASC LIMIT 20"
        ),
        note="MT103 monthly/platform split.",
    ),
    SourceTruthCase(
        case_id="quotes_count",
        question="How many quotes are there?",
        expected_sql="SELECT COUNT(DISTINCT quote_key) AS metric_value FROM datada_mart_quotes WHERE 1=1",
        note="Quote baseline count.",
    ),
    SourceTruthCase(
        case_id="customers_count",
        question="How many customers do we have?",
        expected_sql="SELECT COUNT(DISTINCT customer_key) AS metric_value FROM datada_dim_customers WHERE 1=1",
        note="Customer baseline count.",
    ),
    SourceTruthCase(
        case_id="bookings_count",
        question="How many bookings are there?",
        expected_sql="SELECT COUNT(DISTINCT booking_key) AS metric_value FROM datada_mart_bookings WHERE 1=1",
        note="Bookings baseline count.",
    ),
]


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return round(value, 6)
    return value


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    normalized = [tuple(_normalize_cell(v) for v in row) for row in rows]
    return sorted(normalized, key=lambda r: json.dumps(r, default=str))


def _rows_match(actual: list[tuple[Any, ...]], expected: list[tuple[Any, ...]]) -> bool:
    return _normalize_rows(actual) == _normalize_rows(expected)


def _exec(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple[Any, ...]], str | None]:
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
        return cols, rows, None
    except Exception as exc:
        return [], [], str(exc)


def run_source_truth_suite(
    *,
    team,
    db_path: Path | str,
    runtime: RuntimeSelection,
    max_cases: int = 6,
) -> dict[str, Any]:
    conn = duckdb.connect(str(Path(db_path).expanduser()), read_only=True)
    rows: list[dict[str, Any]] = []
    cases = DEFAULT_CASES[: max(1, min(len(DEFAULT_CASES), int(max_cases)))]

    for case in cases:
        expected_cols, expected_rows, expected_error = _exec(conn, case.expected_sql)
        if expected_error:
            rows.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "note": case.note,
                    "status": "skipped",
                    "reason": f"Expected SQL failed: {expected_error}",
                    "exact_match": False,
                    "latency_ms": 0.0,
                    "expected_sql": case.expected_sql,
                    "actual_sql": "",
                    "expected_cols": [],
                    "actual_cols": [],
                    "expected_rows": [],
                    "actual_rows": [],
                }
            )
            continue

        started = time.perf_counter()
        response = team.run(case.question, runtime)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        actual_sql = response.sql or ""
        actual_cols: list[str] = []
        actual_rows: list[tuple[Any, ...]] = []
        actual_sql_error = None
        if actual_sql:
            actual_cols, actual_rows, actual_sql_error = _exec(conn, actual_sql)

        success = bool(response.success) and not bool(actual_sql_error)
        exact_match = success and _rows_match(actual_rows, expected_rows)
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "note": case.note,
                "status": "ok" if success else "failed",
                "reason": actual_sql_error or response.error or "",
                "exact_match": exact_match,
                "latency_ms": latency_ms,
                "expected_sql": case.expected_sql,
                "actual_sql": actual_sql,
                "expected_cols": expected_cols,
                "actual_cols": actual_cols,
                "expected_rows": expected_rows[:20],
                "actual_rows": actual_rows[:20],
            }
        )

    conn.close()

    evaluated = [r for r in rows if r["status"] != "skipped"]
    matched = [r for r in evaluated if r["exact_match"]]
    avg_latency = round(
        sum(float(r.get("latency_ms") or 0.0) for r in evaluated) / len(evaluated),
        2,
    ) if evaluated else 0.0

    return {
        "cases": len(rows),
        "evaluated_cases": len(evaluated),
        "exact_matches": len(matched),
        "accuracy_pct": round((len(matched) / len(evaluated)) * 100, 2) if evaluated else 0.0,
        "avg_latency_ms": avg_latency,
        "runs": rows,
    }

