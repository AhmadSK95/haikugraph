"""Native SQL execution for v2 runtime."""

from __future__ import annotations

import time
from typing import Any

import duckdb

from haikugraph.v2.exceptions import QueryExecutionError
from haikugraph.v2.types import ExecutionResultV2, QueryPlanV2


def _serialize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def execute_query_plan(
    *,
    db_path: str,
    query_plan: QueryPlanV2,
    max_rows: int = 200,
) -> ExecutionResultV2:
    if not str(query_plan.sql or "").strip():
        raise QueryExecutionError("Missing compiled SQL.")

    started = time.perf_counter()
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(query_plan.sql).fetchall()
        columns = [str(desc[0]) for desc in (conn.description or [])]
        sample_rows: list[dict[str, Any]] = []
        for row in rows[:max_rows]:
            sample_rows.append({col: _serialize_cell(val) for col, val in zip(columns, row)})
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return ExecutionResultV2(
            success=True,
            row_count=len(rows),
            latency_ms=latency_ms,
            columns=columns,
            sample_rows=sample_rows,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return ExecutionResultV2(
            success=False,
            row_count=0,
            latency_ms=latency_ms,
            columns=[],
            sample_rows=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        conn.close()
