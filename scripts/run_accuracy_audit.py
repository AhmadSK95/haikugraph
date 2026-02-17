"""Run SQL-accuracy audit against canonical ground-truth queries.

This script evaluates how often each runtime mode produces SQL whose
result matches a hand-authored expected SQL result for the same question.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


MODES = ["deterministic", "local", "openai", "auto"]


@dataclass(frozen=True)
class AccuracyCase:
    case_id: str
    question: str
    expected_sql: str
    note: str


CASES: list[AccuracyCase] = [
    AccuracyCase(
        "tx_count_total",
        "How many transactions are there?",
        "SELECT COUNT(DISTINCT transaction_key) AS metric_value FROM datada_mart_transactions WHERE 1=1",
        "Transaction count baseline.",
    ),
    AccuracyCase(
        "tx_amount_dec_2025",
        "What is the total amount of transactions in December 2025?",
        (
            "SELECT SUM(amount) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 AND event_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM event_ts) = 2025 AND EXTRACT(MONTH FROM event_ts) = 12"
        ),
        "Amount aggregate scoped to month/year.",
    ),
    AccuracyCase(
        "tx_top5_platform_dec_2025",
        "Top 5 platforms by transaction count in December 2025",
        (
            "SELECT platform_name AS dimension, COUNT(DISTINCT transaction_key) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 AND event_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM event_ts) = 2025 AND EXTRACT(MONTH FROM event_ts) = 12 "
            "GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 5"
        ),
        "Grouped transaction count.",
    ),
    AccuracyCase(
        "tx_refund_count_dec_2025",
        "How many refunds happened in December 2025?",
        (
            "SELECT SUM(CASE WHEN has_refund THEN 1 ELSE 0 END) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 AND event_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM event_ts) = 2025 AND EXTRACT(MONTH FROM event_ts) = 12"
        ),
        "Refund count metric.",
    ),
    AccuracyCase(
        "tx_refund_rate_by_platform",
        "Refund rate by platform",
        (
            "SELECT platform_name AS dimension, AVG(CASE WHEN has_refund THEN 1.0 ELSE 0.0 END) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Refund rate grouped by platform.",
    ),
    AccuracyCase(
        "tx_mt103_rate_by_platform",
        "MT103 rate by platform",
        (
            "SELECT platform_name AS dimension, AVG(CASE WHEN has_mt103 THEN 1.0 ELSE 0.0 END) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "MT103 coverage grouped by platform.",
    ),
    AccuracyCase(
        "tx_unique_customers",
        "How many unique customers made transactions?",
        "SELECT COUNT(DISTINCT customer_id) AS metric_value FROM datada_mart_transactions WHERE 1=1",
        "Distinct customers from transaction mart.",
    ),
    AccuracyCase(
        "tx_count_by_flow",
        "Transaction count by flow",
        (
            "SELECT txn_flow AS dimension, COUNT(DISTINCT transaction_key) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Operational split by txn_flow.",
    ),
    AccuracyCase(
        "tx_compare_dec_vs_prev",
        "Compare transaction count in December 2025 vs previous month",
        (
            "SELECT 'current' AS period, COUNT(DISTINCT transaction_key) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 AND event_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM event_ts) = 2025 AND EXTRACT(MONTH FROM event_ts) = 12 "
            "UNION "
            "SELECT 'comparison' AS period, COUNT(DISTINCT transaction_key) AS metric_value "
            "FROM datada_mart_transactions "
            "WHERE 1=1 AND event_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM event_ts) = 2025 AND EXTRACT(MONTH FROM event_ts) = 11"
        ),
        "Month-over-month explicit comparison.",
    ),
    AccuracyCase(
        "quote_count_total",
        "How many quotes are there?",
        "SELECT COUNT(DISTINCT quote_key) AS metric_value FROM datada_mart_quotes WHERE 1=1",
        "Quote count baseline.",
    ),
    AccuracyCase(
        "quote_count_by_from_currency",
        "Quote volume by from currency",
        (
            "SELECT from_currency AS dimension, COUNT(DISTINCT quote_key) AS metric_value "
            "FROM datada_mart_quotes "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Quote count split by source currency.",
    ),
    AccuracyCase(
        "quote_total_value",
        "What is the total quote value?",
        "SELECT SUM(total_amount_to_be_paid) AS metric_value FROM datada_mart_quotes WHERE 1=1",
        "Total quote value.",
    ),
    AccuracyCase(
        "quote_avg_value_by_currency",
        "Average quote value by from currency",
        (
            "SELECT from_currency AS dimension, AVG(total_amount_to_be_paid) AS metric_value "
            "FROM datada_mart_quotes "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Average quote value grouped by source currency.",
    ),
    AccuracyCase(
        "customer_count_total",
        "How many customers are there?",
        "SELECT COUNT(DISTINCT customer_key) AS metric_value FROM datada_dim_customers WHERE 1=1",
        "Customer count baseline.",
    ),
    AccuracyCase(
        "customer_count_by_country",
        "Customer count by country",
        (
            "SELECT address_country AS dimension, COUNT(DISTINCT customer_key) AS metric_value "
            "FROM datada_dim_customers "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Customer distribution by country.",
    ),
    AccuracyCase(
        "university_count_by_country",
        "University count by country",
        (
            "SELECT address_country AS dimension, SUM(CASE WHEN is_university THEN 1 ELSE 0 END) AS metric_value "
            "FROM datada_dim_customers "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "University subgroup by country.",
    ),
    AccuracyCase(
        "booking_count_total",
        "How many bookings are there?",
        "SELECT COUNT(DISTINCT booking_key) AS metric_value FROM datada_mart_bookings WHERE 1=1",
        "Booking count baseline.",
    ),
    AccuracyCase(
        "booking_total_by_currency",
        "Total booked amount by currency",
        (
            "SELECT currency AS dimension, SUM(booked_amount) AS metric_value "
            "FROM datada_mart_bookings "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Booked amount split by currency.",
    ),
    AccuracyCase(
        "booking_avg_rate_by_deal_type",
        "Average rate by deal type",
        (
            "SELECT deal_type AS dimension, AVG(rate) AS metric_value "
            "FROM datada_mart_bookings "
            "WHERE 1=1 GROUP BY 1 ORDER BY 2 DESC NULLS LAST, 1 ASC LIMIT 20"
        ),
        "Rate behavior by deal type.",
    ),
]


def esc(value: Any) -> str:
    return html.escape(str(value))


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(round(value, 6))
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            return float(round(value, 6))
        return value
    return str(value)


def _value_equal(a: Any, b: Any, tol: float = 1e-5) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def _rows_equal(rows_a: list[tuple[Any, ...]], rows_b: list[tuple[Any, ...]]) -> bool:
    if len(rows_a) != len(rows_b):
        return False
    for ra, rb in zip(rows_a, rows_b):
        if len(ra) != len(rb):
            return False
        for va, vb in zip(ra, rb):
            if not _value_equal(va, vb):
                return False
    return True


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    normalized = [tuple(_normalize_value(v) for v in row) for row in rows]
    return sorted(normalized, key=lambda x: json.dumps(x, default=str))


def _execute_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    cursor = conn.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return columns, rows


def run_audit(db_path: Path, out_dir: Path, local_model: str) -> tuple[Path, dict[str, Any]]:
    app = create_app(db_path=db_path)
    client = TestClient(app)

    providers = client.get("/api/assistant/providers").json()
    health = client.get("/api/assistant/health").json()
    # Use same connection mode as the API executor to avoid DuckDB mixed-mode lock errors.
    conn = duckdb.connect(str(db_path), read_only=False)

    expected_cache: dict[str, dict[str, Any]] = {}
    for case in CASES:
        exp_cols, exp_rows = _execute_sql(conn, case.expected_sql)
        expected_cache[case.case_id] = {
            "columns": exp_cols,
            "rows": exp_rows,
            "rows_norm": _normalize_rows(exp_rows),
        }

    runs: list[dict[str, Any]] = []
    for mode in MODES:
        for case in CASES:
            started = time.perf_counter()
            response = client.post(
                "/api/assistant/query",
                json={
                    "goal": case.question,
                    "llm_mode": mode,
                    "local_model": local_model,
                },
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            payload = response.json()
            runtime = payload.get("runtime", {})

            executed_sql = payload.get("sql")
            execution_error = None
            got_cols: list[str] = []
            got_rows: list[tuple[Any, ...]] = []
            got_rows_norm: list[tuple[Any, ...]] = []

            if executed_sql:
                try:
                    got_cols, got_rows = _execute_sql(conn, executed_sql)
                    got_rows_norm = _normalize_rows(got_rows)
                except Exception as exc:  # pragma: no cover
                    execution_error = str(exc)

            expected = expected_cache[case.case_id]
            exact_match = (
                bool(payload.get("success"))
                and execution_error is None
                and bool(executed_sql)
                and _rows_equal(got_rows_norm, expected["rows_norm"])
            )

            runs.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "note": case.note,
                    "requested_mode": mode,
                    "actual_mode": runtime.get("mode"),
                    "provider": runtime.get("provider"),
                    "llm_effective": bool(runtime.get("llm_effective")),
                    "success": bool(payload.get("success")),
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                    "confidence_score": payload.get("confidence_score"),
                    "expected_sql": case.expected_sql,
                    "actual_sql": executed_sql,
                    "expected_columns": expected["columns"],
                    "actual_columns": got_cols,
                    "expected_rows": expected["rows"][:20],
                    "actual_rows": got_rows[:20],
                    "expected_row_count": len(expected["rows"]),
                    "actual_row_count": len(got_rows),
                    "exact_match": exact_match,
                    "response_error": payload.get("error"),
                    "sql_execution_error": execution_error,
                }
            )

    conn.close()

    summary: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rows = [r for r in runs if r["requested_mode"] == mode]
        matches = [r for r in rows if r["exact_match"]]
        successes = [r for r in rows if r["success"]]
        latencies = [r["latency_ms"] for r in rows]
        summary[mode] = {
            "cases": len(rows),
            "exact_matches": len(matches),
            "accuracy_pct": round((len(matches) / len(rows)) * 100, 2) if rows else 0.0,
            "success_pct": round((len(successes) / len(rows)) * 100, 2) if rows else 0.0,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2)
            if latencies
            else 0.0,
            "llm_effective_pct": round(
                (sum(1 for r in rows if r["llm_effective"]) / len(rows)) * 100, 2
            )
            if rows
            else 0.0,
        }

    failures = [r for r in runs if not r["exact_match"]]
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "db_path": str(db_path),
        "providers": providers,
        "health": health,
        "local_model": local_model,
        "cases": [c.__dict__ for c in CASES],
        "summary": summary,
        "runs": runs,
        "mismatch_count": len(failures),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"accuracy_audit_{ts}.json"
    html_path = out_dir / f"accuracy_audit_{ts}.html"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    html_path.write_text(build_html_report(report))
    return html_path, report


def build_html_report(report: dict[str, Any]) -> str:
    summary_rows = "".join(
        f"""
        <tr>
          <td>{esc(mode)}</td>
          <td>{vals['cases']}</td>
          <td>{vals['exact_matches']}</td>
          <td>{vals['accuracy_pct']}%</td>
          <td>{vals['success_pct']}%</td>
          <td>{vals['avg_latency_ms']}</td>
          <td>{vals['p95_latency_ms']}</td>
          <td>{vals['llm_effective_pct']}%</td>
        </tr>
        """
        for mode, vals in report["summary"].items()
    )

    run_rows = "".join(
        f"""
        <tr>
          <td>{esc(r['case_id'])}</td>
          <td>{esc(r['requested_mode'])}</td>
          <td>{esc(r['actual_mode'])}</td>
          <td>{'YES' if r['exact_match'] else 'NO'}</td>
          <td>{'YES' if r['success'] else 'NO'}</td>
          <td>{r['latency_ms']}</td>
          <td>{esc(r['provider'])}</td>
          <td>{'YES' if r['llm_effective'] else 'NO'}</td>
          <td>{esc(r['response_error'] or r['sql_execution_error'] or '')}</td>
          <td><code>{esc((r['expected_sql'] or '')[:180])}</code></td>
          <td><code>{esc((r['actual_sql'] or '')[:180])}</code></td>
        </tr>
        """
        for r in report["runs"]
    )

    providers = report["providers"]
    provider_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{'YES' if v.get('available') else 'NO'}</td><td>{esc(v.get('reason'))}</td></tr>"
        for k, v in providers.get("checks", {}).items()
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>dataDa Accuracy Audit</title>
  <style>
    body {{ font-family: 'Space Grotesk', 'Segoe UI', sans-serif; background: #eef4f7; color: #102430; margin: 0; padding: 20px; }}
    .section {{ background: white; border: 1px solid #d7e3ea; border-radius: 12px; padding: 14px; margin-bottom: 14px; }}
    h1, h2 {{ margin: 0 0 10px; color: #0d415e; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
    th, td {{ border-bottom: 1px solid #e8eef3; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f9fc; position: sticky; top: 0; }}
    .wrap {{ overflow: auto; max-height: 520px; border: 1px solid #dfe8ef; border-radius: 8px; }}
    code {{ font-family: 'IBM Plex Mono', Menlo, monospace; font-size: 0.75rem; }}
  </style>
</head>
<body>
  <div class="section">
    <h1>dataDa Query Accuracy Audit</h1>
    <p>Generated (UTC): {esc(report['generated_at'])}<br/>
    Database: {esc(report['db_path'])}<br/>
    Local model: {esc(report['local_model'])}<br/>
    Mismatches: {esc(report['mismatch_count'])}</p>
  </div>

  <div class="section">
    <h2>Provider Snapshot</h2>
    <div class="wrap">
      <table>
        <thead><tr><th>Provider</th><th>Available</th><th>Reason</th></tr></thead>
        <tbody>{provider_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Accuracy Summary</h2>
    <div class="wrap">
      <table>
        <thead>
          <tr>
            <th>Mode</th><th>Cases</th><th>Exact Matches</th><th>Accuracy %</th>
            <th>Success %</th><th>Avg Latency (ms)</th><th>P95 Latency (ms)</th><th>LLM Effective %</th>
          </tr>
        </thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Detailed Case Matrix</h2>
    <div class="wrap">
      <table>
        <thead>
          <tr>
            <th>Case</th><th>Requested</th><th>Actual</th><th>Exact</th><th>Success</th><th>Latency</th>
            <th>Provider</th><th>LLM Effective</th><th>Error</th><th>Expected SQL</th><th>Actual SQL</th>
          </tr>
        </thead>
        <tbody>{run_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQL accuracy audit for dataDa")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--local-model", default="qwen2.5:7b-instruct")
    args = parser.parse_args()

    html_path, data = run_audit(Path(args.db_path), Path(args.out_dir), args.local_model)
    print(f"Accuracy audit complete. Report: {html_path}")
    for mode, vals in data["summary"].items():
        print(
            f"- {mode}: accuracy={vals['accuracy_pct']}% "
            f"success={vals['success_pct']}% avg_latency_ms={vals['avg_latency_ms']}"
        )


if __name__ == "__main__":
    main()
