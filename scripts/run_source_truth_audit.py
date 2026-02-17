"""Run source-truth audit against raw SQL tables.

This audit compares assistant query outputs against hand-authored SQL
directly over source tables (test_1_1_merged/test_3_1/test_4_1/test_5_1).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


@dataclass(frozen=True)
class SourceTruthCase:
    case_id: str
    question: str
    expected_sql: str
    note: str


def _ts_expr(raw_col: str) -> str:
    return (
        "COALESCE("
        f"TRY_CAST(NULLIF(TRIM(CAST({raw_col} AS VARCHAR)), '') AS TIMESTAMP),"
        f"TRY_STRPTIME(NULLIF(TRIM(CAST({raw_col} AS VARCHAR)), ''), '%Y-%m-%dT%H:%M:%S.%fZ')"
        ")"
    )


CASES: list[SourceTruthCase] = [
    SourceTruthCase(
        case_id="forex_markup_dec_2025",
        question="What is the forex markup revenue for December 2025?",
        expected_sql=(
            "WITH q AS ("
            f"SELECT {_ts_expr('created_at')} AS created_ts, "
            "TRY_CAST(forex_markup AS DOUBLE) AS forex_markup "
            "FROM test_3_1"
            ") "
            "SELECT SUM(COALESCE(forex_markup, 0.0)) AS metric_value "
            "FROM q WHERE created_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM created_ts)=2025 AND EXTRACT(MONTH FROM created_ts)=12"
        ),
        note="Forex markup must come from source quote table.",
    ),
    SourceTruthCase(
        case_id="charges_dec_2025",
        question="What are total charges in December 2025?",
        expected_sql=(
            "WITH q AS ("
            f"SELECT {_ts_expr('created_at')} AS created_ts, "
            "TRY_CAST(total_additional_charges AS DOUBLE) AS total_additional_charges "
            "FROM test_3_1"
            ") "
            "SELECT SUM(COALESCE(total_additional_charges, 0.0)) AS metric_value "
            "FROM q WHERE created_ts IS NOT NULL "
            "AND EXTRACT(YEAR FROM created_ts)=2025 AND EXTRACT(MONTH FROM created_ts)=12"
        ),
        note="Total charges must map to source quote charges field.",
    ),
    SourceTruthCase(
        case_id="mt103_month_platform",
        question="Total mt103 transactions count split by month wise and platform wise",
        expected_sql=(
            "WITH t AS ("
            f"SELECT {_ts_expr('mt103_created_at')} AS mt103_created_ts, "
            "NULLIF(TRIM(CAST(platform_name AS VARCHAR)), '') AS platform_name, "
            "CASE WHEN NULLIF(TRIM(CAST(mt103_created_at AS VARCHAR)), '') IS NULL THEN 0 ELSE 1 END AS has_mt103 "
            "FROM test_1_1_merged"
            ") "
            "SELECT DATE_TRUNC('month', mt103_created_ts) AS month_bucket, platform_name, SUM(has_mt103) AS metric_value "
            "FROM t WHERE mt103_created_ts IS NOT NULL "
            "GROUP BY 1,2 ORDER BY 3 DESC NULLS LAST, 1 ASC, 2 ASC LIMIT 20"
        ),
        note="Grouped MT103 counts should match source transaction table.",
    ),
    SourceTruthCase(
        case_id="quotes_count_total",
        question="How many quotes are there?",
        expected_sql="SELECT COUNT(*) AS metric_value FROM test_3_1",
        note="Quote count baseline from source quotes table.",
    ),
]


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return round(v, 6)
    return v


def _norm_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted([tuple(_norm(v) for v in row) for row in rows], key=lambda r: json.dumps(r, default=str))


def _rows_equal(a: list[tuple[Any, ...]], b: list[tuple[Any, ...]]) -> bool:
    return _norm_rows(a) == _norm_rows(b)


def _execute(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def esc(v: Any) -> str:
    return html.escape(str(v))


def run_audit(
    db_path: Path,
    out_dir: Path,
    modes: list[str],
    local_model: str,
) -> tuple[Path, dict[str, Any]]:
    app = create_app(db_path=db_path)
    client = TestClient(app)
    conn = duckdb.connect(str(db_path), read_only=False)

    expected_cache: dict[str, dict[str, Any]] = {}
    for case in CASES:
        cols, rows = _execute(conn, case.expected_sql)
        expected_cache[case.case_id] = {"cols": cols, "rows": rows}

    runs: list[dict[str, Any]] = []
    for mode in modes:
        for case in CASES:
            started = time.perf_counter()
            response = client.post(
                "/api/assistant/query",
                json={
                    "goal": case.question,
                    "llm_mode": mode,
                    "local_model": local_model,
                    "session_id": f"audit-{mode}-{case.case_id}",
                },
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            payload = response.json()
            actual_sql = payload.get("sql") or ""
            runtime = payload.get("runtime", {})

            got_cols: list[str] = []
            got_rows: list[tuple[Any, ...]] = []
            exec_error = None
            if actual_sql:
                try:
                    got_cols, got_rows = _execute(conn, actual_sql)
                except Exception as exc:
                    exec_error = str(exc)

            expected = expected_cache[case.case_id]
            exact_match = bool(payload.get("success")) and exec_error is None and _rows_equal(got_rows, expected["rows"])
            runs.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "note": case.note,
                    "mode_requested": mode,
                    "mode_actual": runtime.get("mode"),
                    "provider": runtime.get("provider"),
                    "success": bool(payload.get("success")),
                    "latency_ms": elapsed_ms,
                    "exact_match": exact_match,
                    "expected_sql": case.expected_sql,
                    "actual_sql": actual_sql,
                    "expected_cols": expected["cols"],
                    "actual_cols": got_cols,
                    "expected_rows": expected["rows"][:20],
                    "actual_rows": got_rows[:20],
                    "response_error": payload.get("error"),
                    "sql_error": exec_error,
                    "grounding": (payload.get("data_quality") or {}).get("grounding", {}),
                }
            )

    conn.close()

    summary: dict[str, Any] = {}
    for mode in modes:
        rows = [r for r in runs if r["mode_requested"] == mode]
        matches = [r for r in rows if r["exact_match"]]
        summary[mode] = {
            "cases": len(rows),
            "exact_matches": len(matches),
            "accuracy_pct": round((len(matches) / len(rows)) * 100, 2) if rows else 0.0,
            "avg_latency_ms": round(statistics.mean([r["latency_ms"] for r in rows]), 2) if rows else 0.0,
        }

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "db_path": str(db_path),
        "modes": modes,
        "local_model": local_model,
        "summary": summary,
        "runs": runs,
        "mismatch_count": sum(1 for r in runs if not r["exact_match"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"source_truth_audit_{ts}.json"
    html_path = out_dir / f"source_truth_audit_{ts}.html"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    html_path.write_text(build_html(report))
    return html_path, report


def build_html(report: dict[str, Any]) -> str:
    summary_rows = "".join(
        f"<tr><td>{esc(mode)}</td><td>{vals['cases']}</td><td>{vals['exact_matches']}</td>"
        f"<td>{vals['accuracy_pct']}%</td><td>{vals['avg_latency_ms']}</td></tr>"
        for mode, vals in report["summary"].items()
    )

    detail_rows = "".join(
        f"""
        <tr>
          <td>{esc(r['case_id'])}</td>
          <td>{esc(r['mode_requested'])}</td>
          <td>{esc(r['mode_actual'])}</td>
          <td>{'YES' if r['exact_match'] else 'NO'}</td>
          <td>{'YES' if r['success'] else 'NO'}</td>
          <td>{esc(r['latency_ms'])}</td>
          <td>{esc(r['provider'])}</td>
          <td>{esc(r['response_error'] or r['sql_error'] or '')}</td>
          <td><code>{esc((r['expected_sql'] or '')[:220])}</code></td>
          <td><code>{esc((r['actual_sql'] or '')[:220])}</code></td>
        </tr>
        """
        for r in report["runs"]
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>dataDa Source Truth Audit</title>
  <style>
    body {{ font-family: 'Space Grotesk', 'Segoe UI', sans-serif; background: #eef4f7; color: #102430; margin: 0; padding: 20px; }}
    .section {{ background: white; border: 1px solid #d7e3ea; border-radius: 12px; padding: 14px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
    th, td {{ border-bottom: 1px solid #e8eef3; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f9fc; position: sticky; top: 0; }}
    .wrap {{ overflow: auto; max-height: 520px; border: 1px solid #dfe8ef; border-radius: 8px; }}
    code {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }}
  </style>
</head>
<body>
  <div class="section">
    <h1>dataDa Source-Truth SQL Audit</h1>
    <p>Generated: {esc(report['generated_at'])}<br/>DB: {esc(report['db_path'])}<br/>Modes: {esc(', '.join(report['modes']))}<br/>Mismatches: {esc(report['mismatch_count'])}</p>
  </div>
  <div class="section">
    <h2>Summary</h2>
    <div class="wrap">
      <table>
        <thead><tr><th>Mode</th><th>Cases</th><th>Exact Matches</th><th>Accuracy</th><th>Avg Latency (ms)</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="section">
    <h2>Detailed Runs</h2>
    <div class="wrap">
      <table>
        <thead><tr><th>Case</th><th>Requested</th><th>Actual</th><th>Exact</th><th>Success</th><th>Latency</th><th>Provider</th><th>Error</th><th>Expected SQL</th><th>Actual SQL</th></tr></thead>
        <tbody>{detail_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-truth SQL audit for dataDa")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--modes", default="deterministic,local,auto")
    parser.add_argument("--local-model", default="qwen2.5:7b-instruct")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    html_path, report = run_audit(
        db_path=Path(args.db_path),
        out_dir=Path(args.out_dir),
        modes=modes,
        local_model=args.local_model,
    )
    print(f"Source-truth audit complete. Report: {html_path}")
    for mode, vals in report["summary"].items():
        print(
            f"- {mode}: accuracy={vals['accuracy_pct']}% "
            f"exact={vals['exact_matches']}/{vals['cases']} avg_latency_ms={vals['avg_latency_ms']}"
        )


if __name__ == "__main__":
    main()
