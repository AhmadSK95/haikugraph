"""Run conversational UI regression QA and generate an HTML report.

Focus:
- Session continuity and follow-up rewrites
- Intent/metric/table/filter grounding
- Multi-mode behavior (deterministic/local/auto/openai)
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


DEFAULT_MODES = ["deterministic", "local", "auto", "openai"]


@dataclass(frozen=True)
class QACase:
    case_id: str
    session: str
    goal: str
    expect: dict[str, Any]
    note: str


CASES: list[QACase] = [
    QACase(
        case_id="amount_with_mt103",
        session="thread_main",
        goal="Top 5 platforms by total transaction amount in December 2025 with mt103",
        expect={
            "table": "datada_mart_transactions",
            "metric": "total_amount",
            "filters_present": {"has_mt103": "true"},
            "sql_contains": ["SUM(amount)"],
            "row_count_min": 1,
        },
        note="Amount intent must not collapse into count; mt103 filter should be applied.",
    ),
    QACase(
        case_id="count_mt103_grouped",
        session="thread_main",
        goal="Total mt103 transactions count split by month wise and platform wise",
        expect={
            "table": "datada_mart_transactions",
            "metric": "mt103_count",
            "filters_present": {"has_mt103": "true"},
            "sql_contains": ["SUM(CASE WHEN has_mt103 THEN 1 ELSE 0 END)", "GROUP BY 1, 2"],
            "row_count_min": 1,
        },
        note="Explicit count wording should remain count metric.",
    ),
    QACase(
        case_id="count_not_contaminated_by_prev",
        session="thread_main",
        goal="Total transactions count split by month wise and platform wise",
        expect={
            "table": "datada_mart_transactions",
            "metric": "transaction_count",
            "filters_absent": ["has_mt103"],
            "sql_contains": ["COUNT(DISTINCT transaction_key)", "GROUP BY 1, 2"],
            "row_count_min": 1,
        },
        note="Should not inherit mt103 filter from previous turn.",
    ),
    QACase(
        case_id="followup_time_scope",
        session="thread_main",
        goal="Now show only December 2025",
        expect={
            "table": "datada_mart_transactions",
            "metric": "transaction_count",
            "sql_contains": ["EXTRACT(YEAR FROM event_ts) = 2025", "EXTRACT(MONTH FROM event_ts) = 12"],
            "row_count_min": 1,
        },
        note="Short follow-up should carry previous intent/dimensions and apply month scope.",
    ),
    QACase(
        case_id="vague_data_overview",
        session="thread_discovery",
        goal="What kind of data do I have?",
        expect={
            "answer_contains": ["map of your data", "transactions"],
        },
        note="Vague discovery prompt should produce useful overview.",
    ),
    QACase(
        case_id="forex_markup",
        session="thread_finance",
        goal="What is the forex markup revenue for December 2025?",
        expect={
            "table": "datada_mart_quotes",
            "metric": "forex_markup_revenue",
            "sql_contains": ["SUM(forex_markup)", "EXTRACT(MONTH FROM created_ts) = 12"],
            "row_count_min": 1,
        },
        note="Must route to quote mart and markup metric.",
    ),
    QACase(
        case_id="quotes_count",
        session="thread_quotes",
        goal="How many quotes are there?",
        expect={
            "table": "datada_mart_quotes",
            "metric": "quote_count",
            "sql_contains": ["COUNT(*)"],
            "row_count_min": 1,
        },
        note="Quote count should not default to transactions.",
    ),
    QACase(
        case_id="comparison_mom",
        session="thread_comp",
        goal="Compare this month vs last month transaction count",
        expect={
            "table": "datada_mart_transactions",
            "metric": "transaction_count",
            "sql_contains": ["UNION", "period", "comparison"],
            "row_count_min": 2,
        },
        note="Comparison intent should produce two-period output.",
    ),
]


def _normalize_filter_map(value_filters: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(value_filters, list):
        return out
    for item in value_filters:
        if not isinstance(item, dict):
            continue
        col = item.get("column")
        val = item.get("value")
        if isinstance(col, str) and isinstance(val, str):
            out[col] = val
    return out


def _evaluate_case(case: QACase, payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    grounding = ((payload.get("data_quality") or {}).get("grounding") or {}) if isinstance(payload, dict) else {}
    sql = str(payload.get("sql") or "")
    answer = str(payload.get("answer_markdown") or "")
    row_count = int(payload.get("row_count") or 0)
    value_filters = _normalize_filter_map(grounding.get("value_filters"))

    expected_table = case.expect.get("table")
    if expected_table and grounding.get("table") != expected_table:
        errors.append(f"table expected={expected_table} actual={grounding.get('table')}")

    expected_metric = case.expect.get("metric")
    if expected_metric and grounding.get("metric") != expected_metric:
        errors.append(f"metric expected={expected_metric} actual={grounding.get('metric')}")

    for frag in case.expect.get("sql_contains", []):
        if frag not in sql:
            errors.append(f"sql missing fragment: {frag}")

    for frag in case.expect.get("answer_contains", []):
        if frag.lower() not in answer.lower():
            errors.append(f"answer missing phrase: {frag}")

    for col, val in (case.expect.get("filters_present") or {}).items():
        if value_filters.get(col) != val:
            errors.append(f"filter {col} expected={val} actual={value_filters.get(col)}")

    for col in case.expect.get("filters_absent", []):
        if col in value_filters:
            errors.append(f"unexpected filter present: {col}={value_filters.get(col)}")

    row_count_min = case.expect.get("row_count_min")
    if isinstance(row_count_min, int) and row_count < row_count_min:
        errors.append(f"row_count expected>={row_count_min} actual={row_count}")

    return (len(errors) == 0, errors)


def run_qa(db_path: Path, out_dir: Path, modes: list[str]) -> tuple[Path, dict[str, Any]]:
    app = create_app(db_path=db_path)
    client = TestClient(app)
    generated_at = datetime.now(timezone.utc).isoformat()

    runs: list[dict[str, Any]] = []
    mode_summary: dict[str, dict[str, Any]] = {}

    for mode in modes:
        pass_count = 0
        latencies: list[float] = []
        for case in CASES:
            payload = {
                "goal": case.goal,
                "llm_mode": mode,
                "session_id": f"ui-qa-{mode}-{case.session}",
                "storyteller_mode": True,
            }
            started = time.perf_counter()
            response = client.post("/api/assistant/query", json=payload)
            duration_ms = (time.perf_counter() - started) * 1000

            if response.status_code != 200:
                result = {
                    "mode": mode,
                    "case_id": case.case_id,
                    "goal": case.goal,
                    "note": case.note,
                    "success": False,
                    "status_code": response.status_code,
                    "errors": [f"http_{response.status_code}"],
                    "latency_ms": round(duration_ms, 2),
                    "runtime_mode": "error",
                }
                runs.append(result)
                latencies.append(duration_ms)
                continue

            body = response.json()
            ok, errors = _evaluate_case(case, body)
            if ok:
                pass_count += 1
            latencies.append(duration_ms)
            runs.append(
                {
                    "mode": mode,
                    "case_id": case.case_id,
                    "goal": case.goal,
                    "note": case.note,
                    "success": ok,
                    "errors": errors,
                    "latency_ms": round(duration_ms, 2),
                    "runtime_mode": ((body.get("runtime") or {}).get("mode") or "unknown"),
                    "grounding": ((body.get("data_quality") or {}).get("grounding") or {}),
                    "sql": body.get("sql"),
                    "answer_preview": str(body.get("answer_markdown") or "")[:220],
                    "trace_id": body.get("trace_id"),
                }
            )

        total = len(CASES)
        mode_summary[mode] = {
            "pass_count": pass_count,
            "total": total,
            "pass_rate": round((pass_count / total) * 100, 2) if total else 0.0,
            "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else None,
            "p95_latency_ms": round(_p95(latencies), 2) if latencies else None,
        }

    report = {
        "generated_at": generated_at,
        "db_path": str(db_path),
        "modes": modes,
        "summary": mode_summary,
        "runs": runs,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"ui_regression_qa_{ts}.json"
    html_path = out_dir / f"ui_regression_qa_{ts}.html"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    html_path.write_text(_build_html(report))
    return html_path, report


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _build_html(report: dict[str, Any]) -> str:
    def esc(v: Any) -> str:
        return html.escape(str(v if v is not None else ""))

    summary_rows = "".join(
        f"<tr><td>{esc(mode)}</td><td>{vals['pass_count']}/{vals['total']}</td><td>{vals['pass_rate']}%</td><td>{vals['avg_latency_ms']}</td><td>{vals['p95_latency_ms']}</td></tr>"
        for mode, vals in report["summary"].items()
    )

    run_rows = "".join(
        (
            "<tr>"
            f"<td>{esc(r['mode'])}</td>"
            f"<td>{esc(r['case_id'])}</td>"
            f"<td>{esc(r['runtime_mode'])}</td>"
            f"<td>{esc(r['latency_ms'])}</td>"
            f"<td><span class='badge {'ok' if r['success'] else 'warn'}'>{'PASS' if r['success'] else 'FAIL'}</span></td>"
            f"<td>{esc(' | '.join(r.get('errors', [])) or '-')}</td>"
            "</tr>"
        )
        for r in report["runs"]
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>dataDa UI Regression QA</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background:#0b111b; color:#dce8ff; }}
    h1 {{ margin: 0 0 10px; color:#9fd3ff; }}
    .meta {{ margin-bottom: 14px; color:#9ab0d1; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; }}
    th, td {{ border: 1px solid #2b3c57; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #132236; color:#a8d8ff; }}
    tr:nth-child(even) td {{ background: #0f1a2b; }}
    .badge {{ border-radius: 999px; padding: 2px 8px; font-weight: 700; font-size: 11px; }}
    .badge.ok {{ background: #183e2e; color: #9af5c8; }}
    .badge.warn {{ background: #3d2b18; color: #ffd29c; }}
  </style>
</head>
<body>
  <h1>dataDa UI Regression QA</h1>
  <div class="meta">
    Generated: {esc(report['generated_at'])}<br/>
    DB: {esc(report['db_path'])}<br/>
    Modes: {esc(", ".join(report['modes']))}
  </div>
  <h2>Summary</h2>
  <table>
    <thead><tr><th>Mode</th><th>Pass</th><th>Pass Rate</th><th>Avg Latency (ms)</th><th>P95 Latency (ms)</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  <h2>Case Results</h2>
  <table>
    <thead><tr><th>Mode</th><th>Case</th><th>Runtime</th><th>Latency (ms)</th><th>Status</th><th>Errors</th></tr></thead>
    <tbody>{run_rows}</tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conversational UI regression QA")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    args = parser.parse_args()

    html_path, report = run_qa(Path(args.db_path), Path(args.out_dir), args.modes)
    print(f"UI regression QA complete. Report: {html_path}")
    for mode, vals in report["summary"].items():
        print(
            f"- {mode}: pass={vals['pass_count']}/{vals['total']} "
            f"rate={vals['pass_rate']}% avg_ms={vals['avg_latency_ms']}"
        )


if __name__ == "__main__":
    main()
