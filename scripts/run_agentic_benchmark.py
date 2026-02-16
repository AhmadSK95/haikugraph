"""Run agentic POC benchmarks across runtime modes and generate HTML report."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


QUERY_SUITE = [
    "Total transactions",
    "Total transactions in December 2025",
    "Top 5 platforms by transaction count in December 2025",
    "Refund count in December 2025",
    "Refund rate by platform",
    "MT103 count in December 2025",
    "MT103 rate by platform",
    "Average payment amount by platform",
    "Total amount by state",
    "Total amount by month",
    "Unique customers",
    "Transaction count by txn_flow",
    "Compare this month vs last month transaction count",
    "Compare this year vs last year total amount",
    "Quote volume",
    "Quote volume by from_currency",
    "Total quote value",
    "Average quote value by from_currency",
    "Customer count",
    "Customer count by country",
    "University count by country",
    "Payee count by state",
    "Booking count",
    "Total booked amount by currency",
    "Average rate by deal_type",
]

MODES = ["deterministic", "local", "openai", "auto"]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * p)
    return ordered[idx]


def esc(text: Any) -> str:
    return html.escape(str(text))


def run_benchmark(db_path: Path, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    app = create_app(db_path=db_path)
    client = TestClient(app)

    providers = client.get("/api/assistant/providers").json()
    health = client.get("/api/assistant/health").json()

    runs: list[dict[str, Any]] = []

    for mode in MODES:
        for query in QUERY_SUITE:
            started = time.perf_counter()
            response = client.post(
                "/api/assistant/query",
                json={"goal": query, "llm_mode": mode},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            payload = response.json()
            runtime = payload.get("runtime", {})

            error_text = payload.get("error")
            if not error_text and not payload.get("success", False):
                error_text = payload.get("answer_markdown", "")[:200]

            runs.append(
                {
                    "query": query,
                    "requested_mode": mode,
                    "actual_mode": runtime.get("mode", "unknown"),
                    "provider": runtime.get("provider"),
                    "intent_model": runtime.get("intent_model"),
                    "narrator_model": runtime.get("narrator_model"),
                    "llm_used": bool(runtime.get("use_llm")),
                    "llm_intake_used": bool(runtime.get("llm_intake_used")),
                    "llm_narrative_used": bool(runtime.get("llm_narrative_used")),
                    "llm_effective": bool(runtime.get("llm_effective")),
                    "fallback": mode != runtime.get("mode", mode),
                    "success": bool(payload.get("success", False)),
                    "http_status": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "sql_time_ms": payload.get("execution_time_ms"),
                    "confidence": payload.get("confidence"),
                    "confidence_score": payload.get("confidence_score", 0.0),
                    "row_count": payload.get("row_count"),
                    "error": error_text,
                    "sql": payload.get("sql", ""),
                    "trace_steps": len(payload.get("agent_trace") or []),
                }
            )

    mode_summary: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rows = [r for r in runs if r["requested_mode"] == mode]
        latencies = [r["latency_ms"] for r in rows]
        successes = [r for r in rows if r["success"]]
        mode_summary[mode] = {
            "queries": len(rows),
            "successes": len(successes),
            "success_rate": round((len(successes) / len(rows)) * 100, 2) if rows else 0.0,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50_latency_ms": round(percentile(latencies, 0.5), 2) if latencies else 0.0,
            "p95_latency_ms": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
            "avg_confidence": round(statistics.mean([r["confidence_score"] for r in rows]), 3)
            if rows
            else 0.0,
            "llm_used_rate": round(
                (sum(1 for r in rows if r["llm_used"]) / len(rows)) * 100, 2
            )
            if rows
            else 0.0,
            "llm_effective_rate": round(
                (sum(1 for r in rows if r["llm_effective"]) / len(rows)) * 100, 2
            )
            if rows
            else 0.0,
            "fallback_rate": round(
                (sum(1 for r in rows if r["fallback"]) / len(rows)) * 100, 2
            )
            if rows
            else 0.0,
        }

    failures = [r["error"] for r in runs if not r["success"] and r.get("error")]
    failure_buckets = Counter(failures)

    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        by_query[row["query"]][row["requested_mode"]] = {
            "success": row["success"],
            "latency_ms": row["latency_ms"],
            "actual_mode": row["actual_mode"],
            "provider": row["provider"],
            "confidence_score": row["confidence_score"],
        }

    report_data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "db_path": str(db_path),
        "providers": providers,
        "health": health,
        "queries": QUERY_SUITE,
        "modes": MODES,
        "mode_summary": mode_summary,
        "runs": runs,
        "failure_buckets": dict(failure_buckets.most_common(20)),
        "query_comparison": by_query,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"agentic_benchmark_{ts}.json"
    html_path = out_dir / f"agentic_benchmark_{ts}.html"
    json_path.write_text(json.dumps(report_data, indent=2, default=str))

    html_content = build_html_report(report_data)
    html_path.write_text(html_content)

    return html_path, report_data


def build_html_report(report: dict[str, Any]) -> str:
    providers = report["providers"]
    mode_summary = report["mode_summary"]
    runs = report["runs"]

    provider_rows = "".join(
        f"""
        <tr>
          <td>{esc(name)}</td>
          <td>{'YES' if info.get('available') else 'NO'}</td>
          <td>{esc(info.get('reason'))}</td>
        </tr>
        """
        for name, info in providers.get("checks", {}).items()
    )

    mode_rows = "".join(
        f"""
        <tr>
          <td>{esc(mode)}</td>
          <td>{s['queries']}</td>
          <td>{s['successes']}</td>
          <td>{s['success_rate']}%</td>
          <td>{s['avg_latency_ms']}</td>
          <td>{s['p50_latency_ms']}</td>
          <td>{s['p95_latency_ms']}</td>
          <td>{s['max_latency_ms']}</td>
          <td>{s['avg_confidence']}</td>
          <td>{s['llm_used_rate']}%</td>
          <td>{s['llm_effective_rate']}%</td>
          <td>{s['fallback_rate']}%</td>
        </tr>
        """
        for mode, s in mode_summary.items()
    )

    run_rows = "".join(
        f"""
        <tr>
          <td>{esc(r['query'])}</td>
          <td>{esc(r['requested_mode'])}</td>
          <td>{esc(r['actual_mode'])}</td>
          <td>{esc(r['provider'])}</td>
          <td>{esc(r.get('intent_model') or '')}</td>
          <td>{esc(r.get('narrator_model') or '')}</td>
          <td>{'YES' if r['success'] else 'NO'}</td>
          <td>{'YES' if r.get('llm_effective') else 'NO'}</td>
          <td>{r['latency_ms']}</td>
          <td>{esc(r['row_count'])}</td>
          <td>{esc(r['confidence'])} ({r['confidence_score']:.2f})</td>
          <td>{esc(r['error'] or '')}</td>
          <td><code>{esc((r['sql'] or '')[:220])}</code></td>
        </tr>
        """
        for r in runs
    )

    failure_rows = "".join(
        f"<tr><td>{esc(err)}</td><td>{count}</td></tr>"
        for err, count in report.get("failure_buckets", {}).items()
    )
    if not failure_rows:
        failure_rows = "<tr><td colspan='2'>No failures</td></tr>"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>dataDa Agentic Benchmark Report</title>
  <style>
    body {{
      font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
      margin: 0;
      background: #eef3f8;
      color: #102430;
      padding: 20px;
    }}
    h1, h2 {{ margin: 0 0 10px; color: #0c3d58; }}
    .section {{
      background: #fff;
      border: 1px solid #d5e2eb;
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 10px 26px rgba(7, 45, 62, 0.06);
    }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px; }}
    .chip {{ border: 1px solid #dde8ef; border-radius: 10px; padding: 8px 10px; background: #fbfdff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
    th, td {{ border-bottom: 1px solid #e8eef3; text-align: left; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #f4f9fc; position: sticky; top: 0; }}
    .table-wrap {{ overflow: auto; max-height: 460px; border: 1px solid #dde8ef; border-radius: 10px; }}
    code {{ font-family: 'IBM Plex Mono', Menlo, monospace; font-size: 0.76rem; }}
  </style>
</head>
<body>
  <div class="section">
    <h1>dataDa Agentic POC Benchmark</h1>
    <div class="meta">
      <div class="chip"><strong>Generated (UTC):</strong> {esc(report['generated_at'])}</div>
      <div class="chip"><strong>Database:</strong> {esc(report['db_path'])}</div>
      <div class="chip"><strong>Health:</strong> {esc(report['health'].get('status'))} | semantic_ready={esc(report['health'].get('semantic_ready'))}</div>
      <div class="chip"><strong>Default Mode:</strong> {esc(providers.get('default_mode'))} | <strong>Recommended:</strong> {esc(providers.get('recommended_mode'))}</div>
      <div class="chip"><strong>Total Runs:</strong> {len(runs)} ({len(report['queries'])} queries × {len(report['modes'])} modes)</div>
    </div>
  </div>

  <div class="section">
    <h2>Provider Availability</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Provider</th><th>Available</th><th>Reason</th></tr></thead>
        <tbody>{provider_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Mode Performance Summary</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Mode</th><th>Queries</th><th>Successes</th><th>Success Rate</th>
            <th>Avg Latency (ms)</th><th>P50</th><th>P95</th><th>Max</th>
            <th>Avg Confidence</th><th>LLM Used Rate</th><th>LLM Effective Rate</th><th>Fallback Rate</th>
          </tr>
        </thead>
        <tbody>{mode_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Failure Buckets</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Error</th><th>Count</th></tr></thead>
        <tbody>{failure_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Detailed Run Matrix</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Query</th><th>Requested</th><th>Actual</th><th>Provider</th>
            <th>Intent Model</th><th>Narrator Model</th><th>Success</th><th>LLM Effective</th><th>Latency (ms)</th><th>Rows</th><th>Confidence</th><th>Error</th><th>SQL (prefix)</th>
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
    parser = argparse.ArgumentParser(description="Run dataDa agentic benchmark")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    html_path, data = run_benchmark(Path(args.db_path), Path(args.out_dir))
    print(f"Benchmark complete. Report: {html_path}")
    print("Mode summary:")
    for mode, summary in data["mode_summary"].items():
        print(
            f"- {mode}: success_rate={summary['success_rate']}% "
            f"avg_latency_ms={summary['avg_latency_ms']} fallback_rate={summary['fallback_rate']}%"
        )


if __name__ == "__main__":
    main()
