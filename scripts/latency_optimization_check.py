#!/usr/bin/env python3
"""Quick latency and optimization check for query runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Latency Optimization Check",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Base URL: `{report.get('base_url')}`",
        f"- Runs: `{report.get('runs')}`",
        f"- Successes: `{report.get('successes')}`",
        f"- Cache hit ratio: `{report.get('cache_hit_ratio')}`",
        "",
        "## End-to-end latency (ms)",
        f"- p50: `{report.get('e2e_latency_ms', {}).get('p50', 0.0)}`",
        f"- p95: `{report.get('e2e_latency_ms', {}).get('p95', 0.0)}`",
        "",
        "## Engine execution latency (ms)",
        f"- p50: `{report.get('engine_execution_ms', {}).get('p50', 0.0)}`",
        f"- p95: `{report.get('engine_execution_ms', {}).get('p95', 0.0)}`",
        "",
        "## Notes",
        "- Repeated deterministic queries should show increased cache-hit ratio after first run.",
        "- Use this report as a pre-flight check before full multimode QA.",
    ]
    return "\n".join(lines) + "\n"


def run_check(base_url: str, iterations: int, timeout: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    successes = 0
    cache_hits = 0
    e2e_latencies: list[float] = []
    engine_latencies: list[float] = []
    question = "How many transactions are there?"

    for idx in range(iterations):
        payload = {
            "goal": question,
            "llm_mode": "deterministic",
            "session_id": f"latency-check-{idx}",
        }
        started = time.perf_counter()
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/assistant/query",
            json=payload,
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        e2e_latencies.append(elapsed_ms)

        run_row: dict[str, Any] = {
            "index": idx + 1,
            "status_code": int(resp.status_code),
            "e2e_latency_ms": elapsed_ms,
            "success": False,
            "cache_hit": False,
            "engine_execution_ms": 0.0,
            "error": "",
        }
        if resp.status_code == 200:
            body = resp.json()
            run_row["success"] = bool(body.get("success"))
            if run_row["success"]:
                successes += 1
            runtime = body.get("runtime") or {}
            cache_hit = bool(runtime.get("response_cache_hit"))
            run_row["cache_hit"] = cache_hit
            if cache_hit:
                cache_hits += 1
            engine_ms = float(body.get("execution_time_ms") or 0.0)
            run_row["engine_execution_ms"] = round(engine_ms, 2)
            if engine_ms > 0:
                engine_latencies.append(engine_ms)
        else:
            run_row["error"] = f"HTTP {resp.status_code}"
        runs.append(run_row)

    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "base_url": base_url,
        "runs": iterations,
        "successes": successes,
        "cache_hits": cache_hits,
        "cache_hit_ratio": round(cache_hits / max(1, iterations), 4),
        "e2e_latency_ms": {
            "avg": round(statistics.mean(e2e_latencies), 2) if e2e_latencies else 0.0,
            "p50": round(_percentile(e2e_latencies, 50), 2),
            "p95": round(_percentile(e2e_latencies, 95), 2),
        },
        "engine_execution_ms": {
            "avg": round(statistics.mean(engine_latencies), 2) if engine_latencies else 0.0,
            "p50": round(_percentile(engine_latencies, 50), 2),
            "p95": round(_percentile(engine_latencies, 95), 2),
        },
        "run_details": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast latency optimization checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    report = run_check(
        base_url=args.base_url,
        iterations=max(2, int(args.iterations)),
        timeout=max(5, int(args.timeout)),
    )
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"latency_optimization_check_{stamp}.json"
    md_path = out_dir / f"latency_optimization_check_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"saved_json={json_path}")
    print(f"saved_md={md_path}")
    print(f"cache_hit_ratio={report['cache_hit_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
