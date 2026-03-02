#!/usr/bin/env python3
"""Run a lightweight load/stress trend harness for unified v2 runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SLO_P95_MS = {
    "deterministic": 8000.0,
    "auto": 8000.0,
    "openai": 12000.0,
    "anthropic": 12000.0,
    "local": 15000.0,
}


def _p(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _available_modes(base_url: str, requested_modes: list[str], timeout_s: int) -> tuple[list[str], dict[str, Any]]:
    if not requested_modes:
        requested_modes = ["deterministic", "auto", "local"]
    checks_payload: dict[str, Any] = {}
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/assistant/providers", timeout=timeout_s)
        if resp.status_code == 200:
            checks_payload = dict(resp.json() or {})
    except Exception:
        checks_payload = {}
    checks = dict(checks_payload.get("checks") or {})
    out: list[str] = []
    for mode in requested_modes:
        mode = mode.strip().lower()
        if mode in {"deterministic", "auto"}:
            out.append(mode)
            continue
        provider_map = {"local": "ollama", "openai": "openai", "anthropic": "anthropic"}
        provider = provider_map.get(mode, "")
        if provider and bool((checks.get(provider) or {}).get("available")):
            out.append(mode)
    return out, checks_payload


def _single_request(base_url: str, mode: str, idx: int, timeout_s: int) -> dict[str, Any]:
    prompts = [
        "How many transactions are there?",
        "Show valid transaction spend by month.",
        "What kind of data do I have?",
    ]
    payload = {
        "goal": prompts[idx % len(prompts)],
        "llm_mode": mode,
        "session_id": f"load-trend-{mode}-{idx}",
    }
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/assistant/query",
            json=payload,
            timeout=timeout_s,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "status_code": int(resp.status_code),
            "success": bool(body.get("success")) if int(resp.status_code) == 200 else False,
            "latency_ms": elapsed_ms,
            "execution_ms": float(body.get("execution_time_ms") or 0.0),
            "trace_id": str(body.get("trace_id") or ""),
            "error": str(body.get("error") or body.get("detail") or ""),
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "status_code": 599,
            "success": False,
            "latency_ms": elapsed_ms,
            "execution_ms": 0.0,
            "trace_id": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_round(
    *,
    base_url: str,
    mode: str,
    workers: int,
    requests_per_round: int,
    timeout_s: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(_single_request, base_url, mode, seed + i, timeout_s)
            for i in range(max(1, requests_per_round))
        ]
        for fut in concurrent.futures.as_completed(futures):
            rows.append(dict(fut.result()))
    elapsed_s = max(0.001, (time.perf_counter() - started))
    latencies = [float(r.get("latency_ms") or 0.0) for r in rows]
    successes = [r for r in rows if bool(r.get("success"))]
    return {
        "mode": mode,
        "requests": len(rows),
        "successes": len(successes),
        "success_rate": round(len(successes) / max(1, len(rows)), 4),
        "throughput_rps": round(len(rows) / elapsed_s, 3),
        "latency_ms": {
            "avg": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p50": round(_p(latencies, 50), 2),
            "p95": round(_p(latencies, 95), 2),
        },
        "errors": [r for r in rows if not bool(r.get("success"))][:10],
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# v2 Load/Stress Trend Report",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Base URL: `{report.get('base_url')}`",
        f"- Requested modes: `{', '.join(report.get('requested_modes') or [])}`",
        f"- Effective modes: `{', '.join(report.get('effective_modes') or [])}`",
        f"- Rounds: `{report.get('rounds')}`",
        f"- Requests per round: `{report.get('requests_per_round')}`",
        "",
        "## Mode Summary",
    ]
    for mode_row in (report.get("mode_summary") or []):
        lines.extend(
            [
                f"### {mode_row.get('mode')}",
                f"- Success rate: `{mode_row.get('success_rate')}`",
                f"- p95 latency (ms): `{mode_row.get('latency_ms', {}).get('p95')}`",
                f"- Throughput (rps): `{mode_row.get('throughput_rps')}`",
                f"- SLO target (ms): `{mode_row.get('slo_target_ms')}`",
                f"- SLO pass: `{mode_row.get('slo_pass')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Verdict",
            f"- Overall pass: `{report.get('overall_pass')}`",
            f"- Blocking modes: `{', '.join(report.get('blocking_modes') or []) or 'none'}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2 load/stress trend harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--modes", default="deterministic,auto,local")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests-per-round", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    requested_modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    effective_modes, providers_snapshot = _available_modes(
        args.base_url,
        requested_modes=requested_modes,
        timeout_s=max(5, int(args.timeout)),
    )
    if not effective_modes:
        raise SystemExit("No viable modes available for load/stress trend run.")

    rounds = max(1, int(args.rounds))
    requests_per_round = max(2, int(args.requests_per_round))
    workers = max(1, int(args.workers))
    timeout_s = max(5, int(args.timeout))

    mode_rows: list[dict[str, Any]] = []
    for mode in effective_modes:
        round_rows: list[dict[str, Any]] = []
        for r in range(rounds):
            row = _run_round(
                base_url=args.base_url,
                mode=mode,
                workers=workers,
                requests_per_round=requests_per_round,
                timeout_s=timeout_s,
                seed=(r * requests_per_round),
            )
            row["round_index"] = r + 1
            round_rows.append(row)
        lat_p95 = [float((r.get("latency_ms") or {}).get("p95") or 0.0) for r in round_rows]
        throughput = [float(r.get("throughput_rps") or 0.0) for r in round_rows]
        success_rates = [float(r.get("success_rate") or 0.0) for r in round_rows]
        summary = {
            "mode": mode,
            "rounds": round_rows,
            "success_rate": round(statistics.fmean(success_rates), 4) if success_rates else 0.0,
            "throughput_rps": round(statistics.fmean(throughput), 3) if throughput else 0.0,
            "latency_ms": {
                "p50": round(_p(lat_p95, 50), 2),
                "p95": round(_p(lat_p95, 95), 2),
            },
            "slo_target_ms": float(SLO_P95_MS.get(mode, 12000.0)),
        }
        summary["slo_pass"] = bool(
            float(summary["latency_ms"]["p95"]) <= float(summary["slo_target_ms"])
            and float(summary["success_rate"]) >= 0.98
        )
        mode_rows.append(summary)

    blocking = [str(r.get("mode") or "") for r in mode_rows if not bool(r.get("slo_pass"))]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "requested_modes": requested_modes,
        "effective_modes": effective_modes,
        "rounds": rounds,
        "requests_per_round": requests_per_round,
        "workers": workers,
        "providers_snapshot": providers_snapshot,
        "mode_summary": mode_rows,
        "overall_pass": not blocking,
        "blocking_modes": blocking,
    }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"v2_load_stress_trend_{stamp}.json"
    md_path = out_dir / f"v2_load_stress_trend_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "overall_pass": report["overall_pass"]}, indent=2))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
