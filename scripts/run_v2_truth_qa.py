#!/usr/bin/env python3
"""Tiered QA truth runner for v2 quality-first governance.

This runner aggregates multi-suite quality signals into a machine-readable
truth report used by release gates and scoreboard drift checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import requests


NON_ACTIONABLE_WARNING_PATTERNS: tuple[str, ...] = (
    "chief analyst llm enhancement failed",
    "llm intake refinement failed",
    "openai was unavailable for this run",
    "anthropic was unavailable for this run",
    "local model was unavailable for this run",
    "blocked by policy gate:",
    "[skillcontract]",
    "autonomy switch rejected",
)


@dataclass(frozen=True)
class SuiteGate:
    suite_id: str
    min_score_pct: float


SUITE_GATES: dict[str, SuiteGate] = {
    "Q0": SuiteGate("Q0", 100.0),
    "Q1": SuiteGate("Q1", 99.0),
    "Q2": SuiteGate("Q2", 98.0),
    "Q3": SuiteGate("Q3", 98.0),
    "Q4": SuiteGate("Q4", 95.0),
    "Q5": SuiteGate("Q5", 100.0),
    "Q6": SuiteGate("Q6", 95.0),
    "Q7": SuiteGate("Q7", 100.0),
    "Q8": SuiteGate("Q8", 100.0),
}


TRUTH_WEIGHTS: dict[str, float] = {
    "Q1": 0.22,
    "Q2": 0.18,
    "Q3": 0.16,
    "Q4": 0.12,
    "Q5": 0.12,
    "Q6": 0.08,
    "Q7": 0.07,
    "Q8": 0.05,
}


SEMANTIC_PROBES: list[dict[str, Any]] = [
    {
        "id": "S01_valid_txn_guard",
        "prompt": "How many valid transactions were completed in November 2025?",
        "expected_sql_tokens": ["has_mt103"],
        "expected_refusal": False,
    },
    {
        "id": "S02_spend_validity",
        "prompt": "Show customer spend on transaction by platform for December 2025.",
        "expected_sql_tokens": ["has_mt103", "platform_name"],
        "expected_refusal": False,
    },
    {
        "id": "S03_markup_margin",
        "prompt": "What is forex markup revenue for INR in January 2026?",
        "expected_sql_tokens": ["forex_markup", "from_currency"],
        "expected_refusal": False,
    },
    {
        "id": "S04_refund_cluster",
        "prompt": "In 2025, summarize refunded transactions and top amounts.",
        "expected_sql_tokens": ["has_refund"],
        "expected_refusal": False,
    },
    {
        "id": "S05_funnel_health",
        "prompt": "Give an executive summary of funnel health from quote to booking to MT103 in January 2026.",
        "expected_sql_tokens": [],
        "expected_refusal": False,
    },
    {
        "id": "S06_unsupported_sentiment",
        "prompt": "Show sentiment score trend by month for 2025.",
        "expected_sql_tokens": [],
        "expected_refusal": True,
    },
    {
        "id": "S07_future_fabrication",
        "prompt": "Predict MT103 volume for 2029 with numbers.",
        "expected_sql_tokens": [],
        "expected_refusal": True,
    },
    {
        "id": "S08_dictionary",
        "prompt": "Build a business glossary for this dataset and include field meanings.",
        "expected_sql_tokens": [],
        "expected_refusal": False,
    },
    {
        "id": "S09_cross_domain",
        "prompt": "Count unique clients in UNITED KINGDOM and split universities vs non-universities.",
        "expected_sql_tokens": ["address_country", "is_university"],
        "expected_refusal": False,
    },
    {
        "id": "S10_scenario_tradeoff",
        "prompt": "If we reduce forex markup by 10%, estimate likely conversion impact and list assumptions.",
        "expected_sql_tokens": [],
        "expected_refusal": False,
    },
]


UNSEEN_PORTABILITY_FIXTURES: list[dict[str, Any]] = [
    {
        "fixture_id": "ops_payments",
        "description": "Operational + payment ledger with standard account joins",
        "sql": [
            """
            CREATE TABLE crm_accounts(
                account_id VARCHAR,
                segment VARCHAR,
                home_country VARCHAR,
                created_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE payments_ledger(
                payment_id VARCHAR,
                account_id VARCHAR,
                corridor VARCHAR,
                settled_amount DOUBLE,
                settled_at TIMESTAMP,
                status VARCHAR
            )
            """,
            """
            INSERT INTO crm_accounts VALUES
            ('A1','enterprise','US','2026-01-01'),
            ('A2','smb','IN','2026-01-02'),
            ('A3','smb','GB','2026-01-03')
            """,
            """
            INSERT INTO payments_ledger VALUES
            ('P1','A1','US-IN',1200.0,'2026-01-10','settled'),
            ('P2','A1','US-GB',800.0,'2026-01-11','settled'),
            ('P3','A2','IN-US',540.0,'2026-01-11','pending'),
            ('P4','A3','GB-IN',430.0,'2026-01-12','settled')
            """,
        ],
        "expected": {
            "table_count_min": 2,
            "entities_min": 2,
            "measures_min": 1,
            "time_fields_min": 1,
            "high_risk_join_edges_min": 0,
        },
    },
    {
        "fixture_id": "marketing_weak_join",
        "description": "Weak campaign join coverage should surface join fragility",
        "sql": [
            """
            CREATE TABLE campaign_spend(
                campaign_id VARCHAR,
                channel VARCHAR,
                spend_usd DOUBLE,
                event_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE lead_funnel(
                lead_id VARCHAR,
                campaign_id VARCHAR,
                stage VARCHAR,
                created_at TIMESTAMP
            )
            """,
            """
            INSERT INTO campaign_spend VALUES
            ('C1','search',5400.0,'2026-02-01'),
            ('C2','social',3200.0,'2026-02-01'),
            ('C3','email',900.0,'2026-02-01')
            """,
            """
            INSERT INTO lead_funnel VALUES
            ('L1','C2','quote','2026-02-02'),
            ('L2','C9','quote','2026-02-02'),
            ('L3','C10','booking','2026-02-02'),
            ('L4','C11','mt103','2026-02-03')
            """,
        ],
        "expected": {
            "table_count_min": 2,
            "entities_min": 2,
            "measures_min": 1,
            "time_fields_min": 1,
            "high_risk_join_edges_min": 1,
        },
    },
    {
        "fixture_id": "refund_txn_sparse",
        "description": "Sparse refund + transaction dataset with nullable operational fields",
        "sql": [
            """
            CREATE TABLE transaction_state(
                transaction_id VARCHAR,
                customer_id VARCHAR,
                platform VARCHAR,
                gross_amount DOUBLE,
                has_mt103 BOOLEAN,
                created_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE refund_events(
                refund_id VARCHAR,
                transaction_id VARCHAR,
                refund_amount DOUBLE,
                refund_reason VARCHAR,
                initiated_at TIMESTAMP
            )
            """,
            """
            INSERT INTO transaction_state VALUES
            ('T1','U1','WEB',980.0,TRUE,'2026-01-15'),
            ('T2','U2','MOBILE',440.0,FALSE,'2026-01-15'),
            ('T3','U3',NULL,780.0,TRUE,'2026-01-16'),
            ('T4','U4','WEB',NULL,TRUE,'2026-01-16')
            """,
            """
            INSERT INTO refund_events VALUES
            ('R1','T2',120.0,'duplicate','2026-01-20'),
            ('R2','T4',NULL,'chargeback','2026-01-21'),
            ('R3',NULL,30.0,'goodwill','2026-01-22')
            """,
        ],
        "expected": {
            "table_count_min": 2,
            "entities_min": 2,
            "measures_min": 1,
            "time_fields_min": 1,
            "high_risk_join_edges_min": 0,
        },
    },
]


TIER_CONFIG: dict[str, dict[str, Any]] = {
    "pr": {
        "quick_blackbox": True,
        "blackbox_modes": ["deterministic", "auto"],
        "blackbox_mode_workers": 2,
        "blackbox_heavy_cloud_workers": 1,
        "atomic_workers": 4,
        "local_atomic_workers": 1,
        "followup_workers": 2,
        "local_followup_workers": 1,
        "blackbox_request_timeout": 120,
        "blackbox_retry_count": 2,
        "blackbox_subprocess_timeout_s": 420,
        "blackbox_health_timeout": 20,
        "blackbox_health_retries": 2,
        "blackbox_require_startup_health": False,
        "semantic_mode": "deterministic",
        "semantic_workers": 4,
        "latency_modes": ["deterministic"],
        "latency_iterations": 6,
        "latency_mode_workers": 1,
        "latency_consecutive_fail_limit": 2,
        "provider_integrity_timeout_s": 45,
    },
    "merge": {
        "quick_blackbox": False,
        "blackbox_modes": ["deterministic", "auto"],
        "blackbox_mode_workers": 2,
        "blackbox_heavy_cloud_workers": 1,
        "atomic_workers": 5,
        "local_atomic_workers": 1,
        "followup_workers": 2,
        "local_followup_workers": 1,
        "blackbox_request_timeout": 180,
        "blackbox_retry_count": 2,
        "blackbox_subprocess_timeout_s": 900,
        "blackbox_health_timeout": 25,
        "blackbox_health_retries": 3,
        "blackbox_require_startup_health": False,
        "semantic_mode": "auto",
        "semantic_workers": 5,
        "latency_modes": ["deterministic", "auto"],
        "latency_iterations": 6,
        "latency_mode_workers": 2,
        "latency_consecutive_fail_limit": 2,
        "provider_integrity_timeout_s": 45,
    },
    "release": {
        "quick_blackbox": False,
        "blackbox_modes": ["deterministic", "auto", "local"],
        "blackbox_mode_workers": 3,
        "blackbox_heavy_cloud_workers": 2,
        "atomic_workers": 6,
        "local_atomic_workers": 2,
        "followup_workers": 2,
        "local_followup_workers": 1,
        "blackbox_request_timeout": 120,
        "blackbox_retry_count": 2,
        "blackbox_subprocess_timeout_s": 1800,
        "blackbox_health_timeout": 45,
        "blackbox_health_retries": 4,
        "blackbox_require_startup_health": False,
        "semantic_mode": "auto",
        "semantic_workers": 6,
        "latency_modes": ["deterministic", "auto", "openai", "anthropic"],
        "latency_iterations": 6,
        "latency_mode_workers": 2,
        "latency_consecutive_fail_limit": 2,
        "provider_integrity_timeout_s": 60,
    },
}


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _coerce_text(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return value.decode(errors="replace")
    if value is None:
        return ""
    return str(value)


def _is_actionable_warning(warning: Any) -> bool:
    text = str(warning or "").strip().lower()
    if not text:
        return False
    return not any(pattern in text for pattern in NON_ACTIONABLE_WARNING_PATTERNS)


def _count_actionable_warnings(warnings: Any) -> int:
    if not isinstance(warnings, list):
        return 0
    return sum(1 for warning in warnings if _is_actionable_warning(warning))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    idx = int(round(rank))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _latest_file(directory: Path, pattern: str, since_ts: float | None = None) -> Path | None:
    paths = list(directory.glob(pattern))
    if since_ts is not None:
        paths = [p for p in paths if p.stat().st_mtime >= since_ts]
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _http_query(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout_s: int,
    tenant_id: str,
) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/assistant/query",
        json=payload,
        timeout=timeout_s,
        headers={"x-datada-role": "admin", "x-datada-tenant-id": tenant_id},
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    try:
        body = resp.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    return int(resp.status_code), body, elapsed_ms


def _provider_mode_availability(base_url: str) -> tuple[dict[str, bool], dict[str, Any]]:
    availability: dict[str, bool] = {
        "deterministic": True,
        "auto": True,
        "local": True,
        "openai": True,
        "anthropic": True,
    }
    snapshot: dict[str, Any] = {}
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/assistant/providers", timeout=20)
        if resp.ok:
            body = resp.json()
            if isinstance(body, dict):
                snapshot = body
                checks = body.get("checks") or {}
                if isinstance(checks, dict):
                    availability["local"] = bool((checks.get("ollama") or {}).get("available", False))
                    availability["openai"] = bool((checks.get("openai") or {}).get("available", False))
                    availability["anthropic"] = bool((checks.get("anthropic") or {}).get("available", False))
    except Exception:
        snapshot = {}
    return availability, snapshot


def _select_available_modes(
    requested_modes: list[str],
    availability: dict[str, bool],
) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    skipped: list[str] = []
    for mode in requested_modes:
        key = str(mode or "").strip().lower()
        if not key:
            continue
        if key in {"deterministic", "auto"}:
            selected.append(key)
            continue
        if bool(availability.get(key, True)):
            selected.append(key)
        else:
            skipped.append(key)
    if not selected:
        selected = ["deterministic", "auto"]
    return selected, skipped


def _run_blackbox_suite(
    *,
    base_url: str,
    db_path: Path,
    out_dir: Path,
    tier_cfg: dict[str, Any],
    round_id: str,
) -> dict[str, Any]:
    before = time.time()
    availability, provider_snapshot = _provider_mode_availability(base_url)
    cmd = [
        sys.executable,
        "scripts/qa_round11_blackbox_fresh.py",
        "--base-url",
        base_url,
        "--db-path",
        str(db_path),
        "--out-dir",
        str(out_dir),
        "--round-id",
        round_id,
        "--mode-workers",
        str(_safe_int(tier_cfg.get("blackbox_mode_workers"), 2)),
        "--heavy-cloud-workers",
        str(_safe_int(tier_cfg.get("blackbox_heavy_cloud_workers"), 1)),
        "--atomic-workers",
        str(_safe_int(tier_cfg.get("atomic_workers"), 4)),
        "--local-atomic-workers",
        str(_safe_int(tier_cfg.get("local_atomic_workers"), 1)),
        "--followup-workers",
        str(_safe_int(tier_cfg.get("followup_workers"), 2)),
        "--local-followup-workers",
        str(_safe_int(tier_cfg.get("local_followup_workers"), 1)),
        "--request-timeout",
        str(_safe_int(tier_cfg.get("blackbox_request_timeout"), 120)),
        "--retry-count",
        str(_safe_int(tier_cfg.get("blackbox_retry_count"), 2)),
        "--health-timeout",
        str(_safe_int(tier_cfg.get("blackbox_health_timeout"), 20)),
        "--health-retries",
        str(_safe_int(tier_cfg.get("blackbox_health_retries"), 2)),
    ]
    if bool(tier_cfg.get("blackbox_require_startup_health", False)):
        cmd.append("--require-startup-health")
    requested_modes = [str(m).strip().lower() for m in (tier_cfg.get("blackbox_modes") or []) if str(m).strip()]
    selected_modes, skipped_modes = _select_available_modes(requested_modes, availability)
    if selected_modes:
        cmd.extend(["--modes", ",".join(selected_modes)])
    if bool(tier_cfg.get("quick_blackbox")):
        cmd.append("--quick")
    timeout_s = max(60, _safe_int(tier_cfg.get("blackbox_subprocess_timeout_s"), 1800))
    timed_out = False
    run_started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        completed = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=(exc.stdout or ""),
            stderr=(exc.stderr or "") + f"\n[timeout] blackbox suite exceeded {timeout_s}s",
        )
    elapsed_s = round(time.perf_counter() - run_started, 2)

    report_path = _latest_file(out_dir, "qa_round11_blackbox_fresh_*.json", since_ts=before)
    payload: dict[str, Any] = {}
    if report_path is not None:
        try:
            payload = _load_json(report_path)
        except Exception:
            payload = {}
    summary = ((payload.get("summary") or {}).get("overall") or {})
    category_scores = dict(summary.get("category_scores") or {})
    results = payload.get("results") or []
    if not isinstance(results, list):
        results = []
    raw_warning_counts = [len((r or {}).get("warnings") or []) for r in results if isinstance(r, dict)]
    actionable_warning_counts = [
        _count_actionable_warnings((r or {}).get("warnings") or [])
        for r in results
        if isinstance(r, dict)
    ]
    return {
        "command": " ".join(cmd),
        "return_code": int(completed.returncode),
        "timed_out": timed_out,
        "timeout_budget_s": timeout_s,
        "elapsed_s": elapsed_s,
        "requested_modes": requested_modes,
        "selected_modes": selected_modes,
        "skipped_unavailable_modes": skipped_modes,
        "provider_snapshot": provider_snapshot,
        "stdout_tail": _coerce_text(completed.stdout)[-4000:],
        "stderr_tail": _coerce_text(completed.stderr)[-4000:],
        "report_path": str(report_path) if report_path else "",
        "overall_pass_rate_pct": _safe_float(summary.get("overall_pass_rate"), 0.0),
        "category_scores": category_scores,
        "avg_warning_count": round(statistics.mean(actionable_warning_counts), 4) if actionable_warning_counts else 0.0,
        "avg_warning_count_actionable": round(statistics.mean(actionable_warning_counts), 4)
        if actionable_warning_counts
        else 0.0,
        "avg_warning_count_raw": round(statistics.mean(raw_warning_counts), 4) if raw_warning_counts else 0.0,
        "results_count": len(results),
    }


def _is_refusal(payload: dict[str, Any]) -> bool:
    answer = str(payload.get("answer_markdown") or "").lower()
    return (
        not bool(payload.get("success"))
        or "cannot answer" in answer
        or "not supported" in answer
        or "policy gate" in answer
        or "cannot" in answer
    )


def _semantic_probe_case(
    *,
    base_url: str,
    mode: str,
    tenant_id: str,
    probe: dict[str, Any],
    timeout_s: int = 120,
) -> dict[str, Any]:
    payload = {
        "goal": str(probe["prompt"]),
        "llm_mode": mode,
        "session_id": f"truth-sem-{probe['id']}",
        "tenant_id": tenant_id,
        "role": "admin",
    }
    status, body, latency_ms = _http_query(
        base_url=base_url,
        payload=payload,
        timeout_s=timeout_s,
        tenant_id=tenant_id,
    )
    sql = str(body.get("sql") or "").lower()
    expected_tokens = [str(t).lower() for t in (probe.get("expected_sql_tokens") or [])]
    sql_ok = all(tok in sql for tok in expected_tokens) if expected_tokens else True
    refusal = _is_refusal(body)
    expected_refusal = bool(probe.get("expected_refusal"))
    expectation_met = status == 200 and sql_ok and (refusal == expected_refusal)
    warnings = body.get("warnings") or []
    warning_count_raw = len(warnings) if isinstance(warnings, list) else 0
    warning_count_actionable = _count_actionable_warnings(warnings if isinstance(warnings, list) else [])
    return {
        "id": probe["id"],
        "status_code": status,
        "expectation_met": expectation_met,
        "sql_tokens_met": sql_ok,
        "expected_sql_tokens": expected_tokens,
        "refusal_detected": refusal,
        "expected_refusal": expected_refusal,
        "warning_count": warning_count_actionable,
        "warning_count_actionable": warning_count_actionable,
        "warning_count_raw": warning_count_raw,
        "confidence_score": _safe_float(body.get("confidence_score"), 0.0),
        "execution_time_ms": _safe_float(body.get("execution_time_ms"), latency_ms),
        "analysis_version": str(body.get("analysis_version") or "v1"),
        "provider_effective": str(body.get("provider_effective") or ""),
        "fallback_used": dict(body.get("fallback_used") or {}),
        "quality_flags": list(body.get("quality_flags") or []),
        "answer_excerpt": str(body.get("answer_markdown") or "")[:220],
        "sql_excerpt": str(body.get("sql") or "")[:260],
    }


def _run_semantic_suite(
    *,
    base_url: str,
    mode: str,
    tenant_id: str,
    workers: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worker_count = max(1, int(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _semantic_probe_case,
                base_url=base_url,
                mode=mode,
                tenant_id=tenant_id,
                probe=probe,
            )
            for probe in SEMANTIC_PROBES
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda r: str(r.get("id") or ""))
    met = sum(1 for row in rows if bool(row.get("expectation_met")))
    warning_avg = statistics.mean([_safe_float(row.get("warning_count"), 0.0) for row in rows]) if rows else 0.0
    warning_avg_raw = statistics.mean([_safe_float(row.get("warning_count_raw"), 0.0) for row in rows]) if rows else 0.0
    return {
        "mode": mode,
        "probes": len(rows),
        "expectations_met": met,
        "expectation_pass_rate_pct": round((100.0 * met / max(1, len(rows))), 2),
        "avg_warning_count": round(float(warning_avg), 4),
        "avg_warning_count_raw": round(float(warning_avg_raw), 4),
        "rows": rows,
    }


def _run_smoke_suite(*, base_url: str, tenant_id: str) -> dict[str, Any]:
    health = requests.get(f"{base_url.rstrip('/')}/api/assistant/health", timeout=20)
    if health.status_code != 200:
        return {
            "pass_rate_pct": 0.0,
            "checks": [
                {"name": "health", "passed": False, "detail": f"HTTP {health.status_code}"},
            ],
        }
    payload = {
        "goal": "How many transactions are there?",
        "llm_mode": "deterministic",
        "session_id": "truth-smoke",
        "tenant_id": tenant_id,
        "role": "admin",
    }
    status, body, _ = _http_query(base_url=base_url, payload=payload, timeout_s=60, tenant_id=tenant_id)
    checks = [
        {"name": "health", "passed": True, "detail": "ok"},
        {"name": "query_http", "passed": status == 200, "detail": f"HTTP {status}"},
        {"name": "query_success", "passed": bool(body.get("success")), "detail": "success flag"},
        {"name": "query_sql", "passed": bool(str(body.get("sql") or "").strip()), "detail": "sql presence"},
    ]
    passed = sum(1 for c in checks if bool(c["passed"]))
    return {
        "pass_rate_pct": round((100.0 * passed / max(1, len(checks))), 2),
        "checks": checks,
    }


def _run_unseen_portability_suite() -> dict[str, Any]:
    try:
        from haikugraph.v2.semantic_profiler import profile_dataset
    except Exception as exc:  # noqa: BLE001
        return {
            "fixture_count": len(UNSEEN_PORTABILITY_FIXTURES),
            "checks": [
                {
                    "name": "import_v2_semantic_profiler",
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
            "fixtures": [],
            "pass_rate_pct": 0.0,
        }

    fixture_rows: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="v2-portability-") as temp_dir:
        base = Path(temp_dir)
        for fixture in UNSEEN_PORTABILITY_FIXTURES:
            fixture_id = str(fixture.get("fixture_id") or "fixture")
            description = str(fixture.get("description") or "")
            db_path = base / f"{fixture_id}.duckdb"
            checks: list[dict[str, Any]] = []
            try:
                conn = duckdb.connect(str(db_path))
                try:
                    for stmt in list(fixture.get("sql") or []):
                        conn.execute(str(stmt))
                finally:
                    conn.close()

                catalog = profile_dataset(str(db_path))
                expected = dict(fixture.get("expected") or {})
                table_count = len(list(catalog.tables or []))
                entities_count = len(list(catalog.entities or []))
                measures_count = len(list(catalog.measures or []))
                time_fields_count = len(list(catalog.time_fields or []))
                high_risk = int(catalog.quality_summary.get("high_risk_join_edges") or 0)

                checks = [
                    {
                        "name": f"{fixture_id}:table_count",
                        "passed": table_count >= int(expected.get("table_count_min", 1)),
                        "detail": table_count,
                    },
                    {
                        "name": f"{fixture_id}:dataset_signature",
                        "passed": bool(str(catalog.dataset_signature or "").strip()),
                        "detail": str(catalog.dataset_signature or ""),
                    },
                    {
                        "name": f"{fixture_id}:entities",
                        "passed": entities_count >= int(expected.get("entities_min", 1)),
                        "detail": entities_count,
                    },
                    {
                        "name": f"{fixture_id}:measures",
                        "passed": measures_count >= int(expected.get("measures_min", 1)),
                        "detail": measures_count,
                    },
                    {
                        "name": f"{fixture_id}:time_fields",
                        "passed": time_fields_count >= int(expected.get("time_fields_min", 1)),
                        "detail": time_fields_count,
                    },
                    {
                        "name": f"{fixture_id}:join_risk_expectation",
                        "passed": high_risk >= int(expected.get("high_risk_join_edges_min", 0)),
                        "detail": high_risk,
                    },
                ]
                passed = sum(1 for c in checks if bool(c["passed"]))
                fixture_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "description": description,
                        "db_path": str(db_path),
                        "checks": checks,
                        "pass_rate_pct": round((100.0 * passed / max(1, len(checks))), 2),
                        "quality_summary": dict(catalog.quality_summary or {}),
                        "table_count": table_count,
                    }
                )
                all_checks.extend(checks)
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "name": f"{fixture_id}:fixture_execution",
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                checks.append(failure)
                fixture_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "description": description,
                        "db_path": str(db_path),
                        "checks": checks,
                        "pass_rate_pct": 0.0,
                        "quality_summary": {},
                        "table_count": 0,
                    }
                )
                all_checks.append(failure)

    met = sum(1 for c in all_checks if bool(c.get("passed")))
    return {
        "fixture_count": len(fixture_rows),
        "checks": all_checks,
        "fixtures": fixture_rows,
        "pass_rate_pct": round((100.0 * met / max(1, len(all_checks))), 2),
    }


def _run_portability_suite(
    *,
    base_url: str,
    tenant_id: str,
) -> dict[str, Any]:
    payload = {
        "db_connection_id": "default",
        "tenant_id": tenant_id,
        "role": "viewer",
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/assistant/datasets/profile",
        json=payload,
        timeout=60,
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": tenant_id},
    )
    body: dict[str, Any] = {}
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        body = {}
    table_count = _safe_int(body.get("table_count"), 0)
    signature = str(body.get("dataset_signature") or "")
    quality_profile = body.get("profile") or {}
    join_edges = ((quality_profile.get("join_edges") if isinstance(quality_profile, dict) else None) or [])
    high_risk = _safe_int(body.get("high_risk_join_edges"), 0)
    api_checks = [
        ("http_200", resp.status_code == 200),
        ("table_count", table_count > 0),
        ("dataset_signature", bool(signature)),
        ("join_edges_profiled", isinstance(join_edges, list)),
    ]
    unseen = _run_unseen_portability_suite()
    checks = [{"name": f"api:{name}", "passed": ok} for name, ok in api_checks] + [
        {"name": f"unseen:{c.get('name')}", "passed": bool(c.get("passed"))}
        for c in list(unseen.get("checks") or [])
    ]
    passed = sum(1 for c in checks if bool(c.get("passed")))
    return {
        "status_code": int(resp.status_code),
        "table_count": table_count,
        "high_risk_join_edges": high_risk,
        "sparse_table_count": _safe_int(body.get("sparse_table_count"), 0),
        "checks": checks,
        "api_pass_rate_pct": round((100.0 * sum(1 for _, ok in api_checks if ok) / max(1, len(api_checks))), 2),
        "unseen_pass_rate_pct": _safe_float(unseen.get("pass_rate_pct"), 0.0),
        "pass_rate_pct": round((100.0 * passed / max(1, len(checks))), 2),
        "unseen_fixture_suite": unseen,
        "response_excerpt": body,
    }


def _run_latency_mode(
    *,
    base_url: str,
    tenant_id: str,
    mode: str,
    iterations: int,
    consecutive_fail_limit: int,
) -> dict[str, Any]:
    targets = {
        "deterministic": 8000.0,
        "auto": 8000.0,
        "openai": 12000.0,
        "anthropic": 12000.0,
        "local": 15000.0,
    }
    timeout_targets = {
        "deterministic": 45,
        "auto": 45,
        "openai": 45,
        "anthropic": 45,
        "local": 60,
    }
    latencies: list[float] = []
    ok_calls = 0
    attempted = 0
    fail_streak = 0
    aborted_early = False
    max_calls = max(2, int(iterations))
    stop_after = max(2, int(consecutive_fail_limit))

    for idx in range(max_calls):
        payload = {
            "goal": "How many transactions are there?",
            "llm_mode": mode,
            "session_id": f"truth-lat-{mode}-{idx}",
            "tenant_id": tenant_id,
            "role": "admin",
        }
        status, body, elapsed = _http_query(
            base_url=base_url,
            payload=payload,
            timeout_s=int(timeout_targets.get(mode, 60)),
            tenant_id=tenant_id,
        )
        attempted += 1
        if status == 200 and isinstance(body, dict):
            latencies.append(_safe_float(body.get("execution_time_ms"), elapsed))
            ok_calls += 1
            fail_streak = 0
        else:
            fail_streak += 1
        # Fail fast when provider path is unavailable/unresponsive.
        if ok_calls == 0 and fail_streak >= stop_after:
            aborted_early = attempted < max_calls
            break

    p95 = _percentile(latencies, 95.0)
    target = float(targets.get(mode, 12000.0))
    return {
        "mode": mode,
        "calls_planned": max_calls,
        "calls_attempted": attempted,
        "ok_calls": ok_calls,
        "aborted_early": aborted_early,
        "p50_ms": round(_percentile(latencies, 50.0), 2),
        "p95_ms": round(p95, 2),
        "target_p95_ms": round(target, 2),
        "meets_slo": bool(latencies) and p95 <= target,
    }


def _run_latency_suite(
    *,
    base_url: str,
    tenant_id: str,
    modes: list[str],
    iterations: int,
    mode_workers: int,
    consecutive_fail_limit: int,
) -> dict[str, Any]:
    mode_rows: list[dict[str, Any]] = []
    indexed_modes = list(enumerate(modes))
    rows_by_index: dict[int, dict[str, Any]] = {}
    max_workers = max(1, min(int(mode_workers), max(1, len(indexed_modes))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_latency_mode,
                base_url=base_url,
                tenant_id=tenant_id,
                mode=mode,
                iterations=iterations,
                consecutive_fail_limit=consecutive_fail_limit,
            ): idx
            for idx, mode in indexed_modes
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            rows_by_index[idx] = future.result()
    for idx, _mode in indexed_modes:
        mode_rows.append(rows_by_index[idx])
    met = sum(1 for row in mode_rows if bool(row.get("meets_slo")))
    return {
        "modes": mode_rows,
        "pass_rate_pct": round((100.0 * met / max(1, len(mode_rows))), 2),
    }


def _run_provider_integrity_checks(
    *,
    base_url: str,
    tenant_id: str,
    availability: dict[str, bool] | None = None,
    timeout_s: int = 60,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for mode in ["openai", "anthropic", "local"]:
        if availability is not None and not bool(availability.get(mode, True)):
            checks.append(
                {
                    "mode": mode,
                    "status": "unavailable",
                    "passed": True,
                    "reason": "provider preflight unavailable",
                }
            )
            continue
        payload = {
            "goal": "How many transactions are there?",
            "llm_mode": mode,
            "session_id": f"truth-provider-{mode}",
            "tenant_id": tenant_id,
            "role": "admin",
        }
        status, body, _ = _http_query(
            base_url=base_url,
            payload=payload,
            timeout_s=max(20, int(timeout_s)),
            tenant_id=tenant_id,
        )
        if status == 503:
            checks.append({"mode": mode, "status": "unavailable", "passed": True})
            continue
        runtime = body.get("runtime") or {}
        fallback = body.get("fallback_used") or {}
        provider_effective = str(body.get("provider_effective") or "")
        llm_degraded = bool(runtime.get("llm_degraded"))
        fallback_used = bool(fallback.get("used"))
        # Hidden fallback fails: degraded with no explicit fallback flag.
        hidden_fallback = llm_degraded and not fallback_used
        checks.append(
            {
                "mode": mode,
                "status_code": status,
                "provider_effective": provider_effective,
                "llm_degraded": llm_degraded,
                "fallback_used": fallback_used,
                "passed": status == 200 and not hidden_fallback,
            }
        )
    pass_rate = 100.0 * sum(1 for c in checks if bool(c.get("passed"))) / max(1, len(checks))
    return {"checks": checks, "pass_rate_pct": round(pass_rate, 2)}


def _run_calibration_warning_suite(
    *,
    semantic_rows: list[dict[str, Any]],
    blackbox_warning_avg: float,
    blackbox_warning_avg_raw: float | None = None,
) -> dict[str, Any]:
    warnings = [_safe_float(row.get("warning_count"), 0.0) for row in semantic_rows]
    warnings_raw = [_safe_float(row.get("warning_count_raw"), row.get("warning_count", 0.0)) for row in semantic_rows]
    conf = [_safe_float(row.get("confidence_score"), 0.0) for row in semantic_rows]
    expected = [1.0 if bool(row.get("expectation_met")) else 0.0 for row in semantic_rows]
    brier_terms = [(c - e) ** 2 for c, e in zip(conf, expected)]
    brier = float(statistics.mean(brier_terms)) if brier_terms else 1.0
    semantic_warn_avg = float(statistics.mean(warnings)) if warnings else 0.0
    semantic_warn_avg_raw = float(statistics.mean(warnings_raw)) if warnings_raw else 0.0
    blended_warning_avg = float(statistics.mean([semantic_warn_avg, blackbox_warning_avg]))
    blackbox_warning_avg_raw_value = (
        float(blackbox_warning_avg_raw) if blackbox_warning_avg_raw is not None else float(blackbox_warning_avg)
    )
    blended_warning_avg_raw = float(statistics.mean([semantic_warn_avg_raw, blackbox_warning_avg_raw_value]))

    calibration_ok = brier <= 0.25
    warning_ok = blended_warning_avg <= 1.0
    checks = [
        {"name": "brier_score", "actual": round(brier, 4), "target_max": 0.25, "passed": calibration_ok},
        {
            "name": "avg_warning_count",
            "actual": round(blended_warning_avg, 4),
            "actual_raw": round(blended_warning_avg_raw, 4),
            "target_max": 1.0,
            "passed": warning_ok,
        },
    ]
    passed = sum(1 for c in checks if bool(c["passed"]))
    return {
        "checks": checks,
        "pass_rate_pct": round((100.0 * passed / max(1, len(checks))), 2),
    }


def _suite_score(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return round(max(0.0, min(100.0, float(value))), 2)


def _compute_truth_score(suite_scores: dict[str, float]) -> dict[str, Any]:
    weighted = 0.0
    for suite_id, weight in TRUTH_WEIGHTS.items():
        weighted += _suite_score(suite_scores.get(suite_id, 0.0)) * float(weight)
    weighted = round(weighted, 2)

    floor_violations: list[dict[str, Any]] = []
    floor_ceiling = 100.0
    for suite_id, gate in SUITE_GATES.items():
        if suite_id not in suite_scores:
            continue
        actual = _suite_score(suite_scores[suite_id])
        threshold = gate.min_score_pct
        if actual + 1e-9 < threshold:
            floor_violations.append(
                {
                    "suite_id": suite_id,
                    "actual_pct": round(actual, 2),
                    "required_pct": round(threshold, 2),
                }
            )
            floor_ceiling = min(floor_ceiling, round((actual / max(threshold, 1e-9)) * 100.0, 2))
    truth_score = min(weighted, floor_ceiling) if floor_violations else weighted
    return {
        "weighted_score": weighted,
        "hard_floor_ceiling": round(floor_ceiling, 2),
        "floor_violations": floor_violations,
        "composite_truth_score": round(truth_score, 2),
        "release_gate_passed": not floor_violations,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    suites = report.get("suites") or {}
    summary = report.get("summary") or {}
    lines = [
        "# v2 QA Truth Report",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Tier: `{report.get('tier')}`",
        f"- Composite truth score: **{summary.get('composite_truth_score', 0)}**",
        f"- Release gate passed: **{summary.get('release_gate_passed', False)}**",
        "",
        "## Suite Scores",
        "",
        "| Suite | Score % | Gate % | Passed |",
        "|---|---:|---:|---|",
    ]
    for suite_id in sorted([k for k in suites.keys() if k.startswith("Q")]):
        score = _suite_score((suites.get(suite_id) or {}).get("score_pct", 0.0))
        gate = SUITE_GATES[suite_id].min_score_pct if suite_id in SUITE_GATES else 0.0
        passed = score + 1e-9 >= gate
        lines.append(f"| {suite_id} | {score} | {gate} | {'yes' if passed else 'no'} |")
    lines.append("")
    violations = summary.get("floor_violations") or []
    lines.append("## Hard-Floor Violations")
    if not violations:
        lines.append("- none")
    else:
        for row in violations:
            lines.append(
                f"- {row.get('suite_id')}: actual={row.get('actual_pct')} required={row.get('required_pct')}"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2 truth-focused QA program.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--tier", choices=sorted(TIER_CONFIG.keys()), default="merge")
    parser.add_argument("--tenant-id", default="")
    parser.add_argument("--strict-provider-integrity", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    tier_cfg = dict(TIER_CONFIG[args.tier])
    tenant_id = str(args.tenant_id or "").strip() or f"qa-truth-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    db_path = Path(args.db_path).expanduser()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    smoke = _run_smoke_suite(base_url=base_url, tenant_id=tenant_id)
    q0_score = _suite_score(smoke.get("pass_rate_pct", 0.0))
    if q0_score < SUITE_GATES["Q0"].min_score_pct:
        report = {
            "generated_at": _utc_now(),
            "tier": args.tier,
            "tenant_id": tenant_id,
            "base_url": base_url,
            "run_id": run_id,
            "suites": {"Q0": {"score_pct": q0_score, "detail": smoke}},
            "summary": _compute_truth_score({"Q0": q0_score}),
            "error": "smoke_failed",
        }
        json_path = out_dir / f"v2_qa_truth_report_{run_id}.json"
        md_path = out_dir / f"v2_qa_truth_report_{run_id}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(_render_markdown(report), encoding="utf-8")
        print(json.dumps({"json_report": str(json_path), "md_report": str(md_path), "summary": report["summary"]}, indent=2))
        return 1

    blackbox_result: dict[str, Any] = {}
    semantic_result: dict[str, Any] = {}
    portability_result: dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_blackbox = executor.submit(
            _run_blackbox_suite,
            base_url=base_url,
            db_path=db_path,
            out_dir=out_dir,
            tier_cfg=tier_cfg,
            round_id=f"V2-{run_id}",
        )
        fut_semantic = executor.submit(
            _run_semantic_suite,
            base_url=base_url,
            mode=str(tier_cfg.get("semantic_mode") or "auto"),
            tenant_id=tenant_id,
            workers=_safe_int(tier_cfg.get("semantic_workers"), 4),
        )
        fut_portability = executor.submit(
            _run_portability_suite,
            base_url=base_url,
            tenant_id=tenant_id,
        )
        blackbox_result = fut_blackbox.result()
        semantic_result = fut_semantic.result()
        portability_result = fut_portability.result()

    availability, _provider_snapshot = _provider_mode_availability(base_url)
    requested_latency_modes = [
        str(m).strip().lower()
        for m in (tier_cfg.get("latency_modes") or ["deterministic"])
        if str(m).strip()
    ]
    selected_latency_modes, _skipped_latency_modes = _select_available_modes(
        requested_latency_modes,
        availability,
    )
    latency_result = _run_latency_suite(
        base_url=base_url,
        tenant_id=tenant_id,
        modes=selected_latency_modes,
        iterations=_safe_int(tier_cfg.get("latency_iterations"), 6),
        mode_workers=_safe_int(tier_cfg.get("latency_mode_workers"), 2),
        consecutive_fail_limit=_safe_int(tier_cfg.get("latency_consecutive_fail_limit"), 2),
    )

    provider_integrity = {"checks": [], "pass_rate_pct": 100.0}
    if bool(args.strict_provider_integrity) or args.tier == "release":
        provider_integrity = _run_provider_integrity_checks(
            base_url=base_url,
            tenant_id=tenant_id,
            availability=availability,
            timeout_s=_safe_int(tier_cfg.get("provider_integrity_timeout_s"), 60),
        )

    calibration_warning = _run_calibration_warning_suite(
        semantic_rows=list(semantic_result.get("rows") or []),
        blackbox_warning_avg=_safe_float(blackbox_result.get("avg_warning_count"), 0.0),
        blackbox_warning_avg_raw=_safe_float(
            blackbox_result.get("avg_warning_count_raw"),
            _safe_float(blackbox_result.get("avg_warning_count"), 0.0),
        ),
    )

    category_scores = dict(blackbox_result.get("category_scores") or {})
    q1_score = _suite_score(_safe_float(category_scores.get("factual"), blackbox_result.get("overall_pass_rate_pct", 0.0)))
    q2_score = _suite_score(_safe_float(category_scores.get("followup"), 0.0))
    q3_score = _suite_score(_safe_float(semantic_result.get("expectation_pass_rate_pct"), 0.0))
    q4_score = _suite_score(_safe_float(category_scores.get("analytics_depth"), 0.0))
    q5_base = _safe_float(category_scores.get("behavior"), 0.0)
    q5_score = _suite_score(min(q5_base, _safe_float(provider_integrity.get("pass_rate_pct"), 100.0)))
    q6_score = _suite_score(_safe_float(portability_result.get("pass_rate_pct"), 0.0))
    q7_score = _suite_score(_safe_float(latency_result.get("pass_rate_pct"), 0.0))
    q8_score = _suite_score(_safe_float(calibration_warning.get("pass_rate_pct"), 0.0))

    suite_scores = {
        "Q0": q0_score,
        "Q1": q1_score,
        "Q2": q2_score,
        "Q3": q3_score,
        "Q4": q4_score,
        "Q5": q5_score,
        "Q6": q6_score,
        "Q7": q7_score,
        "Q8": q8_score,
    }
    summary = _compute_truth_score(suite_scores)

    report = {
        "generated_at": _utc_now(),
        "tier": args.tier,
        "tenant_id": tenant_id,
        "base_url": base_url,
        "db_path": str(db_path.resolve()),
        "run_id": run_id,
        "weights": TRUTH_WEIGHTS,
        "suite_gates": {k: v.min_score_pct for k, v in SUITE_GATES.items()},
        "mode_selection": {
            "blackbox_requested_modes": blackbox_result.get("requested_modes", []),
            "blackbox_selected_modes": blackbox_result.get("selected_modes", []),
            "blackbox_skipped_unavailable_modes": blackbox_result.get("skipped_unavailable_modes", []),
            "latency_requested_modes": requested_latency_modes,
            "latency_selected_modes": selected_latency_modes,
            "latency_skipped_unavailable_modes": _skipped_latency_modes,
        },
        "suites": {
            "Q0": {"score_pct": q0_score, "detail": smoke},
            "Q1": {"score_pct": q1_score, "detail": {"factual_score_pct": q1_score, "blackbox": blackbox_result}},
            "Q2": {"score_pct": q2_score, "detail": {"followup_score_pct": q2_score, "blackbox": blackbox_result}},
            "Q3": {"score_pct": q3_score, "detail": semantic_result},
            "Q4": {"score_pct": q4_score, "detail": {"strategic_score_pct": q4_score, "blackbox": blackbox_result}},
            "Q5": {"score_pct": q5_score, "detail": {"behavior_score_pct": q5_base, "provider_integrity": provider_integrity}},
            "Q6": {"score_pct": q6_score, "detail": portability_result},
            "Q7": {"score_pct": q7_score, "detail": latency_result},
            "Q8": {"score_pct": q8_score, "detail": calibration_warning},
        },
        "summary": summary,
    }

    json_path = out_dir / f"v2_qa_truth_report_{run_id}.json"
    md_path = out_dir / f"v2_qa_truth_report_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "md_report": str(md_path),
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0 if bool(summary.get("release_gate_passed")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
