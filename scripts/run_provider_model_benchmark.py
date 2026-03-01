"""Black-box provider/model benchmark for dataDa.

Runs source-truth style checks through HTTP APIs against a running service,
across local/OpenAI/Anthropic models, and emits JSON + HTML reports.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import math
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import requests
from requests.exceptions import RequestException, Timeout

from haikugraph.poc.source_truth import DEFAULT_CASES


DEFAULT_LOCAL_PULL = [
    "qwen2.5:7b-instruct",
    "llama3.1:8b",
    "mistral:7b",
    "llama3.2:latest",
]

LOCAL_MODEL_PRIORITY = [
    "qwen2.5:7b-instruct",
    "mistral:7b",
    "llama3.1:8b",
]

OPENAI_MODEL_PRIORITY = [
    "gpt-5.3",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
]

ANTHROPIC_MODEL_PRIORITY = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


@dataclass(frozen=True)
class Profile:
    profile_id: str
    mode: str
    model: str | None


@dataclass
class RunRow:
    profile_id: str
    mode_requested: str
    model_requested: str | None
    http_status: int
    case_id: str
    question: str
    success: bool
    exact_match: bool
    latency_ms: float
    confidence: float
    mode_actual: str
    provider_actual: str
    intent_model_actual: str
    narrator_model_actual: str
    llm_calls: int
    llm_cache_hit_ratio: float
    llm_total_latency_ms: float
    error: str
    answer_excerpt: str
    expected_sql: str
    actual_sql: str


@dataclass
class SkippedProfile:
    profile_id: str
    reason: str


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


def _exec_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple[Any, ...]], str | None]:
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
        return cols, rows, None
    except Exception as exc:
        return [], [], str(exc)


def _api_get(session: requests.Session, base_url: str, path: str) -> requests.Response:
    return session.get(f"{base_url.rstrip('/')}{path}", timeout=20)


def _api_post(
    session: requests.Session,
    base_url: str,
    path: str,
    body: dict[str, Any],
    *,
    timeout: int = 240,
    tenant_id: str = "public",
) -> requests.Response:
    return session.post(
        f"{base_url.rstrip('/')}{path}",
        json=body,
        timeout=timeout,
        headers={"x-datada-role": "admin", "x-datada-tenant-id": tenant_id},
    )


def _as_list(payload: Any) -> list[Any]:
    return payload if isinstance(payload, list) else []


def _parse_csv_models(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _pick_preferred_models(available: list[str], priority: list[str], *, max_models: int = 1) -> list[str]:
    if not available:
        return []
    selected: list[str] = []
    low_map = {name.lower(): name for name in available if name}
    for item in priority:
        key = item.lower()
        if key in low_map and low_map[key] not in selected:
            selected.append(low_map[key])
            if len(selected) >= max(1, int(max_models)):
                return selected
    if selected:
        return selected
    return available[: max(1, int(max_models))]


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pull_local_models(session: requests.Session, base_url: str, models: list[str]) -> list[str]:
    pulled: list[str] = []
    for model in models:
        resp = _api_post(
            session,
            base_url,
            "/api/assistant/models/local/pull",
            {"model": model, "activate_after_download": False},
            timeout=3600,
            tenant_id="public",
        )
        if resp.ok:
            pulled.append(model)
    return pulled


def _build_profiles(
    session: requests.Session,
    base_url: str,
    *,
    include_deterministic: bool,
    include_auto: bool,
    local_models: list[str],
    openai_models: list[str],
    anthropic_models: list[str],
    max_models_per_provider: int,
    all_advertised_models: bool,
) -> tuple[list[Profile], list[SkippedProfile], dict[str, Any]]:
    profiles: list[Profile] = []
    skipped: list[SkippedProfile] = []

    providers_resp = _api_get(session, base_url, "/api/assistant/providers")
    providers_payload = _safe_json(providers_resp)
    checks = (providers_payload.get("checks") or {}) if providers_resp.ok else {}

    if include_deterministic:
        profiles.append(Profile(profile_id="deterministic", mode="deterministic", model=None))
    if include_auto:
        profiles.append(Profile(profile_id="auto", mode="auto", model=None))

    local_available = bool((checks.get("ollama") or {}).get("available"))
    openai_available = bool((checks.get("openai") or {}).get("available"))
    anthropic_available = bool((checks.get("anthropic") or {}).get("available"))

    if local_available:
        local_resp = _api_get(session, base_url, "/api/assistant/models/local")
        local_payload = _safe_json(local_resp)
        options = _as_list(local_payload.get("options")) if local_resp.ok else []
        installed = [
            str(item.get("name") or "")
            for item in options
            if isinstance(item, dict) and item.get("installed") and str(item.get("name") or "").strip()
        ]
        if local_models:
            chosen = local_models
        elif all_advertised_models:
            chosen = installed
        else:
            chosen = _pick_preferred_models(
                installed,
                LOCAL_MODEL_PRIORITY,
                max_models=max_models_per_provider,
            )
        for model in chosen:
            profiles.append(Profile(profile_id=f"local:{model}", mode="local", model=model))
    else:
        skipped.append(SkippedProfile(profile_id="local:*", reason=str((checks.get("ollama") or {}).get("reason") or "unavailable")))

    if openai_available:
        openai_resp = _api_get(session, base_url, "/api/assistant/models/openai")
        openai_payload = _safe_json(openai_resp)
        options = _as_list(openai_payload.get("options")) if openai_resp.ok else []
        available = [str(item.get("name") or "") for item in options if isinstance(item, dict) and str(item.get("name") or "").strip()]
        if openai_models:
            chosen = openai_models
        elif all_advertised_models:
            chosen = available
        else:
            chosen = _pick_preferred_models(
                available,
                OPENAI_MODEL_PRIORITY,
                max_models=max_models_per_provider,
            )
        for model in chosen:
            if model:
                profiles.append(Profile(profile_id=f"openai:{model}", mode="openai", model=model))
    else:
        skipped.append(SkippedProfile(profile_id="openai:*", reason=str((checks.get("openai") or {}).get("reason") or "unavailable")))

    if anthropic_available:
        anthropic_resp = _api_get(session, base_url, "/api/assistant/models/anthropic")
        anthropic_payload = _safe_json(anthropic_resp)
        options = _as_list(anthropic_payload.get("options")) if anthropic_resp.ok else []
        available = [str(item.get("name") or "") for item in options if isinstance(item, dict) and str(item.get("name") or "").strip()]
        if anthropic_models:
            chosen = anthropic_models
        elif all_advertised_models:
            chosen = available
        else:
            chosen = _pick_preferred_models(
                available,
                ANTHROPIC_MODEL_PRIORITY,
                max_models=max_models_per_provider,
            )
        for model in chosen:
            if model:
                profiles.append(Profile(profile_id=f"anthropic:{model}", mode="anthropic", model=model))
    else:
        skipped.append(SkippedProfile(profile_id="anthropic:*", reason=str((checks.get("anthropic") or {}).get("reason") or "unavailable")))

    return profiles, skipped, providers_payload


def _query_payload(profile: Profile, case_id: str, question: str, tenant_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "goal": question,
        "llm_mode": profile.mode,
        "session_id": f"provider-bench-{profile.profile_id.replace(':', '-')}-{case_id}",
        "tenant_id": tenant_id,
        "role": "admin",
    }
    if profile.mode == "local" and profile.model:
        payload["local_model"] = profile.model
        payload["local_narrator_model"] = profile.model
    elif profile.mode == "openai" and profile.model:
        payload["openai_model"] = profile.model
        payload["openai_narrator_model"] = profile.model
    elif profile.mode == "anthropic" and profile.model:
        payload["anthropic_model"] = profile.model
        payload["anthropic_narrator_model"] = profile.model
    return payload


def _mode_timeout_seconds(profile: Profile, request_timeout: int) -> int:
    base = max(30, int(request_timeout))
    if profile.mode in {"deterministic", "auto"}:
        return min(base, 120)
    if profile.mode == "openai":
        return min(max(base, 120), 180)
    if profile.mode == "anthropic":
        return min(max(base, 120), 180)
    if profile.mode == "local":
        return min(max(base, 120), 180)
    return base


def _should_retry(status_code: int, error: Exception | None) -> bool:
    if isinstance(error, Timeout):
        return True
    if isinstance(error, RequestException):
        return True
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _run_profile_case_request(
    *,
    base_url: str,
    profile: Profile,
    case_id: str,
    question: str,
    tenant_id: str,
    timeout: int = 360,
    retries: int = 1,
    retry_backoff_seconds: float = 0.8,
) -> tuple[str, int, dict[str, Any], float]:
    request_body = _query_payload(profile, case_id, question, tenant_id)
    started = time.perf_counter()
    status_code = 599
    payload: dict[str, Any] = {"error": "request_failed"}
    with requests.Session() as per_call_session:
        for attempt in range(max(1, int(retries) + 1)):
            err: Exception | None = None
            try:
                response = _api_post(
                    per_call_session,
                    base_url,
                    "/api/assistant/query",
                    request_body,
                    timeout=timeout,
                    tenant_id=tenant_id,
                )
                status_code = int(response.status_code)
                payload = _safe_json(response)
            except Exception as exc:  # noqa: BLE001
                status_code = 599
                payload = {"error": str(exc)}
                err = exc

            if attempt >= int(retries) or not _should_retry(status_code, err):
                break
            sleep_s = max(0.0, float(retry_backoff_seconds)) * (2**attempt)
            if sleep_s > 0:
                time.sleep(sleep_s)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return case_id, status_code, payload, latency_ms


def run_benchmark(
    *,
    base_url: str,
    db_path: Path,
    out_dir: Path,
    include_deterministic: bool,
    include_auto: bool,
    local_models: list[str],
    openai_models: list[str],
    anthropic_models: list[str],
    pull_local_models: list[str],
    max_cases: int,
    max_models_per_provider: int,
    all_advertised_models: bool,
    workers: int,
    local_workers: int,
    request_timeout: int,
    request_retries: int,
    retry_backoff_seconds: float,
    tenant_id: str,
) -> tuple[Path, dict[str, Any]]:
    session = requests.Session()

    health_resp = _api_get(session, base_url, "/api/assistant/health")
    if not health_resp.ok:
        raise RuntimeError(f"Service health check failed: {health_resp.status_code} {health_resp.text[:300]}")

    pulled: list[str] = []
    if pull_local_models:
        pulled = _pull_local_models(session, base_url, pull_local_models)

    profiles, skipped, providers_payload = _build_profiles(
        session,
        base_url,
        include_deterministic=include_deterministic,
        include_auto=include_auto,
        local_models=local_models,
        openai_models=openai_models,
        anthropic_models=anthropic_models,
        max_models_per_provider=max(1, int(max_models_per_provider)),
        all_advertised_models=bool(all_advertised_models),
    )
    if not profiles:
        raise RuntimeError("No runnable profiles found. Check provider availability and model lists.")

    source_db = db_path.expanduser()
    with tempfile.NamedTemporaryFile(prefix="datada_bench_", suffix=".duckdb", delete=False) as tmpf:
        shadow_db = Path(tmpf.name)
    shutil.copy2(source_db, shadow_db)

    cases = DEFAULT_CASES[: max(1, min(len(DEFAULT_CASES), int(max_cases)))]

    rows: list[RunRow] = []
    conn = duckdb.connect(str(shadow_db), read_only=False)
    try:
        expected_cache: dict[str, dict[str, Any]] = {}
        for case in cases:
            cols, rows_sql, err = _exec_sql(conn, case.expected_sql)
            if err:
                raise RuntimeError(f"Expected SQL failed for {case.case_id}: {err}")
            expected_cache[case.case_id] = {"cols": cols, "rows": rows_sql}

        for profile in profiles:
            profile_requests: list[tuple[str, int, dict[str, Any], float]] = []
            worker_count = max(1, int(local_workers)) if profile.mode == "local" else max(1, int(workers))
            profile_timeout = _mode_timeout_seconds(profile, request_timeout)
            if worker_count <= 1:
                for case in cases:
                    profile_requests.append(
                        _run_profile_case_request(
                            base_url=base_url,
                            profile=profile,
                            case_id=case.case_id,
                            question=case.question,
                            tenant_id=tenant_id,
                            timeout=profile_timeout,
                            retries=max(0, int(request_retries)),
                            retry_backoff_seconds=max(0.0, float(retry_backoff_seconds)),
                        )
                    )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            _run_profile_case_request,
                            base_url=base_url,
                            profile=profile,
                            case_id=case.case_id,
                            question=case.question,
                            tenant_id=tenant_id,
                            timeout=profile_timeout,
                            retries=max(0, int(request_retries)),
                            retry_backoff_seconds=max(0.0, float(retry_backoff_seconds)),
                        ): idx
                        for idx, case in enumerate(cases)
                    }
                    indexed_results: dict[int, tuple[str, int, dict[str, Any], float]] = {}
                    for future in concurrent.futures.as_completed(future_map):
                        idx = future_map[future]
                        indexed_results[idx] = future.result()
                    for idx in range(len(cases)):
                        profile_requests.append(indexed_results[idx])

            payload_by_case: dict[str, tuple[int, dict[str, Any], float]] = {
                cid: (status, payload, latency_ms)
                for cid, status, payload, latency_ms in profile_requests
            }
            for case in cases:
                status_code, payload, latency_ms = payload_by_case.get(case.case_id, (599, {}, 0.0))
                runtime = payload.get("runtime") or {}
                llm_summary = runtime.get("llm_metrics_summary") or {}
                actual_sql = str(payload.get("sql") or "")
                answer_excerpt = str(payload.get("answer_markdown") or payload.get("detail") or "")[:280]

                _, actual_rows, sql_error = _exec_sql(conn, actual_sql) if actual_sql else ([], [], "missing sql")
                expected_rows = expected_cache[case.case_id]["rows"]
                exact_match = bool(payload.get("success")) and not sql_error and _rows_match(actual_rows, expected_rows)
                error = ""
                if status_code != 200:
                    error = str(payload.get("detail") or payload.get("error") or f"HTTP {status_code}")
                elif payload.get("error"):
                    error = str(payload.get("error"))
                elif sql_error:
                    error = sql_error
                elif payload.get("success") is False:
                    error = answer_excerpt or "query returned success=false"

                rows.append(
                    RunRow(
                        profile_id=profile.profile_id,
                        mode_requested=profile.mode,
                        model_requested=profile.model,
                        http_status=int(status_code),
                        case_id=case.case_id,
                        question=case.question,
                        success=bool(payload.get("success")),
                        exact_match=bool(exact_match),
                        latency_ms=latency_ms,
                        confidence=float(payload.get("confidence_score") or 0.0),
                        mode_actual=str(runtime.get("mode") or ""),
                        provider_actual=str(runtime.get("provider") or ""),
                        intent_model_actual=str(runtime.get("intent_model") or ""),
                        narrator_model_actual=str(runtime.get("narrator_model") or ""),
                        llm_calls=int(llm_summary.get("calls") or 0),
                        llm_cache_hit_ratio=float(llm_summary.get("cache_hit_ratio") or 0.0),
                        llm_total_latency_ms=float(llm_summary.get("total_latency_ms") or 0.0),
                        error=error,
                        answer_excerpt=answer_excerpt,
                        expected_sql=case.expected_sql,
                        actual_sql=actual_sql,
                    )
                )
    finally:
        conn.close()
        shadow_db.unlink(missing_ok=True)

    summary: dict[str, Any] = {}
    for profile in profiles:
        p_rows = [r for r in rows if r.profile_id == profile.profile_id]
        latencies = [r.latency_ms for r in p_rows]
        matches = [r for r in p_rows if r.exact_match]
        successes = [r for r in p_rows if r.success]
        summary[profile.profile_id] = {
            "mode": profile.mode,
            "requested_model": profile.model,
            "cases": len(p_rows),
            "exact_matches": len(matches),
            "factual_accuracy_pct": round((len(matches) / len(p_rows)) * 100, 2) if p_rows else 0.0,
            "success_pct": round((len(successes) / len(p_rows)) * 100, 2) if p_rows else 0.0,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[-1], 2) if len(latencies) >= 2 else (latencies[0] if latencies else 0.0),
            "avg_confidence": round(statistics.mean([r.confidence for r in p_rows]), 4) if p_rows else 0.0,
            "avg_llm_calls": round(statistics.mean([r.llm_calls for r in p_rows]), 2) if p_rows else 0.0,
            "avg_llm_cache_hit_ratio": round(statistics.mean([r.llm_cache_hit_ratio for r in p_rows]), 4) if p_rows else 0.0,
            "avg_llm_total_latency_ms": round(statistics.mean([r.llm_total_latency_ms for r in p_rows]), 2) if p_rows else 0.0,
            "errors": sum(1 for r in p_rows if r.error),
        }

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_url": base_url,
        "db_path": str(db_path),
        "tenant_id": tenant_id,
        "pulled_local_models": pulled,
        "providers": providers_payload,
        "cases_evaluated": [case.case_id for case in cases],
        "profiles": [p.__dict__ for p in profiles],
        "skipped_profiles": [s.__dict__ for s in skipped],
        "summary": summary,
        "runs": [r.__dict__ for r in rows],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"provider_model_benchmark_{ts}.json"
    html_path = out_dir / f"provider_model_benchmark_{ts}.html"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    html_path.write_text(build_html(report))
    return html_path, report


def _esc(value: Any) -> str:
    return html.escape(str(value))


def build_html(report: dict[str, Any]) -> str:
    summary_rows = "".join(
        (
            f"<tr><td>{_esc(profile)}</td>"
            f"<td>{_esc(vals.get('mode'))}</td>"
            f"<td>{_esc(vals.get('requested_model') or '')}</td>"
            f"<td>{_esc(vals.get('exact_matches'))}/{_esc(vals.get('cases'))}</td>"
            f"<td>{_esc(vals.get('factual_accuracy_pct'))}%</td>"
            f"<td>{_esc(vals.get('success_pct'))}%</td>"
            f"<td>{_esc(vals.get('avg_latency_ms'))}</td>"
            f"<td>{_esc(vals.get('p95_latency_ms'))}</td>"
            f"<td>{_esc(vals.get('avg_confidence'))}</td>"
            f"<td>{_esc(vals.get('avg_llm_calls'))}</td>"
            f"<td>{_esc(vals.get('avg_llm_cache_hit_ratio'))}</td>"
            f"<td>{_esc(vals.get('avg_llm_total_latency_ms'))}</td>"
            f"<td>{_esc(vals.get('errors'))}</td></tr>"
        )
        for profile, vals in report.get("summary", {}).items()
    )

    detail_rows = "".join(
        (
            f"<tr><td>{_esc(r.get('profile_id'))}</td>"
            f"<td>{_esc(r.get('case_id'))}</td>"
            f"<td>{_esc(r.get('http_status'))}</td>"
            f"<td>{_esc(r.get('mode_requested'))}</td>"
            f"<td>{_esc(r.get('mode_actual'))}</td>"
            f"<td>{_esc(r.get('provider_actual'))}</td>"
            f"<td>{_esc(r.get('intent_model_actual'))}</td>"
            f"<td>{'YES' if r.get('success') else 'NO'}</td>"
            f"<td>{'YES' if r.get('exact_match') else 'NO'}</td>"
            f"<td>{_esc(r.get('latency_ms'))}</td>"
            f"<td>{_esc(r.get('confidence'))}</td>"
            f"<td>{_esc(r.get('llm_calls'))}</td>"
            f"<td>{_esc(r.get('llm_cache_hit_ratio'))}</td>"
            f"<td>{_esc(r.get('llm_total_latency_ms'))}</td>"
            f"<td>{_esc(r.get('error') or '')}</td>"
            f"<td>{_esc(r.get('answer_excerpt') or '')}</td></tr>"
        )
        for r in report.get("runs", [])
    )

    skipped = report.get("skipped_profiles") or []
    skipped_html = "".join(
        f"<li><code>{_esc(item.get('profile_id'))}</code> — {_esc(item.get('reason'))}</li>"
        for item in skipped
    ) or "<li>None</li>"

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>dataDa Provider Model Benchmark</title>
  <style>
    body {{ font-family: 'Avenir Next', 'Segoe UI', sans-serif; background: #0a1020; color: #eef2ff; margin: 0; padding: 20px; }}
    .card {{ background: #101a33; border: 1px solid #223865; border-radius: 14px; padding: 14px; margin-bottom: 14px; }}
    .muted {{ color: #9db0d8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    th, td {{ border-bottom: 1px solid #223865; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #14203d; position: sticky; top: 0; }}
    .wrap {{ overflow: auto; max-height: 560px; border: 1px solid #223865; border-radius: 10px; }}
    code {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.76rem; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>dataDa Provider/Model Benchmark</h1>
    <div class=\"muted\">Generated: {_esc(report.get('generated_at'))}</div>
    <div class=\"muted\">Service: {_esc(report.get('base_url'))}</div>
    <div class=\"muted\">DB: {_esc(report.get('db_path'))}</div>
    <div class=\"muted\">Cases: {_esc(', '.join(report.get('cases_evaluated') or []))}</div>
    <div class=\"muted\">Pulled local models: {_esc(', '.join(report.get('pulled_local_models') or []))}</div>
  </div>

  <div class=\"card\">
    <h2>Skipped Profiles</h2>
    <ul>{skipped_html}</ul>
  </div>

  <div class=\"card\">
    <h2>Summary</h2>
    <div class=\"wrap\">
      <table>
        <thead>
          <tr>
            <th>Profile</th><th>Mode</th><th>Requested Model</th><th>Exact</th><th>Factual %</th><th>Success %</th><th>Avg Latency (ms)</th><th>P95 (ms)</th><th>Avg Confidence</th><th>Avg LLM Calls</th><th>Avg Cache Hit Ratio</th><th>Avg LLM Latency (ms)</th><th>Errors</th>
          </tr>
        </thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>

  <div class=\"card\">
    <h2>Per-Case Details</h2>
    <div class=\"wrap\">
      <table>
        <thead>
          <tr>
            <th>Profile</th><th>Case</th><th>HTTP</th><th>Mode Req</th><th>Mode Actual</th><th>Provider</th><th>Intent Model</th><th>Success</th><th>Exact</th><th>Latency (ms)</th><th>Confidence</th><th>LLM Calls</th><th>Cache Hit Ratio</th><th>LLM Latency (ms)</th><th>Error</th><th>Answer Excerpt</th>
          </tr>
        </thead>
        <tbody>{detail_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provider/model benchmark against a running dataDa API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--db-path", default="data/haikugraph.db")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--include-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-models", default="", help="Comma-separated local models. Default: all installed models.")
    parser.add_argument("--openai-models", default="", help="Comma-separated OpenAI models. Default: all advertised by API.")
    parser.add_argument("--anthropic-models", default="", help="Comma-separated Anthropic models. Default: all advertised by API.")
    parser.add_argument(
        "--pull-local-models",
        default="",
        help="Comma-separated local models to download before benchmark. Example: qwen2.5:7b-instruct,llama3.1:8b",
    )
    parser.add_argument(
        "--pull-default-local-set",
        action="store_true",
        help="Pull a default balanced local set before benchmark.",
    )
    parser.add_argument("--max-cases", type=int, default=len(DEFAULT_CASES))
    parser.add_argument("--max-models-per-provider", type=int, default=1, help="When model lists are omitted, run top-N preferred models per provider")
    parser.add_argument("--all-advertised-models", action=argparse.BooleanOptionalAction, default=False, help="Run every model advertised by each provider")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers per profile")
    parser.add_argument("--local-workers", type=int, default=1, help="Parallel workers for local profile")
    parser.add_argument("--request-timeout", type=int, default=120, help="Base per-request timeout in seconds")
    parser.add_argument("--request-retries", type=int, default=1, help="Retries for transient request failures")
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.8, help="Base retry backoff in seconds")
    parser.add_argument("--tenant-id", default="", help="Tenant id for this benchmark run (default: auto-generated)")

    args = parser.parse_args()

    pull_models = _parse_csv_models(args.pull_local_models)
    if args.pull_default_local_set:
        pull_models = list(dict.fromkeys([*pull_models, *DEFAULT_LOCAL_PULL]))

    tenant_id = str(args.tenant_id).strip() or f"bench-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    html_path, report = run_benchmark(
        base_url=args.base_url,
        db_path=Path(args.db_path),
        out_dir=Path(args.out_dir),
        include_deterministic=bool(args.include_deterministic),
        include_auto=bool(args.include_auto),
        local_models=_parse_csv_models(args.local_models),
        openai_models=_parse_csv_models(args.openai_models),
        anthropic_models=_parse_csv_models(args.anthropic_models),
        pull_local_models=pull_models,
        max_cases=int(args.max_cases),
        max_models_per_provider=max(1, int(args.max_models_per_provider)),
        all_advertised_models=bool(args.all_advertised_models),
        workers=max(1, int(args.workers)),
        local_workers=max(1, int(args.local_workers)),
        request_timeout=max(30, int(args.request_timeout)),
        request_retries=max(0, int(args.request_retries)),
        retry_backoff_seconds=max(0.0, float(args.retry_backoff_seconds)),
        tenant_id=tenant_id,
    )

    print(f"Provider/model benchmark complete. Report: {html_path}")
    print("Summary:")
    for profile, vals in report.get("summary", {}).items():
        print(
            f"- {profile}: factual={vals.get('factual_accuracy_pct')}% "
            f"success={vals.get('success_pct')}% avg_latency_ms={vals.get('avg_latency_ms')} "
            f"avg_llm_calls={vals.get('avg_llm_calls')} "
            f"avg_cache_hit_ratio={vals.get('avg_llm_cache_hit_ratio')}"
        )


if __name__ == "__main__":
    main()
