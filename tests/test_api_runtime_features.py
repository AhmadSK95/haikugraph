"""Tests for runtime controls: async jobs, trust dashboard, and rollback paths."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest
from fastapi.testclient import TestClient

import haikugraph.api.server as server_module
import haikugraph.v2.orchestrator as v2_orchestrator_module
from haikugraph.api.server import create_app
from haikugraph.io.document_ingest import ingest_documents_to_duckdb
from haikugraph.poc import RuntimeSelection


@pytest.fixture
def seed_db():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = Path(f.name)
    db_path.unlink()

    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE test_1_1_merged (
            transaction_id VARCHAR,
            customer_id VARCHAR,
            payee_id VARCHAR,
            platform_name VARCHAR,
            state VARCHAR,
            txn_flow VARCHAR,
            payment_status VARCHAR,
            mt103_created_at VARCHAR,
            created_at VARCHAR,
            updated_at VARCHAR,
            payment_amount DOUBLE,
            deal_details_amount DOUBLE,
            amount_collected DOUBLE,
            refund_refund_id VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO test_1_1_merged VALUES
        ('t1', 'c1', 'p1', 'B2C-APP', 'NY', 'flow_a', 'completed', '2025-12-03', '2025-12-03', '2025-12-03', 120.0, 120.0, 120.0, NULL),
        ('t2', 'c2', 'p2', 'B2C-WEB', 'CA', 'flow_b', 'completed', NULL, '2025-12-04', '2025-12-04', 90.0, 90.0, 90.0, 'r1')
        """
    )
    conn.execute(
        """
        CREATE TABLE test_3_1 (
            quote_id VARCHAR,
            customer_id VARCHAR,
            source_currency VARCHAR,
            destination_currency VARCHAR,
            exchange_rate DOUBLE,
            total_amount_to_be_paid DOUBLE,
            total_additional_charges DOUBLE,
            forex_markup DOUBLE,
            created_at VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO test_3_1 VALUES
        ('q1', 'c1', 'USD', 'INR', 83.2, 1000.0, 12.0, 5.0, '2025-12-05'),
        ('q2', 'c2', 'EUR', 'USD', 1.1, 700.0, 4.0, 2.0, '2025-12-07')
        """
    )
    conn.execute(
        """
        CREATE TABLE test_4_1 (
            payee_id VARCHAR,
            customer_id VARCHAR,
            is_university BOOLEAN,
            type VARCHAR,
            status VARCHAR,
            created_at VARCHAR,
            address_country VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO test_4_1 VALUES
        ('p1', 'c1', TRUE, 'education', 'active', '2025-01-02', 'US'),
        ('p2', 'c2', FALSE, 'retail', 'active', '2025-01-03', 'US')
        """
    )
    conn.execute(
        """
        CREATE TABLE test_5_1 (
            deal_id VARCHAR,
            quote_id VARCHAR,
            booked_amount DOUBLE,
            rate DOUBLE,
            deal_type VARCHAR,
            customer_id VARCHAR,
            payee_id VARCHAR,
            created_at VARCHAR,
            updated_at VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO test_5_1 VALUES
        ('d1', 'q1', 220.0, 1.02, 'spot', 'c1', 'p1', '2025-11-05', '2025-11-06'),
        ('d2', 'q2', 320.0, 1.03, 'forward', 'c2', 'p2', '2025-11-07', '2025-11-08')
        """
    )
    conn.close()

    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def client(seed_db):
    app = create_app(db_path=seed_db)
    return TestClient(app)


def test_trust_dashboard_updates_after_queries(client):
    for q in [
        "How many transactions are there?",
        "What is total amount in December 2025?",
    ]:
        resp = client.post("/api/assistant/query", json={"goal": q, "llm_mode": "deterministic"})
        assert resp.status_code == 200
        assert resp.json()["runtime"]["tenant_id"] == "public"

    trust = client.get("/api/assistant/trust/dashboard", params={"tenant_id": "public", "hours": 24})
    assert trust.status_code == 200
    payload = trust.json()
    assert payload["runs"] >= 2
    assert payload["window_hours"] == 24
    assert isinstance(payload["by_mode"], list)


def test_query_response_cache_sets_cache_hit_flag(client):
    first = client.post(
        "/api/assistant/query",
        json={
            "goal": "How many transactions are there?",
            "llm_mode": "deterministic",
            "session_id": "cache-run-a",
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["runtime"]["response_cache_hit"] is False

    second = client.post(
        "/api/assistant/query",
        json={
            "goal": "How many transactions are there?",
            "llm_mode": "deterministic",
            "session_id": "cache-run-b",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["runtime"]["response_cache_hit"] is True


def test_capability_scoreboard_endpoint_returns_live_tracker_counts(client):
    resp = client.get("/api/assistant/capability/scoreboard")
    assert resp.status_code == 200
    payload = resp.json()
    counts = payload["counts"]

    assert counts["total"] == 45
    assert counts["done"] + counts["partial"] + counts["gap"] == counts["total"]

    strict_expected = round((counts["done"] / counts["total"]) * 100.0, 2)
    reality_expected = round(((counts["done"] + (0.5 * counts["partial"])) / counts["total"]) * 100.0, 2)
    assert counts["np_strict"] == strict_expected
    assert counts["np_reality"] == reality_expected

    capabilities = payload.get("capabilities") or []
    remaining = payload.get("remaining") or []
    assert len(capabilities) == counts["total"]
    assert len(remaining) == counts["partial"] + counts["gap"]
    assert str(payload.get("tracker_path") or "").endswith("PRODUCT_GAP_TRACKER.md")

    capability_ids = {str(row.get("capability_id") or "") for row in capabilities}
    assert "A01" in capability_ids
    assert "T20" in capability_ids


def test_quality_latest_endpoint_returns_truth_metadata(client):
    resp = client.get("/api/assistant/quality/latest")
    assert resp.status_code == 200
    payload = resp.json()
    assert "generated_at_epoch_ms" in payload
    assert isinstance(payload.get("latest_runs"), list)
    # truth score can be null if no reports exist yet, but field must exist.
    assert "composite_truth_score" in payload


def test_quality_run_detail_endpoint_returns_payload_for_latest_run(client):
    latest = client.get("/api/assistant/quality/latest")
    assert latest.status_code == 200
    rows = latest.json().get("latest_runs") or []
    if not rows:
        pytest.skip("No quality run artifacts in reports directory for this test environment.")
    run_id = str(rows[0].get("run_id") or "")
    detail = client.get(f"/api/assistant/quality/runs/{run_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload.get("run_id") == run_id
    assert str(payload.get("path") or "").endswith(".json")
    assert isinstance(payload.get("payload"), dict)


def test_dataset_profile_endpoint_returns_semantic_summary(client):
    resp = client.post(
        "/api/assistant/datasets/profile",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
        json={"db_connection_id": "default"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("db_connection_id") == "default"
    assert isinstance(payload.get("dataset_signature"), str)
    assert isinstance(payload.get("schema_signature"), str)
    assert int(payload.get("table_count") or 0) >= 1
    assert isinstance(payload.get("semantic_cache_hit"), bool)
    assert isinstance(payload.get("schema_drift_detected"), bool)
    assert isinstance(payload.get("profile"), dict)


def test_dataset_profile_endpoint_uses_semantic_cache_on_repeat(client):
    headers = {"x-datada-role": "viewer", "x-datada-tenant-id": "public"}
    body = {"db_connection_id": "default"}

    first = client.post("/api/assistant/datasets/profile", headers=headers, json=body)
    assert first.status_code == 200
    second = client.post("/api/assistant/datasets/profile", headers=headers, json=body)
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()
    assert isinstance(first_payload.get("dataset_signature"), str)
    assert second_payload.get("dataset_signature") == first_payload.get("dataset_signature")
    assert bool(second_payload.get("semantic_cache_hit")) is True


def test_runtime_stage_slo_endpoint_returns_budget_and_observed_maps(client):
    resp = client.get(
        "/api/assistant/runtime/stage-slo",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
        params={"hours": 24},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "generated_at_epoch_ms" in payload
    assert isinstance(payload.get("stage_budget_ms"), dict)
    assert isinstance(payload.get("observed_p95_ms"), dict)
    # Core stage budgets should always be present.
    for key in ["semantic_profiler", "intent_engine", "planner", "query_compiler", "executor_delegate"]:
        assert key in payload["stage_budget_ms"]


def test_runtime_cutover_readiness_endpoint_returns_governance_state(client):
    resp = client.get(
        "/api/assistant/runtime/cutover/readiness",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "generated_at_epoch_ms" in payload
    assert payload.get("default_runtime_version") == "v2"
    assert isinstance(payload.get("canary_ready"), bool)
    assert isinstance(payload.get("release_gate_passed"), bool)
    assert isinstance(payload.get("artifacts"), list)
    if payload.get("artifacts"):
        first = payload["artifacts"][0]
        assert "name" in first
        assert "path" in first
        assert "exists" in first


def test_runtime_readiness_alias_endpoint_returns_governance_state(client):
    resp = client.get(
        "/api/assistant/runtime/readiness",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "generated_at_epoch_ms" in payload
    assert payload.get("default_runtime_version") == "v2"
    assert isinstance(payload.get("canary_ready"), bool)
    assert isinstance(payload.get("release_gate_passed"), bool)
    assert isinstance(payload.get("artifacts"), list)


def test_runtime_version_decommission_ignores_legacy_env_values(client, monkeypatch):
    monkeypatch.setenv("HG_RUNTIME_VERSION", "v1")
    v1_like = client.get(
        "/api/assistant/runtime/readiness",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
    )
    assert v1_like.status_code == 200
    assert v1_like.json().get("default_runtime_version") == "v2"

    monkeypatch.setenv("HG_RUNTIME_VERSION", "shadow")
    shadow_like = client.get(
        "/api/assistant/runtime/readiness",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
    )
    assert shadow_like.status_code == 200
    assert shadow_like.json().get("default_runtime_version") == "v2"


def test_ui_shell_serves_modular_assets(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    assert "/ui/assets/ui.css" in html
    assert "/ui/assets/ui.js" in html

    css = client.get("/ui/assets/ui.css")
    assert css.status_code == 200
    assert "text/css" in str(css.headers.get("content-type", "")).lower()
    assert ":root" in css.text

    js = client.get("/ui/assets/ui.js")
    assert js.status_code == 200
    assert "javascript" in str(js.headers.get("content-type", "")).lower()
    assert "const apiClient" in js.text


def test_query_response_includes_v2_additive_fields(client):
    response = client.post(
        "/api/assistant/query",
        json={
            "goal": "How many transactions are there?",
            "llm_mode": "deterministic",
            "session_id": "v2-additive-fields",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    for key in [
        "analysis_version",
        "slice_signature",
        "quality_flags",
        "assumptions",
        "truth_score",
        "stage_timings_ms",
        "provider_effective",
        "fallback_used",
        "certainty_tags",
        "decision_memo",
        "grain_signature",
        "denominator_semantics",
    ]:
        assert key in payload
    runtime = payload.get("runtime") or {}
    assert "dataset_signature" in runtime
    assert "schema_signature" in runtime
    assert "semantic_cache_hit" in runtime
    assert "stage_slo_breaches" in runtime


def test_query_response_includes_advanced_analytics_packs(client, monkeypatch):
    monkeypatch.setenv("HG_ADVANCED_FORECAST_ENABLED", "0")
    response = client.post(
        "/api/assistant/query",
        json={
            "goal": "Transaction count by platform_name",
            "llm_mode": "deterministic",
            "session_id": "advanced-pack-smoke",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    advanced = (payload.get("stats_analysis") or {}).get("advanced_packs", {})
    assert isinstance(advanced, dict)
    packs = advanced.get("packs", {})
    assert isinstance(packs, dict)
    assert "variance" in packs
    assert "scenario" in packs
    assert "forecast" in packs


def test_query_response_includes_action_recommendations_block(client):
    response = client.post(
        "/api/assistant/query",
        json={
            "goal": "Transaction count by platform_name",
            "llm_mode": "deterministic",
            "session_id": "recommendation-block-smoke",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    dq = payload.get("data_quality") or {}
    recommendations = dq.get("recommendations") or []
    assert isinstance(recommendations, list)
    if recommendations:
        rec = recommendations[0]
        assert "action" in rec
        assert "expected_impact" in rec
        assert "risk" in rec
    answer = str(payload.get("answer_markdown") or "").lower()
    assert "recommended actions" in answer


def test_query_response_includes_root_cause_ranked_drivers(client):
    response = client.post(
        "/api/assistant/query",
        json={
            "goal": "What is the root cause of transaction count by platform_name?",
            "llm_mode": "deterministic",
            "session_id": "root-cause-smoke",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    root_cause = ((payload.get("data_quality") or {}).get("root_cause") or {})
    ranked = root_cause.get("ranked_drivers") or []
    assert isinstance(ranked, list)
    if ranked:
        first = ranked[0]
        assert "rank" in first
        assert "driver" in first
        assert "evidence_score" in first
        assert "caveat" in first
        assert "root-cause hypotheses" in str(payload.get("answer_markdown") or "").lower()


def test_scenario_set_persistence_and_query_replay(client):
    create = client.post(
        "/api/assistant/scenarios",
        headers={"x-datada-role": "analyst", "x-datada-tenant-id": "public"},
        json={
            "db_connection_id": "default",
            "name": "fx downside",
            "assumptions": ["conversion rate down 8%", "fees up 2%"],
            "status": "active",
        },
    )
    assert create.status_code == 200
    created = create.json()["scenario_set"]
    scenario_set_id = created["scenario_set_id"]
    assert scenario_set_id

    listed = client.get(
        "/api/assistant/scenarios",
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
        params={"db_connection_id": "default"},
    )
    assert listed.status_code == 200
    rows = listed.json().get("scenario_sets") or []
    assert any(str(row.get("scenario_set_id") or "") == scenario_set_id for row in rows)

    replay = client.post(
        "/api/assistant/query",
        json={
            "goal": "Scenario analysis for quote value trend",
            "llm_mode": "deterministic",
            "scenario_set_id": scenario_set_id,
            "session_id": "scenario-replay",
        },
    )
    assert replay.status_code == 200
    payload = replay.json()
    runtime = payload.get("runtime") or {}
    assert runtime.get("scenario_set_id") == scenario_set_id
    dq_scenario = ((payload.get("data_quality") or {}).get("scenario") or {})
    assert dq_scenario.get("assumption_set_id") == scenario_set_id
    assert int(dq_scenario.get("assumption_count") or 0) >= 1


def test_trust_dashboard_includes_parity_summary(client):
    modes = ["deterministic"]
    providers = client.get("/api/assistant/providers").json().get("checks") or {}
    if any(bool((providers.get(name) or {}).get("available")) for name in ("openai", "anthropic", "ollama")):
        modes.append("auto")

    for mode in modes:
        response = client.post(
            "/api/assistant/query",
            json={
                "goal": "How many transactions are there?",
                "llm_mode": mode,
                "session_id": f"parity-{mode}",
            },
        )
        assert response.status_code == 200
    trust = client.get("/api/assistant/trust/dashboard", params={"tenant_id": "public", "hours": 24})
    assert trust.status_code == 200
    payload = trust.json()
    parity = payload.get("parity_summary") or {}
    assert isinstance(parity, dict)
    assert "mode_deltas" in parity


def test_async_query_job_completes(client):
    queued = client.post(
        "/api/assistant/query/async",
        json={
            "goal": "How many customers do we have?",
            "llm_mode": "deterministic",
            "session_id": "async-smoke",
        },
    )
    assert queued.status_code == 200
    job_id = queued.json()["job_id"]

    last = None
    for _ in range(40):
        status = client.get(f"/api/assistant/query/async/{job_id}")
        assert status.status_code == 200
        last = status.json()
        if last["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert last is not None
    assert last["status"] == "completed"
    assert last["response"] is not None
    assert "answer_markdown" in last["response"]


def test_async_query_backpressure_limit_returns_429(seed_db, monkeypatch):
    monkeypatch.setenv("HG_ASYNC_MAX_INFLIGHT", "1")
    monkeypatch.setenv("HG_ASYNC_MAX_INFLIGHT_PER_TENANT", "1")
    app = create_app(db_path=seed_db)
    app.state.runtime_store.create_async_job(
        tenant_id="public",
        connection_id="default",
        session_id="existing-load",
        request_payload={"goal": "preloaded"},
    )
    with TestClient(app) as local_client:
        queued = local_client.post(
            "/api/assistant/query/async",
            json={
                "goal": "How many transactions?",
                "llm_mode": "deterministic",
                "session_id": "blocked-by-load",
            },
        )
    assert queued.status_code == 429
    detail = str(queued.json().get("detail") or "")
    assert "capacity" in detail.lower() or "limit" in detail.lower()


def test_correction_toggle_and_rollback(client):
    feedback = client.post(
        "/api/assistant/feedback",
        json={
            "issue": "rollback test issue",
            "keyword": "forex",
            "target_table": "datada_mart_quotes",
            "target_metric": "forex_markup_revenue",
        },
    )
    assert feedback.status_code == 200
    correction_id = feedback.json()["correction_id"]
    assert correction_id

    disable = client.post(
        "/api/assistant/corrections/toggle",
        json={"db_connection_id": "default", "correction_id": correction_id, "enabled": False},
    )
    assert disable.status_code == 200
    assert disable.json()["enabled"] is False

    rollback = client.post(
        "/api/assistant/corrections/rollback",
        json={"db_connection_id": "default", "correction_id": correction_id},
    )
    assert rollback.status_code == 200
    assert rollback.json()["enabled"] is True


def test_business_rules_list_and_update(client):
    create = client.post(
        "/api/assistant/rules",
        headers={"x-datada-role": "admin", "x-datada-tenant-id": "public"},
        json={
            "db_connection_id": "default",
            "domain": "quotes",
            "name": "fx markup routing",
            "rule_type": "plan_override",
            "triggers": ["forex markup", "fx charge"],
            "action_payload": {
                "target_table": "datada_mart_quotes",
                "target_metric": "forex_markup_revenue",
                "target_dimensions": ["__month__"],
            },
            "notes": "seed rule",
            "priority": 2.0,
            "status": "active",
        },
    )
    assert create.status_code == 200
    rule_id = create.json()["rule_id"]
    assert rule_id

    listed = client.get(
        "/api/assistant/rules",
        headers={"x-datada-role": "admin", "x-datada-tenant-id": "public"},
        params={"db_connection_id": "default", "limit": 50},
    )
    assert listed.status_code == 200
    rules = listed.json()["rules"]
    assert any(r["rule_id"] == rule_id for r in rules)

    updated = client.post(
        "/api/assistant/rules/update",
        headers={"x-datada-role": "admin", "x-datada-tenant-id": "public"},
        json={
            "db_connection_id": "default",
            "rule_id": rule_id,
            "name": "fx markup routing v2",
            "triggers": ["fx markup", "forex markup"],
            "action_payload": {
                "target_table": "datada_mart_quotes",
                "target_metric": "forex_markup_revenue",
                "target_dimensions": ["__month__", "platform_name"],
            },
            "notes": "updated",
            "status": "active",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["rule_id"] == rule_id

    listed2 = client.get(
        "/api/assistant/rules",
        headers={"x-datada-role": "admin", "x-datada-tenant-id": "public"},
        params={"db_connection_id": "default", "limit": 50},
    )
    assert listed2.status_code == 200
    changed = next(r for r in listed2.json()["rules"] if r["rule_id"] == rule_id)
    assert changed["name"] == "fx markup routing v2"
    assert "fx markup" in changed["triggers"]
    assert changed["action_payload"]["target_dimensions"] == ["__month__", "platform_name"]
    assert changed["version"] >= 2


def test_source_truth_endpoint(client):
    resp = client.get(
        "/api/assistant/source-truth/check",
        params={"db_connection_id": "default", "llm_mode": "deterministic", "max_cases": 3},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["cases"] >= 3
    assert payload["mode_actual"] in {"deterministic", "local", "openai", "auto"}
    assert isinstance(payload["runs"], list)


def test_cloud_model_catalog_endpoints(client):
    openai_resp = client.get("/api/assistant/models/openai")
    assert openai_resp.status_code == 200
    openai_payload = openai_resp.json()
    assert openai_payload["provider"] == "openai"
    assert isinstance(openai_payload["options"], list)
    assert len(openai_payload["options"]) >= 1

    anthropic_resp = client.get("/api/assistant/models/anthropic")
    assert anthropic_resp.status_code == 200
    anthropic_payload = anthropic_resp.json()
    assert anthropic_payload["provider"] == "anthropic"
    assert isinstance(anthropic_payload["options"], list)
    assert len(anthropic_payload["options"]) >= 1


def test_query_passes_provider_model_overrides_to_runtime_resolution(client, monkeypatch):
    captured: dict[str, object] = {}

    def _fake_resolve_runtime(mode, **kwargs):
        captured["mode"] = str(mode)
        captured.update(kwargs)
        return RuntimeSelection(
            requested_mode="deterministic",
            mode="deterministic",
            use_llm=False,
            provider=None,
            reason="test_override_capture",
            intent_model=None,
            narrator_model=None,
        )

    monkeypatch.setattr(server_module, "_resolve_runtime", _fake_resolve_runtime)
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "How many transactions are there?",
            "llm_mode": "openai",
            "openai_model": "gpt-4.1-mini",
            "openai_narrator_model": "gpt-4.1-mini",
            "anthropic_model": "claude-sonnet-4-6",
            "anthropic_narrator_model": "claude-sonnet-4-6",
        },
    )
    assert resp.status_code == 200
    assert captured.get("openai_model") == "gpt-4.1-mini"
    assert captured.get("openai_narrator_model") == "gpt-4.1-mini"
    assert captured.get("anthropic_model") == "claude-sonnet-4-6"
    assert captured.get("anthropic_narrator_model") == "claude-sonnet-4-6"


def test_dual_metric_grouped_query_returns_count_and_amount(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "transactions count and amount split by month and platform, only count MT103 transactions",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    sql = (payload.get("sql") or "").lower()
    assert "metric_value" in sql
    assert "secondary_metric_value" in sql
    assert "has_mt103" in sql
    assert "secondary_metric_value" in (payload.get("columns") or [])
    answer = (payload.get("answer_markdown") or "").lower()
    assert "mt103_count" in answer
    assert "total_amount" in answer
    chart_spec = payload.get("chart_spec") or {}
    report = chart_spec.get("report") or {}
    assert isinstance(report.get("panels"), list)
    assert len(report.get("panels")) >= 1


def test_schema_glossary_request_returns_all_marts_dictionary(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "generate me a glossary of schema each field and table and what it means",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    answer = payload.get("answer_markdown") or ""
    assert "datada_mart_transactions" in answer
    assert "datada_mart_quotes" in answer
    assert "datada_dim_customers" in answer
    assert "datada_mart_bookings" in answer
    assert "| Field | Meaning | Notes |" in answer


def test_schema_glossary_includes_business_purpose_and_metric_meanings(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "Generate a full schema dictionary for every mart with business definitions",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    answer = payload.get("answer_markdown") or ""
    assert "Business purpose:" in answer
    assert "has_mt103" in answer
    assert "SWIFT MT103 settlement proof" in answer
    assert "| Metric | SQL expression | Business meaning |" in answer


def test_schema_glossary_detects_table_column_definition_phrasing(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "Give me detailed definitions of tables, columns and their meaning",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    answer = payload.get("answer_markdown") or ""
    assert "Full Schema Dictionary" in answer
    assert "datada_mart_transactions" in answer
    assert "| Field | Meaning | Notes |" in answer


def test_explicit_provider_failure_returns_hard_error(client, monkeypatch):
    runtime = RuntimeSelection(
        requested_mode="anthropic",
        mode="anthropic",
        use_llm=True,
        provider="anthropic",
        reason="anthropic selected",
        intent_model="claude-haiku-4-5-20251001",
        narrator_model="claude-haiku-4-5-20251001",
    )
    monkeypatch.setattr(server_module, "_resolve_runtime", Mock(return_value=runtime))

    def _raise_llm(*_args, **_kwargs):
        raise RuntimeError("401 invalid_api_key")

    monkeypatch.setattr(v2_orchestrator_module, "call_llm", _raise_llm)

    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "what is the total transaction count",
            "llm_mode": "anthropic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is False
    assert "unavailable" in str(payload.get("error") or "").lower()
    runtime_payload = payload.get("runtime") or {}
    assert runtime_payload.get("llm_degraded") is True
    assert runtime_payload.get("llm_degraded_provider") == "anthropic"
    assert "requested mode failed" in (payload.get("answer_markdown") or "").lower()
    warnings = [str(w).lower() for w in (payload.get("warnings") or [])]
    assert any("unavailable" in w for w in warnings)


def test_data_overview_uses_discovery_agents(client):
    resp = client.post(
        "/api/assistant/query",
        json={"goal": "What kind of data do I have?", "llm_mode": "deterministic"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert "Data map" in payload["answer_markdown"]
    assert "Full Schema Dictionary" not in payload["answer_markdown"]


def test_data_overview_includes_rare_pockets_and_semantic_version(client):
    resp = client.post(
        "/api/assistant/query",
        json={"goal": "What kind of data do I have?", "llm_mode": "deterministic"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert "Rare pockets worth exploring" in payload["answer_markdown"]
    quality = payload.get("data_quality", {})
    assert quality.get("semantic_version")
    assert isinstance(quality.get("coverage_by_domain"), dict)
    checks = payload.get("sanity_checks", [])
    assert any(c.get("check_name") == "semantic_versioned" and c.get("passed") for c in checks)


def test_split_month_and_platform_phrase_generates_grouped_dimensions(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "What are the transaction split my month and platform",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    grounding = (payload.get("data_quality") or {}).get("grounding", {})
    dims = grounding.get("dimensions") or []
    assert "__month__" in dims
    assert "platform_name" in dims
    assert payload.get("row_count", 0) > 1
    answer = str(payload.get("answer_markdown") or "")
    assert "**What Drove This**" in answer
    assert "**Evidence**" in answer
    assert "**Caveat**" in answer


def test_cross_domain_markup_vs_spend_query_returns_monthly_comparison(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "Forex markup split by month compared to customer spend on transaction and give me insights",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    sql = str(payload.get("sql") or "").lower()
    assert "primary_agg" in sql
    assert "secondary_agg" in sql
    cols = payload.get("columns") or []
    assert "forex_markup_revenue" in cols
    assert "customer_spend" in cols
    assert payload.get("row_count", 0) >= 1
    trace = payload.get("agent_trace") or []
    assert any(t.get("agent") == "OrganizationalKnowledgeAgent" for t in trace)
    assert any(t.get("skill_contract_enforced") for t in trace if isinstance(t, dict))
    runtime = payload.get("runtime") or {}
    assert (runtime.get("skills_runtime") or {}).get("enforceable_agents", 0) >= 8
    assert "glossary_seed_stats" in runtime


def test_transaction_validity_guard_applies_mt103_filter_for_spend(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "Show valid transaction spend by month",
            "llm_mode": "deterministic",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    sql = str(payload.get("sql") or "").lower()
    assert "has_mt103" in sql
    assert "payment_amount" in sql or "sum(amount)" in sql


def test_document_query_returns_citations(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "policy.txt").write_text(
        "Forex markup policy: markup must be disclosed and capped at 2.5 percent for retail flows.",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.duckdb"
    ingest_result = ingest_documents_to_duckdb(docs_dir=docs_dir, db_path=db_path, force=True)
    assert ingest_result["success"] is True
    assert ingest_result.get("chunks", 0) >= 1

    app = create_app(db_path=db_path)
    client = TestClient(app)
    resp = client.post(
        "/api/assistant/query",
        json={"goal": "What does the policy say about forex markup?", "llm_mode": "deterministic"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["row_count"] >= 1
    assert "D1" in payload["answer_markdown"]
    assert payload["runtime"]["db_kind"] in {"duckdb", "documents"}


def test_tenant_isolation_for_corrections(client):
    feedback = client.post(
        "/api/assistant/feedback",
        headers={"x-datada-tenant-id": "tenant-a", "x-datada-role": "analyst"},
        json={
            "issue": "tenant isolation correction",
            "keyword": "forex",
            "target_table": "datada_mart_quotes",
            "target_metric": "forex_markup_revenue",
        },
    )
    assert feedback.status_code == 200
    correction_id = feedback.json().get("correction_id")
    assert correction_id

    listed_a = client.get(
        "/api/assistant/corrections",
        headers={"x-datada-tenant-id": "tenant-a", "x-datada-role": "viewer"},
    )
    assert listed_a.status_code == 200
    ids_a = {row.get("correction_id") for row in listed_a.json().get("rules", [])}
    assert correction_id in ids_a

    listed_b = client.get(
        "/api/assistant/corrections",
        headers={"x-datada-tenant-id": "tenant-b", "x-datada-role": "viewer"},
    )
    assert listed_b.status_code == 200
    ids_b = {row.get("correction_id") for row in listed_b.json().get("rules", [])}
    assert correction_id not in ids_b


def test_slo_and_incident_endpoints(client):
    bad = client.post(
        "/api/assistant/query",
        json={"goal": "DROP TABLE test_1_1_merged", "llm_mode": "deterministic"},
    )
    assert bad.status_code == 200
    assert bad.json()["success"] is False

    slo = client.get(
        "/api/assistant/slo/evaluate",
        params={"tenant_id": "public", "hours": 24},
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
    )
    assert slo.status_code == 200
    slo_payload = slo.json()
    assert "status" in slo_payload
    assert "targets" in slo_payload

    incidents = client.get(
        "/api/assistant/incidents",
        params={"tenant_id": "public", "limit": 10},
        headers={"x-datada-role": "viewer", "x-datada-tenant-id": "public"},
    )
    assert incidents.status_code == 200
    incident_rows = incidents.json().get("incidents", [])
    assert isinstance(incident_rows, list)
    assert any(str(row.get("source", "")) == "query_failure" for row in incident_rows)


def test_auto_fast_path_prefers_deterministic_for_simple_metric_goal():
    assert server_module._auto_prefers_deterministic_fast_path(
        "total transaction count by month and platform"
    )
    assert not server_module._auto_prefers_deterministic_fast_path(
        "explain the root cause and strategy behind this trend"
    )


def test_resolve_runtime_auto_prefers_available_provider(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "_providers_snapshot",
        lambda force_refresh=False: server_module.ProvidersResponse(
            default_mode=server_module.LLMMode.AUTO,
            recommended_mode=server_module.LLMMode.OPENAI,
            checks={
                "ollama": server_module.ProviderCheck(available=False, reason="down"),
                "openai": server_module.ProviderCheck(available=True, reason="up"),
                "anthropic": server_module.ProviderCheck(available=False, reason="down"),
            },
        ),
    )
    runtime = server_module._resolve_runtime(
        server_module.LLMMode.AUTO,
        goal="show total bookings count by month",
    )
    assert runtime.mode == "openai"
    assert runtime.provider == "openai"
    assert runtime.use_llm is True


def test_resolve_runtime_auto_raises_when_no_provider_available(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "_providers_snapshot",
        lambda force_refresh=False: server_module.ProvidersResponse(
            default_mode=server_module.LLMMode.AUTO,
            recommended_mode=server_module.LLMMode.DETERMINISTIC,
            checks={
                "ollama": server_module.ProviderCheck(available=False, reason="down"),
                "openai": server_module.ProviderCheck(available=False, reason="down"),
                "anthropic": server_module.ProviderCheck(available=False, reason="down"),
            },
        ),
    )
    with pytest.raises(server_module.HTTPException) as exc:
        server_module._resolve_runtime(server_module.LLMMode.AUTO, goal="show total bookings count by month")
    assert exc.value.status_code == 503


def test_provider_snapshot_uses_cache(monkeypatch):
    calls = {"ollama": 0, "openai": 0, "anthropic": 0}

    def _ollama():
        calls["ollama"] += 1
        return server_module.ProviderCheck(available=False, reason="x")

    def _openai():
        calls["openai"] += 1
        return server_module.ProviderCheck(available=False, reason="x")

    def _anthropic():
        calls["anthropic"] += 1
        return server_module.ProviderCheck(available=False, reason="x")

    monkeypatch.setenv("HG_PROVIDER_SNAPSHOT_TTL_SECONDS", "30")
    monkeypatch.setattr(server_module, "_ollama_check", _ollama)
    monkeypatch.setattr(server_module, "_openai_check", _openai)
    monkeypatch.setattr(server_module, "_anthropic_check", _anthropic)
    server_module._PROVIDER_SNAPSHOT_CACHE = None
    server_module._PROVIDER_SNAPSHOT_CACHE_TS = 0.0

    _ = server_module._providers_snapshot(force_refresh=True)
    _ = server_module._providers_snapshot()
    _ = server_module._providers_snapshot()

    assert calls["ollama"] == 1
    assert calls["openai"] == 1
    assert calls["anthropic"] == 1
