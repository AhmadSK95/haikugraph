"""Tests for runtime controls: async jobs, trust dashboard, and rollback paths."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


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

