"""Tests for autonomy transparency and correction governance APIs."""

import tempfile
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
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            payee_id VARCHAR,
            is_university BOOLEAN,
            type VARCHAR,
            status VARCHAR,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO customers VALUES
        (1, 'p1', TRUE, 'education', 'active', '2025-01-10'),
        (2, 'p2', FALSE, 'individual', 'active', '2025-01-12')
        """
    )

    conn.execute(
        """
        CREATE TABLE orders (
            transaction_id VARCHAR,
            customer_id INTEGER,
            payment_amount DOUBLE,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            status VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES
        ('t1', 1, 100.0, '2025-09-01', '2025-09-02', 'completed'),
        ('t2', 2, 120.0, '2025-10-01', '2025-10-02', 'completed')
        """
    )
    conn.close()

    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def client(seed_db):
    app = create_app(db_path=seed_db)
    return TestClient(app)


def test_query_includes_blackboard_and_confidence_decomposition(client):
    resp = client.post(
        "/api/assistant/query",
        json={"goal": "How many customers do we have?", "llm_mode": "deterministic"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "runtime" in data
    assert int(data["runtime"].get("blackboard_entries", 0)) > 0

    packets = data.get("evidence_packets", [])
    blackboard = next((p for p in packets if p.get("agent") == "Blackboard"), None)
    assert blackboard is not None
    assert int(blackboard.get("artifact_count", 0)) > 0
    assert isinstance(blackboard.get("artifacts", []), list)

    autonomy = next((p for p in packets if p.get("agent") == "AutonomyAgent"), None)
    assert autonomy is not None
    assert isinstance(autonomy.get("confidence_decomposition", []), list)
    assert isinstance(autonomy.get("contradiction_resolution", {}), dict)


def test_autonomy_reports_refinement_rounds_and_signatures(client):
    resp = client.post(
        "/api/assistant/query",
        json={
            "goal": "What is the forex markup revenue for December 2025?",
            "llm_mode": "deterministic",
            "max_refinement_rounds": 3,
            "max_candidate_plans": 4,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    autonomy = (data.get("data_quality") or {}).get("autonomy", {})
    rounds = autonomy.get("refinement_rounds", [])
    assert isinstance(rounds, list)
    assert len(rounds) >= 1
    assert len(rounds) <= 3
    assert all("round" in r and "ending_score" in r for r in rounds)

    grounding = (data.get("data_quality") or {}).get("grounding", {})
    assert "concept_coverage_pct" in grounding
    assert grounding.get("execution_signature")


def test_corrections_can_be_listed_and_toggled(client):
    feedback = client.post(
        "/api/assistant/feedback",
        json={
            "db_connection_id": "default",
            "goal": "forex metrics",
            "issue": "Mapped to wrong mart for forex question",
            "keyword": "forex",
            "target_table": "datada_mart_quotes",
            "target_metric": "forex_markup_revenue",
            "target_dimensions": ["__month__"],
        },
    )
    assert feedback.status_code == 200
    feedback_payload = feedback.json()
    correction_id = feedback_payload.get("correction_id")
    assert correction_id

    listed = client.get("/api/assistant/corrections", params={"db_connection_id": "default"})
    assert listed.status_code == 200
    listed_payload = listed.json()
    rules = listed_payload.get("rules", [])
    match = next((r for r in rules if r.get("correction_id") == correction_id), None)
    assert match is not None
    assert match.get("enabled") is True

    toggled = client.post(
        "/api/assistant/corrections/toggle",
        json={
            "db_connection_id": "default",
            "correction_id": correction_id,
            "enabled": False,
        },
    )
    assert toggled.status_code == 200
    assert toggled.json()["enabled"] is False

    listed_after = client.get("/api/assistant/corrections", params={"db_connection_id": "default"})
    assert listed_after.status_code == 200
    rules_after = listed_after.json().get("rules", [])
    match_after = next((r for r in rules_after if r.get("correction_id") == correction_id), None)
    assert match_after is not None
    assert match_after.get("enabled") is False
