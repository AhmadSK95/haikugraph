"""Regression suite for grain and denominator semantics in unified v2 runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


def _seed_db() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = Path(f.name)
    db_path.unlink(missing_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE transactions (
            transaction_id VARCHAR,
            customer_id VARCHAR,
            quote_id VARCHAR,
            platform_name VARCHAR,
            payment_amount DOUBLE,
            forex_markup DOUBLE,
            has_mt103 BOOLEAN,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO transactions VALUES
        ('t1','c1','q1','WEB',120.0,2.0,TRUE,'2025-12-01'),
        ('t2','c1','q2','WEB',80.0,1.0,TRUE,'2025-12-10'),
        ('t3','c2','q3','APP',50.0,0.5,FALSE,'2025-12-12')
        """
    )
    conn.close()
    return db_path


def _client() -> tuple[TestClient, Path]:
    db_path = _seed_db()
    app = create_app(db_path=db_path)
    return TestClient(app), db_path


def test_denominator_semantics_per_customer() -> None:
    client, db_path = _client()
    try:
        resp = client.post(
            "/api/assistant/query",
            json={"goal": "Average payment amount per customer", "llm_mode": "deterministic"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["success"] is True
        assert payload.get("denominator_semantics") == "per_customer"
        assert str(payload.get("grain_signature") or "").strip()
    finally:
        client.close()
        db_path.unlink(missing_ok=True)


def test_denominator_semantics_per_transaction() -> None:
    client, db_path = _client()
    try:
        resp = client.post(
            "/api/assistant/query",
            json={"goal": "Average payment amount per transaction", "llm_mode": "deterministic"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["success"] is True
        assert payload.get("denominator_semantics") == "per_transaction"
        assert str(payload.get("grain_signature") or "").strip()
    finally:
        client.close()
        db_path.unlink(missing_ok=True)


def test_denominator_semantics_per_quote() -> None:
    client, db_path = _client()
    try:
        resp = client.post(
            "/api/assistant/query",
            json={"goal": "Average payment amount per quote", "llm_mode": "deterministic"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["success"] is True
        assert payload.get("denominator_semantics") == "per_quote"
        assert str(payload.get("grain_signature") or "").strip()
    finally:
        client.close()
        db_path.unlink(missing_ok=True)


def test_conflicting_denominator_requests_clarification() -> None:
    client, db_path = _client()
    try:
        resp = client.post(
            "/api/assistant/query",
            json={"goal": "Show spend per customer and per transaction", "llm_mode": "deterministic"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["success"] is False
        assert "Clarification needed" in str(payload.get("answer_markdown") or "")
        assert "contradiction_detected" in (payload.get("quality_flags") or [])
    finally:
        client.close()
        db_path.unlink(missing_ok=True)
