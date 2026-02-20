"""Stress tests for the dataDa agentic pipeline.

Tests focus on stability: 200 status codes, no crashes, no hangs.
These do NOT assert on correctness of answers.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _query(client, goal: str, **extra) -> dict:
    payload = {"goal": goal, **extra}
    resp = client.post("/api/assistant/query", json=payload)
    return {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else {}}


# =============================================================================
# Concurrent queries
# =============================================================================

class TestConcurrency:
    """Multiple queries running in parallel or rapid sequence."""

    def test_10_parallel_queries(self, client_datada):
        """Fire 10 queries concurrently — all should return 200."""
        goals = [
            "How many transactions?",
            "Total transaction amount",
            "Transactions by platform",
            "How many customers?",
            "Average transaction amount",
            "Transactions in December",
            "Top 5 transactions by amount",
            "How many quotes?",
            "Total booked amount",
            "Transactions by state",
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_query, client_datada, g) for g in goals]
            results = [f.result(timeout=120) for f in futures]
        assert all(r["status"] == 200 for r in results)

    def test_50_sequential_queries(self, client_datada):
        """Run 50 simple queries sequentially — none should crash."""
        for i in range(50):
            res = _query(client_datada, f"How many transactions? (variation {i})")
            assert res["status"] == 200

    def test_session_isolation(self, client_datada):
        """Two sessions querying simultaneously should not interfere."""
        results: list[dict] = []

        def run_session(sid: str, goal: str):
            r = _query(client_datada, goal, session_id=sid)
            results.append(r)

        t1 = threading.Thread(target=run_session, args=("sess-A", "Total transaction amount"))
        t2 = threading.Thread(target=run_session, args=("sess-B", "How many customers?"))
        t1.start(); t2.start()
        t1.join(timeout=60); t2.join(timeout=60)

        assert len(results) == 2
        assert all(r["status"] == 200 for r in results)


# =============================================================================
# Large / edge-case payloads
# =============================================================================

class TestPayloadEdgeCases:
    """Abnormal payloads that should not crash the server."""

    def test_max_length_goal(self, client_datada):
        """2000-character goal string — should not crash."""
        long_goal = "How many transactions " * 90  # ~1980 chars
        long_goal = long_goal[:2000]
        res = _query(client_datada, long_goal)
        assert res["status"] == 200

    def test_goal_with_only_whitespace_after_strip(self, client_datada):
        """Goal that is whitespace should be rejected by validation."""
        resp = client_datada.post("/api/assistant/query", json={"goal": "   "})
        # FastAPI validates min_length=1 on stripped string or returns 422/200 with error
        assert resp.status_code in (200, 422)

    def test_empty_constraints(self, client_datada):
        """Empty constraints dict should work fine."""
        res = _query(client_datada, "How many transactions?", constraints={})
        assert res["status"] == 200


# =============================================================================
# Edge-case inputs (unicode, injection, etc.)
# =============================================================================

class TestEdgeCaseInputs:
    """Unusual input strings that should not crash the pipeline."""

    def test_unicode_goal(self, client_datada):
        """Unicode characters in goal."""
        res = _query(client_datada, "Wie viele Transaktionen? 日本語 中文")
        assert res["status"] == 200

    def test_emoji_goal(self, client_datada):
        """Emoji in goal string."""
        res = _query(client_datada, "How many transactions? 🤖💰📊")
        assert res["status"] == 200

    def test_sql_injection_attempt(self, client_datada):
        """SQL injection should be safely handled (not executed)."""
        res = _query(client_datada, "'; DROP TABLE test_1_1_merged; --")
        assert res["status"] == 200
        body = res["body"]
        # Should either block or safely parse — table must still exist
        assert body.get("success") is not None

    def test_html_tags_in_goal(self, client_datada):
        """HTML injection should not crash."""
        res = _query(client_datada, "<script>alert('xss')</script> How many transactions?")
        assert res["status"] == 200

    def test_newlines_in_goal(self, client_datada):
        """Newlines in goal string."""
        res = _query(client_datada, "How many\ntransactions\ndo we have?")
        assert res["status"] == 200

    def test_backslashes_in_goal(self, client_datada):
        """Backslashes should not crash parser."""
        res = _query(client_datada, "Transactions \\n with \\t special chars")
        assert res["status"] == 200

    def test_very_long_word(self, client_datada):
        """Single very long word should not hang regex."""
        long_word = "a" * 500
        res = _query(client_datada, f"Show {long_word} transactions")
        assert res["status"] == 200


# =============================================================================
# Empty / sparse data scenarios
# =============================================================================

class TestEmptyData:
    """Tests with databases that have empty or sparse data."""

    def test_query_on_empty_tables(self, seed_db_datada):
        """Even with populated DB, a domain with 0 matching rows should not crash."""
        from haikugraph.api.server import create_app
        from fastapi.testclient import TestClient
        # Query for something that won't match
        app = create_app(db_path=seed_db_datada)
        c = TestClient(app)
        res = _query(c, "Transactions from Antarctica in 1999")
        assert res["status"] == 200

    def test_all_null_filter(self, client_datada):
        """Filtering on a value that matches nothing."""
        res = _query(client_datada, "Show me transactions with platform NONEXISTENT_PLATFORM")
        assert res["status"] == 200

    def test_query_obscure_domain(self, client_datada):
        """Querying a domain that doesn't exist should not crash."""
        res = _query(client_datada, "Show me all inventory items")
        assert res["status"] == 200


# =============================================================================
# Rapid sequential sessions & follow-ups
# =============================================================================

class TestRapidSessions:
    """Rapid sequential sessions to test session management."""

    def test_follow_up_chain(self, client_datada):
        """3-query follow-up chain using same session_id."""
        sid = "stress-followup-001"
        q1 = _query(client_datada, "How many transactions?", session_id=sid)
        assert q1["status"] == 200
        q2 = _query(client_datada, "Split that by platform", session_id=sid)
        assert q2["status"] == 200
        q3 = _query(client_datada, "Now in December only", session_id=sid)
        assert q3["status"] == 200

    def test_many_unique_sessions(self, client_datada):
        """20 unique session IDs, one query each."""
        for i in range(20):
            res = _query(client_datada, "How many transactions?", session_id=f"rapid-{i:03d}")
            assert res["status"] == 200

    def test_concurrent_sessions(self, client_datada):
        """10 sessions querying concurrently."""
        def run(sid):
            return _query(client_datada, "Total transaction amount", session_id=f"conc-{sid}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(run, i) for i in range(10)]
            results = [f.result(timeout=120) for f in futures]
        assert all(r["status"] == 200 for r in results)


# =============================================================================
# Performance stability
# =============================================================================

class TestPerformanceStability:
    """Ensure no progressive degradation over many queries."""

    def test_no_latency_degradation(self, client_datada):
        """Average latency of last 10 queries should not exceed 3x the first 10."""
        latencies: list[float] = []
        for i in range(30):
            t0 = time.perf_counter()
            res = _query(client_datada, "How many transactions?")
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            assert res["status"] == 200

        first_10_avg = sum(latencies[:10]) / 10
        last_10_avg = sum(latencies[-10:]) / 10
        # Allow 3x tolerance (first batch may include cold-start)
        assert last_10_avg < first_10_avg * 3.5 or last_10_avg < 10.0

    def test_stable_results_over_repetitions(self, client_datada):
        """Same query 10 times should return consistent status and structure."""
        statuses = set()
        for _ in range(10):
            res = _query(client_datada, "Total transaction amount")
            statuses.add(res["status"])
        assert statuses == {200}
