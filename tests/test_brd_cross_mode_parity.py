"""BRD §7.1.5 — Cross-mode parity suite.

Verifies: deterministic mode produces consistent contract checks.
(LLM modes tested when providers available.)
"""
from __future__ import annotations

import pytest
from haikugraph.poc.agentic_team import AgenticAnalyticsTeam


def _rt():
    from haikugraph.poc.agentic_team import RuntimeSelection

    return RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )


class TestDeterministicParity:
    """Cross-mode parity: deterministic mode must be consistent."""

    def test_parity_01_same_query_same_result(self, known_data_db):
        """Same query run twice should produce the same SQL."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many transactions?", _rt())
            r2 = team.run("How many transactions?", _rt())
            assert r1.sql == r2.sql

    def test_parity_02_contract_consistent(self, known_data_db):
        """Contract spec should be identical for identical queries."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Transaction count by platform", _rt())
            r2 = team.run("Transaction count by platform", _rt())
            assert r1.contract_spec.get("metric") == r2.contract_spec.get("metric")
            assert r1.contract_spec.get("dimensions") == r2.contract_spec.get("dimensions")

    def test_parity_03_confidence_stable(self, known_data_db):
        """Confidence should be stable across reruns."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Customer count by country", _rt())
            r2 = team.run("Customer count by country", _rt())
            assert abs(r1.confidence_score - r2.confidence_score) < 0.05

    def test_parity_04_month_filter_applied(self, known_data_db):
        """Dec-2025 filter applied consistently."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Transactions in Dec-2025", _rt())
            r2 = team.run("Dec-2025 transaction count", _rt())
            # Both should filter to month=12
            assert "12" in (r1.sql or "")
            assert "12" in (r2.sql or "")

    def test_parity_05_data_overview_consistent(self, known_data_db):
        """Data overview query should be consistent."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("What kind of data do I have?", _rt())
            r2 = team.run("What kind of data do I have?", _rt())
            assert r1.success == r2.success
