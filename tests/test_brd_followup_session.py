"""BRD §7.1.3 — Follow-up/session suite (>=15 chains).

Covers: carry-over intent, narrowing/widening filters, metric continuity.
Acceptance: >=95% pass.
"""
from __future__ import annotations

import pytest
from haikugraph.poc.agentic_team import AgenticAnalyticsTeam
from haikugraph.agents.contracts import ConfidenceLevel


def _rt():
    from haikugraph.poc.agentic_team import RuntimeSelection

    return RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )


# ---------------------------------------------------------------------------
# Suite: Follow-up chains
# ---------------------------------------------------------------------------

class TestFollowUpChains:
    """15+ follow-up chains testing context carry-over."""

    def test_chain01_narrow_to_month(self, known_data_db):
        """Base query → narrow to December."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many transactions are there?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many transactions are there?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Now only for December 2025", _rt(), conversation_context=ctx)
            assert r2.success
            sql = r2.sql or ""
            assert "12" in sql or "december" in sql.lower() or "MONTH" in sql.upper()

    def test_chain02_narrow_to_platform(self, known_data_db):
        """Base query → filter by platform."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Transaction count by platform_name", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "Transaction count by platform_name"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Just B2C-APP", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain03_widen_after_narrow(self, known_data_db):
        """Narrow to Dec → widen back to all."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Transaction count in Dec-2025", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "Transaction count in Dec-2025"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("What about all months?", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain04_switch_metric(self, known_data_db):
        """Count → amount on same scope."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many transactions in Dec-2025?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many transactions in Dec-2025?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("What about the total amount instead?", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain05_switch_domain(self, known_data_db):
        """Transactions → quotes domain switch."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many transactions?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many transactions?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("How many quotes are there?", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain06_add_grouping(self, known_data_db):
        """Simple count → add grouping."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many transactions?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many transactions?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Split that by platform", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain07_comparison_followup(self, known_data_db):
        """Dec-2025 → compare with previous month."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Transaction count in Dec-2025", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "Transaction count in Dec-2025"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Compare with November", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain08_mt103_drill(self, known_data_db):
        """MT103 count → by platform."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many MT103 transactions?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many MT103 transactions?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Now by platform", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain09_booking_chain(self, known_data_db):
        """Booking count → amount → by deal type."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many bookings?", _rt())
            ctx = [{"role": "user", "content": "How many bookings?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("What's the total booked amount?", _rt(), conversation_context=ctx)
            ctx.extend([{"role": "user", "content": "What's the total booked amount?"},
                       {"role": "assistant", "content": r2.answer_markdown}])
            r3 = team.run("Break that down by deal type", _rt(), conversation_context=ctx)
            assert r3.success

    def test_chain10_customer_chain(self, known_data_db):
        """Customer count → by country."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many customers?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many customers?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("By country", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain11_quote_narrowing(self, known_data_db):
        """Quotes → Dec only."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Total forex markup across all quotes", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "Total forex markup across all quotes"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Only for December 2025", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain12_refund_drill(self, known_data_db):
        """Refund count → by platform."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many refunded transactions?", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "How many refunded transactions?"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("Split by platform", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain13_three_step_chain(self, known_data_db):
        """Three-step: total → by month → Dec only."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Total transaction amount", _rt())
            ctx = [{"role": "user", "content": "Total transaction amount"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("By month", _rt(), conversation_context=ctx)
            ctx.extend([{"role": "user", "content": "By month"},
                       {"role": "assistant", "content": r2.answer_markdown}])
            r3 = team.run("Just December", _rt(), conversation_context=ctx)
            assert r3.success

    def test_chain14_show_then_explain(self, known_data_db):
        """Show data → ask for explanation."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("Transaction count by platform", _rt())
            assert r1.success
            ctx = [{"role": "user", "content": "Transaction count by platform"},
                   {"role": "assistant", "content": r1.answer_markdown}]
            r2 = team.run("What data do we have?", _rt(), conversation_context=ctx)
            assert r2.success

    def test_chain15_metric_continuity(self, known_data_db):
        """Same metric, same time — should get consistent numbers."""
        with AgenticAnalyticsTeam(known_data_db) as team:
            r1 = team.run("How many transactions in Dec-2025?", _rt())
            r2 = team.run("Transaction count for December 2025", _rt())
            # Both should succeed
            assert r1.success
            assert r2.success
            # Both should reference month=12
            assert "12" in (r1.sql or "")
            assert "12" in (r2.sql or "")
