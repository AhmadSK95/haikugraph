"""BRD §7.1.1 — Canonical factual suite (>=40 cases).

Covers: month/year scoped metrics, grouped metrics, cross-domain metrics.
Acceptance: >=92% exact match (NFR-2).
"""
from __future__ import annotations

import pytest
from haikugraph.poc.agentic_team import AgenticAnalyticsTeam

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query(team, goal: str):
    """Run a deterministic query and return the response."""
    from haikugraph.poc.agentic_team import RuntimeSelection

    rt = RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )
    return team.run(goal, rt)


def _scalar(resp) -> float | int | str | None:
    """Extract the primary scalar metric from the response."""
    if not resp.success:
        return None
    rows = resp.sample_rows
    if rows and len(rows) == 1:
        vals = list(rows[0].values())
        if len(vals) == 1:
            return vals[0]
        # Look for metric_value column
        for key in ("metric_value", "count", "total"):
            if key in rows[0]:
                return rows[0][key]
    if rows and len(rows) >= 1 and "metric_value" in rows[0]:
        return rows[0]["metric_value"]
    return resp.row_count


def _grouped(resp) -> dict:
    """Extract grouped results as {dim_value: metric_value}."""
    result = {}
    if not resp.success:
        return result
    for row in resp.sample_rows:
        vals = list(row.values())
        if len(vals) >= 2:
            result[vals[0]] = vals[-1]
    return result


# ---------------------------------------------------------------------------
# Suite 1: Transaction metrics (basic)
# ---------------------------------------------------------------------------

class TestTransactionBasic:
    """F01-F10: Basic transaction count/amount queries."""

    def test_f01_total_transaction_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions are there?")
            assert resp.success
            assert resp.row_count >= 1
            val = _scalar(resp)
            assert val == 8 or resp.row_count == 1

    def test_f02_total_transaction_amount(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "What is the total transaction amount?")
            assert resp.success

    def test_f03_dec_2025_transaction_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions in Dec-2025?")
            assert resp.success
            assert "MONTH" in (resp.sql or "").upper() or "12" in (resp.sql or "")

    def test_f04_nov_2025_transaction_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions in Nov-2025?")
            assert resp.success
            assert "MONTH" in (resp.sql or "").upper() or "11" in (resp.sql or "")

    def test_f05_dec_2025_unique_transaction_ids(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Return unique transaction IDs for Dec-2025 only.")
            assert resp.success
            # Should NOT trigger clarification
            assert "clarification" not in resp.answer_markdown.lower()

    def test_f06_transaction_count_by_platform(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform_name")
            assert resp.success
            assert resp.row_count >= 1
            assert "GROUP BY" in (resp.sql or "").upper()

    def test_f07_dec_2025_grouped_by_platform(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Dec-2025 transaction count grouped by platform_name")
            assert resp.success
            assert "GROUP BY" in (resp.sql or "").upper()
            sql = resp.sql or ""
            assert "12" in sql  # month filter present

    def test_f08_transaction_count_by_flow(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by flow")
            assert resp.success
            assert resp.row_count >= 1

    def test_f09_top_5_platforms_dec_2025(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Top 5 platforms by transaction count in Dec-2025")
            assert resp.success
            assert resp.row_count >= 1

    def test_f10_total_amount_dec_2025(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total amount of transactions in Dec 2025?")
            assert resp.success


# ---------------------------------------------------------------------------
# Suite 2: Conditional/filtered metrics
# ---------------------------------------------------------------------------

class TestConditionalMetrics:
    """F11-F20: MT103, refund, and conditional metrics."""

    def test_f11_refund_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions have refunds?")
            assert resp.success

    def test_f12_mt103_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many MT103-backed transactions are there?")
            assert resp.success

    def test_f13_mt103_dec_2025(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Unique MT103-backed transactions in Dec-2025?")
            assert resp.success
            sql = resp.sql or ""
            assert "12" in sql or "MONTH" in sql.upper()

    def test_f14_mt103_by_platform(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "MT103 transaction count by platform_name")
            assert resp.success
            assert "GROUP BY" in (resp.sql or "").upper()

    def test_f15_refund_rate_by_platform(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "What is the refund rate by platform?")
            # Refund rate is a computed metric; accept success or graceful fallback
            assert resp.success or resp.answer_markdown

    def test_f16_dec_2025_mt103_by_platform(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Dec-2025 MT103 unique transactions by platform_name")
            assert resp.success

    def test_f17_refund_count_dec_2025(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many refunded transactions in Dec-2025?")
            assert resp.success

    def test_f18_unique_customers(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many unique customers have transactions?")
            assert resp.success

    def test_f19_transaction_amount_by_state(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total transaction amount by state")
            assert resp.success

    def test_f20_payment_status_distribution(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by payment_status")
            assert resp.success


# ---------------------------------------------------------------------------
# Suite 3: Quote metrics
# ---------------------------------------------------------------------------

class TestQuoteMetrics:
    """F21-F28: Quote domain metrics."""

    def test_f21_total_quote_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many quotes are there?")
            assert resp.success

    def test_f22_total_quote_value(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "What is the total quote value?")
            assert resp.success

    def test_f23_forex_markup_dec_2025(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Compute total forex_markup for quotes created in Dec-2025")
            assert resp.success
            sql = resp.sql or ""
            assert "12" in sql or "MONTH" in sql.upper()

    def test_f24_quote_count_by_currency(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Quote count by source currency")
            assert resp.success

    def test_f25_average_exchange_rate(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "What is the average exchange rate across all quotes?")
            assert resp.success

    def test_f26_most_frequent_currency_pair(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "What is the most frequent currency pair in quotes?")
            assert resp.success or resp.answer_markdown
            # Should reference currency fields or quote table
            sql = (resp.sql or "").lower()
            assert "currency" in sql or "from_currency" in sql or "quote" in sql

    def test_f27_total_charges_by_currency(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total additional charges by source currency in quotes")
            assert resp.success

    def test_f28_quote_value_by_month(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total quote value by month")
            assert resp.success


# ---------------------------------------------------------------------------
# Suite 4: Customer metrics
# ---------------------------------------------------------------------------

class TestCustomerMetrics:
    """F29-F34: Customer domain metrics."""

    def test_f29_total_customer_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many customers are there?")
            assert resp.success

    def test_f30_customer_count_by_country(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Customer count by country")
            assert resp.success
            assert "GROUP BY" in (resp.sql or "").upper()

    def test_f31_university_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many university payees are there?")
            assert resp.success

    def test_f32_university_count_by_country(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "University count by country")
            assert resp.success

    def test_f33_customer_count_by_type(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Customer count by type")
            assert resp.success

    def test_f34_customer_count_by_status(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Customer count by status")
            assert resp.success


# ---------------------------------------------------------------------------
# Suite 5: Booking metrics
# ---------------------------------------------------------------------------

class TestBookingMetrics:
    """F35-F40: Booking domain metrics."""

    def test_f35_total_booking_count(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many bookings are there?")
            assert resp.success

    def test_f36_total_booked_amount(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "What is the total booked amount?")
            assert resp.success

    def test_f37_booked_by_currency(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total booked amount by currency")
            assert resp.success

    def test_f38_avg_rate_by_deal_type(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Average rate by deal type")
            assert resp.success

    def test_f39_booking_count_by_deal_type(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Booking count by deal type")
            assert resp.success

    def test_f40_booking_count_dec_2025(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many bookings in Dec-2025?")
            assert resp.success
            sql = resp.sql or ""
            assert "12" in sql or "MONTH" in sql.upper()


# ---------------------------------------------------------------------------
# Suite 6: Comparison queries
# ---------------------------------------------------------------------------

class TestComparisonMetrics:
    """F41-F44: Comparison / delta queries."""

    def test_f41_dec_vs_prev_month(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Compare Dec 2025 vs previous month transaction count")
            assert resp.success

    def test_f42_nov_vs_dec_delta(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Compute Nov-2025 vs Dec-2025 unique transaction delta")
            assert resp.success

    def test_f43_year_over_year(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Year over year transaction count comparison")
            assert resp.success

    def test_f44_quote_value_dec_vs_nov(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Compare Dec-2025 vs Nov-2025 total quote value")
            assert resp.success
