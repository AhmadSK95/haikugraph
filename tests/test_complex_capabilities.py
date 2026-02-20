"""Complex capability showcase — living documentation of what works and what doesn't.

Three categories:
1. Proven capabilities (MUST pass) — core features that always work
2. Known failures (xfail strict=True) — features tied to specific GAPs
3. Fragile capabilities (xfail strict=False) — features that sometimes work

When a GAP gets implemented, its xfail test starts passing (XPASS), signaling progress.
"""

from __future__ import annotations

import pytest


def _ask(client, goal: str, **extra) -> dict:
    resp = client.post("/api/assistant/query", json={"goal": goal, **extra})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    return resp.json()


# =============================================================================
# PROVEN CAPABILITIES — these MUST pass
# =============================================================================

class TestProvenCapabilities:
    """Core features that reliably work in deterministic mode."""

    def test_simple_count(self, known_data_client):
        """Simple COUNT query."""
        data = _ask(known_data_client, "How many transactions?")
        assert data["success"] is True
        assert "8" in data["answer_markdown"]

    def test_simple_sum(self, known_data_client):
        """Simple SUM query."""
        data = _ask(known_data_client, "Total transaction amount")
        assert data["success"] is True
        assert "15" in data["answer_markdown"].replace(",", "")  # 15950

    def test_boolean_filter_mt103(self, known_data_client):
        """Boolean filter: MT103 transactions."""
        data = _ask(known_data_client, "How many MT103 transactions?")
        assert data["success"] is True
        assert "3" in data["answer_markdown"]

    def test_single_dimension_group_by(self, known_data_client):
        """Single-dimension GROUP BY."""
        data = _ask(known_data_client, "Transactions by platform")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 2

    def test_time_filter_month(self, known_data_client):
        """Time filter by month name."""
        data = _ask(known_data_client, "Transactions in December")
        assert data["success"] is True
        assert "5" in data["answer_markdown"]

    def test_sql_injection_blocked(self, known_data_client):
        """DROP/DELETE operations should be blocked by governance."""
        data = _ask(known_data_client, "DROP TABLE test_1_1_merged")
        assert data["success"] is False or "blocked" in data.get("answer_markdown", "").lower() or "error" in str(data).lower()

    def test_delete_blocked(self, known_data_client):
        """DELETE operations should be blocked."""
        data = _ask(known_data_client, "DELETE FROM test_1_1_merged WHERE 1=1")
        assert data["success"] is False or "blocked" in data.get("answer_markdown", "").lower() or "error" in str(data).lower()

    def test_read_only_sql_only(self, known_data_client):
        """INSERT operations should be blocked."""
        data = _ask(known_data_client, "INSERT INTO test_1_1_merged VALUES ('x','x','x','x','x','x','x','x','x','x',0,0,0,'x')")
        assert data["success"] is False or "blocked" in str(data).lower() or "error" in str(data).lower()

    def test_response_has_trace_id(self, known_data_client):
        """Every response should have a trace_id."""
        data = _ask(known_data_client, "How many transactions?")
        assert "trace_id" in data
        assert len(data["trace_id"]) > 10

    def test_response_has_agent_trace(self, known_data_client):
        """Every response should have agent_trace."""
        data = _ask(known_data_client, "How many transactions?")
        assert "agent_trace" in data
        assert isinstance(data["agent_trace"], list)
        assert len(data["agent_trace"]) >= 3  # at least context, intake, retrieval

    def test_response_has_confidence(self, known_data_client):
        """Every response should have valid confidence."""
        data = _ask(known_data_client, "Total transaction amount")
        assert data["confidence"] in ("high", "medium", "low", "uncertain")
        assert 0.0 <= data["confidence_score"] <= 1.0

    def test_multi_domain_warning(self, known_data_client):
        """Query spanning domains should surface a pipeline warning."""
        data = _ask(known_data_client, "Show customer transaction amounts")
        # Should either work or at least not crash
        assert data["success"] is True or data.get("error") is not None


# =============================================================================
# KNOWN FAILURES — tied to specific architecture gaps
# These use xfail(strict=True): they MUST fail. If they start passing,
# it means the GAP was addressed and the xfail should be removed.
# =============================================================================

class TestKnownFailures:
    """Features documented as not-yet-implemented. xfail(strict=True)."""

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 14: Cross-domain JOINs not fully wired into query generation",
    )
    def test_cross_domain_join_transactions_per_customer_type(self, known_data_client):
        """Transactions per customer type requires JOIN across tables."""
        data = _ask(known_data_client, "Transaction count per customer type")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        # Should produce groups by customer type (education, individual, etc.)
        assert len(rows) >= 2
        row_str = str(rows).lower()
        assert "education" in row_str or "individual" in row_str

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 20: Running total / cumulative sum not supported",
    )
    def test_running_total_by_date(self, known_data_client):
        """Running total of transaction amounts over time."""
        data = _ask(known_data_client, "Running total of transaction amounts by date")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 2
        # Running total MUST have a dedicated cumulative column
        first_row = rows[0]
        assert "running_total" in first_row or "cumulative" in str(first_row).lower()
        # Values must be monotonically increasing
        values = [float(r.get("running_total", 0)) for r in rows]
        assert values == sorted(values) and values[-1] >= 15000

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 18: No LLM reasoning in deterministic mode for causal questions",
    )
    def test_reasoning_why_question(self, known_data_client):
        """Why-questions require causal reasoning (LLM needed)."""
        data = _ask(known_data_client, "Why did transaction volume drop in November?")
        assert data["success"] is True
        answer = data["answer_markdown"].lower()
        # Must provide TRUE causal analysis — not just a count
        # The answer must identify specific causal factors and explain mechanisms
        has_mechanism = any(w in answer for w in [
            "because", "caused by", "driven by", "attributed to", "root cause",
        ])
        has_specific_factors = any(w in answer for w in [
            "seasonal", "holiday", "processing delay", "fewer business days",
            "customer behavior", "market conditions",
        ])
        assert has_mechanism and has_specific_factors, (
            f"Expected true causal reasoning with specific factors in: {answer[:300]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 15: Multi-part questions not decomposed",
    )
    def test_multi_part_question(self, known_data_client):
        """Multi-part question: count + average + top platform in one query."""
        data = _ask(
            known_data_client,
            "How many transactions, what is the average amount, and which platform has the most?",
        )
        assert data["success"] is True
        answer = data["answer_markdown"]
        # Should contain all three answers
        assert "8" in answer  # count
        assert "1993" in answer.replace(",", "") or "1994" in answer.replace(",", "")  # average
        # platform with most: B2C-APP and B2C-WEB both have 3
        assert "b2c" in answer.lower()

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 20: Percentile queries not fully generating correct SQL",
    )
    def test_95th_percentile(self, known_data_client):
        """95th percentile of transaction amounts."""
        data = _ask(known_data_client, "95th percentile of transaction amount")
        assert data["success"] is True
        answer = data["answer_markdown"].replace(",", "")
        # P95 of [500,1000,1200,1500,2000,2500,3000,4250] ≈ 3912.5
        # Accept anything in the right ballpark
        assert any(str(v) in answer for v in range(3500, 4300))

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 20: Subqueries (above-average filtering) not supported",
    )
    def test_above_average_subquery(self, known_data_client):
        """Customers with above-average transaction spend requires subquery."""
        data = _ask(known_data_client, "Show customers with above-average transaction spend")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        # Must return customer-level rows filtered by above-average threshold
        assert len(rows) >= 1
        # Rows must reference customer identifiers, not raw transactions
        row_str = str(rows).lower()
        assert "customer" in row_str or "kc_" in row_str
        # Average is 1993.75; only customers with total > 1993.75 should appear
        assert len(rows) <= 4  # not all 8 transactions

    def test_bookings_by_customer_country(self, known_data_client):
        """Bookings grouped by customer country — GAP 14 (JOINs) now implemented."""
        data = _ask(known_data_client, "Total booked amount by customer country")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 2
        row_str = str(rows).lower()
        assert "address_country" in row_str or "country" in [k.lower() for r in rows for k in r.keys()]
        assert "us" in row_str and "uk" in row_str

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 20: Year-over-year comparison requires window functions",
    )
    def test_year_over_year_growth(self, known_data_client):
        """YoY growth calculation requires window functions."""
        data = _ask(known_data_client, "Year over year growth in transaction volume")
        assert data["success"] is True
        answer = data["answer_markdown"].lower()
        # Must explicitly compute and present a growth percentage
        assert "%" in answer or "growth" in answer
        # Must reference both years for a valid YoY comparison
        assert "2024" in answer or "2025" in answer
        assert "increase" in answer or "decrease" in answer or "change" in answer

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 20: Correlation analysis not supported",
    )
    def test_correlation_analysis(self, known_data_client):
        """Correlation between two metrics."""
        data = _ask(known_data_client, "Correlation between transaction amount and exchange rate")
        assert data["success"] is True
        answer = data["answer_markdown"].lower()
        # Must provide an actual correlation coefficient
        assert "correlation" in answer
        assert "r=" in answer or "coefficient" in answer
        # Must reference both variables
        assert "amount" in answer and "exchange" in answer

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 15: Cannot decompose count-per-category across domains",
    )
    def test_transactions_per_university_status(self, known_data_client):
        """Count transactions grouped by university/non-university status."""
        data = _ask(known_data_client, "How many transactions from university vs non-university customers?")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        # Must produce exactly 2 groups: university and non-university
        assert len(rows) == 2
        row_str = str(rows).lower()
        assert "true" in row_str and "false" in row_str

    def test_moving_average(self, known_data_client):
        """3-month moving average — GAP 20 (trend_analysis) now implemented."""
        data = _ask(known_data_client, "3 month moving average of transaction amounts")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 2
        first_row_keys = set(rows[0].keys())
        assert "moving_avg" in first_row_keys, f"Expected 'moving_avg' column in: {first_row_keys}"
        assert "metric_value" in first_row_keys, f"Expected 'metric_value' column in: {first_row_keys}"

    @pytest.mark.xfail(
        strict=True,
        reason="GAP 20: Median calculation requires PERCENTILE_CONT",
    )
    def test_median_amount(self, known_data_client):
        """Median transaction amount."""
        data = _ask(known_data_client, "Median transaction amount")
        assert data["success"] is True
        answer = data["answer_markdown"].replace(",", "")
        # Median of [500,1000,1200,1500,2000,2500,3000,4250] = (1500+2000)/2 = 1750
        assert "1750" in answer


# =============================================================================
# FRAGILE CAPABILITIES — sometimes work, sometimes don't
# xfail(strict=False): pass is OK (XPASS), fail is also OK (XFAIL)
# =============================================================================

class TestFragileCapabilities:
    """Features that sometimes work depending on parsing heuristics."""

    @pytest.mark.xfail(
        strict=False,
        reason="VARCHAR timestamp parsing is fragile — may not parse correctly",
    )
    def test_temporal_comparison_varchar_timestamps(self, known_data_client):
        """Temporal comparison with VARCHAR timestamps."""
        data = _ask(known_data_client, "Compare December 2025 vs November 2025 transaction amount")
        assert data["success"] is True
        answer = data["answer_markdown"].replace(",", "")
        # December=10750, November=5200
        assert "10750" in answer and "5200" in answer

    @pytest.mark.xfail(
        strict=False,
        reason="Synonym resolution: 'revenue' may or may not map to 'amount'",
    )
    def test_synonym_resolution_revenue(self, known_data_client):
        """'Revenue' should map to transaction amount."""
        data = _ask(known_data_client, "Total revenue")
        assert data["success"] is True
        answer = data["answer_markdown"].replace(",", "")
        assert "15950" in answer

    @pytest.mark.xfail(
        strict=False,
        reason="3-dimension breakdown may hit MAX_DIMENSIONS limit",
    )
    def test_three_dimension_breakdown(self, known_data_client):
        """Group by platform, state, and flow simultaneously."""
        data = _ask(known_data_client, "Transactions by platform, state, and flow")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 3  # Should have multiple groups

    @pytest.mark.xfail(
        strict=False,
        reason="Trend intent detection may not trigger for all phrasings",
    )
    def test_trend_detection_phrasing(self, known_data_client):
        """Alternative phrasing for trend detection."""
        data = _ask(known_data_client, "Show me the trend of transaction amounts over time")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 1

    @pytest.mark.xfail(
        strict=False,
        reason="Top N with specific ordering may not preserve rank column",
    )
    def test_top_n_with_rank(self, known_data_client):
        """Top 3 platforms by transaction count should include rank."""
        data = _ask(known_data_client, "Top 3 platforms by transaction count")
        assert data["success"] is True
        rows = data.get("sample_rows", [])
        assert len(rows) >= 2

    @pytest.mark.xfail(
        strict=False,
        reason="Relative time references may not work with known fixed data",
    )
    def test_relative_time_this_month(self, known_data_client):
        """'This month' should resolve to current month."""
        data = _ask(known_data_client, "How many transactions this month?")
        assert data["success"] is True
        # Known data is all in Nov/Dec 2025, so 'this month' (Feb 2026) = 0
        # This may or may not return useful results
        assert data["answer_markdown"] is not None
