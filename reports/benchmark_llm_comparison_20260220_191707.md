# LLM Mode Benchmark Report — 20260220_191707

## Provider Availability

| Provider | Available |
|----------|-----------|
| anthropic | Yes |
| deterministic | Yes |
| local | Yes |
| openai | Yes |

## Overall Intelligence Scores

| Mode | Correctness (50%) | Confidence (15%) | Latency (15%) | Narrative (20%) | **Composite** |
|------|-------------------|------------------|---------------|-----------------|---------------|
| deterministic | 94.44% | 94.44% | 99.56% | 100.00% | **96.32%** |
| local | 94.44% | 94.44% | 58.83% | 94.44% | **89.10%** |
| openai | 100.00% | 100.00% | 30.88% | 98.33% | **89.30%** |
| anthropic | 100.00% | 100.00% | 69.00% | 88.33% | **93.02%** |

## Accuracy by Category

| Category | deterministic | local | openai | anthropic |
|----------|------|------|------|------|
| aggregations | 100% (4/4) | 75% (3/4) | 100% (4/4) | 100% (4/4) |
| boolean_filters | 67% (2/3) | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| complex | 100% (2/2) | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| grouping | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| simple_counts | 100% (4/4) | 100% (4/4) | 100% (4/4) | 100% (4/4) |
| time_filters | 100% (2/2) | 100% (2/2) | 100% (2/2) | 100% (2/2) |

## Latency Comparison

| Mode | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | P95 (ms) |
|------|-------------|-----------|----------|----------|----------|
| deterministic | 130 | 133 | 118 | 159 | 154 |
| local | 12125 | 12350 | 10104 | 15648 | 15451 |
| openai | 20042 | 26499 | 11919 | 85686 | 49175 |
| anthropic | 9216 | 9301 | 7688 | 12587 | 12398 |

## Head-to-Head Comparison

**local vs deterministic** — Uplift: -7.22%

- deterministic_wins: 1
- local_wins: 1
- ties: 16
- deterministic_composite: 0.9632
- local_composite: 0.891

**openai vs deterministic** — Uplift: -7.02%

- deterministic_wins: 0
- openai_wins: 1
- ties: 17
- deterministic_composite: 0.9632
- openai_composite: 0.893

**anthropic vs deterministic** — Uplift: -3.30%

- deterministic_wins: 0
- anthropic_wins: 1
- ties: 17
- deterministic_composite: 0.9632
- anthropic_composite: 0.9302

**openai vs local** — Uplift: +0.20%

- local_wins: 0
- openai_wins: 1
- ties: 17
- local_composite: 0.891
- openai_composite: 0.893

**anthropic vs local** — Uplift: +3.92%

- local_wins: 0
- anthropic_wins: 1
- ties: 17
- local_composite: 0.891
- anthropic_composite: 0.9302

**anthropic vs openai** — Uplift: +3.72%

- openai_wins: 0
- anthropic_wins: 0
- ties: 18
- openai_composite: 0.893
- anthropic_composite: 0.9302

## LLM Step Activation

| Mode | Queries | Intake Used | Narrative Used | Effective |
|------|---------|-------------|----------------|-----------|
| deterministic | 18 | 0 | 0 | 0 |
| local | 18 | 18 | 17 | 18 |
| openai | 18 | 18 | 18 | 18 |
| anthropic | 18 | 18 | 18 | 18 |

## Warnings & Issues

- [local] agg_tx_avg: The query is selecting 'payment_status' which is not relevant to the goal of finding the average payment amount per transaction.
- [local] agg_tx_avg: The WHERE clause with 1=1 is unnecessary and can be removed.
- [local] time_dec_total: The WHERE clause condition `AND EXTRACT(YEAR FROM event_ts) = 2025 AND EXTRACT(MONTH FROM event_ts) = 12` is correct for December 2025, but ensure that the column `event_ts` stores dates in a format compatible with this extraction.
- [openai] bool_univ: Dimension 'customer_type' from intake is not in datada_dim_customers schema; removed.
- [openai] bool_univ: The query does not filter for universities specifically, which may lead to inaccurate results.
- [openai] bool_univ: Using '1=1' in the WHERE clause is unnecessary and does not add value.
- [openai] time_dec_cnt: The query is asking for data from December 2025, which is in the future and may not be available in the database.
- [anthropic] cnt_tx: Query uses DISTINCT on transaction_key which may be unnecessary if transaction_key is already a primary key
- [anthropic] cnt_tx: WHERE 1=1 is a code smell suggesting dynamic query building - verify no filters were accidentally omitted
- [anthropic] cnt_tx: No time period specified - ensure this captures the intended scope (all-time vs. specific period)
- [anthropic] cnt_cust: Query uses 'WHERE 1=1' which is a code smell - suggests dynamic query building that may leave room for SQL injection if not properly parameterized
- [anthropic] cnt_cust: No temporal filter applied - result includes all customers regardless of status or date, which may not reflect 'active' customers if that's the intent
- [anthropic] cnt_quote: Query uses 'datada_mart_quotes' table name which appears to contain a typo ('datada' instead of 'data'). Verify table name is correct.
- [anthropic] cnt_quote: WHERE 1=1 clause is unnecessary and suggests auto-generated or templated SQL. Consider removing for clarity.
- [anthropic] cnt_book: Query uses COUNT(DISTINCT booking_key) which may be slower than COUNT(*) if booking_key has no nulls
- [anthropic] cnt_book: WHERE 1=1 is unnecessary and suggests auto-generated or templated SQL
- [anthropic] cnt_book: No time period specified - result represents all-time bookings with no temporal context
- [anthropic] agg_tx_total: WHERE 1=1 clause is unnecessary and suggests auto-generated or templated SQL
- [anthropic] agg_tx_total: No date range filtering applied - result includes all historical transactions which may not be the intended scope
- [anthropic] agg_tx_total: Consider verifying if this should be filtered by transaction status (e.g., completed, not cancelled)
- [anthropic] agg_tx_avg: WHERE 1=1 clause is redundant and suggests auto-generated or templated SQL
- [anthropic] agg_tx_avg: No time period filtering applied - result represents all historical data without context
- [anthropic] agg_tx_avg: Missing transaction count or volume metrics to contextualize the average
- [anthropic] agg_tx_avg: No data quality checks (e.g., filtering out zero/negative amounts or outliers)
- [anthropic] bool_refund: Query uses SUM(CASE...) instead of COUNT(CASE...) - while functionally equivalent here, COUNT is more idiomatic for counting rows
- [anthropic] bool_refund: WHERE 1=1 is unnecessary and suggests auto-generated or templated SQL - consider removing for clarity
- [anthropic] bool_univ: Dimension 'customer_type' from intake is not in datada_dim_customers schema; removed.
- [anthropic] bool_univ: Query does not filter for universities - WHERE clause contains only '1=1' placeholder
- [anthropic] bool_univ: Goal asks for university customers but no customer_type, industry, or organization_type filter is applied
- [anthropic] bool_univ: Query will return total customer count, not university customer count
- [anthropic] bool_univ: Missing critical business logic to distinguish universities from other customer types
- [anthropic] grp_country: Only 3 rows returned despite LIMIT 20 - verify data completeness
- [anthropic] grp_country: NULLS LAST clause may hide null country values - consider explicit NULL handling
- [anthropic] grp_country: No date filtering applied - ensure this reflects current customer state
- [anthropic] grp_deal: Only 2 rows returned - consider if deal_type has limited cardinality or if filtering is too restrictive
- [anthropic] grp_deal: LIMIT 20 applied but only 2 results suggests data may be sparse or heavily filtered
- [anthropic] time_dec_cnt: Query uses EXTRACT functions which may have performance implications on large datasets; consider using date range predicates for better index utilization
- [anthropic] time_dec_cnt: COUNT(DISTINCT transaction_key) assumes transaction_key is the appropriate unique identifier; verify this matches business definition of 'transactions'
- [anthropic] time_dec_total: Query filters for future date (December 2025) - verify this is intentional and data exists for this period
- [anthropic] time_dec_total: SUM aggregation with single row result - confirm no NULL amounts are being excluded unintentionally
- [anthropic] cplx_amt_plat: Dimension 'platform' from intake is not in datada_mart_transactions schema; removed.
- [anthropic] cplx_amt_plat: Query uses NULLS LAST clause which may not be supported in all SQL dialects
- [anthropic] cplx_amt_plat: LIMIT 20 applied but only 3 rows returned - consider if this is expected behavior
- [anthropic] cplx_amt_plat: No date filtering applied - ensure this captures the intended time period for payment analysis
- [anthropic] cplx_mt103_plat: Query uses LIMIT 20 but only returns 3 rows - consider if filtering is too restrictive
- [anthropic] cplx_mt103_plat: No date range filter applied - ensure MT103 counts represent intended time period
- [anthropic] cplx_mt103_plat: NULLS LAST ordering may hide platforms with NULL names - verify data quality

---
	*Generated by test_benchmark_llm_modes.py*