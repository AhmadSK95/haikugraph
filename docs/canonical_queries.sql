-- Canonical reference queries for dataDa accuracy validation.
-- These are ground-truth SQL definitions used by scripts/run_accuracy_audit.py.

-- 1) How many transactions are there?
SELECT COUNT(DISTINCT transaction_key) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1;

-- 2) Total amount of transactions in December 2025
SELECT SUM(amount) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
  AND event_ts IS NOT NULL
  AND EXTRACT(YEAR FROM event_ts) = 2025
  AND EXTRACT(MONTH FROM event_ts) = 12;

-- 3) Top 5 platforms by transaction count in December 2025
SELECT platform_name AS dimension, COUNT(DISTINCT transaction_key) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
  AND event_ts IS NOT NULL
  AND EXTRACT(YEAR FROM event_ts) = 2025
  AND EXTRACT(MONTH FROM event_ts) = 12
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 5;

-- 4) Refund count in December 2025
SELECT SUM(CASE WHEN has_refund THEN 1 ELSE 0 END) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
  AND event_ts IS NOT NULL
  AND EXTRACT(YEAR FROM event_ts) = 2025
  AND EXTRACT(MONTH FROM event_ts) = 12;

-- 5) Refund rate by platform
SELECT platform_name AS dimension, AVG(CASE WHEN has_refund THEN 1.0 ELSE 0.0 END) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 6) MT103 rate by platform
SELECT platform_name AS dimension, AVG(CASE WHEN has_mt103 THEN 1.0 ELSE 0.0 END) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 7) Unique customers from transactions
SELECT COUNT(DISTINCT customer_id) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1;

-- 8) Transaction count by flow
SELECT txn_flow AS dimension, COUNT(DISTINCT transaction_key) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 9) Compare transaction count (Dec 2025 vs Nov 2025)
SELECT 'current' AS period, COUNT(DISTINCT transaction_key) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
  AND event_ts IS NOT NULL
  AND EXTRACT(YEAR FROM event_ts) = 2025
  AND EXTRACT(MONTH FROM event_ts) = 12
UNION
SELECT 'comparison' AS period, COUNT(DISTINCT transaction_key) AS metric_value
FROM datada_mart_transactions
WHERE 1 = 1
  AND event_ts IS NOT NULL
  AND EXTRACT(YEAR FROM event_ts) = 2025
  AND EXTRACT(MONTH FROM event_ts) = 11;

-- 10) Quote count
SELECT COUNT(DISTINCT quote_key) AS metric_value
FROM datada_mart_quotes
WHERE 1 = 1;

-- 11) Quote volume by source currency
SELECT from_currency AS dimension, COUNT(DISTINCT quote_key) AS metric_value
FROM datada_mart_quotes
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 12) Total quote value
SELECT SUM(total_amount_to_be_paid) AS metric_value
FROM datada_mart_quotes
WHERE 1 = 1;

-- 13) Average quote value by source currency
SELECT from_currency AS dimension, AVG(total_amount_to_be_paid) AS metric_value
FROM datada_mart_quotes
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 14) Customer count
SELECT COUNT(DISTINCT customer_key) AS metric_value
FROM datada_dim_customers
WHERE 1 = 1;

-- 15) Customer count by country
SELECT address_country AS dimension, COUNT(DISTINCT customer_key) AS metric_value
FROM datada_dim_customers
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 16) University count by country
SELECT address_country AS dimension, SUM(CASE WHEN is_university THEN 1 ELSE 0 END) AS metric_value
FROM datada_dim_customers
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 17) Booking count
SELECT COUNT(DISTINCT booking_key) AS metric_value
FROM datada_mart_bookings
WHERE 1 = 1;

-- 18) Total booked amount by currency
SELECT currency AS dimension, SUM(booked_amount) AS metric_value
FROM datada_mart_bookings
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;

-- 19) Average booking rate by deal type
SELECT deal_type AS dimension, AVG(rate) AS metric_value
FROM datada_mart_bookings
WHERE 1 = 1
GROUP BY 1
ORDER BY 2 DESC NULLS LAST, 1 ASC
LIMIT 20;
