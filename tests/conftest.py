"""Shared test fixtures for the haikugraph test suite.

Provides two database fixtures and two TestClient fixtures:

* ``seed_db_datada``   -- 500+ rows of realistic data across 4 tables
* ``known_data_db``    -- small, precisely-counted data for deterministic assertions
* ``client_datada``    -- FastAPI TestClient backed by ``seed_db_datada``
* ``known_data_client``-- FastAPI TestClient backed by ``known_data_db``

The fixture names are intentionally different from the local ``seed_db`` /
``client`` fixtures defined inside individual test files so that pytest's
"local fixture wins" rule is never triggered by accident.
"""

from __future__ import annotations

import random
import tempfile
import uuid
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLATFORMS = ["B2C-APP", "B2C-WEB", "B2B"]
_STATES = ["NY", "CA", "TX", "FL", "IL"]
_TXN_FLOWS = ["flow_a", "flow_b", "flow_c"]
_PAYMENT_STATUSES = ["completed", "pending", "failed", "processing"]
_SOURCE_CURRENCIES = ["USD", "EUR", "GBP", "INR", "CAD"]
_DEST_CURRENCIES = ["INR", "USD", "EUR", "GBP", "AUD"]
_CUSTOMER_TYPES = ["education", "individual", "business", "retail"]
_CUSTOMER_STATUSES = ["active", "inactive", "pending"]
_COUNTRIES = ["US", "UK", "IN", "CA", "DE"]
_DEAL_TYPES = ["spot", "forward", "option"]


def _random_timestamp(month: int, year: int = 2025) -> str:
    day = random.randint(1, 28)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all four tables that SemanticLayerManager auto-detects."""
    conn.execute("""
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
    """)
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE TABLE test_4_1 (
            payee_id VARCHAR,
            customer_id VARCHAR,
            is_university BOOLEAN,
            type VARCHAR,
            status VARCHAR,
            created_at VARCHAR,
            address_country VARCHAR
        )
    """)
    conn.execute("""
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
    """)


def _make_db_path() -> Path:
    """Create a temporary file for DuckDB and remove it so DuckDB can own it."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = Path(f.name)
    db_path.unlink()  # DuckDB needs to create the file itself
    return db_path


# ---------------------------------------------------------------------------
# Fixture 1 -- large seed data (500+ rows)
# ---------------------------------------------------------------------------

@pytest.fixture()
def seed_db_datada():
    """Create a DuckDB database with 500+ rows of realistic data.

    Table row counts (approximate):
        test_1_1_merged  ~200 transactions
        test_3_1         ~150 quotes
        test_4_1         ~100 customers
        test_5_1         ~100 bookings
    Total: ~550 rows
    """
    rng = random.Random(42)  # deterministic seed for reproducibility

    db_path = _make_db_path()
    conn = duckdb.connect(str(db_path))
    _create_schema(conn)

    # --- Customers (100 rows) -------------------------------------------
    customer_ids: list[str] = []
    payee_ids: list[str] = []
    for i in range(100):
        cid = f"cust_{i:04d}"
        pid = f"payee_{i:04d}"
        customer_ids.append(cid)
        payee_ids.append(pid)
        is_uni = rng.choice([True, False])
        ctype = rng.choice(_CUSTOMER_TYPES)
        cstatus = rng.choice(_CUSTOMER_STATUSES)
        month = rng.choice([10, 11, 12])
        ts = _random_timestamp(month)
        country = rng.choice(_COUNTRIES)
        conn.execute(
            "INSERT INTO test_4_1 VALUES (?, ?, ?, ?, ?, ?, ?)",
            [pid, cid, is_uni, ctype, cstatus, ts, country],
        )

    # --- Transactions (200 rows) ----------------------------------------
    for i in range(200):
        tid = f"txn_{i:05d}"
        cid = rng.choice(customer_ids)
        pid = rng.choice(payee_ids)
        platform = rng.choice(_PLATFORMS)
        state = rng.choice(_STATES)
        flow = rng.choice(_TXN_FLOWS)
        pstatus = rng.choice(_PAYMENT_STATUSES)

        # ~30 % get mt103
        mt103 = _random_timestamp(rng.choice([11, 12])) if rng.random() < 0.30 else None
        # ~15 % get refund
        refund = f"ref_{_uid()}" if rng.random() < 0.15 else None

        month = rng.choice([11, 12])
        ts = _random_timestamp(month)
        updated = _random_timestamp(month)
        amount = round(rng.uniform(50.0, 10000.0), 2)
        deal_amt = round(amount * rng.uniform(0.95, 1.05), 2)
        collected = round(amount * rng.uniform(0.90, 1.0), 2)

        conn.execute(
            "INSERT INTO test_1_1_merged VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [tid, cid, pid, platform, state, flow, pstatus,
             mt103, ts, updated, amount, deal_amt, collected, refund],
        )

    # --- Quotes (150 rows) ----------------------------------------------
    quote_ids: list[str] = []
    for i in range(150):
        qid = f"quote_{i:04d}"
        quote_ids.append(qid)
        cid = rng.choice(customer_ids)
        src = rng.choice(_SOURCE_CURRENCIES)
        dst = rng.choice(_DEST_CURRENCIES)
        rate = round(rng.uniform(0.5, 90.0), 4)
        total_pay = round(rng.uniform(100.0, 8000.0), 2)
        charges = round(rng.uniform(1.0, 50.0), 2)
        markup = round(rng.uniform(0.5, 10.0), 2)
        month = rng.choice([11, 12])
        ts = _random_timestamp(month)
        conn.execute(
            "INSERT INTO test_3_1 VALUES (?,?,?,?,?,?,?,?,?)",
            [qid, cid, src, dst, rate, total_pay, charges, markup, ts],
        )

    # --- Bookings (100 rows) --------------------------------------------
    for i in range(100):
        did = f"deal_{i:04d}"
        qid = rng.choice(quote_ids)
        booked = round(rng.uniform(100.0, 5000.0), 2)
        rate = round(rng.uniform(0.8, 1.5), 4)
        dtype = rng.choice(_DEAL_TYPES)
        cid = rng.choice(customer_ids)
        pid = rng.choice(payee_ids)
        month = rng.choice([11, 12])
        ts = _random_timestamp(month)
        updated = _random_timestamp(month)
        conn.execute(
            "INSERT INTO test_5_1 VALUES (?,?,?,?,?,?,?,?,?)",
            [did, qid, booked, rate, dtype, cid, pid, ts, updated],
        )

    conn.close()

    yield db_path

    db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Fixture 2 -- small, precisely known data
# ---------------------------------------------------------------------------

@pytest.fixture()
def known_data_db():
    """Create a DuckDB database with exactly-known row counts and totals.

    Transactions (8 rows) -- test_1_1_merged:
        payment_amounts: 1000, 2000, 1500, 3000, 2500, 500, 1200, 4250
        total = 15,950
        Platforms: B2C-APP (3), B2C-WEB (3), B2B (2)
        Months: Dec-2025 (5), Nov-2025 (3)
        has_mt103 (mt103_created_at IS NOT NULL): 3  (rows 0, 2, 4)
        has_refund (refund_refund_id IS NOT NULL): 2  (rows 5, 7)

    Customers (5 rows) -- test_4_1:
        is_university: 2 TRUE, 3 FALSE
        Countries: US (3), UK (1), IN (1)

    Quotes (4 rows) -- test_3_1:
        total_amount_to_be_paid: 1000, 2000, 3000, 500  => total 6500

    Bookings (3 rows) -- test_5_1:
        booked_amount: 500, 1000, 1500  => total 3000
    """
    db_path = _make_db_path()
    conn = duckdb.connect(str(db_path))
    _create_schema(conn)

    # ------------------------------------------------------------------
    # Transactions  (8 rows)
    # ------------------------------------------------------------------
    #   idx  amount  platform  month  mt103             refund
    #   0    1000    B2C-APP   Dec    2025-12-01 10:00  NULL
    #   1    2000    B2C-WEB   Dec    NULL              NULL
    #   2    1500    B2C-APP   Nov    2025-11-15 09:00  NULL
    #   3    3000    B2B       Dec    NULL              NULL
    #   4    2500    B2C-WEB   Nov    2025-11-20 14:30  NULL
    #   5    500     B2C-APP   Dec    NULL              ref_001
    #   6    1200    B2C-WEB   Nov    NULL              NULL
    #   7    4250    B2B       Dec    NULL              ref_002
    # ------------------------------------------------------------------
    transactions = [
        ("kt_001", "kc_01", "kp_01", "B2C-APP", "NY", "flow_a", "completed",
         "2025-12-01 10:00:00", "2025-12-01 08:00:00", "2025-12-01 10:30:00",
         1000.0, 1000.0, 1000.0, None),
        ("kt_002", "kc_02", "kp_02", "B2C-WEB", "CA", "flow_b", "completed",
         None, "2025-12-03 11:00:00", "2025-12-03 12:00:00",
         2000.0, 2000.0, 2000.0, None),
        ("kt_003", "kc_01", "kp_01", "B2C-APP", "TX", "flow_c", "completed",
         "2025-11-15 09:00:00", "2025-11-15 07:00:00", "2025-11-15 09:30:00",
         1500.0, 1500.0, 1500.0, None),
        ("kt_004", "kc_03", "kp_03", "B2B", "FL", "flow_a", "pending",
         None, "2025-12-05 14:00:00", "2025-12-05 14:30:00",
         3000.0, 3000.0, 3000.0, None),
        ("kt_005", "kc_02", "kp_02", "B2C-WEB", "IL", "flow_b", "completed",
         "2025-11-20 14:30:00", "2025-11-20 13:00:00", "2025-11-20 15:00:00",
         2500.0, 2500.0, 2500.0, None),
        ("kt_006", "kc_04", "kp_04", "B2C-APP", "NY", "flow_a", "completed",
         None, "2025-12-08 09:00:00", "2025-12-08 09:30:00",
         500.0, 500.0, 500.0, "ref_001"),
        ("kt_007", "kc_05", "kp_05", "B2C-WEB", "CA", "flow_c", "processing",
         None, "2025-11-25 16:00:00", "2025-11-25 16:30:00",
         1200.0, 1200.0, 1200.0, None),
        ("kt_008", "kc_03", "kp_03", "B2B", "TX", "flow_b", "completed",
         None, "2025-12-10 10:00:00", "2025-12-10 10:30:00",
         4250.0, 4250.0, 4250.0, "ref_002"),
    ]
    for row in transactions:
        conn.execute(
            "INSERT INTO test_1_1_merged VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            list(row),
        )

    # ------------------------------------------------------------------
    # Customers  (5 rows)
    # ------------------------------------------------------------------
    #   payee  customer  is_university  type        status  country
    #   kp_01  kc_01     TRUE           education   active  US
    #   kp_02  kc_02     FALSE          individual  active  US
    #   kp_03  kc_03     TRUE           education   active  UK
    #   kp_04  kc_04     FALSE          business    active  US
    #   kp_05  kc_05     FALSE          retail      active  IN
    # ------------------------------------------------------------------
    customers = [
        ("kp_01", "kc_01", True,  "education",  "active", "2025-01-10 08:00:00", "US"),
        ("kp_02", "kc_02", False, "individual", "active", "2025-02-15 09:00:00", "US"),
        ("kp_03", "kc_03", True,  "education",  "active", "2025-03-20 10:00:00", "UK"),
        ("kp_04", "kc_04", False, "business",   "active", "2025-04-25 11:00:00", "US"),
        ("kp_05", "kc_05", False, "retail",     "active", "2025-05-30 12:00:00", "IN"),
    ]
    for row in customers:
        conn.execute(
            "INSERT INTO test_4_1 VALUES (?,?,?,?,?,?,?)",
            list(row),
        )

    # ------------------------------------------------------------------
    # Quotes  (4 rows)
    # ------------------------------------------------------------------
    #   total_amount_to_be_paid: 1000 + 2000 + 3000 + 500 = 6500
    # ------------------------------------------------------------------
    quotes = [
        ("kq_01", "kc_01", "USD", "INR", 83.25,  1000.0, 12.0, 3.0, "2025-12-02 10:00:00"),
        ("kq_02", "kc_02", "EUR", "USD", 1.10,   2000.0, 8.0,  2.5, "2025-12-04 11:00:00"),
        ("kq_03", "kc_03", "GBP", "INR", 105.50, 3000.0, 15.0, 4.0, "2025-11-18 12:00:00"),
        ("kq_04", "kc_04", "USD", "EUR", 0.92,   500.0,  5.0,  1.5, "2025-12-09 09:00:00"),
    ]
    for row in quotes:
        conn.execute(
            "INSERT INTO test_3_1 VALUES (?,?,?,?,?,?,?,?,?)",
            list(row),
        )

    # ------------------------------------------------------------------
    # Bookings  (3 rows)
    # ------------------------------------------------------------------
    #   booked_amount: 500 + 1000 + 1500 = 3000
    # ------------------------------------------------------------------
    bookings = [
        ("kd_01", "kq_01", 500.0,  1.02, "spot",    "kc_01", "kp_01",
         "2025-12-03 08:00:00", "2025-12-03 08:30:00"),
        ("kd_02", "kq_02", 1000.0, 1.05, "forward", "kc_02", "kp_02",
         "2025-12-05 09:00:00", "2025-12-05 09:30:00"),
        ("kd_03", "kq_03", 1500.0, 1.08, "spot",    "kc_03", "kp_03",
         "2025-11-19 10:00:00", "2025-11-19 10:30:00"),
    ]
    for row in bookings:
        conn.execute(
            "INSERT INTO test_5_1 VALUES (?,?,?,?,?,?,?,?,?)",
            list(row),
        )

    conn.close()

    yield db_path

    db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_datada(seed_db_datada):
    """FastAPI TestClient backed by the large seed database."""
    app = create_app(db_path=seed_db_datada)
    return TestClient(app)


@pytest.fixture()
def known_data_client(known_data_db):
    """FastAPI TestClient backed by the small, precisely-known database."""
    app = create_app(db_path=known_data_db)
    return TestClient(app)
