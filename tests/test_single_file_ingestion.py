"""Regression tests for the single-file / partial-file ingestion recursion bug.

Verifies that SemanticLayerManager.prepare() succeeds without infinite
recursion when only a subset of source tables is present in the database.

Bug: when only one file is ingested (e.g. transactions), the empty-stub
objects for missing domains (e.g. datada_dim_customers) were created as
VIEWs.  On a second prepare() call, _detect_source_tables() matched those
stubs as source tables, producing self-referential CREATE OR REPLACE VIEW
statements that DuckDB rejected with
  "Binder Error: infinite recursion detected".

The fix:
  1. Drop existing datada_* objects before detection (clean slate).
  2. Filter datada_* names from the information_schema scan.
  3. Use empty TABLEs instead of VIEWs for missing-domain stubs.
  4. Post-creation validation query on every mart.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pytest

from haikugraph.poc.agentic_team import SemanticLayerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tables: dict[str, str]) -> Path:
    """Create a temp DuckDB file with the given CREATE TABLE + INSERT DDL.

    *tables* maps a human label to a SQL string that is executed verbatim.
    Returns the path; caller is responsible for cleanup.
    """
    fd = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
    db_path = Path(fd.name)
    fd.close()
    db_path.unlink()  # DuckDB creates it fresh

    conn = duckdb.connect(str(db_path))
    for ddl in tables.values():
        conn.execute(ddl)
    conn.close()
    return db_path


_TXN_DDL = """
CREATE TABLE raw_transactions (
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
);
INSERT INTO raw_transactions VALUES
  ('t1','c1','p1','B2C','NY','flow_a','completed',NULL,'2025-12-01','2025-12-01',1000,1000,1000,NULL),
  ('t2','c2','p2','B2B','CA','flow_b','pending',NULL,'2025-12-02','2025-12-02',2000,2000,2000,NULL);
"""

_CUST_DDL = """
CREATE TABLE raw_customers (
    payee_id VARCHAR,
    customer_id VARCHAR,
    is_university BOOLEAN,
    type VARCHAR,
    status VARCHAR,
    created_at VARCHAR,
    address_country VARCHAR
);
INSERT INTO raw_customers VALUES
  ('p1','c1',TRUE,'education','active','2025-01-01','US'),
  ('p2','c2',FALSE,'business','active','2025-02-01','UK');
"""

_QUOTES_DDL = """
CREATE TABLE raw_quotes (
    quote_id VARCHAR,
    customer_id VARCHAR,
    source_currency VARCHAR,
    destination_currency VARCHAR,
    exchange_rate DOUBLE,
    total_amount_to_be_paid DOUBLE,
    total_additional_charges DOUBLE,
    forex_markup DOUBLE,
    created_at VARCHAR
);
INSERT INTO raw_quotes VALUES
  ('q1','c1','USD','INR',83.0,1000,10,3,'2025-12-01');
"""

_BOOKINGS_DDL = """
CREATE TABLE raw_bookings (
    deal_id VARCHAR,
    quote_id VARCHAR,
    booked_amount DOUBLE,
    rate DOUBLE,
    deal_type VARCHAR,
    customer_id VARCHAR,
    payee_id VARCHAR,
    created_at VARCHAR,
    updated_at VARCHAR
);
INSERT INTO raw_bookings VALUES
  ('d1','q1',500,1.02,'spot','c1','p1','2025-12-01','2025-12-01');
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleFileIngestion:
    """Only one source table in the database; all other domains are stubs."""

    def test_transactions_only(self, tmp_path):
        db = _make_db({"txn": _TXN_DDL})
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            assert "datada_mart_transactions" in cat["marts"]
            # Empty stubs must be queryable
            conn = duckdb.connect(str(db), read_only=True)
            assert conn.execute("SELECT COUNT(*) FROM datada_dim_customers").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM datada_mart_quotes").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM datada_mart_bookings").fetchone()[0] == 0
            conn.close()
        finally:
            db.unlink(missing_ok=True)

    def test_customers_only(self, tmp_path):
        db = _make_db({"cust": _CUST_DDL})
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            assert "datada_dim_customers" in cat["marts"]
            conn = duckdb.connect(str(db), read_only=True)
            assert conn.execute("SELECT COUNT(*) FROM datada_dim_customers").fetchone()[0] == 2
            conn.close()
        finally:
            db.unlink(missing_ok=True)

    def test_quotes_only(self, tmp_path):
        db = _make_db({"quotes": _QUOTES_DDL})
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            assert "datada_mart_quotes" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)

    def test_bookings_only(self, tmp_path):
        db = _make_db({"bookings": _BOOKINGS_DDL})
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            assert "datada_mart_bookings" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)


class TestRepeatPrepareNoRecursion:
    """Calling prepare() twice must not trigger infinite recursion.

    This is the exact scenario that caused the original bug: the second
    prepare() re-discovers the stub objects from the first run.
    """

    def test_double_prepare_transactions_only(self, tmp_path):
        db = _make_db({"txn": _TXN_DDL})
        try:
            slm = SemanticLayerManager(db)
            slm.prepare(force=True)
            # Second prepare must succeed (was raising BinderException before fix)
            cat = slm.prepare(force=True)
            assert "datada_mart_transactions" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)

    def test_double_prepare_customers_only(self, tmp_path):
        db = _make_db({"cust": _CUST_DDL})
        try:
            slm = SemanticLayerManager(db)
            slm.prepare(force=True)
            cat = slm.prepare(force=True)
            assert "datada_dim_customers" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)

    def test_triple_prepare_single_file(self, tmp_path):
        """Three consecutive prepares should all succeed."""
        db = _make_db({"txn": _TXN_DDL})
        try:
            slm = SemanticLayerManager(db)
            for _ in range(3):
                cat = slm.prepare(force=True)
            assert "datada_mart_transactions" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)


class TestPartialFileIngestion:
    """Two of four source tables present."""

    def test_transactions_and_customers(self, tmp_path):
        db = _make_db({"txn": _TXN_DDL, "cust": _CUST_DDL})
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            assert "datada_mart_transactions" in cat["marts"]
            assert "datada_dim_customers" in cat["marts"]
            # The transactions view should JOIN on customers successfully
            conn = duckdb.connect(str(db), read_only=True)
            rows = conn.execute("SELECT COUNT(*) FROM datada_mart_transactions").fetchone()[0]
            assert rows == 2
            conn.close()
        finally:
            db.unlink(missing_ok=True)

    def test_transactions_and_quotes(self, tmp_path):
        db = _make_db({"txn": _TXN_DDL, "quotes": _QUOTES_DDL})
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            assert "datada_mart_transactions" in cat["marts"]
            assert "datada_mart_quotes" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)

    def test_double_prepare_partial(self, tmp_path):
        """Repeat prepare with partial data must not recurse."""
        db = _make_db({"txn": _TXN_DDL, "cust": _CUST_DDL})
        try:
            slm = SemanticLayerManager(db)
            slm.prepare(force=True)
            cat = slm.prepare(force=True)
            assert "datada_mart_transactions" in cat["marts"]
            assert "datada_dim_customers" in cat["marts"]
        finally:
            db.unlink(missing_ok=True)


class TestFullFileIngestion:
    """All four source tables present -- the happy path."""

    def test_all_four_tables(self, tmp_path):
        db = _make_db({
            "txn": _TXN_DDL,
            "cust": _CUST_DDL,
            "quotes": _QUOTES_DDL,
            "bookings": _BOOKINGS_DDL,
        })
        try:
            slm = SemanticLayerManager(db)
            cat = slm.prepare()
            for mart in [
                "datada_mart_transactions",
                "datada_mart_quotes",
                "datada_dim_customers",
                "datada_mart_bookings",
            ]:
                assert mart in cat["marts"], f"{mart} missing from catalog"
        finally:
            db.unlink(missing_ok=True)

    def test_double_prepare_full(self, tmp_path):
        db = _make_db({
            "txn": _TXN_DDL,
            "cust": _CUST_DDL,
            "quotes": _QUOTES_DDL,
            "bookings": _BOOKINGS_DDL,
        })
        try:
            slm = SemanticLayerManager(db)
            slm.prepare(force=True)
            cat = slm.prepare(force=True)
            for mart in [
                "datada_mart_transactions",
                "datada_mart_quotes",
                "datada_dim_customers",
                "datada_mart_bookings",
            ]:
                assert mart in cat["marts"]
        finally:
            db.unlink(missing_ok=True)


class TestStubsAreTablesNotViews:
    """Verify that empty stubs are created as TABLEs, not VIEWs."""

    def test_stub_is_table(self, tmp_path):
        db = _make_db({"txn": _TXN_DDL})
        try:
            slm = SemanticLayerManager(db)
            slm.prepare()
            conn = duckdb.connect(str(db), read_only=True)
            # datada_dim_customers should be a BASE TABLE, not a VIEW
            result = conn.execute(
                "SELECT table_type FROM information_schema.tables "
                "WHERE table_name = 'datada_dim_customers'"
            ).fetchone()
            assert result is not None
            assert result[0] == "BASE TABLE", (
                f"Expected BASE TABLE, got {result[0]}. "
                "Empty stubs must be tables to prevent self-referential recursion."
            )
            conn.close()
        finally:
            db.unlink(missing_ok=True)


class TestDetectSourceTablesFiltering:
    """_detect_source_tables must never return a datada_* managed object."""

    def test_datada_objects_excluded(self, tmp_path):
        db = _make_db({"txn": _TXN_DDL})
        try:
            slm = SemanticLayerManager(db)
            slm.prepare()
            # Now call prepare again with force -- _detect_source_tables runs
            # and must not pick up any datada_* objects
            conn = duckdb.connect(str(db), read_only=False)
            # Drop the mart objects first, then detect
            SemanticLayerManager._drop_marts(conn)
            source = slm._detect_source_tables(conn)
            conn.close()
            for domain, table_name in source.items():
                if table_name:
                    assert not table_name.startswith("datada_"), (
                        f"Domain '{domain}' mapped to managed object '{table_name}'"
                    )
        finally:
            db.unlink(missing_ok=True)
