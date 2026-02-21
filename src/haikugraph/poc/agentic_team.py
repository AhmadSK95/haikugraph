"""Agentic analytics team runtime for dataDa.

This module implements a hierarchical, role-based multi-agent workflow that
simulates a compact analytics and data-engineering team.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from haikugraph.agents.contracts import (
    AssistantQueryResponse,
    ConfidenceLevel,
    EvidenceItem,
    SanityCheck,
)
from haikugraph.agents.stats_agent import run_stats_analysis
from haikugraph.llm.router import call_llm
import yaml

from haikugraph.poc.autonomy import AutonomyConfig, AgentMemoryStore
from haikugraph.sql.safe_executor import SafeSQLExecutor


MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MAX_DIMENSIONS = 3

DOMAIN_TO_MART: dict[str, str] = {
    "transactions": "datada_mart_transactions",
    "quotes": "datada_mart_quotes",
    "customers": "datada_dim_customers",
    "bookings": "datada_mart_bookings",
    "documents": "datada_document_chunks",
}

# ── GAP 14: Well-defined JOIN paths between known marts ──────────────────
JOIN_PATHS: dict[tuple[str, str], dict[str, Any]] = {
    ("datada_mart_transactions", "datada_dim_customers"): {
        "type": "LEFT JOIN",
        "on": "t.customer_id = c.customer_id",
        "aliases": ("t", "c"),
    },
    ("datada_dim_customers", "datada_mart_transactions"): {
        "type": "LEFT JOIN",
        "on": "c.customer_id = t.customer_id",
        "aliases": ("c", "t"),
    },
    ("datada_mart_transactions", "datada_mart_quotes"): {
        "type": "LEFT JOIN",
        "on": "t.quote_id = q.quote_id",
        "aliases": ("t", "q"),
    },
    ("datada_mart_quotes", "datada_mart_transactions"): {
        "type": "LEFT JOIN",
        "on": "q.quote_id = t.quote_id",
        "aliases": ("q", "t"),
    },
    ("datada_mart_quotes", "datada_dim_customers"): {
        "type": "LEFT JOIN",
        "on": "q.customer_id = c.customer_id",
        "aliases": ("q", "c"),
    },
    ("datada_dim_customers", "datada_mart_quotes"): {
        "type": "LEFT JOIN",
        "on": "c.customer_id = q.customer_id",
        "aliases": ("c", "q"),
    },
    ("datada_mart_bookings", "datada_dim_customers"): {
        "type": "LEFT JOIN",
        "on": "b.customer_id = c.customer_id",
        "aliases": ("b", "c"),
    },
    ("datada_dim_customers", "datada_mart_bookings"): {
        "type": "LEFT JOIN",
        "on": "c.customer_id = b.customer_id",
        "aliases": ("c", "b"),
    },
}


# ── GAP 35: Built-in domain knowledge (fallback when YAML not available) ──
_BUILTIN_DOMAIN_KNOWLEDGE: dict[str, Any] = {
    "version": "1.0",
    "domains": {
        "transactions": {
            "description": "Financial transactions including wire transfers, MT103, refunds",
            "primary_table": "datada_mart_transactions",
            "key_column": "transaction_key",
        },
        "customers": {
            "description": "Customer/payee records, beneficiaries, account holders",
            "primary_table": "datada_dim_customers",
            "key_column": "customer_key",
        },
        "quotes": {
            "description": "Currency exchange quotes, forex rates, markup",
            "primary_table": "datada_mart_quotes",
            "key_column": "quote_key",
        },
        "bookings": {
            "description": "Booked deals, confirmed currency exchanges",
            "primary_table": "datada_mart_bookings",
            "key_column": "booking_key",
        },
    },
    "synonyms": {
        "user": "customers",
        "users": "customers",
        "client": "customers",
        "clients": "customers",
        "sender": "customers",
        "receiver": "customers",
        "payer": "customers",
        "person": "customers",
        "people": "customers",
        "payment": "transactions",
        "payments": "transactions",
        "transfer": "transactions",
        "transfers": "transactions",
        "remittance": "transactions",
        "wire": "transactions",
        "transaction": "transactions",
        "transactions": "transactions",
        "fx": "quotes",
        "rate": "quotes",
        "deal": "bookings",
        "deals": "bookings",
    },
    "relationships": [
        {"parent": "customers", "child": "transactions", "join_key": "customer_id"},
        {"parent": "customers", "child": "quotes", "join_key": "customer_id"},
        {"parent": "quotes", "child": "bookings", "join_key": "quote_id"},
    ],
    "business_rules": {
        "unique_intent": {
            "triggers": ["unique", "distinct", "different", "individual"],
            "action": "force_count_distinct",
            "entity_key_map": {
                "customers": "customer_id",
                "transactions": "transaction_key",
                "quotes": "quote_key",
                "bookings": "booking_key",
            },
        },
        "successful_mt103": {
            "triggers": ["successful mt103", "completed mt103"],
            "action": "add_filter",
            "filter": {"column": "has_mt103", "value": "true"},
        },
    },
}


def _load_domain_knowledge() -> dict[str, Any]:
    """Load domain knowledge from YAML file, falling back to built-in dict."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "domain_knowledge.yaml",
        Path("data") / "domain_knowledge.yaml",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "domains" in data:
                    return data
            except Exception:
                pass
    return dict(_BUILTIN_DOMAIN_KNOWLEDGE)


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _compact(value: Any, *, max_len: int = 160) -> Any:
    if isinstance(value, dict):
        data = {k: value[k] for k in list(value)[:8]}
        text = json.dumps(data, default=str)
        return text[:max_len]
    if isinstance(value, list):
        text = json.dumps(value[:4], default=str)
        return text[:max_len]
    text = str(value)
    return text[:max_len]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _fmt_number(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return str(value)
    if abs(num) >= 1000 or num.is_integer():
        return f"{num:,.0f}" if num.is_integer() else f"{num:,.2f}"
    return f"{num:.4g}"


def _semantic_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


@dataclass
class RuntimeSelection:
    requested_mode: str
    mode: str
    use_llm: bool
    provider: str | None
    reason: str
    intent_model: str | None = None
    narrator_model: str | None = None
    fallback_warning: str | None = None


class SemanticLayerManager:
    """Builds and maintains typed semantic marts for agent use."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._prepared_mtime: float | None = None
        self._catalog: dict[str, Any] | None = None
        self._lock = threading.RLock()

    # Mart names managed by the semantic layer -- used to avoid
    # self-referential view creation and to clean up before rebuilds.
    _MART_NAMES = frozenset({
        "datada_mart_transactions",
        "datada_mart_quotes",
        "datada_dim_customers",
        "datada_mart_bookings",
    })

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")

            mtime = self.db_path.stat().st_mtime
            if not force and self._catalog is not None and self._prepared_mtime == mtime:
                return self._catalog

            conn = duckdb.connect(str(self.db_path), read_only=False)
            try:
                # Drop existing mart objects to prevent self-referential
                # views when _detect_source_tables re-discovers them.
                self._drop_marts(conn)

                source_tables = self._detect_source_tables(conn)
                self._build_customers_view(conn, source_tables.get("customers"))
                self._build_transactions_view(conn, source_tables.get("transactions"))
                self._build_quotes_view(conn, source_tables.get("quotes"))
                self._build_bookings_view(conn, source_tables.get("bookings"))

                # Validate that all mart objects can be queried without
                # recursion errors before building the catalog.
                self._validate_marts(conn)

                catalog = self._build_catalog(conn, source_tables)
            finally:
                conn.close()

            self._catalog = catalog
            self._prepared_mtime = mtime
            return catalog

    @staticmethod
    def _drop_marts(conn: duckdb.DuckDBPyConnection) -> None:
        """Drop all managed mart/dim objects regardless of their type."""
        existing = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT table_name, table_type FROM information_schema.tables "
                "WHERE table_schema='main' AND table_name LIKE 'datada_%'"
            ).fetchall()
        }
        for mart in SemanticLayerManager._MART_NAMES:
            if mart not in existing:
                continue
            if existing[mart] == "VIEW":
                conn.execute(f"DROP VIEW {_q(mart)}")
            else:
                conn.execute(f"DROP TABLE {_q(mart)}")

    def _detect_source_tables(self, conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
        tables = [
            t
            for (t,) in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            if not t.startswith("datada_")  # exclude managed mart/dim objects
        ]
        columns_by_table: dict[str, set[str]] = {}
        for table in tables:
            cols = {
                c
                for (c,) in conn.execute(
                    f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'"
                ).fetchall()
            }
            columns_by_table[table] = cols

        def find_table(*required: str) -> str | None:
            required_set = set(required)
            for table in tables:
                if required_set.issubset(columns_by_table[table]):
                    return table
            return None

        transactions = find_table("transaction_id")
        quotes = find_table("quote_id")
        customers = find_table("payee_id", "customer_id", "is_university")
        bookings = find_table("booked_amount", "deal_id")

        # Reasonable fallback by row size / naming if exact signatures do not exist.
        if transactions is None and tables:
            transactions = max(tables, key=lambda t: self._table_count(conn, t))
        if quotes is None:
            quotes = next((t for t in tables if "3" in t), None)
        if customers is None:
            customers = next((t for t in tables if "4" in t), None)
        if bookings is None:
            bookings = next((t for t in tables if "5" in t), None)

        return {
            "transactions": transactions or "",
            "quotes": quotes or "",
            "customers": customers or "",
            "bookings": bookings or "",
        }

    def _table_count(self, conn: duckdb.DuckDBPyConnection, table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) FROM {_q(table)}").fetchone()[0])

    def _table_cols(self, conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
        if not table:
            return set()
        return {
            c
            for (c,) in conn.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'"
            ).fetchall()
        }

    def _txt(self, cols: set[str], source_col: str, alias: str | None = None) -> str:
        out_alias = alias or source_col
        if source_col in cols:
            return (
                f"NULLIF(TRIM(CAST({_q(source_col)} AS VARCHAR)), '') AS {_q(out_alias)}"
            )
        return f"NULL::VARCHAR AS {_q(out_alias)}"

    def _num(self, cols: set[str], source_col: str, alias: str) -> str:
        if source_col in cols:
            return (
                "TRY_CAST(REGEXP_REPLACE(NULLIF(TRIM(CAST({col} AS VARCHAR)), ''), '[^0-9\\\\.-]', '', 'g') "
                "AS DOUBLE) AS {alias}"
            ).format(col=_q(source_col), alias=_q(alias))
        return f"NULL::DOUBLE AS {_q(alias)}"

    def _ts(self, cols: set[str], source_col: str, alias: str) -> str:
        if source_col in cols:
            return (
                "COALESCE("
                "TRY_CAST(NULLIF(TRIM(CAST({col} AS VARCHAR)), '') AS TIMESTAMP),"
                "TRY_STRPTIME(NULLIF(TRIM(CAST({col} AS VARCHAR)), ''), '%Y-%m-%dT%H:%M:%S.%fZ')"
                ") AS {alias}"
            ).format(col=_q(source_col), alias=_q(alias))
        return f"NULL::TIMESTAMP AS {_q(alias)}"

    def _build_transactions_view(self, conn: duckdb.DuckDBPyConnection, table: str | None) -> None:
        if not table:
            conn.execute(
                """
                CREATE TABLE datada_mart_transactions (
                    transaction_key VARCHAR,
                    transaction_id VARCHAR,
                    customer_id VARCHAR,
                    payee_id VARCHAR,
                    platform_name VARCHAR,
                    state VARCHAR,
                    txn_flow VARCHAR,
                    payment_status VARCHAR,
                    account_details_status VARCHAR,
                    deal_details_status VARCHAR,
                    quote_status VARCHAR,
                    event_ts TIMESTAMP,
                    mt103_created_ts TIMESTAMP,
                    payment_created_ts TIMESTAMP,
                    created_ts TIMESTAMP,
                    updated_ts TIMESTAMP,
                    payment_amount DOUBLE,
                    deal_amount DOUBLE,
                    amount_collected DOUBLE,
                    amount DOUBLE,
                    has_mt103 BOOLEAN,
                    has_refund BOOLEAN,
                    address_country VARCHAR,
                    address_state VARCHAR,
                    customer_type VARCHAR,
                    is_university BOOLEAN
                )
                """
            )
            return

        cols = self._table_cols(conn, table)
        sql = f"""
        CREATE OR REPLACE VIEW datada_mart_transactions AS
        WITH base AS (
            SELECT
                {self._txt(cols, 'transaction_id')},
                {self._txt(cols, 'customer_id')},
                {self._txt(cols, 'payee_id')},
                {self._txt(cols, 'platform_name')},
                {self._txt(cols, 'state')},
                {self._txt(cols, 'txn_flow')},
                {self._txt(cols, 'payment_status')},
                {self._txt(cols, 'account_details_status')},
                {self._txt(cols, 'deal_details_status')},
                {self._txt(cols, 'quote_status')},
                {self._txt(cols, 'mt103_created_at', 'mt103_created_at_raw')},
                {self._txt(cols, 'refund_refund_id', 'refund_id_raw')},
                {self._ts(cols, 'mt103_created_at', 'mt103_created_ts')},
                {self._ts(cols, 'payment_created_at', 'payment_created_ts')},
                {self._ts(cols, 'created_at', 'created_ts')},
                {self._ts(cols, 'updated_at', 'updated_ts')},
                {self._num(cols, 'payment_amount', 'payment_amount_num')},
                {self._num(cols, 'deal_details_amount', 'deal_amount_num')},
                {self._num(cols, 'amount_collected', 'amount_collected_num')}
            FROM {_q(table)}
        )
        SELECT
            COALESCE(b.transaction_id, CONCAT('row_', CAST(ROW_NUMBER() OVER () AS VARCHAR))) AS transaction_key,
            b.transaction_id,
            b.customer_id,
            b.payee_id,
            b.platform_name,
            b.state,
            b.txn_flow,
            b.payment_status,
            b.account_details_status,
            b.deal_details_status,
            b.quote_status,
            COALESCE(b.payment_created_ts, b.created_ts, b.updated_ts) AS event_ts,
            b.mt103_created_ts,
            b.payment_created_ts,
            b.created_ts,
            b.updated_ts,
            COALESCE(b.payment_amount_num, 0.0) AS payment_amount,
            COALESCE(b.deal_amount_num, 0.0) AS deal_amount,
            COALESCE(b.amount_collected_num, 0.0) AS amount_collected,
            COALESCE(b.payment_amount_num, b.deal_amount_num, b.amount_collected_num, 0.0) AS amount,
            CASE WHEN b.mt103_created_ts IS NULL THEN FALSE ELSE TRUE END AS has_mt103,
            CASE WHEN refund_id_raw IS NULL THEN FALSE ELSE TRUE END AS has_refund,
            c.address_country,
            c.address_state,
            c.type AS customer_type,
            c.is_university
        FROM base b
        LEFT JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY customer_key) AS _dedup_rn
            FROM datada_dim_customers
        ) c ON b.customer_id = c.customer_id AND c._dedup_rn = 1
        """
        conn.execute(sql)

    def _build_quotes_view(self, conn: duckdb.DuckDBPyConnection, table: str | None) -> None:
        if not table:
            conn.execute(
                """
                CREATE TABLE datada_mart_quotes (
                    quote_key VARCHAR,
                    quote_id VARCHAR,
                    ref_id VARCHAR,
                    customer_id VARCHAR,
                    status VARCHAR,
                    from_currency VARCHAR,
                    to_currency VARCHAR,
                    purpose_code VARCHAR,
                    transaction_range VARCHAR,
                    created_ts TIMESTAMP,
                    updated_ts TIMESTAMP,
                    amount_at_source DOUBLE,
                    amount_at_destination DOUBLE,
                    total_amount_to_be_paid DOUBLE,
                    exchange_rate DOUBLE,
                    total_additional_charges DOUBLE,
                    forex_markup DOUBLE,
                    amount_without_markup DOUBLE,
                    swift_charges DOUBLE,
                    platform_charges DOUBLE
                )
                """
            )
            return

        cols = self._table_cols(conn, table)
        sql = f"""
        CREATE OR REPLACE VIEW datada_mart_quotes AS
        WITH base AS (
            SELECT
                {self._txt(cols, 'quote_id')},
                {self._txt(cols, 'ref_id')},
                {self._txt(cols, 'customer_id')},
                {self._txt(cols, 'status')},
                {self._txt(cols, 'from_currency')},
                {self._txt(cols, 'to_currency')},
                {self._txt(cols, 'purpose_code')},
                {self._txt(cols, 'transaction_range')},
                {self._ts(cols, 'created_at', 'created_ts')},
                {self._ts(cols, 'updated_at', 'updated_ts')},
                {self._num(cols, 'amount_at_source', 'amount_at_source_num')},
                {self._num(cols, 'amount_at_destination', 'amount_at_destination_num')},
                {self._num(cols, 'total_amount_to_be_paid', 'total_amount_to_be_paid_num')},
                {self._num(cols, 'exchange_rate', 'exchange_rate_num')},
                {self._num(cols, 'total_additional_charges', 'total_additional_charges_num')},
                {self._num(cols, 'forex_markup', 'forex_markup_num')},
                {self._num(cols, 'amount_without_markup', 'amount_without_markup_num')},
                {self._num(cols, 'swift_charges', 'swift_charges_num')},
                {self._num(cols, 'platform_charges', 'platform_charges_num')}
            FROM {_q(table)}
        )
        SELECT
            COALESCE(quote_id, ref_id, CONCAT('quote_', CAST(ROW_NUMBER() OVER () AS VARCHAR))) AS quote_key,
            quote_id,
            ref_id,
            customer_id,
            status,
            from_currency,
            to_currency,
            purpose_code,
            transaction_range,
            created_ts,
            updated_ts,
            COALESCE(amount_at_source_num, 0.0) AS amount_at_source,
            COALESCE(amount_at_destination_num, 0.0) AS amount_at_destination,
            COALESCE(total_amount_to_be_paid_num, 0.0) AS total_amount_to_be_paid,
            COALESCE(exchange_rate_num, 0.0) AS exchange_rate,
            COALESCE(total_additional_charges_num, 0.0) AS total_additional_charges,
            COALESCE(forex_markup_num, 0.0) AS forex_markup,
            COALESCE(amount_without_markup_num, 0.0) AS amount_without_markup,
            COALESCE(swift_charges_num, 0.0) AS swift_charges,
            COALESCE(platform_charges_num, 0.0) AS platform_charges
        FROM base
        """
        conn.execute(sql)

    def _build_customers_view(self, conn: duckdb.DuckDBPyConnection, table: str | None) -> None:
        if not table:
            conn.execute(
                """
                CREATE TABLE datada_dim_customers (
                    customer_key VARCHAR,
                    payee_key VARCHAR,
                    customer_id VARCHAR,
                    payee_id VARCHAR,
                    is_university BOOLEAN,
                    type VARCHAR,
                    address_country VARCHAR,
                    address_state VARCHAR,
                    status VARCHAR,
                    created_ts TIMESTAMP,
                    updated_ts TIMESTAMP
                )
                """
            )
            return

        cols = self._table_cols(conn, table)
        is_uni_expr = (
            f"CAST({_q('is_university')} AS BOOLEAN) AS is_university"
            if "is_university" in cols
            else "NULL::BOOLEAN AS is_university"
        )
        sql = f"""
        CREATE OR REPLACE VIEW datada_dim_customers AS
        WITH base AS (
            SELECT
                {self._txt(cols, 'customer_id')},
                {self._txt(cols, 'payee_id')},
                {self._txt(cols, 'type')},
                {self._txt(cols, 'address_country')},
                {self._txt(cols, 'address_state')},
                {self._txt(cols, 'status')},
                {self._ts(cols, 'created_at', 'created_ts')},
                {self._ts(cols, 'updated_at', 'updated_ts')},
                {is_uni_expr}
            FROM {_q(table)}
        )
        SELECT
            COALESCE(customer_id, CONCAT('cust_', CAST(ROW_NUMBER() OVER () AS VARCHAR))) AS customer_key,
            COALESCE(payee_id, CONCAT('payee_', CAST(ROW_NUMBER() OVER () AS VARCHAR))) AS payee_key,
            customer_id,
            payee_id,
            is_university,
            type,
            address_country,
            address_state,
            status,
            created_ts,
            updated_ts
        FROM base
        """
        conn.execute(sql)

    def _build_bookings_view(self, conn: duckdb.DuckDBPyConnection, table: str | None) -> None:
        if not table:
            conn.execute(
                """
                CREATE TABLE datada_mart_bookings (
                    booking_key VARCHAR,
                    customer_id VARCHAR,
                    booked_ts TIMESTAMP,
                    value_date DATE,
                    currency VARCHAR,
                    deal_type VARCHAR,
                    status VARCHAR,
                    linked_txn_status VARCHAR,
                    booked_amount DOUBLE,
                    available_balance DOUBLE,
                    amount_on_hold DOUBLE,
                    linked_txn_amount DOUBLE,
                    rate DOUBLE,
                    deal_id VARCHAR
                )
                """
            )
            return

        cols = self._table_cols(conn, table)
        sql = f"""
        CREATE OR REPLACE VIEW datada_mart_bookings AS
        WITH base AS (
            SELECT
                {self._txt(cols, 'deal_id')},
                {self._txt(cols, 'customer_id')},
                {self._ts(cols, 'booked_at', 'booked_ts')},
                {self._txt(cols, 'value_date', 'value_date_raw')},
                {self._txt(cols, 'currency')},
                {self._txt(cols, 'deal_type')},
                {self._txt(cols, 'status')},
                {self._txt(cols, 'linked_txn_status')},
                {self._num(cols, 'booked_amount', 'booked_amount_num')},
                {self._num(cols, 'available_balance', 'available_balance_num')},
                {self._num(cols, 'amount_on_hold', 'amount_on_hold_num')},
                {self._num(cols, 'linked_txn_amount', 'linked_txn_amount_num')},
                {self._num(cols, 'rate', 'rate_num')}
            FROM {_q(table)}
        )
        SELECT
            CONCAT('booking_', CAST(ROW_NUMBER() OVER () AS VARCHAR)) AS booking_key,
            customer_id,
            booked_ts,
            TRY_CAST(value_date_raw AS DATE) AS value_date,
            currency,
            deal_type,
            status,
            linked_txn_status,
            COALESCE(booked_amount_num, 0.0) AS booked_amount,
            COALESCE(available_balance_num, 0.0) AS available_balance,
            COALESCE(amount_on_hold_num, 0.0) AS amount_on_hold,
            COALESCE(linked_txn_amount_num, 0.0) AS linked_txn_amount,
            COALESCE(rate_num, 0.0) AS rate,
            deal_id
        FROM base
        """
        conn.execute(sql)

    def _validate_marts(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Run a lightweight SELECT on each mart to catch recursion errors early."""
        for mart in self._MART_NAMES:
            try:
                conn.execute(f"SELECT 1 FROM {_q(mart)} LIMIT 0")
            except duckdb.BinderException as exc:
                raise RuntimeError(
                    f"Mart validation failed for {mart}: {exc}. "
                    "This usually means a view references itself."
                ) from exc

    def _build_catalog(
        self,
        conn: duckdb.DuckDBPyConnection,
        source_tables: dict[str, str],
    ) -> dict[str, Any]:
        marts = [
            "datada_mart_transactions",
            "datada_mart_quotes",
            "datada_dim_customers",
            "datada_mart_bookings",
        ]

        catalog: dict[str, Any] = {
            "source_tables": source_tables,
            "marts": {},
            "dimension_values": {},
            "semantic_version": "",
            "metrics_by_table": {
                "datada_mart_transactions": {
                    "transaction_count": "COUNT(DISTINCT transaction_key)",
                    "unique_customers": "COUNT(DISTINCT customer_id)",
                    "total_amount": "SUM(amount)",
                    "avg_amount": "AVG(amount)",
                    "refund_count": "COUNT(DISTINCT CASE WHEN has_refund THEN transaction_key END)",
                    "refund_rate": "AVG(CASE WHEN has_refund THEN 1.0 ELSE 0.0 END)",
                    "mt103_count": "COUNT(DISTINCT CASE WHEN has_mt103 THEN transaction_key END)",
                    "mt103_rate": "AVG(CASE WHEN has_mt103 THEN 1.0 ELSE 0.0 END)",
                },
                "datada_mart_quotes": {
                    "quote_count": "COUNT(*)",
                    "total_quote_value": "SUM(total_amount_to_be_paid)",
                    "avg_quote_value": "AVG(total_amount_to_be_paid)",
                    "forex_markup_revenue": "SUM(forex_markup)",
                    "avg_forex_markup": "AVG(forex_markup)",
                    "total_charges": "SUM(total_additional_charges)",
                },
                "datada_dim_customers": {
                    "customer_count": "COUNT(DISTINCT customer_key)",
                    "payee_count": "COUNT(DISTINCT payee_key)",
                    "university_count": "SUM(CASE WHEN is_university THEN 1 ELSE 0 END)",
                },
                "datada_mart_bookings": {
                    "booking_count": "COUNT(DISTINCT deal_id)",
                    "total_booked_amount": "SUM(booked_amount)",
                    "avg_rate": "AVG(rate)",
                },
            },
            "preferred_time_column": {
                "datada_mart_transactions": "event_ts",
                "datada_mart_quotes": "created_ts",
                "datada_dim_customers": "created_ts",
                "datada_mart_bookings": "booked_ts",
            },
            "preferred_time_column_by_metric": {
                "datada_mart_transactions": {
                    "mt103_count": "mt103_created_ts",
                    "mt103_rate": "mt103_created_ts",
                }
            },
            "quality": {
                "coverage_by_domain": {},
                "semantic_version": "",
            },
        }

        for mart in marts:
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {_q(mart)}").fetchone()[0])
            columns = [
                c
                for (c,) in conn.execute(
                    f"SELECT column_name FROM information_schema.columns WHERE table_name='{mart}' ORDER BY ordinal_position"
                ).fetchall()
            ]
            catalog["marts"][mart] = {
                "row_count": row_count,
                "columns": columns,
            }

        dim_map = {
            "datada_mart_transactions": [
                "platform_name", "state", "payment_status", "txn_flow",
                "address_country", "address_state",
            ],
            "datada_mart_quotes": ["status", "from_currency", "to_currency", "purpose_code"],
            "datada_dim_customers": ["type", "address_country", "address_state", "status"],
            "datada_mart_bookings": ["currency", "deal_type", "status", "linked_txn_status", "deal_id"],
        }

        for mart, dimensions in dim_map.items():
            catalog["dimension_values"][mart] = {}
            for dim_col in dimensions:
                if dim_col not in catalog["marts"][mart]["columns"]:
                    continue
                values = [
                    str(v)
                    for (v,) in conn.execute(
                        f"""
                        SELECT { _q(dim_col) }
                        FROM { _q(mart) }
                        WHERE { _q(dim_col) } IS NOT NULL AND TRIM(CAST({ _q(dim_col) } AS VARCHAR)) != ''
                        GROUP BY 1
                        ORDER BY COUNT(*) DESC
                        LIMIT 15
                        """
                    ).fetchall()
                ]
                catalog["dimension_values"][mart][dim_col] = values

        tx_rows = catalog["marts"]["datada_mart_transactions"]["row_count"]
        tx_quality = {
            "transactions_row_count": tx_rows,
            "transaction_key_null_rate": 0.0,
            "event_ts_null_rate": 0.0,
            "amount_nonzero_ratio": 0.0,
        }
        if tx_rows > 0:
            q = conn.execute(
                """
                SELECT
                    AVG(CASE WHEN transaction_key IS NULL THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN event_ts IS NULL THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN amount IS NULL OR amount = 0 THEN 0.0 ELSE 1.0 END)
                FROM datada_mart_transactions
                """
            ).fetchone()
            tx_quality["transaction_key_null_rate"] = float(q[0] or 0.0)
            tx_quality["event_ts_null_rate"] = float(q[1] or 0.0)
            tx_quality["amount_nonzero_ratio"] = float(q[2] or 0.0)

        quote_rows = catalog["marts"]["datada_mart_quotes"]["row_count"]
        quote_quality = {
            "quotes_row_count": quote_rows,
            "quote_key_null_rate": 0.0,
            "markup_nonzero_ratio": 0.0,
        }
        if quote_rows > 0:
            q = conn.execute(
                """
                SELECT
                    AVG(CASE WHEN quote_key IS NULL THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN forex_markup IS NULL OR forex_markup = 0 THEN 0.0 ELSE 1.0 END)
                FROM datada_mart_quotes
                """
            ).fetchone()
            quote_quality["quote_key_null_rate"] = float(q[0] or 0.0)
            quote_quality["markup_nonzero_ratio"] = float(q[1] or 0.0)

        customer_rows = catalog["marts"]["datada_dim_customers"]["row_count"]
        customer_quality = {
            "customers_row_count": customer_rows,
            "customer_key_null_rate": 0.0,
            "university_ratio": 0.0,
        }
        if customer_rows > 0:
            q = conn.execute(
                """
                SELECT
                    AVG(CASE WHEN customer_key IS NULL THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN is_university THEN 1.0 ELSE 0.0 END)
                FROM datada_dim_customers
                """
            ).fetchone()
            customer_quality["customer_key_null_rate"] = float(q[0] or 0.0)
            customer_quality["university_ratio"] = float(q[1] or 0.0)

        booking_rows = catalog["marts"]["datada_mart_bookings"]["row_count"]
        booking_quality = {
            "bookings_row_count": booking_rows,
            "booking_key_null_rate": 0.0,
            "booked_amount_nonzero_ratio": 0.0,
        }
        if booking_rows > 0:
            q = conn.execute(
                """
                SELECT
                    AVG(CASE WHEN booking_key IS NULL THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN booked_amount IS NULL OR booked_amount = 0 THEN 0.0 ELSE 1.0 END)
                FROM datada_mart_bookings
                """
            ).fetchone()
            booking_quality["booking_key_null_rate"] = float(q[0] or 0.0)
            booking_quality["booked_amount_nonzero_ratio"] = float(q[1] or 0.0)

        coverage_by_domain = {
            "transactions": tx_rows,
            "quotes": quote_rows,
            "customers": customer_rows,
            "bookings": booking_rows,
        }

        semantic_version_payload = {
            "marts": {k: catalog["marts"][k]["columns"] for k in sorted(catalog["marts"])},
            "metrics_by_table": catalog["metrics_by_table"],
            "preferred_time_column": catalog["preferred_time_column"],
            "preferred_time_column_by_metric": catalog["preferred_time_column_by_metric"],
        }
        semantic_version = _semantic_signature(semantic_version_payload)
        catalog["semantic_version"] = semantic_version
        catalog["quality"] = {
            **tx_quality,
            **quote_quality,
            **customer_quality,
            **booking_quality,
            "coverage_by_domain": coverage_by_domain,
            "semantic_version": semantic_version,
            "semantic_marts_ready": all(v > 0 for v in coverage_by_domain.values()),
        }
        return catalog


class AgenticAnalyticsTeam:
    """Hierarchical agentic runtime for analytics queries."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.semantic = SemanticLayerManager(self.db_path)
        default_memory_db = self.db_path.with_name(f"{self.db_path.stem}_agent_memory.duckdb")
        self.memory_db_path = Path(os.environ.get("HG_MEMORY_DB_PATH", str(default_memory_db))).expanduser()
        self.memory = AgentMemoryStore(self.memory_db_path)
        # Use a writable-mode connection for compatibility with semantic layer refresh.
        # SQL safety is still enforced by SafeSQLExecutor guardrails.
        self.executor = SafeSQLExecutor(self.db_path, read_only=False)
        # GAP 35: Load domain expertise knowledge base
        self._domain_knowledge = _load_domain_knowledge()

    def close(self) -> None:
        self.executor.close()

    def __enter__(self) -> "AgenticAnalyticsTeam":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def record_feedback(
        self,
        *,
        tenant_id: str = "public",
        trace_id: str | None,
        session_id: str | None,
        goal: str | None,
        issue: str,
        suggested_fix: str | None = None,
        severity: str = "medium",
        keyword: str | None = None,
        target_table: str | None = None,
        target_metric: str | None = None,
        target_dimensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist explicit user feedback and optional correction rules."""

        feedback_id = self.memory.store_feedback(
            tenant_id=tenant_id,
            trace_id=trace_id,
            session_id=session_id,
            goal=goal,
            issue=issue,
            suggested_fix=suggested_fix,
            severity=severity,
            metadata={
                "keyword": keyword,
                "target_table": target_table,
                "target_metric": target_metric,
                "target_dimensions": target_dimensions or [],
            },
        )

        correction_id = ""
        if keyword and target_table and target_metric:
            correction_id = self.memory.upsert_correction(
                tenant_id=tenant_id,
                keyword=keyword,
                target_table=target_table,
                target_metric=target_metric,
                target_dimensions=target_dimensions or [],
                notes=f"User feedback: {issue}",
                source="user_feedback",
                weight=1.8,
            )

        return {
            "feedback_id": feedback_id,
            "correction_id": correction_id,
        }

    def list_corrections(
        self,
        *,
        tenant_id: str = "public",
        limit: int = 250,
        include_disabled: bool = True,
    ) -> list[dict[str, Any]]:
        return self.memory.list_corrections(
            tenant_id=tenant_id,
            limit=limit,
            include_disabled=include_disabled,
        )

    def set_correction_enabled(self, correction_id: str, enabled: bool, *, tenant_id: str = "public") -> bool:
        return self.memory.set_correction_enabled(correction_id, enabled, tenant_id=tenant_id)

    def rollback_correction(self, correction_id: str, *, tenant_id: str = "public") -> dict[str, Any]:
        return self.memory.rollback_correction(correction_id, tenant_id=tenant_id)

    def list_tool_candidates(
        self,
        *,
        tenant_id: str = "public",
        status: str | None = None,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        return self.memory.list_tool_candidates(tenant_id=tenant_id, status=status, limit=limit)

    def stage_tool_candidate(self, tool_id: str, *, tenant_id: str = "public") -> dict[str, Any]:
        return self.memory.stage_tool_candidate(tool_id, tenant_id=tenant_id, db_path=self.db_path)

    def promote_tool_candidate(self, tool_id: str, *, tenant_id: str = "public") -> dict[str, Any]:
        return self.memory.promote_tool_candidate(tool_id, tenant_id=tenant_id)

    def rollback_tool_candidate(
        self,
        tool_id: str,
        *,
        tenant_id: str = "public",
        reason: str = "",
    ) -> dict[str, Any]:
        return self.memory.rollback_tool_candidate(tool_id, tenant_id=tenant_id, reason=reason)

    # -- Glossary delegation --
    def upsert_glossary_term(self, **kw: Any) -> dict[str, Any]:
        return self.memory.upsert_glossary_term(**kw)

    def list_glossary(self, **kw: Any) -> list[dict[str, Any]]:
        return self.memory.list_glossary(**kw)

    def resolve_glossary(self, text: str, **kw: Any) -> list[dict[str, Any]]:
        return self.memory.resolve_glossary(text, **kw)

    # -- Teaching delegation --
    def add_teaching(self, **kw: Any) -> dict[str, Any]:
        return self.memory.add_teaching(**kw)

    def list_teachings(self, **kw: Any) -> list[dict[str, Any]]:
        return self.memory.list_teachings(**kw)

    def run(
        self,
        goal: str,
        runtime: RuntimeSelection,
        *,
        tenant_id: str = "public",
        conversation_context: list[dict[str, Any]] | None = None,
        storyteller_mode: bool = False,
        autonomy: AutonomyConfig | None = None,
    ) -> AssistantQueryResponse:
        trace_id = str(uuid.uuid4())
        trace: list[dict[str, Any]] = []
        blackboard: list[dict[str, Any]] = []
        started = time.perf_counter()
        history = conversation_context or []
        autonomy_cfg = autonomy or AutonomyConfig()
        self._pipeline_warnings: list[str] = []
        if runtime.fallback_warning:
            self._pipeline_warnings.append(runtime.fallback_warning)

        try:
            effective_goal = self._run_agent(
                trace,
                "ContextAgent",
                "followup_resolution",
                self._resolve_contextual_goal,
                goal,
                history,
                runtime,
            )
            self._blackboard_post(
                blackboard,
                producer="ContextAgent",
                artifact_type="resolved_goal",
                payload={"goal": goal, "resolved_goal": effective_goal},
                consumed_by=["ChiefAnalystAgent", "IntakeAgent"],
            )
            catalog = self._run_agent(trace, "DataEngineeringTeam", "semantic_layer", self.semantic.prepare)
            self._blackboard_post(
                blackboard,
                producer="DataEngineeringTeam",
                artifact_type="semantic_catalog",
                payload={
                    "marts": list((catalog.get("marts") or {}).keys()),
                    "source_tables": catalog.get("source_tables", {}),
                    "quality": catalog.get("quality", {}),
                },
                consumed_by=["ChiefAnalystAgent", "SemanticRetrievalAgent", "PlanningAgent"],
            )
            mission = self._run_agent(
                trace,
                "ChiefAnalystAgent",
                "supervision",
                self._chief_analyst,
                effective_goal,
                runtime,
                catalog,
            )
            self._blackboard_post(
                blackboard,
                producer="ChiefAnalystAgent",
                artifact_type="mission_brief",
                payload=mission,
                consumed_by=["IntakeAgent"],
            )
            memory_hints = self._run_agent(
                trace,
                "MemoryAgent",
                "episodic_recall",
                self.memory.recall,
                effective_goal,
                tenant_id=tenant_id,
            )
            self._blackboard_post(
                blackboard,
                producer="MemoryAgent",
                artifact_type="episodic_memory_hints",
                payload={"count": len(memory_hints), "top": (memory_hints or [])[:2]},
                consumed_by=["IntakeAgent", "AutonomyAgent"],
            )
            learned_corrections = self._run_agent(
                trace,
                "MemoryAgent",
                "correction_rule_recall",
                self.memory.get_matching_corrections,
                effective_goal,
                tenant_id=tenant_id,
            )
            self._blackboard_post(
                blackboard,
                producer="MemoryAgent",
                artifact_type="correction_rules",
                payload={"count": len(learned_corrections), "top": (learned_corrections or [])[:3]},
                consumed_by=["AutonomyAgent", "PlanningAgent"],
            )
            intake = self._run_agent(
                trace,
                "IntakeAgent",
                "goal_structuring",
                self._intake_agent,
                effective_goal,
                runtime,
                mission,
                catalog,
                memory_hints,
            )
            self._blackboard_post(
                blackboard,
                producer="IntakeAgent",
                artifact_type="structured_goal",
                payload=intake,
                consumed_by=["SemanticRetrievalAgent", "GovernanceAgent"],
            )
            # ── GAP 36d: Consume multi_domain_hint from ChiefAnalyst ──────────
            if mission.get("multi_domain_hint") and len(mission.get("detected_domains", [])) > 1:
                chief_domains = mission["detected_domains"]
                intake_domains = set(intake.get("domains_detected", []))
                for cd in chief_domains:
                    if cd not in intake_domains:
                        intake.setdefault("domains_detected", []).append(cd)
                intake_secondary = [d for d in intake.get("domains_detected", []) if d != intake.get("domain")]
                if intake_secondary and not intake.get("secondary_domains"):
                    intake["secondary_domains"] = intake_secondary

            # ── GAP 13: Surface multi-domain detection ──────────────────
            if intake.get("secondary_domains"):
                self._pipeline_warnings.append(
                    f"Multiple data domains detected: {', '.join(intake['domains_detected'])}. "
                    f"Primary domain '{intake['domain']}' used; secondary domains "
                    f"({', '.join(intake['secondary_domains'])}) may require cross-domain JOINs."
                )

            # ── GAP 32: Multi-part question decomposition ──────────────────
            if runtime.use_llm and runtime.provider:
                sub_questions = self._detect_multi_part(effective_goal, runtime)
                if sub_questions and len(sub_questions) >= 2:
                    sub_results = []
                    for sq in sub_questions:
                        sr = self._run_sub_query(
                            sq, runtime, catalog, mission, memory_hints or [],
                        )
                        sub_results.append(sr)
                    if any(r["success"] for r in sub_results):
                        plan, query_plan, execution, audit = self._merge_multi_part_results(
                            effective_goal, sub_results, runtime,
                        )
                        # Build multi-part narrative from sub-results
                        narr_parts: list[str] = []
                        for r in sub_results:
                            sub_narr = self._run_agent(
                                trace,
                                "NarrativeAgent",
                                "multi_part_sub_narrative",
                                self._narrative_agent,
                                r["sub_goal"],
                                r["plan"],
                                r["execution"],
                                r["audit"],
                                runtime,
                                storyteller_mode,
                                history,
                            )
                            narr_parts.append(sub_narr.get("answer_markdown", ""))
                        combined_answer = "\n\n---\n\n".join(
                            f"**{r['sub_goal']}**\n\n{narr}"
                            for r, narr in zip(sub_results, narr_parts)
                            if narr
                        )
                        confidence_score = max(0.0, min(1.0, float(audit.get("score", 0.6))))
                        confidence = self._to_confidence(confidence_score)
                        total_ms = (time.perf_counter() - started) * 1000
                        trace.append({
                            "agent": "ChiefAnalystAgent",
                            "role": "finalize_response",
                            "status": "success",
                            "duration_ms": round(total_ms, 2),
                            "summary": f"multi-part response assembled ({len(sub_results)} sub-queries)",
                        })
                        self._memory_write_turn(
                            trace=trace, trace_id=trace_id, goal=goal,
                            resolved_goal=effective_goal, tenant_id=tenant_id,
                            runtime=runtime, success=execution["success"],
                            confidence_score=confidence_score,
                            row_count=execution.get("row_count"),
                            plan=plan, sql=query_plan.get("sql"),
                            audit_warnings=audit.get("warnings", []),
                            correction_applied=False, correction_reason="",
                            metadata={"autonomy_mode": autonomy_cfg.mode, "multi_part": True},
                        )
                        return AssistantQueryResponse(
                            success=execution["success"],
                            answer_markdown=combined_answer,
                            confidence=confidence,
                            confidence_score=confidence_score,
                            definition_used=plan["definition_used"],
                            evidence=[
                                EvidenceItem(
                                    description=f"Sub-query: {r['sub_goal']}",
                                    value=str(r["execution"].get("row_count", 0)),
                                    source="multi_part_decomposition",
                                    sql_reference=r["query_plan"].get("sql"),
                                )
                                for r in sub_results
                            ],
                            sanity_checks=[
                                SanityCheck(
                                    check_name=f"sub_query_{i+1}",
                                    passed=r["success"],
                                    message=f"{r['sub_goal']}: {'OK' if r['success'] else r['execution'].get('error', 'failed')}",
                                )
                                for i, r in enumerate(sub_results)
                            ],
                            sql=query_plan["sql"],
                            row_count=execution["row_count"],
                            columns=execution["columns"],
                            sample_rows=execution["sample_rows"],
                            execution_time_ms=execution["execution_time_ms"],
                            trace_id=trace_id,
                            runtime=self._runtime_payload(
                                runtime,
                                llm_intake_used=True,
                                autonomy=autonomy_cfg,
                            ) | {"blackboard_entries": len(blackboard)},
                            agent_trace=trace,
                            chart_spec=None,
                            evidence_packets=[
                                {"agent": "MultiPartDecomposer", "sub_queries": len(sub_results)},
                                {"agent": "Blackboard", "artifact_count": len(blackboard),
                                 "artifacts": blackboard, "edges": self._blackboard_edges(blackboard)},
                            ],
                            data_quality={
                                **catalog.get("quality", {}),
                                "audit_score": confidence_score,
                                "multi_part": True,
                                "blackboard": {"artifact_count": len(blackboard), "edges": self._blackboard_edges(blackboard)},
                            },
                            suggested_questions=[
                                f"Tell me more about: {r['sub_goal']}" for r in sub_results[:3]
                            ],
                            warnings=self._pipeline_warnings,
                        )

            clarification = self._run_agent(
                trace,
                "ClarificationAgent",
                "ambiguity_gate",
                self._clarification_agent,
                effective_goal,
                intake,
                history,
                memory_hints,
            )
            self._blackboard_post(
                blackboard,
                producer="ClarificationAgent",
                artifact_type="ambiguity_gate",
                payload=clarification,
                consumed_by=["ChiefAnalystAgent", "PlanningAgent"],
            )
            if clarification.get("needs_clarification"):
                total_ms = (time.perf_counter() - started) * 1000
                trace.append(
                    {
                        "agent": "ChiefAnalystAgent",
                        "role": "finalize_response",
                        "status": "success",
                        "duration_ms": round(total_ms, 2),
                        "summary": "clarification requested",
                    }
                )
                questions = list(clarification.get("questions") or [])
                suggested = list(clarification.get("suggested_questions") or [])
                self._memory_write_turn(
                    trace=trace,
                    trace_id=trace_id,
                    goal=goal,
                    resolved_goal=effective_goal,
                    tenant_id=tenant_id,
                    runtime=runtime,
                    success=False,
                    confidence_score=0.18,
                    row_count=0,
                    plan={
                        "table": "",
                        "metric": "",
                        "dimensions": [],
                        "time_filter": intake.get("time_filter"),
                        "value_filters": [],
                    },
                    sql=None,
                    audit_warnings=[f"clarification_required: {clarification.get('reason', '')}"],
                    correction_applied=False,
                    correction_reason="",
                    metadata={
                        "autonomy_mode": autonomy_cfg.mode,
                        "clarification_required": True,
                        "clarification_reason": clarification.get("reason", ""),
                    },
                )
                return AssistantQueryResponse(
                    success=False,
                    answer_markdown=(
                        f"**I need one quick clarification before I run this query.**\n\n"
                        + "\n".join(f"- {q}" for q in questions[:3])
                    ),
                    confidence=ConfidenceLevel.UNCERTAIN,
                    confidence_score=0.18,
                    definition_used="clarification_required",
                    evidence=[],
                    sanity_checks=[
                        SanityCheck(
                            check_name="clarification_required",
                            passed=False,
                            message=str(clarification.get("reason") or "Question is ambiguous."),
                        )
                    ],
                    sql=None,
                    row_count=0,
                    columns=[],
                    sample_rows=[],
                    execution_time_ms=0.0,
                    trace_id=trace_id,
                    runtime=self._runtime_payload(
                        runtime,
                        llm_intake_used=bool(intake.get("_llm_intake_used", False)),
                        autonomy=autonomy_cfg,
                    )
                    | {"blackboard_entries": len(blackboard)},
                    agent_trace=trace,
                    chart_spec=None,
                    evidence_packets=[
                        {
                            "agent": "ClarificationAgent",
                            "result": clarification,
                        },
                        {
                            "agent": "Blackboard",
                            "artifact_count": len(blackboard),
                            "artifacts": blackboard,
                            "edges": self._blackboard_edges(blackboard),
                        },
                    ],
                    data_quality={
                        **catalog.get("quality", {}),
                        "clarification": clarification,
                        "blackboard": {"artifact_count": len(blackboard), "edges": self._blackboard_edges(blackboard)},
                    },
                    suggested_questions=suggested[:6],
                    error="clarification_required",
                    warnings=self._pipeline_warnings,
                )

            pre_gov = self._run_agent(
                trace,
                "GovernanceAgent",
                "policy_gate",
                self._governance_precheck,
                effective_goal,
            )
            self._blackboard_post(
                blackboard,
                producer="GovernanceAgent",
                artifact_type="policy_gate",
                payload=pre_gov,
                consumed_by=["ChiefAnalystAgent", "PlanningAgent"],
            )
            if not pre_gov["allowed"]:
                return AssistantQueryResponse(
                    success=False,
                    answer_markdown=f"**Request blocked by governance**\n\n{pre_gov['reason']}",
                    confidence=ConfidenceLevel.UNCERTAIN,
                    confidence_score=0.0,
                    definition_used=goal,
                    evidence=[],
                    sanity_checks=[
                        SanityCheck(
                            check_name="governance",
                            passed=False,
                            message=pre_gov["reason"],
                        )
                    ],
                    trace_id=trace_id,
                    runtime=self._runtime_payload(
                        runtime,
                        llm_intake_used=bool(intake.get("_llm_intake_used", False)),
                        autonomy=autonomy_cfg,
                    ),
                    agent_trace=trace,
                    suggested_questions=["Ask an aggregated business metric instead."],
                    data_quality=catalog.get("quality", {}),
                    warnings=self._pipeline_warnings,
                )

            if intake.get("intent") == "data_overview":
                discovery_plan = self._run_agent(
                    trace,
                    "DiscoveryPlannerAgent",
                    "overview_planning",
                    self._data_overview_plan_agent,
                    goal,
                    catalog,
                )
                self._blackboard_post(
                    blackboard,
                    producer="DiscoveryPlannerAgent",
                    artifact_type="overview_plan",
                    payload=discovery_plan,
                    consumed_by=["CatalogProfilerAgent", "CatalogExplainerAgent"],
                )
                overview_profile = self._run_agent(
                    trace,
                    "CatalogProfilerAgent",
                    "dataset_profiling",
                    self._data_overview_profile_agent,
                    discovery_plan,
                    catalog,
                )
                self._blackboard_post(
                    blackboard,
                    producer="CatalogProfilerAgent",
                    artifact_type="overview_profile",
                    payload=overview_profile,
                    consumed_by=["CatalogExplainerAgent", "ChiefAnalystAgent"],
                )
                overview = self._run_agent(
                    trace,
                    "CatalogExplainerAgent",
                    "dataset_overview",
                    self._data_overview_agent,
                    goal,
                    catalog,
                    storyteller_mode,
                    runtime,
                    overview_profile,
                )
                self._blackboard_post(
                    blackboard,
                    producer="CatalogExplainerAgent",
                    artifact_type="overview_answer",
                    payload=overview,
                    consumed_by=["ChiefAnalystAgent"],
                )
                total_ms = (time.perf_counter() - started) * 1000
                trace.append(
                    {
                        "agent": "ChiefAnalystAgent",
                        "role": "finalize_response",
                        "status": "success",
                        "duration_ms": round(total_ms, 2),
                        "summary": "overview response assembled",
                    }
                )
                return AssistantQueryResponse(
                    success=True,
                    answer_markdown=overview["answer_markdown"],
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.95,
                    definition_used="semantic catalog overview",
                    evidence=[
                        EvidenceItem(
                            description="Semantic marts available",
                            value=str(len(catalog.get("marts", {}))),
                            source="semantic layer",
                            sql_reference=None,
                        )
                    ],
                    sanity_checks=[
                        SanityCheck(
                            check_name="catalog_ready",
                            passed=True,
                            message="Semantic marts built and discoverable.",
                        ),
                        SanityCheck(
                            check_name="semantic_versioned",
                            passed=bool(catalog.get("semantic_version")),
                            message=(
                                f"Semantic profile version {catalog.get('semantic_version')}."
                                if catalog.get("semantic_version")
                                else "Semantic profile version missing."
                            ),
                        ),
                    ],
                    trace_id=trace_id,
                    runtime={
                        **self._runtime_payload(runtime, autonomy=autonomy_cfg),
                        "blackboard_entries": len(blackboard),
                    },
                    agent_trace=trace,
                    chart_spec={
                        "type": "bar",
                        "x": "mart",
                        "y": "row_count",
                        "title": "Data footprint by mart",
                    },
                    evidence_packets=[
                        {
                            "agent": "DiscoveryPlannerAgent",
                            "tasks": discovery_plan.get("tasks", []),
                        },
                        {
                            "agent": "CatalogProfilerAgent",
                            "profile": overview_profile,
                        },
                        {
                            "agent": "Blackboard",
                            "artifact_count": len(blackboard),
                            "artifacts": blackboard,
                            "edges": self._blackboard_edges(blackboard),
                        },
                    ],
                    data_quality={
                        **catalog.get("quality", {}),
                        "overview_profile": overview_profile,
                    },
                    sample_rows=overview.get("sample_rows", []),
                    columns=overview.get("columns", ["mart", "row_count"]),
                    suggested_questions=overview.get("suggested_questions", []),
                    warnings=self._pipeline_warnings,
                )

            if intake.get("intent") == "schema_exploration":
                schema_domain = intake.get("domain", "transactions")
                schema_result = self._run_agent(
                    trace,
                    "SchemaExplorationAgent",
                    "schema_exploration",
                    self._schema_exploration_agent,
                    effective_goal,
                    schema_domain,
                    catalog,
                )
                self._blackboard_post(
                    blackboard,
                    producer="SchemaExplorationAgent",
                    artifact_type="schema_description",
                    payload=schema_result,
                    consumed_by=["ChiefAnalystAgent"],
                )
                total_ms = (time.perf_counter() - started) * 1000
                trace.append(
                    {
                        "agent": "ChiefAnalystAgent",
                        "role": "finalize_response",
                        "status": "success",
                        "duration_ms": round(total_ms, 2),
                        "summary": "schema exploration response assembled",
                    }
                )
                return AssistantQueryResponse(
                    success=True,
                    answer_markdown=schema_result["answer_markdown"],
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.92,
                    definition_used=f"schema exploration for {schema_domain}",
                    evidence=[
                        EvidenceItem(
                            description=f"Schema for {schema_result.get('table', '')}",
                            value=f"{schema_result.get('row_count', 0):,} rows, {len(schema_result.get('columns', []))} columns",
                            source="semantic catalog",
                            sql_reference=None,
                        )
                    ],
                    sanity_checks=[
                        SanityCheck(
                            check_name="catalog_ready",
                            passed=True,
                            message="Schema derived from semantic catalog.",
                        ),
                    ],
                    trace_id=trace_id,
                    runtime={
                        **self._runtime_payload(runtime, autonomy=autonomy_cfg),
                        "blackboard_entries": len(blackboard),
                    },
                    agent_trace=trace,
                    chart_spec=None,
                    evidence_packets=[
                        {
                            "agent": "SchemaExplorationAgent",
                            "result": schema_result,
                        },
                        {
                            "agent": "Blackboard",
                            "artifact_count": len(blackboard),
                            "artifacts": blackboard,
                            "edges": self._blackboard_edges(blackboard),
                        },
                    ],
                    data_quality=catalog.get("quality", {}),
                    suggested_questions=schema_result.get("suggested_questions", []),
                    warnings=self._pipeline_warnings,
                )

            if intake.get("intent") == "document_qa" or intake.get("domain") == "documents":
                doc_retrieval = self._run_agent(
                    trace,
                    "DocumentRetrievalAgent",
                    "citation_retrieval",
                    self._document_retrieval_agent,
                    effective_goal,
                    intake,
                )
                self._blackboard_post(
                    blackboard,
                    producer="DocumentRetrievalAgent",
                    artifact_type="document_matches",
                    payload={
                        "match_count": int(doc_retrieval.get("match_count") or 0),
                        "source_count": int(doc_retrieval.get("source_count") or 0),
                        "citations": doc_retrieval.get("citations", [])[:6],
                    },
                    consumed_by=["NarrativeAgent", "ChiefAnalystAgent"],
                )
                doc_answer = self._run_agent(
                    trace,
                    "NarrativeAgent",
                    "document_briefing",
                    self._document_answer_agent,
                    goal,
                    doc_retrieval,
                    runtime,
                    storyteller_mode,
                )
                self._blackboard_post(
                    blackboard,
                    producer="NarrativeAgent",
                    artifact_type="business_answer",
                    payload=doc_answer,
                    consumed_by=["ChiefAnalystAgent"],
                )
                match_count = int(doc_retrieval.get("match_count") or 0)
                doc_ready = bool(doc_retrieval.get("document_table_ready", False))
                doc_success = doc_ready
                if not doc_ready:
                    confidence_score = 0.12
                else:
                    confidence_score = (
                        0.22 if match_count <= 0 else max(0.55, min(0.93, 0.58 + 0.07 * min(5, match_count)))
                    )
                confidence = self._to_confidence(confidence_score)
                total_ms = (time.perf_counter() - started) * 1000
                checks = [
                    SanityCheck(
                        check_name="document_table_available",
                        passed=bool(doc_retrieval.get("document_table_ready", False)),
                        message=(
                            "Document chunks table is available."
                            if doc_retrieval.get("document_table_ready", False)
                            else "Document chunks table not found."
                        ),
                    ),
                    SanityCheck(
                        check_name="citation_matches",
                        passed=match_count > 0,
                        message=(
                            f"Retrieved {match_count} citation snippets."
                            if match_count > 0
                            else "No citation snippets matched this question."
                        ),
                    ),
                ]
                trace.append(
                    {
                        "agent": "ChiefAnalystAgent",
                        "role": "finalize_response",
                        "status": "success",
                        "duration_ms": round(total_ms, 2),
                        "summary": "document response assembled",
                    }
                )
                self._memory_write_turn(
                    trace=trace,
                    trace_id=trace_id,
                    goal=goal,
                    resolved_goal=effective_goal,
                    tenant_id=tenant_id,
                    runtime=runtime,
                    success=doc_success,
                    confidence_score=confidence_score,
                    row_count=match_count,
                    plan={
                        "table": "datada_document_chunks",
                        "metric": "document_match_count",
                        "dimensions": ["source_path"],
                        "time_filter": None,
                        "value_filters": [],
                    },
                    sql=doc_retrieval.get("sql_used"),
                    audit_warnings=[] if match_count > 0 else ["No document citations matched."],
                    correction_applied=False,
                    correction_reason="",
                    metadata={
                        "autonomy_mode": autonomy_cfg.mode,
                        "document_citations": doc_retrieval.get("citations", []),
                    },
                )
                return AssistantQueryResponse(
                    success=doc_success,
                    answer_markdown=doc_answer.get("answer_markdown", ""),
                    confidence=confidence,
                    confidence_score=confidence_score,
                    definition_used="citation-backed document retrieval",
                    evidence=[
                        EvidenceItem(
                            description="Matched document snippets",
                            value=str(match_count),
                            source="document_retrieval",
                            sql_reference=doc_retrieval.get("sql_used"),
                        ),
                        EvidenceItem(
                            description="Sources covered",
                            value=str(doc_retrieval.get("source_count") or 0),
                            source="document_retrieval",
                            sql_reference=doc_retrieval.get("sql_used"),
                        ),
                    ],
                    sanity_checks=checks,
                    sql=doc_retrieval.get("sql_used"),
                    row_count=match_count,
                    columns=list(doc_retrieval.get("columns") or []),
                    sample_rows=list(doc_retrieval.get("sample_rows") or []),
                    execution_time_ms=float(doc_retrieval.get("execution_time_ms") or 0.0),
                    trace_id=trace_id,
                    runtime=self._runtime_payload(runtime, autonomy=autonomy_cfg)
                    | {"blackboard_entries": len(blackboard)},
                    agent_trace=trace,
                    chart_spec={"type": "table", "columns": ["citation", "source", "snippet"]},
                    evidence_packets=[
                        {
                            "agent": "DocumentRetrievalAgent",
                            "citations": doc_retrieval.get("citations", []),
                            "source_count": doc_retrieval.get("source_count", 0),
                            "match_count": doc_retrieval.get("match_count", 0),
                        },
                        {
                            "agent": "Blackboard",
                            "artifact_count": len(blackboard),
                            "artifacts": blackboard,
                            "edges": self._blackboard_edges(blackboard),
                        },
                    ],
                    data_quality={
                        "audit_score": confidence_score,
                        "grounding": {
                            "table": "datada_document_chunks",
                            "metric": "document_match_count",
                            "goal_terms_detected": doc_retrieval.get("matched_terms", []),
                            "goal_term_misses": doc_retrieval.get("missing_terms", []),
                            "query_columns_used": ["source_path", "chunk_index", "content"],
                            "replay_match": None,
                            "intent": "document_qa",
                            "dimensions": ["source_path"],
                            "time_filter": None,
                            "value_filters": [],
                        },
                        "document_retrieval": {
                            "source_count": doc_retrieval.get("source_count", 0),
                            "match_count": doc_retrieval.get("match_count", 0),
                            "citations": doc_retrieval.get("citations", []),
                        },
                        "blackboard": {"artifact_count": len(blackboard), "edges": self._blackboard_edges(blackboard)},
                    },
                    suggested_questions=doc_answer.get("suggested_questions", []),
                    warnings=self._pipeline_warnings,
                )

            retrieval = self._run_agent(
                trace,
                "SemanticRetrievalAgent",
                "domain_mapping",
                self._semantic_retrieval_agent,
                intake,
                catalog,
            )
            self._blackboard_post(
                blackboard,
                producer="SemanticRetrievalAgent",
                artifact_type="domain_mapping",
                payload=retrieval,
                consumed_by=["PlanningAgent"],
            )
            plan = self._run_agent(
                trace,
                "PlanningAgent",
                "task_graph",
                self._planning_agent,
                intake,
                retrieval,
                catalog,
                runtime,
            )
            self._blackboard_post(
                blackboard,
                producer="PlanningAgent",
                artifact_type="execution_plan",
                payload=plan,
                consumed_by=[
                    "TransactionsSpecialistAgent",
                    "CustomerSpecialistAgent",
                    "RevenueSpecialistAgent",
                    "RiskSpecialistAgent",
                    "QueryEngineerAgent",
                    "AutonomyAgent",
                ],
            )

            tx_findings = self._run_agent(
                trace,
                "TransactionsSpecialistAgent",
                "domain_analysis",
                self._transactions_specialist,
                plan,
                catalog,
            )
            customer_findings = self._run_agent(
                trace,
                "CustomerSpecialistAgent",
                "domain_analysis",
                self._customer_specialist,
                plan,
                catalog,
            )
            revenue_findings = self._run_agent(
                trace,
                "RevenueSpecialistAgent",
                "domain_analysis",
                self._revenue_specialist,
                plan,
                catalog,
            )
            risk_findings = self._run_agent(
                trace,
                "RiskSpecialistAgent",
                "domain_analysis",
                self._risk_specialist,
                plan,
                catalog,
            )
            # Surface specialist warnings into pipeline
            for findings in [tx_findings, customer_findings, revenue_findings, risk_findings]:
                self._pipeline_warnings.extend(findings.get("warnings", []))
            self._blackboard_post(
                blackboard,
                producer="SpecialistAgents",
                artifact_type="domain_findings",
                payload={
                    "transactions": tx_findings,
                    "customers": customer_findings,
                    "revenue": revenue_findings,
                    "risk": risk_findings,
                },
                consumed_by=["QueryEngineerAgent", "AutonomyAgent"],
            )

            query_plan = self._run_agent(
                trace,
                "QueryEngineerAgent",
                "sql_compilation",
                self._query_engine_agent,
                plan,
                [tx_findings, customer_findings, revenue_findings, risk_findings],
                runtime,
                catalog,
            )
            self._blackboard_post(
                blackboard,
                producer="QueryEngineerAgent",
                artifact_type="query_plan",
                payload=query_plan,
                consumed_by=["ExecutionAgent", "AuditAgent", "AutonomyAgent"],
            )
            execution = self._run_agent(
                trace,
                "ExecutionAgent",
                "safe_execution",
                self._execution_agent,
                query_plan,
            )
            # ── GAP 33: SQL Error Recovery ──────────────────
            if not execution["success"] and runtime.use_llm and runtime.provider:
                recovery = self._recover_failed_sql(plan, query_plan, execution, runtime)
                if recovery and recovery.get("sql"):
                    validation = self.executor.validate(recovery["sql"])
                    if validation.is_valid:
                        recovered_qp = {"sql": recovery["sql"], "table": query_plan["table"]}
                        recovered_exec = self._execution_agent(recovered_qp)
                        if recovered_exec["success"]:
                            query_plan = recovered_qp
                            execution = recovered_exec
                            self._pipeline_warnings.append(
                                f"SQL error recovered: {recovery.get('fix_description', 'auto-fixed')}"
                            )
            self._blackboard_post(
                blackboard,
                producer="ExecutionAgent",
                artifact_type="execution_result",
                payload=execution,
                consumed_by=["AuditAgent", "AutonomyAgent", "NarrativeAgent", "VisualizationAgent"],
            )
            audit = self._run_agent(
                trace,
                "AuditAgent",
                "consistency_validation",
                self._audit_agent,
                plan,
                query_plan,
                execution,
                runtime,
            )
            self._blackboard_post(
                blackboard,
                producer="AuditAgent",
                artifact_type="audit_report",
                payload=audit,
                consumed_by=["AutonomyAgent", "NarrativeAgent"],
            )
            self._pipeline_warnings.extend(audit.get("warnings", []))

            # ── GAP 9: Post-audit feedback loop ──────────────────
            # If the first audit score is poor and there are concept misses,
            # re-invoke planning→query→execution→audit once to self-correct.
            audit_score = float(audit.get("score", 1.0))
            goal_term_misses = audit.get("grounding", {}).get("goal_term_misses", [])
            if audit_score < 0.5 and goal_term_misses and not intake.get("_replan_applied"):
                # GAP 34: Use LLM-enhanced diagnosis when available
                if runtime.use_llm and runtime.provider:
                    diagnosis = self._analyze_audit_failure_with_llm(
                        effective_goal, plan, query_plan, execution, audit, runtime,
                    )
                    if diagnosis:
                        enriched_intake = self._apply_llm_diagnosis_to_intake(intake, diagnosis, catalog)
                    else:
                        enriched_intake = self._enrich_intake_from_audit(intake, audit)
                else:
                    enriched_intake = self._enrich_intake_from_audit(intake, audit)
                enriched_intake["_replan_applied"] = True

                replan = self._run_agent(
                    trace,
                    "PlanningAgent",
                    "replan_after_audit",
                    self._planning_agent,
                    enriched_intake,
                    retrieval,
                    catalog,
                    runtime,
                )
                replan_qp = self._run_agent(
                    trace,
                    "QueryEngineerAgent",
                    "replan_sql_compilation",
                    self._query_engine_agent,
                    replan,
                    [tx_findings, customer_findings, revenue_findings, risk_findings],
                    runtime,
                    catalog,
                )
                replan_exec = self._run_agent(
                    trace,
                    "ExecutionAgent",
                    "replan_execution",
                    self._execution_agent,
                    replan_qp,
                )
                replan_audit = self._run_agent(
                    trace,
                    "AuditAgent",
                    "replan_audit",
                    self._audit_agent,
                    replan,
                    replan_qp,
                    replan_exec,
                    runtime,
                )
                replan_score = float(replan_audit.get("score", 0.0))
                if replan_score > audit_score:
                    plan = replan
                    query_plan = replan_qp
                    execution = replan_exec
                    audit = replan_audit
                    self._pipeline_warnings.append(
                        f"Query was automatically refined based on audit feedback "
                        f"(score {audit_score:.2f} → {replan_score:.2f})."
                    )
                else:
                    self._pipeline_warnings.append(
                        "Audit-driven re-plan attempted but did not improve results; "
                        "keeping original query."
                    )

            all_specialist_findings = [tx_findings, customer_findings, revenue_findings, risk_findings]
            autonomy_result = self._run_agent(
                trace,
                "AutonomyAgent",
                "bounded_reconciliation",
                self._autonomous_refinement_agent,
                effective_goal,
                plan,
                query_plan,
                execution,
                audit,
                catalog,
                memory_hints,
                learned_corrections,
                autonomy_cfg,
                tenant_id,
                specialist_findings=all_specialist_findings,
            )
            self._blackboard_post(
                blackboard,
                producer="AutonomyAgent",
                artifact_type="refinement_decision",
                payload={
                    "correction_applied": autonomy_result.get("correction_applied", False),
                    "correction_reason": autonomy_result.get("correction_reason", ""),
                    "evaluated_candidates": autonomy_result.get("evaluated_candidates", []),
                    "confidence_decomposition": autonomy_result.get("confidence_decomposition", []),
                    "contradiction_resolution": autonomy_result.get("contradiction_resolution", {}),
                    "refinement_rounds": autonomy_result.get("refinement_rounds", []),
                    "toolsmith_candidates": autonomy_result.get("toolsmith_candidates", []),
                },
                consumed_by=["NarrativeAgent", "ChiefAnalystAgent"],
            )
            plan = autonomy_result.get("plan", plan)
            query_plan = autonomy_result.get("query_plan", query_plan)
            execution = autonomy_result.get("execution", execution)
            audit = autonomy_result.get("audit", audit)

            narration = self._run_agent(
                trace,
                "NarrativeAgent",
                "business_communication",
                self._narrative_agent,
                goal,
                plan,
                execution,
                audit,
                runtime,
                storyteller_mode,
                history,
            )
            self._blackboard_post(
                blackboard,
                producer="NarrativeAgent",
                artifact_type="business_answer",
                payload=narration,
                consumed_by=["ChiefAnalystAgent"],
            )
            chart_spec = self._run_agent(
                trace,
                "VisualizationAgent",
                "chart_suggestion",
                self._viz_agent,
                plan,
                execution,
            )
            self._blackboard_post(
                blackboard,
                producer="VisualizationAgent",
                artifact_type="visualization_spec",
                payload=chart_spec,
                consumed_by=["ChiefAnalystAgent"],
            )

            confidence_score = max(0.0, min(1.0, float(audit.get("score", 0.6))))
            confidence = self._to_confidence(confidence_score)

            # ── GAP 11: Build contribution map from trace ─────────
            contribution_map = [
                {
                    "agent": entry["agent"],
                    "role": entry.get("role", ""),
                    "contribution": entry.get("contribution", ""),
                    "dropped_items": entry.get("dropped_items", []),
                    "status": entry.get("status", "unknown"),
                }
                for entry in trace
                if entry.get("contribution")
            ]
            audit_warnings = audit.get("warnings", [])
            grounding = audit.get("grounding", {})
            coverage_pct = grounding.get("concept_coverage_pct", 100.0)
            misses = grounding.get("goal_term_misses", [])
            confidence_reasoning = (
                f"Score {confidence_score:.2f} ({confidence.value}). "
                f"Concept coverage {coverage_pct}%"
                + (f", missed terms: {misses}" if misses else "")
                + (f", {len(audit_warnings)} audit warning(s)" if audit_warnings else "")
                + "."
            )

            evidence_packets = [
                {
                    "agent": "PlanningAgent",
                    "goal": plan.get("goal"),
                    "intent": plan.get("intent"),
                    "table": plan.get("table"),
                    "metric": plan.get("metric"),
                    "metric_expr": plan.get("metric_expr"),
                    "dimensions": plan.get("dimensions", []),
                    "time_filter": plan.get("time_filter"),
                    "value_filters": plan.get("value_filters", []),
                },
                {
                    "agent": "QueryEngineerAgent",
                    "table": plan["table"],
                    "metric": plan["metric"],
                    "sql": query_plan["sql"],
                    "row_count": execution["row_count"],
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                },
                {
                    "agent": "AuditAgent",
                    "score": confidence_score,
                    "warnings": audit.get("warnings", []),
                },
                {
                    "agent": "AutonomyAgent",
                    "mode": autonomy_cfg.mode,
                    "auto_correction": autonomy_cfg.auto_correction,
                    "strict_truth": autonomy_cfg.strict_truth,
                    "applied": autonomy_result.get("correction_applied", False),
                    "reason": autonomy_result.get("correction_reason", ""),
                    "evaluated_candidates": autonomy_result.get("evaluated_candidates", []),
                    "confidence_decomposition": autonomy_result.get("confidence_decomposition", []),
                    "contradiction_resolution": autonomy_result.get("contradiction_resolution", {}),
                    "refinement_rounds": autonomy_result.get("refinement_rounds", []),
                    "probe_findings": autonomy_result.get("probe_findings", []),
                    "toolsmith_candidates": autonomy_result.get("toolsmith_candidates", []),
                },
                {
                    "agent": "Blackboard",
                    "artifact_count": len(blackboard),
                    "artifacts": blackboard,
                    "edges": self._blackboard_edges(blackboard),
                },
            ]

            sanity_checks = [
                SanityCheck(check_name=check["name"], passed=check["passed"], message=check["message"])
                for check in audit.get("checks", [])
            ]

            evidence = [
                EvidenceItem(
                    description="Primary metric result",
                    value=str(narration.get("headline_value", execution.get("row_count", 0))),
                    source="query execution",
                    sql_reference=query_plan["sql"],
                ),
                EvidenceItem(
                    description="Metric mapping",
                    value=f"{plan.get('metric')} on {plan.get('table')}",
                    source="planning",
                    sql_reference=query_plan["sql"],
                ),
            ]

            total_ms = (time.perf_counter() - started) * 1000
            trace.append(
                {
                    "agent": "ChiefAnalystAgent",
                    "role": "finalize_response",
                    "status": "success",
                    "duration_ms": round(total_ms, 2),
                    "summary": "response assembled",
                }
            )
            self._memory_write_turn(
                trace=trace,
                trace_id=trace_id,
                goal=goal,
                resolved_goal=effective_goal,
                tenant_id=tenant_id,
                runtime=runtime,
                success=execution["success"],
                confidence_score=confidence_score,
                row_count=execution.get("row_count"),
                plan=plan,
                sql=query_plan.get("sql"),
                audit_warnings=audit.get("warnings", []),
                correction_applied=bool(autonomy_result.get("correction_applied", False)),
                correction_reason=autonomy_result.get("correction_reason"),
                metadata={
                    "autonomy_mode": autonomy_cfg.mode,
                    "strict_truth": autonomy_cfg.strict_truth,
                    "llm_intake_used": bool(intake.get("_llm_intake_used", False)),
                    "llm_narrative_used": bool(narration.get("llm_narrative_used", False)),
                    "evaluated_candidates": autonomy_result.get("evaluated_candidates", []),
                    "refinement_rounds": autonomy_result.get("refinement_rounds", []),
                    "toolsmith_candidates": autonomy_result.get("toolsmith_candidates", []),
                },
            )

            if execution["success"] and confidence_score >= 0.78:
                self._memory_learn_from_success(
                    trace,
                    tenant_id=tenant_id,
                    goal=effective_goal,
                    plan=plan,
                    score=confidence_score,
                )

            # ── Statistical analysis ──────────────────────────────
            stats_dict: dict[str, Any] = {}
            if execution["success"] and execution.get("sample_rows"):
                try:
                    import pandas as _pd

                    stats_df = _pd.DataFrame(execution["sample_rows"])
                    if not stats_df.empty:
                        stats_result = run_stats_analysis(stats_df)
                        stats_dict = stats_result.to_dict()
                        trace.append(
                            {
                                "agent": "StatsAgent",
                                "role": "statistical_analysis",
                                "status": "success",
                                "duration_ms": 0,
                                "summary": stats_result.summary,
                            }
                        )
                except Exception as stats_exc:
                    trace.append(
                        {
                            "agent": "StatsAgent",
                            "role": "statistical_analysis",
                            "status": "warning",
                            "duration_ms": 0,
                            "summary": f"Stats analysis skipped: {stats_exc}",
                        }
                    )

            return AssistantQueryResponse(
                success=execution["success"],
                answer_markdown=narration["answer_markdown"],
                confidence=confidence,
                confidence_score=confidence_score,
                definition_used=plan["definition_used"],
                evidence=evidence,
                sanity_checks=sanity_checks,
                sql=query_plan["sql"],
                row_count=execution["row_count"],
                columns=execution["columns"],
                sample_rows=execution["sample_rows"],
                execution_time_ms=execution["execution_time_ms"],
                trace_id=trace_id,
                runtime=self._runtime_payload(
                    runtime,
                    llm_intake_used=bool(intake.get("_llm_intake_used", False)),
                    llm_narrative_used=bool(narration.get("llm_narrative_used", False)),
                    autonomy=autonomy_cfg,
                    correction_applied=bool(autonomy_result.get("correction_applied", False)),
                )
                | {"blackboard_entries": len(blackboard)},
                agent_trace=trace,
                chart_spec=chart_spec,
                contribution_map=contribution_map,
                confidence_reasoning=confidence_reasoning,
                evidence_packets=evidence_packets,
                data_quality={
                    **catalog.get("quality", {}),
                    "audit_score": confidence_score,
                    "audit_warnings": audit.get("warnings", []),
                    "grounding": audit.get("grounding", {}),
                    "autonomy": {
                        "correction_applied": bool(autonomy_result.get("correction_applied", False)),
                        "correction_reason": autonomy_result.get("correction_reason"),
                        "evaluated_candidates": autonomy_result.get("evaluated_candidates", []),
                        "confidence_decomposition": autonomy_result.get("confidence_decomposition", []),
                        "contradiction_resolution": autonomy_result.get("contradiction_resolution", {}),
                        "refinement_rounds": autonomy_result.get("refinement_rounds", []),
                        "toolsmith_candidates": autonomy_result.get("toolsmith_candidates", []),
                    },
                    "blackboard": {"artifact_count": len(blackboard), "edges": self._blackboard_edges(blackboard)},
                },
                stats_analysis=stats_dict,
                suggested_questions=narration.get("suggested_questions", []),
                warnings=self._pipeline_warnings,
            )
        except Exception as exc:
            total_ms = (time.perf_counter() - started) * 1000
            trace.append(
                {
                    "agent": "ChiefAnalystAgent",
                    "role": "error_handler",
                    "status": "failed",
                    "duration_ms": round(total_ms, 2),
                    "summary": str(exc),
                }
            )
            return AssistantQueryResponse(
                success=False,
                answer_markdown=f"**Error executing dataDa agentic pipeline**\n\n{exc}",
                confidence=ConfidenceLevel.UNCERTAIN,
                confidence_score=0.0,
                definition_used=goal,
                evidence=[],
                sanity_checks=[
                    SanityCheck(check_name="pipeline", passed=False, message=str(exc))
                ],
                trace_id=trace_id,
                runtime=self._runtime_payload(runtime, autonomy=autonomy_cfg),
                error=str(exc),
                agent_trace=trace,
                suggested_questions=["Try a simpler business question."],
                warnings=getattr(self, "_pipeline_warnings", []),
            )

    @staticmethod
    def _extract_contribution(agent: str, out: Any) -> tuple[str, list[str], str]:
        """Derive a contribution string, dropped_items, and reasoning from agent output.

        Returns (contribution, dropped_items, reasoning) — GAP 39 adds the reasoning
        field so trace entries explain WHY decisions were made, not just WHAT.
        """
        contribution = ""
        dropped: list[str] = []
        reasoning = ""
        if not isinstance(out, dict):
            return contribution, dropped, reasoning

        if agent == "IntakeAgent":
            dims = out.get("dimensions", [])
            metric = out.get("metric", "")
            domain = out.get("domain", "")
            secondary = out.get("secondary_domains", [])
            contribution = f"domain={domain}; metric={metric}; dims={dims}"
            if secondary:
                contribution += f"; secondary_domains={secondary}"
            # Reasoning: explain defaulting
            reasons: list[str] = []
            if not dims:
                reasons.append("no grouping keywords detected")
            if metric and "count" in metric and not any(
                k in str(out.get("_raw_goal", "")).lower()
                for k in ["count", "how many", "total"]
            ):
                reasons.append(f"metric '{metric}' defaulted (no explicit metric term)")
            if secondary:
                reasons.append(f"cross-domain detected → will trigger JOIN planning")
            reasoning = "; ".join(reasons) if reasons else "standard intake classification"
        elif agent == "ChiefAnalystAgent":
            detected = out.get("detected_domains", [])
            multi = out.get("multi_domain_hint", False)
            analysis = out.get("analysis_type", "")
            contribution = f"detected_domains={detected}; multi_domain={multi}; analysis_type={analysis}"
            if multi:
                reasoning = f"detected_domains={detected}; multi_domain=true → will trigger JOIN planning"
            else:
                reasoning = f"single domain: {detected[0] if detected else 'unknown'}"
        elif agent == "PlanningAgent":
            override = out.get("_specialist_metric_override", "")
            contribution = (
                f"table={out.get('table')}, metric={out.get('metric')}, "
                f"dims={out.get('dimensions', [])}"
            )
            if override:
                contribution += f"; specialist_override: {override}"
                reasoning = f"metric_expr overridden by specialist to: {override}"
            else:
                reasoning = "standard plan from intake"
            dropped = [
                d for d in out.get("_dropped_dimensions", [])
            ] if out.get("_dropped_dimensions") else []
        elif agent == "QueryEngineerAgent":
            sql = str(out.get("sql", ""))[:120]
            contribution = f"sql={sql}"
            reasoning = "SQL generated from plan"
        elif agent == "ExecutionAgent":
            contribution = (
                f"rows={out.get('row_count')}, "
                f"cols={out.get('columns', [])}"
            )
            reasoning = f"executed successfully, {out.get('row_count', 0)} rows returned"
        elif agent == "AuditAgent":
            contribution = (
                f"score={out.get('score')}, "
                f"warnings={len(out.get('warnings', []))}, "
                f"misses={out.get('grounding', {}).get('goal_term_misses', [])}"
            )
            misses = out.get("grounding", {}).get("goal_term_misses", [])
            if misses:
                reasoning = f"goal terms not found in results: {misses}"
            else:
                reasoning = f"all goal terms grounded, score={out.get('score')}"
        elif agent == "AutonomyAgent":
            contribution = (
                f"correction={out.get('correction_applied', False)}, "
                f"reason={out.get('correction_reason', '')[:80]}"
            )
            if out.get("correction_applied"):
                reasoning = f"applied correction: {out.get('correction_reason', '')[:80]}"
            else:
                reasoning = "no correction needed"
        elif agent == "NarrativeAgent":
            contribution = f"headline={str(out.get('headline_value', ''))[:60]}"
            reasoning = "narrative formatted"
        elif agent == "VisualizationAgent":
            contribution = f"chart_type={out.get('type', 'unknown')}"
            reasoning = f"chart type: {out.get('type', 'unknown')}"
        elif agent == "MemoryAgent":
            # GAP 41b: Show correction signal when present
            if out.get("_memory_correction_applied"):
                contribution = f"memory_correction=true, applied learned correction"
                reasoning = "applied learned correction from similar past query"
            elif isinstance(out, dict) and "count" in out:
                contribution = f"matches={out.get('count', 0)}"
                reasoning = f"found {out.get('count', 0)} similar past queries"
            elif isinstance(out, list):
                contribution = f"matches={len(out)}"
                reasoning = f"found {len(out)} similar past queries"
            else:
                contribution = f"keys={list(out.keys())[:6]}"
                reasoning = "memory recall"
        elif agent.endswith("SpecialistAgent"):
            warnings = out.get("warnings", [])
            directives = out.get("directives", [])
            d_types = [d.get("type", "") for d in directives] if directives else []
            contribution = f"warnings={len(warnings)}"
            if directives:
                contribution += f", directives={len(directives)} ({', '.join(d_types)})"
                reasoning = "; ".join(d.get("reason", "") for d in directives if d.get("reason"))
            else:
                reasoning = f"{len(warnings)} warnings, no directives"
        elif agent == "ClarificationAgent":
            needs = out.get("needs_clarification", False)
            reason_str = out.get("reason", "")
            contribution = f"needs_clarification={needs}"
            if reason_str:
                contribution += f", reasons={reason_str}"
            reasoning = reason_str if reason_str else "no clarification needed"
        else:
            # Generic: summarise top-level keys
            contribution = f"keys={list(out.keys())[:6]}"

        return contribution, dropped, reasoning

    def _run_agent(
        self,
        trace: list[dict[str, Any]],
        agent: str,
        role: str,
        fn,
        *args,
        **kwargs,
    ) -> Any:
        start = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
            contribution, dropped_items, reasoning = self._extract_contribution(agent, out)
            trace.append(
                {
                    "agent": agent,
                    "role": role,
                    "status": "success",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": _compact(out),
                    "contribution": contribution,
                    "dropped_items": dropped_items,
                    "reasoning": reasoning,
                }
            )
            return out
        except Exception as exc:
            trace.append(
                {
                    "agent": agent,
                    "role": role,
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": str(exc),
                    "contribution": "",
                    "dropped_items": [],
                }
            )
            raise

    def _enrich_intake_from_audit(
        self,
        intake: dict[str, Any],
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge audit feedback into intake for a re-plan cycle (GAP 9).

        Extracts ``goal_term_misses`` and ``concept_misses`` from the audit
        grounding report, maps them to known dimension/metric keywords, and
        merges them into a copy of the intake payload.  The caller is
        responsible for setting ``_replan_applied`` on the result to prevent
        infinite loops.
        """
        enriched = dict(intake)
        grounding = audit.get("grounding", {})
        goal_term_misses = grounding.get("goal_term_misses", [])
        existing_dims: list[str] = list(enriched.get("dimensions") or [])

        # Map missed concepts to dimensions/metric hints
        concept_dim_hints: dict[str, str] = {
            "forex_domain": "datada_mart_quotes",
            "mt103": "has_mt103",
            "refund": "has_refund",
        }
        for miss in goal_term_misses:
            miss_lower = str(miss).lower().strip()
            if miss_lower in concept_dim_hints:
                hint = concept_dim_hints[miss_lower]
                if hint.startswith("datada_mart_"):
                    enriched["domain"] = hint.replace("datada_mart_", "")
                elif hint not in existing_dims:
                    existing_dims.append(hint)

        enriched["dimensions"] = existing_dims
        enriched["_audit_feedback"] = {
            "goal_term_misses": goal_term_misses,
            "original_score": audit.get("score", 0),
        }
        return enriched

    # ── GAP 34: LLM-Enhanced Audit Retry ──────────────────────────────
    def _analyze_audit_failure_with_llm(
        self,
        goal: str,
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
        runtime: RuntimeSelection,
    ) -> dict[str, Any] | None:
        """Use LLM to diagnose why the audit scored poorly and suggest fixes."""
        schema_info = ""
        table = plan.get("table", "")
        _cat = getattr(self.semantic, '_catalog', None) or {}
        if _cat:
            mart_meta = _cat.get("marts", {}).get(table, {})
            cols = mart_meta.get("columns", [])
            if cols:
                schema_info = f"Table {table} columns: {', '.join(cols)}"

        audit_summary = {
            "score": audit.get("score"),
            "warnings": audit.get("warnings", [])[:5],
            "goal_term_misses": audit.get("grounding", {}).get("goal_term_misses", []),
            "concept_coverage_pct": audit.get("grounding", {}).get("concept_coverage_pct"),
        }
        plan_summary = {
            "intent": plan.get("intent"),
            "domain": plan.get("domain"),
            "metric": plan.get("metric"),
            "dimensions": plan.get("dimensions", []),
            "table": table,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an audit diagnostician for a data analytics pipeline. "
                    "A query produced poor results. Diagnose the root cause and suggest specific fixes.\n\n"
                    f"Schema: {schema_info}\n\n"
                    "Available domains: transactions, customers, quotes, bookings\n\n"
                    "Return JSON with keys:\n"
                    "- diagnosis: string explaining what went wrong\n"
                    "- suggested_domain: string or null (correct domain if wrong)\n"
                    "- suggested_metric: string or null (correct metric if wrong)\n"
                    "- suggested_intent: string or null (correct intent if wrong)\n"
                    "- add_dimensions: list of dimension columns to add (empty if none)\n"
                    "- add_value_filters: list of {column, value} filters to add (empty if none)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {goal}\n"
                    f"Current plan: {json.dumps(plan_summary)}\n"
                    f"SQL executed: {query_plan.get('sql', '')}\n"
                    f"Row count: {execution.get('row_count', 0)}\n"
                    f"Audit findings: {json.dumps(audit_summary)}\n"
                    "Diagnose and suggest fixes. JSON only."
                ),
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="planner",
                provider=runtime.provider,
                timeout=25,
            )
            return _extract_json_payload(raw)
        except Exception as exc:
            self._pipeline_warnings.append(
                f"Audit failure analysis LLM call failed ({type(exc).__name__})"
            )
            return None

    def _apply_llm_diagnosis_to_intake(
        self,
        intake: dict[str, Any],
        diagnosis: dict[str, Any],
        catalog: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply validated LLM diagnosis suggestions to a copy of intake."""
        enriched = dict(intake)

        # Validate and apply suggested domain
        valid_domains = set(DOMAIN_TO_MART.keys())
        suggested_domain = diagnosis.get("suggested_domain")
        if isinstance(suggested_domain, str) and suggested_domain in valid_domains:
            enriched["domain"] = suggested_domain

        # Validate and apply suggested metric
        suggested_metric = diagnosis.get("suggested_metric")
        if isinstance(suggested_metric, str) and suggested_metric.strip():
            # Verify metric exists in catalog for the target domain
            target_domain = enriched.get("domain", "")
            mart_name = DOMAIN_TO_MART.get(target_domain, "")
            if catalog and mart_name:
                mart_meta = catalog.get("marts", {}).get(mart_name, {})
                known_metrics = set(mart_meta.get("metrics", {}).keys())
                if suggested_metric in known_metrics:
                    enriched["metric"] = suggested_metric

        # Validate and apply suggested intent
        valid_intents = {
            "metric", "grouped_metric", "comparison", "lookup", "trend_analysis",
            "percentile", "ranked_grouped", "running_total", "subquery_filter",
            "yoy_growth", "correlation", "data_overview", "schema_exploration",
        }
        suggested_intent = diagnosis.get("suggested_intent")
        if isinstance(suggested_intent, str) and suggested_intent in valid_intents:
            enriched["intent"] = suggested_intent

        # Validate and add dimensions
        add_dims = diagnosis.get("add_dimensions", [])
        if isinstance(add_dims, list):
            existing_dims = list(enriched.get("dimensions") or [])
            target_domain = enriched.get("domain", "")
            mart_name = DOMAIN_TO_MART.get(target_domain, "")
            valid_cols = set()
            if catalog and mart_name:
                valid_cols = set(catalog.get("marts", {}).get(mart_name, {}).get("columns", []))
            for dim in add_dims:
                if isinstance(dim, str) and dim in valid_cols and dim not in existing_dims:
                    existing_dims.append(dim)
            enriched["dimensions"] = existing_dims[:MAX_DIMENSIONS]

        # Validate and add value filters
        add_filters = diagnosis.get("add_value_filters", [])
        if isinstance(add_filters, list):
            existing_filters = list(enriched.get("value_filters") or [])
            for f in add_filters:
                if isinstance(f, dict) and f.get("column") and f.get("value"):
                    existing_filters.append({"column": str(f["column"]), "value": str(f["value"])})
            enriched["value_filters"] = existing_filters

        enriched["_audit_feedback"] = {
            "goal_term_misses": diagnosis.get("diagnosis", ""),
            "original_score": 0,
            "llm_diagnosis": True,
        }
        return enriched

    def _chief_analyst(
        self,
        goal: str,
        runtime: RuntimeSelection,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        result = self._chief_analyst_deterministic(goal, catalog)
        result["mission"] = goal
        result["runtime_mode"] = runtime.mode
        result["available_marts"] = list(catalog.get("marts", {}).keys())

        if runtime.use_llm and runtime.provider:
            llm_result = self._chief_analyst_with_llm(goal, runtime, catalog, result)
            if llm_result:
                result.update(llm_result)
        return result

    def _chief_analyst_deterministic(
        self,
        goal: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        lower = goal.lower()
        domain_keywords = {
            "transactions": ["transaction", "payment", "transfer", "mt103", "refund", "swift"],
            "customers": ["customer", "payee", "beneficiary", "university", "address"],
            "quotes": ["quote", "forex", "exchange", "markup", "spread", "charge"],
            "bookings": ["booking", "booked", "deal", "value date"],
        }
        # GAP 36a: Enrich domain keywords from domain knowledge synonyms
        for term, target_domain in self._domain_knowledge.get("synonyms", {}).items():
            if target_domain in domain_keywords and term not in domain_keywords[target_domain]:
                domain_keywords[target_domain].append(term)

        detected_domains: list[str] = []
        for domain, keywords in domain_keywords.items():
            if any(kw in lower for kw in keywords):
                detected_domains.append(domain)
        if not detected_domains:
            detected_domains = ["transactions"]

        priority = "normal"
        if any(kw in lower for kw in ["urgent", "critical", "asap", "immediately"]):
            priority = "high"
        elif any(kw in lower for kw in ["when you can", "low priority", "nice to have"]):
            priority = "low"

        multi_domain_hint = len(detected_domains) > 1
        specialists = ["transactions", "customers", "revenue", "risk"]

        analysis_type = "metric"
        if any(kw in lower for kw in ["compare", " vs ", "versus"]):
            analysis_type = "comparison"
        elif any(kw in lower for kw in ["trend", "over time", "historical"]):
            analysis_type = "trend"
        elif any(kw in lower for kw in ["top ", "rank", "best", "worst"]):
            analysis_type = "ranking"
        elif any(kw in lower for kw in ["breakdown", "by ", "per ", "split"]):
            analysis_type = "grouped"

        return {
            "specialists": specialists,
            "detected_domains": detected_domains,
            "primary_domain": detected_domains[0],
            "multi_domain_hint": multi_domain_hint,
            "priority": priority,
            "analysis_type": analysis_type,
        }

    def _chief_analyst_with_llm(
        self,
        goal: str,
        runtime: RuntimeSelection,
        catalog: dict[str, Any],
        deterministic_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        available_marts = list(catalog.get("marts", {}).keys())
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Chief Analyst agent. Analyze the user's data question and determine:\n"
                    "1. Which data domains are needed (transactions, customers, quotes, bookings)\n"
                    "2. What type of analysis: metric, grouped, comparison, trend, ranking\n"
                    "3. Whether the question is ambiguous or needs clarification\n\n"
                    "Return JSON with keys: detected_domains (list), analysis_type (string), "
                    "specialists (list), reasoning (string), is_ambiguous (bool).\n\n"
                    f"Available data tables: {available_marts}"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {goal}\nInitial analysis: {json.dumps(deterministic_result)}\nRefine if needed. JSON only.",
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="intent",
                provider=runtime.provider,
                model=runtime.intent_model,
                timeout=20,
            )
            parsed = _extract_json_payload(raw)
            if parsed and isinstance(parsed.get("detected_domains"), list):
                return {"_llm_chief_reasoning": parsed.get("reasoning", ""), **{
                    k: v for k, v in parsed.items()
                    if k in ("detected_domains", "analysis_type", "specialists")
                }}
        except Exception as exc:
            self._pipeline_warnings.append(
                f"Chief analyst LLM enhancement failed ({type(exc).__name__}); using deterministic analysis."
            )
        return None

    def _resolve_contextual_goal(
        self,
        goal: str,
        conversation_context: list[dict[str, Any]],
        runtime: RuntimeSelection,
    ) -> str:
        if not conversation_context:
            return goal

        lower = goal.lower().strip()
        tokens = re.findall(r"[a-z0-9_]+", lower)
        token_set = set(tokens)
        explicit_subject_terms = [
            "transaction",
            "quote",
            "booking",
            "customer",
            "refund",
            "mt103",
            "markup",
            "forex",
            "revenue",
            "amount",
            "rate",
            "count",
        ]
        has_explicit_subject = any(term in lower for term in explicit_subject_terms)
        has_reference_pronoun = any(term in token_set for term in {"same", "those", "it", "them", "previous"})
        has_reference_phrase = any(phrase in lower for phrase in ["that result", "that one", "same as above"])
        followup = (
            lower.startswith(("and ", "also ", "what about", "now ", "for that"))
            or has_reference_pronoun
            or has_reference_phrase
            or (len(tokens) <= 6 and not has_explicit_subject)
        )
        if not followup:
            return goal

        last_turn = conversation_context[-1]
        last_goal = str(last_turn.get("goal") or last_turn.get("user_goal") or "").strip()
        if not last_goal:
            return goal

        if runtime.use_llm and runtime.provider:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the follow-up question as a standalone analytics question. "
                        "Return JSON only with key standalone_goal."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "previous_question": last_goal,
                            "follow_up": goal,
                        }
                    ),
                },
            ]
            try:
                raw = call_llm(
                    messages,
                    role="intent",
                    provider=runtime.provider,
                    model=runtime.intent_model,
                    timeout=20,
                )
                parsed = _extract_json_payload(raw)
                rewritten = parsed.get("standalone_goal") if isinstance(parsed, dict) else None
                if isinstance(rewritten, str) and rewritten.strip():
                    return rewritten.strip()
            except Exception:
                pass

        return f"{goal}. Context: previous question was '{last_goal}'."

    def _data_overview_plan_agent(
        self,
        goal: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        marts = catalog.get("marts", {})
        tasks: list[dict[str, Any]] = []
        for mart, meta in marts.items():
            row_count = int(meta.get("row_count") or 0)
            if row_count <= 0:
                continue
            tasks.append(
                {
                    "mart": mart,
                    "row_count": row_count,
                    "priority": "high" if row_count > 10000 else "normal",
                    "focus": "distribution + timeline + top dimensions",
                }
            )
        tasks.sort(key=lambda x: x.get("row_count", 0), reverse=True)
        return {
            "goal": goal,
            "tasks": tasks,
            "plan_summary": f"profile {len(tasks)} active marts",
        }

    def _profile_sql_value(
        self,
        sql: str,
        *,
        column: str = "metric_value",
        default: Any = None,
    ) -> Any:
        result = self.executor.execute(sql)
        if not result.success or not result.rows:
            return default
        return result.rows[0].get(column, default)

    def _profile_sql_rows(self, sql: str, *, limit: int = 6) -> list[dict[str, Any]]:
        result = self.executor.execute(sql)
        if not result.success:
            return []
        return list(result.rows[: max(1, min(20, int(limit)))])

    def _data_overview_profile_agent(
        self,
        discovery_plan: dict[str, Any],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        del discovery_plan  # currently deterministic; plan reserved for richer strategy negotiation.
        marts = catalog.get("marts", {})
        profile: dict[str, Any] = {
            "marts": [],
            "domains": {},
            "highlights": [],
            "semantic_version": str(catalog.get("semantic_version") or ""),
            "quality": dict(catalog.get("quality") or {}),
        }
        for mart, meta in marts.items():
            profile["marts"].append(
                {
                    "mart": mart,
                    "row_count": int(meta.get("row_count") or 0),
                    "columns": list(meta.get("columns", []))[:24],
                }
            )
        profile["marts"].sort(key=lambda x: x["row_count"], reverse=True)

        if "datada_mart_transactions" in marts and int(marts["datada_mart_transactions"].get("row_count") or 0) > 0:
            tx = {
                "top_platforms": self._profile_sql_rows(
                    """
                    SELECT platform_name AS dimension, COUNT(DISTINCT transaction_key) AS metric_value
                    FROM datada_mart_transactions
                    WHERE platform_name IS NOT NULL AND TRIM(platform_name) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "top_states": self._profile_sql_rows(
                    """
                    SELECT state AS dimension, COUNT(DISTINCT transaction_key) AS metric_value
                    FROM datada_mart_transactions
                    WHERE state IS NOT NULL AND TRIM(state) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "mt103_count": int(
                    self._profile_sql_value(
                        """
                        SELECT COUNT(DISTINCT CASE WHEN has_mt103 THEN transaction_key END) AS metric_value
                        FROM datada_mart_transactions
                        """,
                        default=0,
                    )
                    or 0
                ),
                "refund_count": int(
                    self._profile_sql_value(
                        """
                        SELECT COUNT(DISTINCT CASE WHEN has_refund THEN transaction_key END) AS metric_value
                        FROM datada_mart_transactions
                        """,
                        default=0,
                    )
                    or 0
                ),
                "amount_total": float(
                    self._profile_sql_value(
                        "SELECT SUM(amount) AS metric_value FROM datada_mart_transactions",
                        default=0.0,
                    )
                    or 0.0
                ),
                "time_window": self._profile_sql_rows(
                    """
                    SELECT MIN(event_ts) AS min_ts, MAX(event_ts) AS max_ts
                    FROM datada_mart_transactions
                    """,
                    limit=1,
                ),
                "monthly_activity": self._profile_sql_rows(
                    """
                    SELECT
                        DATE_TRUNC('month', event_ts) AS dimension,
                        COUNT(DISTINCT transaction_key) AS metric_value
                    FROM datada_mart_transactions
                    WHERE event_ts IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1 DESC
                    LIMIT 12
                    """,
                    limit=12,
                ),
                "rare_platforms": self._profile_sql_rows(
                    """
                    SELECT platform_name AS dimension, COUNT(DISTINCT transaction_key) AS metric_value
                    FROM datada_mart_transactions
                    WHERE platform_name IS NOT NULL AND TRIM(platform_name) != ''
                    GROUP BY 1
                    ORDER BY 2 ASC, 1 ASC
                    LIMIT 4
                    """,
                    limit=4,
                ),
            }
            profile["domains"]["transactions"] = tx

        if "datada_mart_quotes" in marts and int(marts["datada_mart_quotes"].get("row_count") or 0) > 0:
            quotes = {
                "top_currency_pairs": self._profile_sql_rows(
                    """
                    SELECT CONCAT(COALESCE(from_currency, '?'), '->', COALESCE(to_currency, '?')) AS dimension,
                           COUNT(DISTINCT quote_key) AS metric_value
                    FROM datada_mart_quotes
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "forex_markup_revenue": float(
                    self._profile_sql_value(
                        "SELECT SUM(forex_markup) AS metric_value FROM datada_mart_quotes",
                        default=0.0,
                    )
                    or 0.0
                ),
                "charges_total": float(
                    self._profile_sql_value(
                        "SELECT SUM(total_additional_charges) AS metric_value FROM datada_mart_quotes",
                        default=0.0,
                    )
                    or 0.0
                ),
                "time_window": self._profile_sql_rows(
                    "SELECT MIN(created_ts) AS min_ts, MAX(created_ts) AS max_ts FROM datada_mart_quotes",
                    limit=1,
                ),
                "monthly_activity": self._profile_sql_rows(
                    """
                    SELECT
                        DATE_TRUNC('month', created_ts) AS dimension,
                        COUNT(DISTINCT quote_key) AS metric_value
                    FROM datada_mart_quotes
                    WHERE created_ts IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1 DESC
                    LIMIT 12
                    """,
                    limit=12,
                ),
                "rare_currency_pairs": self._profile_sql_rows(
                    """
                    SELECT CONCAT(COALESCE(from_currency, '?'), '->', COALESCE(to_currency, '?')) AS dimension,
                           COUNT(DISTINCT quote_key) AS metric_value
                    FROM datada_mart_quotes
                    GROUP BY 1
                    ORDER BY 2 ASC, 1 ASC
                    LIMIT 4
                    """,
                    limit=4,
                ),
            }
            profile["domains"]["quotes"] = quotes

        if "datada_dim_customers" in marts and int(marts["datada_dim_customers"].get("row_count") or 0) > 0:
            customers = {
                "top_countries": self._profile_sql_rows(
                    """
                    SELECT address_country AS dimension, COUNT(DISTINCT customer_key) AS metric_value
                    FROM datada_dim_customers
                    WHERE address_country IS NOT NULL AND TRIM(address_country) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "top_customer_types": self._profile_sql_rows(
                    """
                    SELECT customer_type AS dimension, COUNT(DISTINCT customer_key) AS metric_value
                    FROM datada_dim_customers
                    WHERE customer_type IS NOT NULL AND TRIM(customer_type) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "university_count": int(
                    self._profile_sql_value(
                        """
                        SELECT SUM(CASE WHEN is_university THEN 1 ELSE 0 END) AS metric_value
                        FROM datada_dim_customers
                        """,
                        default=0,
                    )
                    or 0
                ),
                "rare_countries": self._profile_sql_rows(
                    """
                    SELECT address_country AS dimension, COUNT(DISTINCT customer_key) AS metric_value
                    FROM datada_dim_customers
                    WHERE address_country IS NOT NULL AND TRIM(address_country) != ''
                    GROUP BY 1
                    ORDER BY 2 ASC, 1 ASC
                    LIMIT 4
                    """,
                    limit=4,
                ),
                "monthly_new_customers": self._profile_sql_rows(
                    """
                    SELECT
                        DATE_TRUNC('month', created_ts) AS dimension,
                        COUNT(DISTINCT customer_key) AS metric_value
                    FROM datada_dim_customers
                    WHERE created_ts IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1 DESC
                    LIMIT 12
                    """,
                    limit=12,
                ),
            }
            profile["domains"]["customers"] = customers

        if "datada_mart_bookings" in marts and int(marts["datada_mart_bookings"].get("row_count") or 0) > 0:
            bookings = {
                "top_deal_types": self._profile_sql_rows(
                    """
                    SELECT deal_type AS dimension, COUNT(DISTINCT booking_key) AS metric_value
                    FROM datada_mart_bookings
                    WHERE deal_type IS NOT NULL AND TRIM(deal_type) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "currency_mix": self._profile_sql_rows(
                    """
                    SELECT currency AS dimension, SUM(booked_amount) AS metric_value
                    FROM datada_mart_bookings
                    WHERE currency IS NOT NULL AND TRIM(currency) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "booked_total": float(
                    self._profile_sql_value(
                        "SELECT SUM(booked_amount) AS metric_value FROM datada_mart_bookings",
                        default=0.0,
                    )
                    or 0.0
                ),
                "monthly_activity": self._profile_sql_rows(
                    """
                    SELECT
                        DATE_TRUNC('month', booked_ts) AS dimension,
                        COUNT(DISTINCT booking_key) AS metric_value
                    FROM datada_mart_bookings
                    WHERE booked_ts IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1 DESC
                    LIMIT 12
                    """,
                    limit=12,
                ),
                "rare_deal_types": self._profile_sql_rows(
                    """
                    SELECT deal_type AS dimension, COUNT(DISTINCT booking_key) AS metric_value
                    FROM datada_mart_bookings
                    WHERE deal_type IS NOT NULL AND TRIM(deal_type) != ''
                    GROUP BY 1
                    ORDER BY 2 ASC, 1 ASC
                    LIMIT 4
                    """,
                    limit=4,
                ),
            }
            profile["domains"]["bookings"] = bookings

        has_docs_table = int(
            self._profile_sql_value(
                """
                SELECT COUNT(*) AS metric_value
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = 'datada_documents'
                """,
                default=0,
            )
            or 0
        ) > 0
        has_doc_chunks_table = int(
            self._profile_sql_value(
                """
                SELECT COUNT(*) AS metric_value
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = 'datada_document_chunks'
                """,
                default=0,
            )
            or 0
        ) > 0
        if has_docs_table:
            documents = {
                "document_count": int(
                    self._profile_sql_value(
                        "SELECT COUNT(DISTINCT doc_id) AS metric_value FROM datada_documents",
                        default=0,
                    )
                    or 0
                ),
                "top_file_types": self._profile_sql_rows(
                    """
                    SELECT file_type AS dimension, COUNT(*) AS metric_value
                    FROM datada_documents
                    WHERE file_type IS NOT NULL AND TRIM(file_type) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "top_titles": self._profile_sql_rows(
                    """
                    SELECT title AS dimension, token_count AS metric_value
                    FROM datada_documents
                    WHERE title IS NOT NULL AND TRIM(title) != ''
                    ORDER BY token_count DESC NULLS LAST, title ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
                "top_sources": self._profile_sql_rows(
                    """
                    SELECT source_path AS dimension, MAX(token_count) AS metric_value
                    FROM datada_documents
                    WHERE source_path IS NOT NULL AND TRIM(source_path) != ''
                    GROUP BY 1
                    ORDER BY 2 DESC NULLS LAST, 1 ASC
                    LIMIT 6
                    """,
                    limit=6,
                ),
            }
            if has_doc_chunks_table:
                documents["chunk_count"] = int(
                    self._profile_sql_value(
                        "SELECT COUNT(*) AS metric_value FROM datada_document_chunks",
                        default=0,
                    )
                    or 0
                )
            profile["domains"]["documents"] = documents

        if profile["marts"]:
            biggest = profile["marts"][0]
            profile["highlights"].append(
                f"Largest mart is {biggest['mart']} with {_fmt_number(biggest['row_count'])} rows."
            )
        if profile["domains"].get("quotes", {}).get("forex_markup_revenue", 0.0) > 0:
            markup = profile["domains"]["quotes"]["forex_markup_revenue"]
            profile["highlights"].append(f"Forex markup signal exists with total {_fmt_number(markup)}.")
        if profile["domains"].get("transactions", {}).get("mt103_count", 0) > 0:
            mt103 = profile["domains"]["transactions"]["mt103_count"]
            profile["highlights"].append(f"MT103 activity detected: {_fmt_number(mt103)} transactions.")
        if profile["domains"].get("documents", {}).get("document_count", 0) > 0:
            doc_count = profile["domains"]["documents"]["document_count"]
            profile["highlights"].append(
                f"Document corpus available with {_fmt_number(doc_count)} files for citation-backed Q&A."
            )
        semantic_version = str(catalog.get("semantic_version") or "")
        if semantic_version:
            profile["highlights"].append(f"Semantic profile version `{semantic_version}` is active.")
        coverage = dict((catalog.get("quality") or {}).get("coverage_by_domain") or {})
        if coverage:
            sparse_domain = min(coverage, key=lambda k: int(coverage.get(k) or 0))
            sparse_rows = int(coverage.get(sparse_domain) or 0)
            profile["highlights"].append(
                f"Smallest active domain is `{sparse_domain}` with {_fmt_number(sparse_rows)} rows."
            )
        return profile

    def _data_overview_agent(
        self,
        goal: str,
        catalog: dict[str, Any],
        storyteller_mode: bool,
        runtime: RuntimeSelection,
        overview_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        marts = list((overview_profile or {}).get("marts") or [])
        domains = dict((overview_profile or {}).get("domains") or {})
        highlights = list((overview_profile or {}).get("highlights") or [])
        if not marts:
            marts = [
                {"mart": mart, "row_count": int(meta.get("row_count") or 0)}
                for mart, meta in (catalog.get("marts", {}) or {}).items()
            ]
            marts.sort(key=lambda x: x["row_count"], reverse=True)

        map_lines = [f"- `{row['mart']}`: {_fmt_number(row['row_count'])} rows" for row in marts[:8]]

        tx = domains.get("transactions", {})
        quotes = domains.get("quotes", {})
        customers = domains.get("customers", {})
        bookings = domains.get("bookings", {})
        documents = domains.get("documents", {})
        quality = dict((overview_profile or {}).get("quality") or {})
        semantic_version = str(
            (overview_profile or {}).get("semantic_version")
            or catalog.get("semantic_version")
            or quality.get("semantic_version")
            or "n/a"
        )
        coverage_by_domain = dict(quality.get("coverage_by_domain") or {})

        def _rare_dimension(rows: Any) -> str:
            if not isinstance(rows, list) or not rows:
                return "n/a"
            row0 = rows[0] if isinstance(rows[0], dict) else {}
            return str(row0.get("dimension") or "n/a")

        insight_lines: list[str] = []
        if tx:
            top_platform = (tx.get("top_platforms") or [{}])[0]
            insight_lines.append(
                "Transactions: "
                f"{_fmt_number(tx.get('mt103_count', 0))} MT103, "
                f"{_fmt_number(tx.get('refund_count', 0))} refunds, "
                f"top platform `{top_platform.get('dimension', 'n/a')}`."
            )
        if quotes:
            top_pair = (quotes.get("top_currency_pairs") or [{}])[0]
            insight_lines.append(
                "Quotes: "
                f"forex markup total `{_fmt_number(quotes.get('forex_markup_revenue', 0.0))}`, "
                f"charges `{_fmt_number(quotes.get('charges_total', 0.0))}`, "
                f"leading pair `{top_pair.get('dimension', 'n/a')}`."
            )
        if customers:
            top_country = (customers.get("top_countries") or [{}])[0]
            insight_lines.append(
                "Customers: "
                f"{_fmt_number(customers.get('university_count', 0))} universities, "
                f"largest country segment `{top_country.get('dimension', 'n/a')}`."
            )
        if bookings:
            top_type = (bookings.get("top_deal_types") or [{}])[0]
            insight_lines.append(
                "Bookings: "
                f"booked total `{_fmt_number(bookings.get('booked_total', 0.0))}`, "
                f"top deal type `{top_type.get('dimension', 'n/a')}`."
            )
        if documents:
            top_type = (documents.get("top_file_types") or [{}])[0]
            insight_lines.append(
                "Documents: "
                f"{_fmt_number(documents.get('document_count', 0))} files"
                + (
                    f", {_fmt_number(documents.get('chunk_count', 0))} citation chunks"
                    if documents.get("chunk_count") is not None
                    else ""
                )
                + f", leading type `{top_type.get('dimension', 'n/a')}`."
            )

        rare_lines: list[str] = []
        if tx:
            rare_lines.append(f"Transactions rare platforms: `{_rare_dimension(tx.get('rare_platforms'))}`.")
        if quotes:
            rare_lines.append(f"Quotes rare currency pairs: `{_rare_dimension(quotes.get('rare_currency_pairs'))}`.")
        if customers:
            rare_lines.append(f"Customers rare countries: `{_rare_dimension(customers.get('rare_countries'))}`.")
        if bookings:
            rare_lines.append(f"Bookings rare deal types: `{_rare_dimension(bookings.get('rare_deal_types'))}`.")

        schema_lines = []
        for row in marts[:6]:
            columns = list(row.get("columns") or [])
            if not columns:
                continue
            schema_lines.append(f"- `{row['mart']}` key columns: {', '.join(columns[:6])}")

        intro = (
            "Here is the richer map of your data universe."
            if storyteller_mode
            else "Here is the full map of what data you have and where deeper signals live."
        )
        answer = (
            f"**{goal}**\n\n"
            f"{intro}\n\n"
            "**Data map**\n"
            + "\n".join(map_lines)
            + "\n\n**What is inside each stream**\n"
            + ("\n".join(f"- {line}" for line in insight_lines) if insight_lines else "- Core marts are available.")
        )
        if schema_lines:
            answer += "\n\n**Schema landmarks**\n" + "\n".join(schema_lines)
        if rare_lines:
            answer += "\n\n**Rare pockets worth exploring**\n" + "\n".join(f"- {line}" for line in rare_lines)
        if highlights:
            answer += "\n\n**Notable signals**\n" + "\n".join(f"- {line}" for line in highlights[:5])
        if coverage_by_domain:
            coverage_text = ", ".join(
                f"{k}={_fmt_number(v)}" for k, v in sorted(coverage_by_domain.items())
            )
            answer += f"\n\n**Coverage snapshot**\n- {coverage_text}\n- semantic version: `{semantic_version}`"
        answer += (
            "\n\nI can now drill down by platform, month, state, currency pair, customer segment, or deal type."
        )
        if runtime.use_llm and runtime.provider:
            answer += "\n(Generated with agentic profiling + narrative synthesis.)"

        sample_rows = marts[:8]
        sample_columns = ["mart", "row_count"]

        return {
            "answer_markdown": answer,
            "sample_rows": sample_rows,
            "columns": sample_columns,
            "suggested_questions": [
                "Show monthly trend of transaction amount and MT103 side by side",
                "Which currency pairs and platforms contribute most to forex markup?",
                "Where do refunds cluster by state and platform?",
                "Explain customer mix by country and type like a business briefing",
            ],
        }

    # ── GAP 32: Multi-part question decomposition ──────────────────
    def _detect_multi_part(
        self,
        goal: str,
        runtime: RuntimeSelection,
    ) -> list[str] | None:
        """Detect and decompose multi-part questions into sub-questions.

        Uses a deterministic pre-check first, then LLM decomposition if
        multi-part signals are found.  Returns None for single-part questions.
        """
        # Deterministic pre-check: skip LLM if clearly single-part
        multi_signals = [
            " and what", " and how", " and which", " and show",
            ", what", ", how", ", which", "? also", "? and ",
        ]
        if not any(s in goal.lower() for s in multi_signals) or len(goal) < 40:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a question decomposer for a data analytics pipeline.\n\n"
                    "Decompose the user's question into 2-4 independent sub-questions, "
                    "each answerable with a single SQL query.\n\n"
                    "RULES:\n"
                    "- Do NOT decompose multi-dimension GROUP BY ('by platform and month' = 1 question)\n"
                    "- Do NOT decompose comparisons ('this vs that' = 1 question)\n"
                    "- Each sub-question must be self-contained and reference the domain explicitly\n"
                    "- Maximum 4 sub-questions\n\n"
                    "Return JSON: {\"is_multi_part\": true, \"sub_questions\": [\"q1\", \"q2\", ...]} "
                    "or {\"is_multi_part\": false}"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {goal}\nJSON only.",
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="intent",
                provider=runtime.provider,
                model=runtime.intent_model,
                timeout=20,
            )
            parsed = _extract_json_payload(raw)
            if parsed and parsed.get("is_multi_part"):
                subs = parsed.get("sub_questions", [])
                if isinstance(subs, list) and 2 <= len(subs) <= 4:
                    return [str(s) for s in subs if isinstance(s, str) and s.strip()]
            return None
        except Exception as exc:
            self._pipeline_warnings.append(
                f"Multi-part detection LLM call failed ({type(exc).__name__})"
            )
            return None

    def _run_sub_query(
        self,
        sub_goal: str,
        runtime: RuntimeSelection,
        catalog: dict[str, Any],
        mission: dict[str, Any],
        memory_hints: list,
        retrieval_cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single sub-question through the intake→plan→query→exec→audit pipeline."""
        try:
            intake = self._intake_agent(sub_goal, runtime, mission, catalog, memory_hints)
            retrieval = self._semantic_retrieval_agent(intake, catalog)
            plan = self._planning_agent(intake, retrieval, catalog, runtime)
            query_plan = self._query_engine_agent(plan, [])
            execution = self._execution_agent(query_plan)

            # GAP 33: attempt recovery on sub-query failure too
            if not execution["success"] and runtime.use_llm and runtime.provider:
                recovery = self._recover_failed_sql(plan, query_plan, execution, runtime)
                if recovery and recovery.get("sql"):
                    validation = self.executor.validate(recovery["sql"])
                    if validation.is_valid:
                        recovered_qp = {"sql": recovery["sql"], "table": query_plan["table"]}
                        recovered_exec = self._execution_agent(recovered_qp)
                        if recovered_exec["success"]:
                            query_plan = recovered_qp
                            execution = recovered_exec

            audit = self._audit_agent(plan, query_plan, execution, runtime)
            return {
                "sub_goal": sub_goal,
                "plan": plan,
                "query_plan": query_plan,
                "execution": execution,
                "audit": audit,
                "success": execution["success"],
            }
        except Exception as exc:
            return {
                "sub_goal": sub_goal,
                "plan": {"goal": sub_goal, "intent": "metric", "table": "", "metric": "", "metric_expr": "", "dimensions": [], "definition_used": sub_goal, "top_n": 20, "value_filters": [], "time_filter": None},
                "query_plan": {"sql": "", "table": ""},
                "execution": {"success": False, "error": str(exc), "row_count": 0, "columns": [], "sample_rows": [], "execution_time_ms": 0, "sql_executed": "", "warnings": []},
                "audit": {"score": 0, "warnings": [str(exc)], "checks": [], "grounding": {}},
                "success": False,
            }

    def _merge_multi_part_results(
        self,
        goal: str,
        sub_results: list[dict[str, Any]],
        runtime: RuntimeSelection,
    ) -> tuple[dict, dict, dict, dict]:
        """Merge sub-query results into composite plan/query_plan/execution/audit."""
        sub_plans = [r["plan"] for r in sub_results]
        all_sample_rows: list[dict] = []
        all_columns: list[str] = []
        total_row_count = 0
        all_warnings: list[str] = []
        min_score = 1.0
        combined_sql_parts: list[str] = []
        any_success = False

        for r in sub_results:
            exec_r = r["execution"]
            audit_r = r["audit"]
            if exec_r.get("success"):
                any_success = True
            total_row_count += exec_r.get("row_count", 0)
            all_sample_rows.extend(exec_r.get("sample_rows", []))
            for col in exec_r.get("columns", []):
                if col not in all_columns:
                    all_columns.append(col)
            all_warnings.extend(audit_r.get("warnings", []))
            score = float(audit_r.get("score", 0))
            if score < min_score:
                min_score = score
            sql = r["query_plan"].get("sql", "")
            if sql:
                combined_sql_parts.append(f"/* Sub-query: {r['sub_goal']} */\n{sql}")

        # Build composite plan
        plan = {
            "goal": goal,
            "intent": "multi_part",
            "table": sub_plans[0].get("table", "") if sub_plans else "",
            "metric": "multi_part_composite",
            "metric_expr": "N/A",
            "dimensions": [],
            "definition_used": goal,
            "top_n": 20,
            "value_filters": [],
            "time_filter": None,
            "sub_plans": sub_plans,
            "_multi_part": True,
        }

        query_plan = {
            "sql": "\n;\n".join(combined_sql_parts),
            "table": plan["table"],
        }

        execution = {
            "success": any_success,
            "sql_executed": query_plan["sql"],
            "error": None if any_success else "All sub-queries failed",
            "row_count": total_row_count,
            "columns": all_columns,
            "sample_rows": all_sample_rows[:25],
            "execution_time_ms": sum(
                r["execution"].get("execution_time_ms", 0) for r in sub_results
            ),
            "warnings": [],
            "_sub_results": sub_results,
        }

        audit = {
            "score": min_score,
            "warnings": list(dict.fromkeys(all_warnings))[:10],
            "checks": [],
            "grounding": {
                "concept_coverage_pct": 100.0 if any_success else 0.0,
                "goal_term_misses": [],
            },
        }

        return plan, query_plan, execution, audit

    def _intake_agent(
        self,
        goal: str,
        runtime: RuntimeSelection,
        mission: dict[str, Any],
        catalog: dict[str, Any],
        memory_hints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        parsed = self._intake_deterministic(goal, catalog)
        parsed["_llm_intake_used"] = False
        deterministic = dict(parsed)
        explicit_top = bool(re.search(r"\btop\s+\d+\b", goal.lower()))
        if runtime.use_llm and runtime.provider:
            llm_parsed = self._intake_with_llm(goal, runtime, parsed, catalog)
            if llm_parsed:
                cleaned = self._sanitize_intake_payload(llm_parsed)
                if not explicit_top:
                    cleaned.pop("top_n", None)
                if cleaned:
                    merged = {
                        **parsed,
                        **{k: v for k, v in cleaned.items() if v not in (None, "")},
                    }
                    _is_llm = bool(runtime.use_llm and runtime.provider)
                    merged = self._enforce_intake_consistency(goal, deterministic, merged, catalog, is_llm_mode=_is_llm)
                    parsed.update(merged)
                    parsed["_llm_intake_used"] = True
        _is_llm = bool(runtime.use_llm and runtime.provider)
        parsed = self._enforce_intake_consistency(goal, deterministic, parsed, catalog, is_llm_mode=_is_llm)
        if memory_hints:
            parsed = self._apply_memory_hints(goal, parsed, memory_hints)
        return parsed

    def _clarification_agent(
        self,
        goal: str,
        intake: dict[str, Any],
        conversation_context: list[dict[str, Any]],
        memory_hints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Detect ambiguous goals and request clarification before execution."""

        lower = goal.lower().strip()
        tokens = re.findall(r"[a-z0-9_]+", lower)
        grouped_signal = any(
            marker in lower for marker in [" by ", "split", "breakdown", "wise", "trend", "per "]
        )
        followup_signal = any(
            marker in lower
            for marker in [
                "what about",
                "show me",
                "same",
                "that one",
                "for that",
                "now ",
                "only ",
                "this month",
                "last month",
            ]
        )
        domain_terms = [
            "transaction",
            "quote",
            "booking",
            "customer",
            "order",
            "product",
            "region",
            "refund",
            "mt103",
            "markup",
            "forex",
            "charge",
            "payee",
            "university",
            "document",
            "pdf",
            "currency pair",
            "currency combination",
        ]
        metric_terms = [
            "count",
            "how many",
            "amount",
            "sum",
            "total",
            "average",
            "avg",
            "rate",
            "revenue",
            "value",
            "volume",
            "most common",
            "most popular",
        ]
        has_domain_term = any(t in lower for t in domain_terms)
        has_metric_term = any(t in lower for t in metric_terms)
        explicit_time = bool(intake.get("time_filter"))
        intent = str(intake.get("intent") or "")
        metric = str(intake.get("metric") or "")

        reasons: list[str] = []
        if intent in {"data_overview", "document_qa", "schema_exploration"}:
            return {
                "needs_clarification": False,
                "reason": "",
                "questions": [],
                "suggested_questions": [],
                "ambiguity_score": 0.0,
            }

        if followup_signal and not conversation_context and not (has_domain_term or has_metric_term):
            reasons.append("follow_up_without_history")
        if (
            grouped_signal
            and not has_metric_term
            and not has_domain_term
            and metric in {"transaction_count", "quote_count", "booking_count"}
        ):
            reasons.append("grouping_without_metric_choice")
        if len(tokens) <= 7 and not has_domain_term and not has_metric_term:
            reasons.append("goal_too_brief_without_scope")
        if lower in {"only december", "this month", "last month", "what about this month"}:
            reasons.append("time_reference_without_business_scope")
        if not has_domain_term and metric == "transaction_count" and len(tokens) <= 9:
            reasons.append("default_metric_fallback_without_explicit_domain")
        # Detect when value filters reference terms not in the schema
        vf_terms = [str(vf.get("value", "")).lower() for vf in (intake.get("value_filters") or [])]
        if not vf_terms and has_domain_term:
            # Check if goal mentions filtering concepts that couldn't be resolved
            filter_hints = ["only", "just", "specific", "particular", "where"]
            if any(h in lower for h in filter_hints) and not any(t in lower for t in metric_terms):
                reasons.append("possible_filter_intent_unresolved")

        # ── GAP 38a: Unique intent without DISTINCT ──────────────────
        unique_triggers = self._domain_knowledge.get("business_rules", {}).get(
            "unique_intent", {}
        ).get("triggers", ["unique", "distinct", "individual", "different"])
        has_unique_intent = any(t in lower for t in unique_triggers)
        metric_expr_upper = str(intake.get("metric_expr", "")).upper()
        if has_unique_intent and "DISTINCT" not in metric_expr_upper:
            reasons.append("unique_intent_without_distinct_metric")

        # ── GAP 38b: Cross-domain entities without join plan ──────────────────
        intake_domains = intake.get("domains_detected", [])
        if len(intake_domains) > 1 and not intake.get("secondary_domains"):
            reasons.append("cross_domain_entities_without_join_plan")

        # ── GAP 38c: Metric-domain mismatch (synonym check) ──────────────────
        entity_domains: set[str] = set()
        for token in re.findall(r"[a-z]+", lower):
            if token in self._domain_knowledge.get("synonyms", {}):
                entity_domains.add(self._domain_knowledge["synonyms"][token])
        # Only fire mismatch if entity domains aren't already captured as
        # detected domains (cross-domain queries are handled by secondary_domains)
        all_detected = set(intake.get("domains_detected", []))
        unresolved_entity_domains = entity_domains - all_detected - {intake.get("domain")}
        if unresolved_entity_domains:
            reasons.append("metric_domain_mismatch")

        # ── GAP 38d: unique_intent is auto-fixable by specialists ──────────────
        # Always remove unique_intent_without_distinct_metric as a soft warning —
        # specialists (GAP 37) will apply COUNT(DISTINCT) override automatically
        if "unique_intent_without_distinct_metric" in reasons:
            self._pipeline_warnings.append(
                "Unique/distinct intent detected — specialists will apply COUNT(DISTINCT) override."
            )
            reasons.remove("unique_intent_without_distinct_metric")

        needs_clarification = bool(reasons)
        questions: list[str] = []
        if needs_clarification:
            if not has_domain_term:
                questions.append(
                    "Which domain should I analyze: transactions, quotes, customers, bookings, or documents?"
                )
            if grouped_signal:
                questions.append(
                    "Which metric should I split (for example transaction count, total amount, MT103 count, refunds, or markup revenue)?"
                )
            if not explicit_time:
                questions.append("What time window should I use (all time, this month, or a specific month/year)?")
            # GAP 38e: Questions for new reason types
            if "cross_domain_entities_without_join_plan" in reasons:
                questions.append(
                    "This query spans multiple data domains. Should I join them or focus on a single domain?"
                )
            if "metric_domain_mismatch" in reasons:
                questions.append(
                    "The entities you mentioned belong to a different domain than the primary query. "
                    "Should I include a cross-domain JOIN?"
                )
            if "unique_intent_without_distinct_metric" in reasons:
                questions.append(
                    "You asked for unique/distinct results. Should I count distinct entities (e.g., unique customers)?"
                )
            questions = questions[:3]

        suggestions: list[str] = []
        if questions:
            suggestions.extend(
                [
                    "Transaction count split by month and platform for December 2025",
                    "Forex markup revenue by month from quotes",
                    "Booking amount trend by month for this year",
                ]
            )
        if memory_hints:
            top = memory_hints[0] if memory_hints else {}
            hinted_goal = str(top.get("goal") or "").strip()
            if hinted_goal:
                suggestions.insert(0, hinted_goal)
        suggestions = list(dict.fromkeys(suggestions))[:6]

        return {
            "needs_clarification": needs_clarification,
            "reason": ", ".join(reasons),
            "questions": questions,
            "suggested_questions": suggestions,
            "ambiguity_score": round(min(1.0, len(reasons) * 0.25), 2),
        }

    def _intake_with_llm(
        self,
        goal: str,
        runtime: RuntimeSelection,
        fallback: dict[str, Any],
        catalog: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        # GAP 30: Build schema summary from catalog for LLM context
        schema_lines: list[str] = []
        if catalog:
            for mart_name, mart_meta in catalog.get("marts", {}).items():
                cols = mart_meta.get("columns", [])
                schema_lines.append(f"  {mart_name}: {', '.join(cols[:20])}")
        schema_block = "\n".join(schema_lines) if schema_lines else "  (no schema available)"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are IntakeAgent in a data analytics pipeline. Classify the user's question independently into a structured query plan.\n\n"
                    "Return ONLY valid JSON with these keys:\n"
                    "- intent: one of 'metric', 'grouped_metric', 'comparison', 'lookup', 'trend_analysis', "
                    "'percentile', 'ranked_grouped', 'running_total', 'subquery_filter', 'yoy_growth', "
                    "'correlation', 'data_overview', 'schema_exploration'\n"
                    "- domain: one of 'transactions', 'customers', 'quotes', 'bookings'\n"
                    "- metric: the metric name (e.g. 'transaction_count', 'total_amount', 'avg_amount', "
                    "'refund_count', 'mt103_count', 'quote_count', 'total_quote_value', 'booking_count', "
                    "'total_booked_amount', 'forex_markup_revenue', 'customer_count')\n"
                    "- dimensions: list of column names to GROUP BY (empty list if not a grouped query)\n"
                    "- time_filter: time period if mentioned (e.g. 'december', 'last month'), null if not\n"
                    "- value_filters: list of {column, operator, value} for WHERE conditions\n\n"
                    "DOMAIN SYNONYMS:\n"
                    "- transactions = payments, transfers, remittances, wire transfers, money sent, money received\n"
                    "- quotes = currency exchange, forex quotes, exchange rates, FX\n"
                    "- customers = clients, users, accounts, payers, payees\n"
                    "- bookings = reservations, booked deals, confirmed orders\n\n"
                    f"SCHEMA:\n{schema_block}\n\n"
                    "RULES:\n"
                    "- Only add dimensions if the user explicitly asks for a breakdown/grouping (by/per/split/grouped)\n"
                    "- For 'how many' or 'total' questions WITHOUT 'by/per/split', use intent='metric' with empty dimensions\n"
                    "- For 'average' questions, use the avg_ metric variant, NOT the total with a GROUP BY\n"
                    "- Use 'running_total' for cumulative/running total questions\n"
                    "- Use 'subquery_filter' for above/below average filtering questions\n"
                    "- Use 'yoy_growth' for year-over-year or period growth questions\n"
                    "- Use 'correlation' for correlation/relationship between two variables\n"
                    "- Use 'trend_analysis' for trend/moving average questions\n"
                    "- Use 'percentile' for percentile/median/P95 questions\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {goal}\n"
                    f"Deterministic hint (cross-check only): {json.dumps(fallback)}\n"
                    "Classify independently. JSON only."
                ),
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="intent",
                provider=runtime.provider,
                model=runtime.intent_model,
                timeout=30,
            )
        except Exception as exc:
            self._pipeline_warnings.append(
                f"LLM intake refinement failed ({type(exc).__name__}: {exc})"
            )
            return None
        return _extract_json_payload(raw)

    def _goal_has_count_intent(self, lower: str) -> bool:
        return any(k in lower for k in ["count", "how many", "number of", "volume", "qty", "quantity"])

    def _goal_has_amount_intent(self, lower: str) -> bool:
        return any(
            k in lower
            for k in [
                "amount",
                "revenue",
                "value",
                "sum",
                "markup",
                "charge",
                "charges",
                "fee",
                "fees",
                "booked",
            ]
        )

    def _build_dim_candidates_from_catalog(
        self,
        domain: str,
        catalog: dict[str, Any],
    ) -> dict[str, str]:
        """Dynamically build keyword→column dimension mapping from catalog."""
        mart = DOMAIN_TO_MART.get(domain, "")
        dim_value_cols = set(catalog.get("dimension_values", {}).get(mart, {}).keys())
        all_columns = set(catalog.get("marts", {}).get(mart, {}).get("columns", []))

        # Candidate columns: dimension value columns + columns with dimension-like suffixes
        dim_suffixes = ("_name", "_type", "_status", "_code", "_country", "_state", "_flow")
        exclude_suffixes = ("_key", "_id", "_ts")

        candidate_cols: set[str] = set(dim_value_cols)
        for col in all_columns:
            if any(col.endswith(s) for s in dim_suffixes):
                candidate_cols.add(col)
            if any(col.endswith(s) for s in exclude_suffixes):
                candidate_cols.discard(col)

        # Remove key/id/ts columns from candidates
        candidate_cols = {
            c for c in candidate_cols
            if not any(c.endswith(s) for s in exclude_suffixes)
        }

        mapping: dict[str, str] = {}
        for col in sorted(candidate_cols):
            # Exact column name
            mapping[col] = col

            # Prefix-stripped alias: address_country -> country, txn_flow -> flow
            parts = col.split("_")
            if len(parts) > 1:
                suffix = parts[-1]
                if suffix not in mapping:
                    mapping[suffix] = col

            # Split tokens: platform_name -> platform
            for part in parts:
                if len(part) >= 3 and part not in mapping:
                    mapping[part] = col

        # Always add month
        mapping["month"] = "__month__"

        return mapping

    def _resolve_metric_from_catalog(
        self,
        lower: str,
        domain: str,
        catalog: dict[str, Any],
    ) -> str:
        """Score candidate metrics from the catalog against goal tokens."""
        mart = DOMAIN_TO_MART.get(domain, "")
        metrics = catalog.get("metrics_by_table", {}).get(mart, {})
        if not metrics:
            return "transaction_count"

        domain_defaults = {
            "transactions": "transaction_count",
            "quotes": "quote_count",
            "customers": "customer_count",
            "bookings": "booking_count",
        }

        high_affinity: dict[str, list[str]] = {
            "refund": ["refund_count", "refund_rate"],
            "mt103": ["mt103_count", "mt103_rate"],
            "markup": ["forex_markup_revenue", "avg_forex_markup"],
            "payee": ["payee_count"],
            "university": ["university_count"],
            "charge": ["total_charges"],
            "fee": ["total_charges"],
        }

        has_count_words = self._goal_has_count_intent(lower)
        has_amount_words = self._goal_has_amount_intent(lower)
        has_avg = any(k in lower for k in ["avg", "average", "mean"])

        best_metric = domain_defaults.get(domain, "transaction_count")
        best_score = 0

        for metric_name in metrics:
            score = 0
            name_tokens = re.split(r"[_\s]+", metric_name)

            # Token overlap: each metric name token found in goal gets +2
            for token in name_tokens:
                if token and token in lower:
                    score += 2

            # High-affinity keywords
            for keyword, affinity_metrics in high_affinity.items():
                if keyword in lower and metric_name in affinity_metrics:
                    score += 3

            # Aggregation affinity
            if has_count_words and metric_name.endswith("_count"):
                score += 1
            if has_amount_words and any(
                t in metric_name for t in ["amount", "value", "revenue", "charges", "booked"]
            ):
                score += 1
            if has_avg and metric_name.startswith("avg_"):
                score += 2

            if score > best_score:
                best_score = score
                best_metric = metric_name

        return best_metric

    def _detect_schema_exploration(self, lower: str, catalog: dict[str, Any]) -> str | None:
        """Return domain string if the goal is a schema exploration question, else None."""
        schema_signals = [
            r"\bwhat\s+(?:do\s+we|does\s+it|do\s+i)\s+(?:capture|track|store|have|record)\b",
            r"\bwhat\s+(?:fields?|columns?|attributes?|demographics?|properties|info|information)\b",
            r"\bdescribe\s+the\b",
            r"\bwhat\s+are\s+the\b.*\b(?:fields?|columns?|attributes?|demographics?)\b",
            r"\bwhat\s+information\b",
            r"\bschema\b",
            r"\bstructure\s+of\b",
            r"\bwhat\s+(?:does|is\s+in)\s+the\s+\w+\s+table\b",
            r"\bwhat\s+(?:data|details?)\s+(?:do|does|is)\b.*\b(?:have|contain|include|store)\b",
        ]
        if not any(re.search(p, lower) for p in schema_signals):
            return None
        domain_keywords = {
            "customer": "customers",
            "transaction": "transactions",
            "quote": "quotes",
            "booking": "bookings",
        }
        for keyword, domain in domain_keywords.items():
            if keyword in lower:
                return domain
        return None

    def _schema_exploration_agent(
        self,
        goal: str,
        domain: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a rich markdown description of a domain's schema from the catalog."""
        mart = DOMAIN_TO_MART.get(domain, "")
        mart_meta = catalog.get("marts", {}).get(mart, {})
        columns = mart_meta.get("columns", [])
        row_count = mart_meta.get("row_count", 0)
        metrics = catalog.get("metrics_by_table", {}).get(mart, {})
        dim_values = catalog.get("dimension_values", {}).get(mart, {})
        time_col = catalog.get("preferred_time_column", {}).get(mart, "")

        lines: list[str] = []
        lines.append(f"## {domain.title()} Schema (`{mart}`)")
        lines.append(f"**Rows:** {row_count:,}  |  **Columns:** {len(columns)}")
        lines.append("")

        lines.append("### Columns")
        for col in columns:
            annotations: list[str] = []
            if col == time_col:
                annotations.append("time column")
            if col in dim_values:
                top_vals = dim_values[col][:5]
                annotations.append(f"values: {', '.join(top_vals)}")
            if col.endswith(("_key", "_id")):
                annotations.append("identifier")
            suffix = f"  _({'; '.join(annotations)})_" if annotations else ""
            lines.append(f"- `{col}`{suffix}")
        lines.append("")

        if metrics:
            lines.append("### Available Metrics")
            for name, expr in metrics.items():
                lines.append(f"- **{name}**: `{expr}`")
            lines.append("")

        suggested: list[str] = []
        if metrics:
            first_metric = next(iter(metrics))
            suggested.append(f"How many {domain} do we have?")
            suggested.append(f"Show me {first_metric} by month")
        if dim_values:
            first_dim = next(iter(dim_values))
            suggested.append(f"What is the breakdown of {domain} by {first_dim}?")
        suggested.append(f"What is the total {domain} count this year?")

        lines.append("### Suggested Follow-Up Queries")
        for q in suggested[:4]:
            lines.append(f"- {q}")

        return {
            "answer_markdown": "\n".join(lines),
            "domain": domain,
            "table": mart,
            "columns": columns,
            "row_count": row_count,
            "suggested_questions": suggested[:4],
        }

    def _intake_deterministic(self, goal: str, catalog: dict[str, Any]) -> dict[str, Any]:
        g = goal.strip()
        lower = g.lower()

        data_overview_patterns = [
            r"\bwhat\s+(kind\s+of\s+)?data\b",
            r"\bwhat\s+can\s+i\s+ask\b",
            r"\bshow\s+available\s+data\b",
            r"\bdescribe\s+(my\s+)?data\b",
            r"\bwhat\s+tables?\b",
            r"\bdata\s+overview\b",
            r"\bwhat\s+do\s+we\s+have\b",
            r"\bwhat\s+is\s+available\b",
        ]

        intent = "metric"
        schema_domain = self._detect_schema_exploration(lower, catalog)
        if schema_domain is not None:
            intent = "schema_exploration"
        elif any(re.search(p, lower) for p in data_overview_patterns):
            intent = "data_overview"
        elif any(
            k in lower
            for k in [
                "document",
                "pdf",
                "docx",
                "policy",
                "contract",
                "handbook",
                "manual",
                "what does this file say",
                "from the docs",
            ]
        ):
            intent = "document_qa"
        elif any(k in lower for k in [" vs ", " versus ", "compare", "compared"]):
            intent = "comparison"
        # ── GAP 20: Complex analytical patterns (checked before grouped_metric
        #    to avoid false matches on keywords like "by", "per", etc.) ──
        elif any(k in lower for k in ["running total", "cumulative sum", "cumulative"]):
            intent = "running_total"
        elif any(k in lower for k in ["above average", "below average", "above-average", "below-average"]):
            intent = "subquery_filter"
        elif any(k in lower for k in ["year over year", "yoy", "year-over-year"]):
            intent = "yoy_growth"
        elif any(k in lower for k in ["correlation", "correlate"]):
            intent = "correlation"
        # RC5: Superlative patterns like "which country has the most payees"
        elif re.search(r'\bwhich\s+\w+\s+(?:has|have|had)\s+(?:the\s+)?(?:most|least|highest|lowest|fewest|largest|smallest)', lower):
            intent = "grouped_metric"
        elif any(
            k in lower
            for k in [
                "top ",
                "split by",
                "split ",
                "split",
                "breakdown",
                " by ",
                " per ",
                "each ",
                "month wise",
                "platform wise",
                "country wise",
            ]
        ):
            intent = "grouped_metric"
        elif any(k in lower for k in ["list", "show rows", "display rows", "show me all"]):
            intent = "lookup"
        elif any(k in lower for k in ["trend", "over time", "moving average"]):
            intent = "trend_analysis"
        elif any(k in lower for k in ["percentile", "median", "p90", "p95"]):
            intent = "percentile"
        elif any(k in lower for k in ["top", "bottom", "rank"]) and " by " in lower:
            intent = "ranked_grouped"

        domain = "transactions"
        if any(
            k in lower
            for k in [
                "document",
                "pdf",
                "docx",
                "policy",
                "contract",
                "handbook",
                "manual",
                "from the docs",
                "from documents",
            ]
        ):
            domain = "documents"
        elif any(
            k in lower
            for k in [
                "quote",
                "forex",
                "exchange rate",
                "amount to be paid",
                "markup",
                "charge",
                "spread",
                "currency pair",
                "currency combination",
            ]
        ):
            domain = "quotes"
        elif any(k in lower for k in ["booking", "booked", "deal type", "value date"]):
            domain = "bookings"
        elif any(k in lower for k in ["payee", "university", "address", "beneficiary"]):
            domain = "customers"
        elif "customer" in lower and "transaction" not in lower and "amount" not in lower:
            domain = "customers"
        # GAP 36c: Resolve synonyms from domain knowledge for domain routing
        elif any(
            term in lower
            for term, target in self._domain_knowledge.get("synonyms", {}).items()
            if target == "customers"
        ):
            # "users", "clients", etc. → customers domain (when not already matched above)
            if "transaction" not in lower and "amount" not in lower:
                domain = "customers"

        if intent == "schema_exploration" and schema_domain:
            domain = schema_domain

        # ── GAP 13: Multi-domain detection (early, needed by dimension discovery) ──
        domain_keyword_lists = {
            "transactions": ["transaction", "payment", "transfer", "mt103", "refund", "swift"],
            "quotes": ["quote", "forex", "exchange rate", "amount to be paid", "markup", "charge", "spread", "currency pair", "currency combination"],
            "bookings": ["booking", "booked", "deal type", "value date"],
            "customers": ["customer", "payee", "university", "address", "beneficiary"],
        }
        # GAP 36b: Enrich from domain knowledge synonyms
        for term, target_domain in self._domain_knowledge.get("synonyms", {}).items():
            if target_domain in domain_keyword_lists and term not in domain_keyword_lists[target_domain]:
                domain_keyword_lists[target_domain].append(term)

        domains_detected: list[str] = []
        for d, keywords in domain_keyword_lists.items():
            if any(kw in lower for kw in keywords):
                domains_detected.append(d)
        if not domains_detected:
            domains_detected = [domain]
        elif domain not in domains_detected:
            domains_detected.insert(0, domain)
        secondary_domains = [d for d in domains_detected if d != domain]

        has_count_words = self._goal_has_count_intent(lower)
        has_amount_words = self._goal_has_amount_intent(lower)
        has_mt103 = "mt103" in lower
        has_refund = "refund" in lower

        if domain == "documents":
            metric = "document_relevance"
        else:
            metric = self._resolve_metric_from_catalog(lower, domain, catalog)

            # Rate metric override: when user asks for a rate/ratio, prefer _rate variant
            if any(k in lower for k in ["rate", "ratio", "percentage", "coverage", "%"]):
                rate_metrics = [m for m in catalog.get("metrics_by_table", {}).get(
                    DOMAIN_TO_MART.get(domain, ""), {}) if m.endswith("_rate")]
                for rm in rate_metrics:
                    prefix = rm.replace("_rate", "")
                    if prefix in lower:
                        metric = rm
                        break
            # Amount override: when asking about amount of filtered items
            if has_amount_words and not has_count_words and (has_mt103 or has_refund):
                metric = "total_amount"

        # RC4a: Detect "currency pair" as compound dimension for quotes domain
        _currency_pair_detected = False
        if domain == "quotes" and any(kw in lower for kw in ["currency pair", "currency combination", "most common currency"]):
            _currency_pair_detected = True
            if any(kw in lower for kw in ["most common", "top", "most popular"]):
                intent = "grouped_metric"
                if metric == "quote_count" or not metric:
                    metric = "quote_count"

        dim_candidates = self._build_dim_candidates_from_catalog(domain, catalog)

        dimensions: list[str] = []
        dim_signal = any(
            t in lower
            for t in [" by ", "split", "breakdown", "top", "wise", "per ", "month wise", "grouped", "group by"]
        )
        has_month_signal = bool(
            re.search(r"\b(by month|monthly|month[\s-]?wise)\b", lower)
            or ("month" in lower and dim_signal)
            or ("split my month" in lower)
        )
        has_trend_signal = bool(re.search(r"\btrends?\b", lower))
        if domain != "documents" and has_month_signal:
            dimensions.append("__month__")

        # RC4a (cont): Inject virtual currency_pair dimension
        if _currency_pair_detected and "currency_pair" not in dimensions:
            dimensions.append("currency_pair")

        matched_dim_keys: set[str] = set()
        for key, col in dim_candidates.items():
            if key == "month":
                continue
            if key in lower and dim_signal and col not in dimensions:
                dimensions.append(col)
                matched_dim_keys.add(key)

        # Dynamic dimension discovery: for unmatched keywords, try to
        # resolve against actual catalog columns via substring matching.
        if dim_signal and domain != "documents":
            by_match_single = re.findall(r"\bby\s+(\w+)", lower)
            # Also capture two-word phrases (e.g., "customer country" → "address_country")
            by_match_double = re.findall(r"\bby\s+(\w+\s+\w+)", lower)
            split_match = re.findall(r"\bsplit\s+(?:by\s+)?(\w+)", lower)
            # Prioritize longer (multi-word) matches over single-word
            candidate_words_ordered: list[str] = list(by_match_double) + list(by_match_single) + list(split_match)
            noise = {"the", "a", "an", "and", "or", "each", "every", "month", "monthly"}
            seen: set[str] = set()
            candidate_words: list[str] = []
            for w in candidate_words_ordered:
                w_clean = w.strip()
                if w_clean not in seen and w_clean not in noise and w_clean not in matched_dim_keys:
                    candidate_words.append(w_clean)
                    seen.add(w_clean)
            known_keys = set(dim_candidates.keys())
            mart_name = DOMAIN_TO_MART.get(domain, "")
            dim_value_cols = set(catalog.get("dimension_values", {}).get(mart_name, {}).keys())
            mart_cols = set(catalog.get("marts", {}).get(mart_name, {}).get("columns", []))
            # Include secondary domain columns for cross-table dimension resolution (GAP 14).
            for sec_d in domains_detected:
                if sec_d != domain:
                    sec_mart = DOMAIN_TO_MART.get(sec_d, "")
                    sec_dim_vals = set(catalog.get("dimension_values", {}).get(sec_mart, {}).keys())
                    sec_cols = set(catalog.get("marts", {}).get(sec_mart, {}).get("columns", []))
                    dim_value_cols |= sec_dim_vals
                    mart_cols |= sec_cols
            available_dim_cols = dim_value_cols | mart_cols

            resolved_dims: set[str] = set()
            consumed_words: set[str] = set()  # words consumed by multi-word phrases
            for word in candidate_words:
                if word in known_keys:
                    continue
                # Skip single words already consumed by a successful multi-word match
                if word in consumed_words:
                    continue
                # Try substring match: "country" -> "address_country"
                resolved = None
                # For multi-word phrases, try each individual word as well
                words_to_try = word.split() if " " in word else [word]
                for w in words_to_try:
                    for col in sorted(available_dim_cols):
                        if w in col or col in w:
                            if col not in dimensions and col not in resolved_dims:
                                # Prefer columns that are NOT id/key columns
                                if col.endswith("_id") or col.endswith("_key"):
                                    continue
                                resolved = col
                                break
                    if resolved:
                        break
                if not resolved:
                    # Fallback: try id columns too if nothing better found
                    for w in words_to_try:
                        for col in sorted(available_dim_cols):
                            if w in col or col in w:
                                if col not in dimensions and col not in resolved_dims:
                                    resolved = col
                                    break
                        if resolved:
                            break
                if resolved:
                    dimensions.append(resolved)
                    resolved_dims.add(resolved)
                    # Mark individual words as consumed so single-word duplicates are skipped
                    if " " in word:
                        for part in word.split():
                            consumed_words.add(part)
                    self._pipeline_warnings.append(
                        f"Dimension '{word}' dynamically resolved to column "
                        f"'{resolved}' via catalog lookup."
                    )
                else:
                    self._pipeline_warnings.append(
                        f"Dimension keyword '{word}' was not recognized for the "
                        f"'{domain}' domain and was ignored."
                    )

        # "trend" implies time-series but should not override explicit dimensions.
        # Only add __month__ when "trend" appears and no month signal already added it.
        if has_trend_signal and domain != "documents" and "__month__" not in dimensions:
            dimensions.append("__month__")

        if intent == "grouped_metric" and not dimensions and dim_candidates:
            # Fallback grouped intent: prefer first stable dimension for readable grouping.
            default_col = next(iter(dim_candidates.values()))
            if default_col != "__month__":
                dimensions.append(default_col)

        if len(dimensions) > MAX_DIMENSIONS:
            dropped_dims = dimensions[MAX_DIMENSIONS:]
            dimensions = dimensions[:MAX_DIMENSIONS]
            self._pipeline_warnings.append(
                f"Requested {len(dimensions) + len(dropped_dims)} dimensions but "
                f"only {MAX_DIMENSIONS} are supported. Dropped: {', '.join(dropped_dims)}."
            )
        if metric == "university_count":
            dimensions = [d for d in dimensions if d != "is_university"]
        dimension = dimensions[0] if dimensions else None

        top_n = 20
        top_match = re.search(r"\btop\s+(\d+)\b", lower)
        if top_match:
            top_n = max(1, min(100, int(top_match.group(1))))

        time_filter: dict[str, Any] | None = None
        month = None
        for name, number in MONTH_MAP.items():
            if re.search(rf"\b{name}\b", lower):
                month = number
                break

        years = re.findall(r"\b(20\d{2})\b", lower)
        year = int(years[0]) if years else None

        if month is not None:
            time_filter = {"kind": "month_year", "month": month, "year": year}
        elif "this month" in lower:
            time_filter = {"kind": "relative", "value": "this_month"}
        elif "last month" in lower:
            time_filter = {"kind": "relative", "value": "last_month"}
        elif "this year" in lower:
            time_filter = {"kind": "relative", "value": "this_year"}
        elif "last year" in lower:
            time_filter = {"kind": "relative", "value": "last_year"}
        elif year:
            time_filter = {"kind": "year_only", "year": year}

        value_filters: list[dict[str, str]] = []
        domain_table = DOMAIN_TO_MART.get(domain, "")

        for col, values in catalog.get("dimension_values", {}).get(domain_table, {}).items():
            for value in values:
                if len(value) < 3:
                    continue
                if value.lower() in lower:
                    value_filters.append({"column": col, "value": value})
                    break

        def add_value_filter(column: str, value: str) -> None:
            for vf in value_filters:
                if vf.get("column") == column:
                    return
            value_filters.append({"column": column, "value": value})

        # Boolean filter inference: detect boolean columns from catalog and match
        # keywords rather than hardcoding every "if mt103 do that" pattern.
        mart_cols = set(catalog.get("marts", {}).get(domain_table, {}).get("columns", []))
        _BOOLEAN_COLUMN_KEYWORDS = {
            "has_mt103": ["mt103"],
            "has_refund": ["refund"],
            "is_university": ["university", "universities"],
        }
        for bool_col, keywords in _BOOLEAN_COLUMN_KEYWORDS.items():
            if bool_col not in mart_cols:
                continue
            if any(kw in lower for kw in keywords):
                if any(neg in lower for neg in ["without", "excluding", "no ", "non-"]):
                    add_value_filter(bool_col, "false")
                else:
                    add_value_filter(bool_col, "true")

        # Detect unresolved MT-code filters
        mt_match = re.search(r'\bmt\d{3}\b', lower)
        if mt_match:
            mt_code = mt_match.group(0)
            expected_col = f"has_{mt_code}"
            already_filtered = any(vf.get("column") == expected_col for vf in value_filters)
            if expected_col not in mart_cols and not already_filtered:
                available_bool = [c for c in sorted(mart_cols) if c.startswith("has_") or c.startswith("is_")]
                self._pipeline_warnings.append(
                    f"Filter for '{mt_code.upper()}' transactions requested, but column "
                    f"'{expected_col}' does not exist in the data. Results include ALL "
                    f"transactions, not just {mt_code.upper()}. "
                    f"Available boolean filters: {available_bool}"
                )

        return {
            "goal": g,
            "intent": intent,
            "domain": domain,
            "metric": metric,
            "dimension": dimension,
            "dimensions": dimensions,
            "top_n": top_n,
            "time_filter": time_filter,
            "value_filters": value_filters,
            "domains_detected": domains_detected,
            "secondary_domains": secondary_domains,
        }

    def _enforce_intake_consistency(
        self,
        goal: str,
        deterministic: dict[str, Any],
        parsed: dict[str, Any],
        catalog: dict[str, Any] | None = None,
        is_llm_mode: bool = False,
    ) -> dict[str, Any]:
        out = dict(parsed)
        lower = goal.lower()

        # RC6: Hard domain guard — explicit entity nouns override LLM drift
        if "transaction" in lower and not any(k in lower for k in ["booking", "booked", "deal type", "value date"]):
            if out.get("domain") != "transactions":
                self._pipeline_warnings.append(
                    f"Domain corrected from '{out.get('domain')}' to 'transactions' -- goal explicitly mentions 'transaction'."
                )
                out["domain"] = "transactions"

        # GAP 20 / GAP 30: Preserve deterministic complex-analytic intents.
        # In LLM mode, trust the LLM's classification for complex intents
        # since the LLM now knows all 12+ intent types.
        _complex_intents = {
            "running_total", "subquery_filter", "yoy_growth", "correlation",
        }
        det_intent = deterministic.get("intent")
        if det_intent in _complex_intents and not is_llm_mode:
            out["intent"] = det_intent

        compare_terms = [" vs ", " versus ", "compare", "compared"]
        if out.get("intent") == "comparison" and not any(t in lower for t in compare_terms):
            out["intent"] = deterministic.get("intent", out.get("intent"))

        amount_terms = ["amount", "revenue", "value", "sum", "booked", "markup"]
        has_count_words = self._goal_has_count_intent(lower)
        if any(t in lower for t in amount_terms):
            if str(out.get("metric", "")).endswith("_count") and str(
                deterministic.get("metric", "")
            ) in {"total_amount", "total_quote_value", "total_booked_amount", "forex_markup_revenue"}:
                out["metric"] = deterministic.get("metric")

        if "how many quotes" in lower or ("quote" in lower and "how many" in lower):
            out["domain"] = "quotes"
            out["metric"] = "quote_count"

        value_filters = list(out.get("value_filters") or [])

        def has_filter(col: str, val: str) -> bool:
            sval = val.lower()
            for vf in value_filters:
                if str(vf.get("column", "")).lower() == col.lower() and str(vf.get("value", "")).lower() == sval:
                    return True
            return False

        def add_filter(col: str, val: str) -> None:
            if not has_filter(col, val):
                value_filters.append({"column": col, "value": val})

        # Infer boolean filters from goal keywords using catalog-aware pattern
        _BOOL_KW = {"has_mt103": ["mt103"], "has_refund": ["refund"], "is_university": ["university", "universities"]}
        mart_name_for_filter = DOMAIN_TO_MART.get(out.get("domain", ""), "")
        filter_cols = set(catalog.get("marts", {}).get(mart_name_for_filter, {}).get("columns", [])) if catalog else set()
        for bool_col, kws in _BOOL_KW.items():
            if bool_col not in filter_cols:
                continue
            if any(kw in lower for kw in kws):
                if any(neg in lower for neg in ["without", "excluding", "no ", "non-"]):
                    add_filter(bool_col, "false")
                else:
                    add_filter(bool_col, "true")
                # When asking for amount of filtered items, ensure metric matches
                if any(t in lower for t in amount_terms) and not has_count_words:
                    out["metric"] = "total_amount"

        if "split" in lower and str(out.get("intent") or "") == "metric":
            out["intent"] = "grouped_metric"

        raw_dims = list(out.get("dimensions") or [])
        dim0 = out.get("dimension")
        if not raw_dims and isinstance(dim0, str) and dim0.strip():
            raw_dims = [dim0.strip()]

        # Validate LLM-provided dims against catalog columns (preserve valid ones).
        mart_name = DOMAIN_TO_MART.get(out.get("domain", ""), "")
        available_cols: set[str] = set()
        if catalog:
            available_cols = set(
                catalog.get("marts", {}).get(mart_name, {}).get("columns", [])
            )
            # Include secondary domain columns for cross-table dimension resolution (GAP 14).
            for sec_d in out.get("secondary_domains", []):
                sec_mart = DOMAIN_TO_MART.get(sec_d, "")
                sec_cols = set(catalog.get("marts", {}).get(sec_mart, {}).get("columns", []))
                available_cols |= sec_cols
        # Virtual dimensions that are expanded at query time (e.g., currency_pair)
        _VIRTUAL_DIMS = {"__month__", "currency_pair"}
        dims: list[str] = []
        for dim in raw_dims:
            if dim in _VIRTUAL_DIMS or dim in available_cols:
                if dim not in dims:
                    dims.append(dim)
            elif available_cols:
                self._pipeline_warnings.append(
                    f"Dimension '{dim}' from intake is not in {mart_name} schema; removed."
                )
            else:
                # No catalog available -- keep dim as-is (best effort).
                if dim not in dims:
                    dims.append(dim)

        # Additive keyword enrichment: add missing dims based on goal keywords.
        dim_signal = any(
            token in lower for token in [" by ", "split", "breakdown", "wise", "per ", "grouped", "group by"]
        )
        if "month" in lower and dim_signal and "__month__" not in dims:
            dims.append("__month__")
        if out.get("domain") == "transactions":
            if "platform" in lower and dim_signal and "platform_name" not in dims:
                dims.append("platform_name")
            if "state" in lower and dim_signal and "state" not in dims:
                dims.append("state")
            if ("region" in lower or "country" in lower) and dim_signal and "address_country" not in dims:
                dims.append("address_country")
        if len(dims) > MAX_DIMENSIONS:
            dropped = dims[MAX_DIMENSIONS:]
            dims = dims[:MAX_DIMENSIONS]
            self._pipeline_warnings.append(
                f"Requested {len(dims) + len(dropped)} dimensions but only "
                f"{MAX_DIMENSIONS} are supported. Dropped: {', '.join(dropped)}."
            )
        out["dimensions"] = dims
        out["dimension"] = dims[0] if dims else None

        out["value_filters"] = value_filters
        return out

    def _apply_memory_hints(
        self,
        goal: str,
        parsed: dict[str, Any],
        memory_hints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Use similar historical runs to stabilize vague follow-up prompts."""

        if not memory_hints:
            return parsed

        lower = goal.lower()
        explicit_subject_terms = [
            "transaction",
            "quote",
            "customer",
            "booking",
            "mt103",
            "refund",
            "forex",
            "markup",
            "charge",
            "revenue",
        ]
        has_explicit_subject = any(term in lower for term in explicit_subject_terms)
        best = memory_hints[0]
        if float(best.get("similarity", 0.0)) < 0.55:
            return parsed

        # GAP 41a: For explicit queries, still apply learned corrections if available
        if has_explicit_subject:
            if not best.get("correction_applied") or float(best.get("similarity", 0)) < 0.75:
                return parsed
            # Apply learned correction from a previous corrected run
            merged = dict(parsed)
            past_metric = str(best.get("metric") or "").strip()
            if past_metric:
                merged["metric"] = past_metric
            past_dims = list(best.get("dimensions") or [])
            if past_dims:
                merged["dimensions"] = past_dims[:MAX_DIMENSIONS]
                merged["dimension"] = merged["dimensions"][0]
            if isinstance(best.get("time_filter"), dict):
                merged["time_filter"] = best["time_filter"]
            if isinstance(best.get("value_filters"), list) and best["value_filters"]:
                merged["value_filters"] = best["value_filters"][:4]
            merged["_memory_correction_applied"] = True
            return merged

        merged = dict(parsed)
        past_metric = str(best.get("metric") or "").strip()
        past_dims = list(best.get("dimensions") or [])
        if past_metric and merged.get("metric") == "transaction_count":
            merged["metric"] = past_metric
        if past_dims and not merged.get("dimensions"):
            merged["dimensions"] = past_dims[:MAX_DIMENSIONS]
            merged["dimension"] = merged["dimensions"][0]
        if not merged.get("time_filter") and isinstance(best.get("time_filter"), dict):
            merged["time_filter"] = best["time_filter"]
        if not merged.get("value_filters") and isinstance(best.get("value_filters"), list):
            merged["value_filters"] = best["value_filters"][:4]
        return merged

    def _document_query_terms(self, goal: str, *, limit: int = 10) -> list[str]:
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "from",
            "with",
            "that",
            "this",
            "have",
            "does",
            "says",
            "about",
            "into",
            "for",
            "the",
            "and",
            "are",
            "you",
            "data",
            "document",
            "documents",
            "file",
            "files",
            "pdf",
            "docx",
            "show",
            "tell",
            "give",
            "please",
            "explain",
            "summary",
        }
        terms: list[str] = []
        for token in re.findall(r"[a-z0-9_]{3,}", goal.lower()):
            if token in stopwords:
                continue
            if token not in terms:
                terms.append(token)
            if len(terms) >= max(1, min(20, int(limit))):
                break
        return terms

    def _document_chunks_table_ready(self) -> bool:
        check_sql = """
        SELECT COUNT(*) AS metric_value
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name='datada_document_chunks'
        """
        val = self._profile_sql_value(check_sql, default=0)
        return int(val or 0) > 0

    def _document_retrieval_agent(self, goal: str, intake: dict[str, Any]) -> dict[str, Any]:
        del intake  # currently goal-driven retrieval with chunk-level lexical scoring.
        if not self._document_chunks_table_ready():
            return {
                "success": False,
                "document_table_ready": False,
                "message": "Document index is not available. Run `haikugraph ingest-docs` first.",
                "match_count": 0,
                "source_count": 0,
                "citations": [],
                "sample_rows": [],
                "columns": ["citation", "source", "snippet", "score"],
                "matched_terms": [],
                "missing_terms": [],
                "sql_used": "",
                "execution_time_ms": 0.0,
            }

        terms = self._document_query_terms(goal, limit=8)
        if terms:
            escaped_terms = [term.replace("'", "''") for term in terms]
            score_expr = " + ".join(
                [f"CASE WHEN LOWER(content) LIKE '%{term}%' THEN 1 ELSE 0 END" for term in escaped_terms]
            )
        else:
            score_expr = "1"

        sql = f"""
        SELECT
            source_path,
            file_name,
            title,
            chunk_index,
            char_start,
            char_end,
            content,
            token_count,
            ({score_expr}) AS term_score
        FROM datada_document_chunks
        WHERE content IS NOT NULL AND TRIM(content) != ''
        ORDER BY term_score DESC, token_count DESC NULLS LAST, source_path ASC, chunk_index ASC
        LIMIT 60
        """
        result = self.executor.execute(sql)
        if not result.success:
            return {
                "success": False,
                "document_table_ready": True,
                "message": result.error or "Document query failed.",
                "match_count": 0,
                "source_count": 0,
                "citations": [],
                "sample_rows": [],
                "columns": ["citation", "source", "snippet", "score"],
                "matched_terms": [],
                "missing_terms": terms,
                "sql_used": sql.strip(),
                "execution_time_ms": float(result.execution_time_ms or 0.0),
            }

        rows = list(result.rows or [])
        filtered = [r for r in rows if int(r.get("term_score") or 0) > 0] if terms else rows
        top = filtered[:8]
        citation_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(top, start=1):
            citation = f"[D{idx}]"
            source = str(row.get("file_name") or row.get("source_path") or "unknown")
            source_path = str(row.get("source_path") or "")
            snippet = " ".join(str(row.get("content") or "").split())
            if len(snippet) > 320:
                snippet = snippet[:320].rsplit(" ", 1)[0] + "..."
            chunk_index = int(row.get("chunk_index") or 0)
            char_start = int(row.get("char_start") or 0)
            char_end = int(row.get("char_end") or 0)
            citation_rows.append(
                {
                    "citation": citation,
                    "source": source,
                    "source_path": source_path,
                    "chunk_index": chunk_index,
                    "char_span": f"{char_start}-{char_end}",
                    "score": int(row.get("term_score") or 0),
                    "snippet": snippet,
                }
            )
            sample_rows.append(
                {
                    "citation": citation,
                    "source": source,
                    "snippet": snippet,
                    "score": int(row.get("term_score") or 0),
                }
            )

        combined = " ".join(str(row.get("content") or "").lower() for row in top)
        matched_terms = [term for term in terms if term in combined]
        missing_terms = [term for term in terms if term not in matched_terms]
        source_count = len({str(row.get("source_path") or "") for row in top if str(row.get("source_path") or "")})
        return {
            "success": True,
            "document_table_ready": True,
            "message": "ok",
            "match_count": len(citation_rows),
            "source_count": source_count,
            "citations": citation_rows,
            "sample_rows": sample_rows,
            "columns": ["citation", "source", "snippet", "score"],
            "matched_terms": matched_terms,
            "missing_terms": missing_terms,
            "sql_used": result.sql_executed or sql.strip(),
            "execution_time_ms": float(result.execution_time_ms or 0.0),
        }

    def _document_answer_agent(
        self,
        goal: str,
        retrieval: dict[str, Any],
        runtime: RuntimeSelection,
        storyteller_mode: bool,
    ) -> dict[str, Any]:
        citations = list(retrieval.get("citations") or [])
        match_count = int(retrieval.get("match_count") or 0)
        if not retrieval.get("document_table_ready", False):
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    "I do not have a document index yet. "
                    "Run `haikugraph ingest-docs --docs-dir <path> --db-path <db>` and ask again."
                ),
                "suggested_questions": [
                    "Which documents were ingested?",
                    "Show top document types and titles",
                    "Summarize key themes from the uploaded docs",
                ],
            }
        if match_count <= 0:
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    "I checked the indexed documents but found no citation-level matches for this phrasing. "
                    "Try a specific keyword, clause, or file name."
                ),
                "suggested_questions": [
                    "List all available document titles",
                    "Search for 'markup' across documents",
                    "Show snippets that mention policy or compliance",
                ],
            }

        top = citations[:4]
        fallback_lines = []
        for item in top:
            fallback_lines.append(
                f"- {item['citation']} `{item['source']}`: {item['snippet']}"
            )
        intro = (
            "Here are the strongest evidence snippets from your document set:"
            if storyteller_mode
            else "Top citation-backed snippets:"
        )
        fallback_answer = (
            f"**{goal}**\n\n"
            f"{intro}\n"
            + "\n".join(fallback_lines)
            + "\n\nSources are citation-anchored; ask a follow-up to drill into any citation."
        )
        fallback = {
            "answer_markdown": fallback_answer,
            "suggested_questions": [
                "Show me more snippets from the top cited source",
                "Summarize this in 3 bullet points for business stakeholders",
                "What terms are missing from the current evidence?",
            ],
        }

        if not runtime.use_llm or not runtime.provider:
            return fallback
        try:
            payload = {
                "question": goal,
                "citations": [
                    {
                        "id": item.get("citation"),
                        "source": item.get("source"),
                        "snippet": item.get("snippet"),
                    }
                    for item in top
                ],
                "style": "storyteller" if storyteller_mode else "concise_professional",
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are NarrativeAgent. Create a concise answer from citations only. "
                        "Do not invent facts. Keep <=120 words and include citation ids like [D1]. "
                        "Return JSON with keys answer_markdown and suggested_questions."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, default=str)},
            ]
            raw = call_llm(
                messages,
                role="narrator",
                provider=runtime.provider,
                model=runtime.narrator_model,
                timeout=40,
            )
            parsed = _extract_json_payload(raw)
            if parsed and isinstance(parsed.get("answer_markdown"), str):
                answer_text = self._compact_answer_markdown(
                    self._clean_llm_answer_markdown(parsed["answer_markdown"]),
                    max_lines=9,
                    max_chars=850,
                )
                if not any(cit.get("citation") in answer_text for cit in top):
                    answer_text += "\n\nCitations: " + ", ".join(str(cit.get("citation")) for cit in top)
                return {
                    "answer_markdown": answer_text,
                    "suggested_questions": self._normalize_suggested_questions(
                        parsed.get("suggested_questions"),
                        {"table": "datada_document_chunks", "metric": "document_match_count"},
                    ),
                }
        except Exception:
            pass
        return fallback

    def _semantic_retrieval_agent(
        self,
        intake: dict[str, Any],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        table = DOMAIN_TO_MART.get(intake["domain"], "datada_mart_transactions")
        table_meta = catalog["marts"][table]
        return {
            "table": table,
            "columns": table_meta["columns"],
            "row_count": table_meta["row_count"],
            "metrics": catalog["metrics_by_table"][table],
            "preferred_time_column": catalog["preferred_time_column"].get(table),
            "preferred_time_by_metric": catalog.get("preferred_time_column_by_metric", {}).get(table, {}),
        }

    def _planning_agent(
        self,
        intake: dict[str, Any],
        retrieval: dict[str, Any],
        catalog: dict[str, Any],
        runtime: RuntimeSelection | None = None,
    ) -> dict[str, Any]:
        metric_expr = retrieval["metrics"].get(intake["metric"])
        if metric_expr is None:
            fallback_metric, metric_expr = next(iter(retrieval["metrics"].items()))
            intake["metric"] = fallback_metric

        raw_dims = intake.get("dimensions")
        if not isinstance(raw_dims, list):
            raw_dims = [intake.get("dimension")] if intake.get("dimension") else []
        dimensions: list[str] = []
        # Build set of columns available in secondary domain tables for
        # cross-table dimension resolution (GAP 14).
        secondary_cols: set[str] = set()
        for sec_domain in intake.get("secondary_domains", []):
            sec_table = DOMAIN_TO_MART.get(sec_domain, "")
            sec_meta = catalog.get("marts", {}).get(sec_table, {})
            secondary_cols.update(sec_meta.get("columns", []))
        _virtual_dims = {"__month__", "currency_pair"}
        for dim in raw_dims:
            if not isinstance(dim, str):
                continue
            if dim in _virtual_dims:
                dimensions.append(dim)
                continue
            if dim in retrieval["columns"]:
                dimensions.append(dim)
            elif dim in secondary_cols:
                # Dimension lives in a secondary (joined) table — keep it.
                dimensions.append(dim)
            else:
                self._pipeline_warnings.append(
                    f"Dimension '{dim}' was requested but is not available in "
                    f"{retrieval['table']}. It was removed from the query."
                )
        if len(dimensions) > MAX_DIMENSIONS:
            dimensions = dimensions[:MAX_DIMENSIONS]
        dimension = dimensions[0] if dimensions else None

        value_filters = list(intake.get("value_filters", []))
        valid_cols = set(retrieval.get("columns", []))
        value_filters = [
            vf
            for vf in value_filters
            if isinstance(vf, dict)
            and isinstance(vf.get("column"), str)
            and vf.get("column") in valid_cols
            and isinstance(vf.get("value"), str)
            and vf.get("value") not in ("[]", "{}", "None", "null", "")
        ]

        if intake["domain"] == "transactions":
            goal_lower = str(intake.get("goal", "")).lower()
            has_amount_words = self._goal_has_amount_intent(goal_lower)
            has_count_words = self._goal_has_count_intent(goal_lower)
            if "mt103" in goal_lower and has_amount_words and not has_count_words:
                if not any(vf["column"] == "has_mt103" for vf in value_filters):
                    value_filters.append({"column": "has_mt103", "value": "true"})
                if intake.get("metric") in {"transaction_count", "mt103_count"}:
                    intake["metric"] = "total_amount"
                    metric_expr = retrieval["metrics"].get("total_amount", metric_expr)
            if "refund" in goal_lower and has_amount_words and not has_count_words:
                if not any(vf["column"] == "has_refund" for vf in value_filters):
                    value_filters.append({"column": "has_refund", "value": "true"})
                if intake.get("metric") in {"transaction_count", "refund_count"}:
                    intake["metric"] = "total_amount"
                    metric_expr = retrieval["metrics"].get("total_amount", metric_expr)

        deterministic_plan = {
            "goal": intake.get("goal", ""),
            "intent": intake["intent"],
            "table": retrieval["table"],
            "metric": intake["metric"],
            "metric_expr": metric_expr,
            "dimension": dimension,
            "dimensions": dimensions,
            "available_columns": retrieval.get("columns", []),
            "secondary_domains": intake.get("secondary_domains", []),
            "time_column": retrieval.get("preferred_time_by_metric", {}).get(
                intake["metric"],
                retrieval.get("preferred_time_column"),
            ),
            "time_filter": intake.get("time_filter"),
            "value_filters": value_filters,
            "top_n": max(1, min(100, int(intake.get("top_n", 20)))),
            "definition_used": (
                f"{intake['metric']} on {retrieval['table']}"
                + (f" grouped by {', '.join(dimensions)}" if dimensions else "")
            ),
            "row_count_hint": retrieval["row_count"],
        }

        if runtime and runtime.use_llm and runtime.provider and not intake.get("_llm_intake_used"):
            llm_refinement = self._planning_agent_with_llm(
                deterministic_plan, retrieval, catalog, runtime
            )
            if llm_refinement:
                deterministic_plan["_llm_planning_reasoning"] = llm_refinement.get("reasoning", "")
                # Apply LLM suggestions only if they reference valid catalog entries
                if llm_refinement.get("metric") and llm_refinement["metric"] in retrieval["metrics"]:
                    deterministic_plan["metric"] = llm_refinement["metric"]
                    deterministic_plan["metric_expr"] = retrieval["metrics"][llm_refinement["metric"]]
                if llm_refinement.get("dimensions") and isinstance(llm_refinement["dimensions"], list):
                    valid_dims = [
                        d for d in llm_refinement["dimensions"]
                        if d in retrieval.get("columns", []) or d in _virtual_dims
                    ]
                    if valid_dims:
                        deterministic_plan["dimensions"] = valid_dims[:MAX_DIMENSIONS]
                        deterministic_plan["dimension"] = valid_dims[0]

        return deterministic_plan

    def _planning_agent_with_llm(
        self,
        plan: dict[str, Any],
        retrieval: dict[str, Any],
        catalog: dict[str, Any],
        runtime: RuntimeSelection,
    ) -> dict[str, Any] | None:
        """Use LLM to evaluate and optionally refine the deterministic plan."""
        plan_summary = {k: v for k, v in plan.items() if k not in ("available_columns", "row_count_hint")}
        messages = [
            {
                "role": "system",
                "content": (
                    "You are PlanningAgent. Evaluate the query plan and suggest refinements.\n\n"
                    "Return JSON with keys: metric (string or null), dimensions (list or null), reasoning (string).\n\n"
                    "RULES:\n"
                    "- Only suggest changes if the current plan is clearly misaligned with the question\n"
                    "- Do NOT add dimensions/GROUP BY unless the user explicitly asks for a breakdown (by/per/split/grouped)\n"
                    "- For scalar questions ('what is the average', 'how many total'), dimensions MUST be empty []\n"
                    "- For 'average' questions, ensure the metric is an avg_ variant, not a sum grouped by something\n"
                    "- Return null for metric/dimensions if no change is needed\n\n"
                    f"Available metrics: {list(retrieval['metrics'].keys())}\n"
                    f"Available columns: {retrieval.get('columns', [])}"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {plan.get('goal', '')}\nCurrent plan: {json.dumps(plan_summary)}\nJSON only.",
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="intent",
                provider=runtime.provider,
                model=runtime.intent_model,
                timeout=20,
            )
            return _extract_json_payload(raw)
        except Exception as exc:
            self._pipeline_warnings.append(
                f"Planning agent LLM refinement failed ({type(exc).__name__}); using deterministic plan."
            )
        return None

    def _autonomous_refinement_agent(
        self,
        goal: str,
        base_plan: dict[str, Any],
        base_query_plan: dict[str, Any],
        base_execution: dict[str, Any],
        base_audit: dict[str, Any],
        catalog: dict[str, Any],
        memory_hints: list[dict[str, Any]] | None,
        learned_corrections: list[dict[str, Any]] | None,
        autonomy: AutonomyConfig,
        tenant_id: str,
        specialist_findings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate candidate plans and autonomously switch when evidence improves."""

        base_breakdown = self._candidate_score_breakdown(goal, base_plan, base_execution, base_audit)
        base_score = float(base_breakdown["total"])
        selected = {
            "plan": base_plan,
            "query_plan": base_query_plan,
            "execution": base_execution,
            "audit": base_audit,
            "score": base_score,
            "score_breakdown": base_breakdown,
            "reason": "base_plan",
        }
        candidate_evals: list[dict[str, Any]] = [
            {
                "candidate": "base_plan",
                "table": base_plan.get("table"),
                "metric": base_plan.get("metric"),
                "score": round(base_score, 4),
                "decomposition": base_breakdown,
                "row_count": int(base_execution.get("row_count") or 0),
                "goal_term_misses": list(base_audit.get("grounding", {}).get("goal_term_misses", [])),
                "round": 0,
            }
        ]
        correction_applied = False
        correction_reason = ""
        refinement_rounds: list[dict[str, Any]] = []

        if not autonomy.auto_correction:
            return {
                **selected,
                "evaluated_candidates": candidate_evals,
                "confidence_decomposition": candidate_evals,
                "contradiction_resolution": self._resolve_candidate_contradictions(candidate_evals),
                "correction_applied": False,
                "correction_reason": "",
                "refinement_rounds": refinement_rounds,
                "probe_findings": [],
                "toolsmith_candidates": [],
            }

        seen = {self._plan_signature(base_plan)}
        max_rounds = max(1, int(autonomy.max_refinement_rounds))
        seed_plan = base_plan
        seed_execution = base_execution
        for round_idx in range(max_rounds):
            round_no = round_idx + 1
            round_started_score = float(selected["score"])
            evaluated = 0
            improved = False
            candidates = self._generate_candidate_plans(
                goal=goal,
                base_plan=seed_plan,
                base_execution=seed_execution,
                base_audit=base_audit,
                catalog=catalog,
                memory_hints=memory_hints or [],
                learned_corrections=learned_corrections or [],
                autonomy=autonomy,
            )
            for candidate in candidates[: max(1, autonomy.max_candidate_plans)]:
                signature = self._plan_signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                evaluated += 1
                # GAP 37: Pass specialist findings so directives apply to candidate plans
                query_plan = self._query_engine_agent(candidate, specialist_findings or [])
                execution = self._execution_agent(query_plan)
                audit = self._audit_agent(candidate, query_plan, execution)
                breakdown = self._candidate_score_breakdown(goal, candidate, execution, audit)
                score = float(breakdown["total"])
                candidate_evals.append(
                    {
                        "candidate": candidate.get("_variant_reason", "variant"),
                        "table": candidate.get("table"),
                        "metric": candidate.get("metric"),
                        "score": round(score, 4),
                        "decomposition": breakdown,
                        "row_count": int(execution.get("row_count") or 0),
                        "goal_term_misses": list(audit.get("grounding", {}).get("goal_term_misses", [])),
                        "round": round_no,
                    }
                )
                if score > selected["score"]:
                    improved = True
                    selected = {
                        "plan": candidate,
                        "query_plan": query_plan,
                        "execution": execution,
                        "audit": audit,
                        "score": score,
                        "score_breakdown": breakdown,
                        "reason": candidate.get("_variant_reason", "higher_score"),
                    }

            refinement_rounds.append(
                {
                    "round": round_no,
                    "evaluated_candidates": evaluated,
                    "improved": improved,
                    "starting_score": round(round_started_score, 4),
                    "ending_score": round(float(selected["score"]), 4),
                    "selected_reason": selected.get("reason", "base_plan"),
                }
            )
            if not improved:
                break
            seed_plan = selected["plan"]
            seed_execution = selected["execution"]

        base_misses = len(base_audit.get("grounding", {}).get("goal_term_misses", []))
        selected_misses = len(selected["audit"].get("grounding", {}).get("goal_term_misses", []))
        score_delta = selected["score"] - base_score
        if (
            selected is not None
            and (
                score_delta >= autonomy.min_score_delta_for_switch
                or (autonomy.strict_truth and selected_misses < base_misses)
            )
            and selected["reason"] != "base_plan"
        ):
            correction_applied = True
            correction_reason = (
                f"Switched to {selected['reason']} "
                f"(score {selected['score']:.2f} vs {base_score:.2f}, misses {base_misses}->{selected_misses})."
            )
        else:
            selected = {
                "plan": base_plan,
                "query_plan": base_query_plan,
                "execution": base_execution,
                "audit": base_audit,
                "score": base_score,
                "score_breakdown": base_breakdown,
                "reason": "base_plan",
            }

        probe_findings = self._toolsmith_probe_findings(
            goal=goal,
            plan=selected["plan"],
            audit=selected["audit"],
            max_probes=max(0, autonomy.max_probe_queries),
        )
        toolsmith_candidates = self._register_toolsmith_candidates(
            goal=goal,
            plan=selected["plan"],
            probe_findings=probe_findings,
            tenant_id=tenant_id,
        )
        return {
            "plan": selected["plan"],
            "query_plan": selected["query_plan"],
            "execution": selected["execution"],
            "audit": selected["audit"],
            "score": selected["score"],
            "evaluated_candidates": candidate_evals,
            "confidence_decomposition": candidate_evals,
            "contradiction_resolution": self._resolve_candidate_contradictions(candidate_evals),
            "correction_applied": correction_applied,
            "correction_reason": correction_reason,
            "refinement_rounds": refinement_rounds,
            "probe_findings": probe_findings,
            "toolsmith_candidates": toolsmith_candidates,
        }

    def _generate_candidate_plans(
        self,
        *,
        goal: str,
        base_plan: dict[str, Any],
        base_execution: dict[str, Any],
        base_audit: dict[str, Any] | None = None,
        catalog: dict[str, Any],
        memory_hints: list[dict[str, Any]],
        learned_corrections: list[dict[str, Any]],
        autonomy: AutonomyConfig,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        lower = goal.lower()

        for rule in learned_corrections[:3]:
            target_table = str(rule.get("target_table") or "").strip()
            target_metric = str(rule.get("target_metric") or "").strip()
            target_dims = list(rule.get("target_dimensions") or [])
            if not target_table or not target_metric:
                continue
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table=target_table,
                metric=target_metric,
                dimensions=target_dims or None,
                reason=f"learned_rule:{rule.get('keyword', '')}",
            )
            if variant:
                candidates.append(variant)

        for hint in memory_hints[:3]:
            if float(hint.get("similarity", 0.0)) < 0.5:
                continue
            table = str(hint.get("table") or "")
            metric = str(hint.get("metric") or "")
            dims = list(hint.get("dimensions") or [])
            time_filter = hint.get("time_filter")
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table=table,
                metric=metric,
                dimensions=dims or None,
                time_filter=time_filter if isinstance(time_filter, dict) else None,
                reason=f"memory_replay:{hint.get('similarity')}",
            )
            if variant:
                candidates.append(variant)

        if any(k in lower for k in ["forex", "markup", "charges", "quote"]):
            metric = "forex_markup_revenue"
            if any(k in lower for k in ["avg", "average", "mean"]):
                metric = "avg_forex_markup"
            if "charge" in lower:
                metric = "total_charges"
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table="datada_mart_quotes",
                metric=metric,
                reason="keyword_quotes_domain",
            )
            if variant:
                candidates.append(variant)

        # GAP 36e: Include synonym terms for customer detection
        _customer_terms = ["customer", "payee", "university", "beneficiary", "user", "users", "client", "clients"]
        if any(k in lower for k in _customer_terms):
            metric = "customer_count"
            if "payee" in lower:
                metric = "payee_count"
            elif "university" in lower:
                metric = "university_count"
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table="datada_dim_customers",
                metric=metric,
                reason="keyword_customers_domain",
            )
            if variant:
                candidates.append(variant)

        if any(k in lower for k in ["booked", "booking", "deal", "value date"]):
            metric = "total_booked_amount" if self._goal_has_amount_intent(lower) else "booking_count"
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table="datada_mart_bookings",
                metric=metric,
                reason="keyword_bookings_domain",
            )
            if variant:
                candidates.append(variant)

        if "mt103" in lower:
            metric = "mt103_count" if self._goal_has_count_intent(lower) else "total_amount"
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table="datada_mart_transactions",
                metric=metric,
                reason="keyword_mt103_metric",
            )
            if variant:
                candidates.append(variant)
        if "refund" in lower:
            metric = "refund_count" if self._goal_has_count_intent(lower) else "total_amount"
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                table="datada_mart_transactions",
                metric=metric,
                reason="keyword_refund_metric",
            )
            if variant:
                candidates.append(variant)

        enrich_dims: list[str] = []
        existing_dims = list(base_plan.get("dimensions") or [])
        if re.search(r"\b(by month|month wise|monthly|trends?)\b", lower) and "__month__" not in existing_dims:
            enrich_dims.append("__month__")
        if "platform" in lower and "platform_name" not in existing_dims:
            enrich_dims.append("platform_name")
        if "state" in lower and "state" not in existing_dims:
            enrich_dims.append("state")
        if ("region" in lower or "country" in lower) and "address_country" not in existing_dims:
            enrich_dims.append("address_country")
        if enrich_dims:
            merged_dims = (existing_dims + enrich_dims)[:MAX_DIMENSIONS]
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                dimensions=merged_dims,
                reason="dimension_enrichment",
            )
            if variant:
                candidates.append(variant)

        if int(base_execution.get("row_count") or 0) == 0 and base_plan.get("time_filter"):
            variant = self._build_plan_variant(
                base_plan,
                catalog,
                time_filter=None,
                reason="relax_time_filter",
            )
            if variant:
                candidates.append(variant)

        # Audit-driven dimension variants: if audit detected goal_term_misses
        # that could be dimension names, generate candidates with those added.
        if base_audit:
            goal_misses = base_audit.get("grounding", {}).get("goal_term_misses", [])
            table = base_plan.get("table", "")
            available_cols = set(
                catalog.get("marts", {}).get(table, {}).get("columns", [])
            )
            dim_value_cols = set(
                catalog.get("dimension_values", {}).get(table, {}).keys()
            )
            discoverable = available_cols | dim_value_cols
            for missed_term in goal_misses[:3]:
                missed_lower = str(missed_term).lower().strip()
                if not missed_lower:
                    continue
                resolved_col = None
                for col in sorted(discoverable):
                    if missed_lower in col or col in missed_lower:
                        resolved_col = col
                        break
                if resolved_col and resolved_col not in existing_dims:
                    new_dims = existing_dims + [resolved_col]
                    variant = self._build_plan_variant(
                        base_plan,
                        catalog,
                        dimensions=new_dims[:MAX_DIMENSIONS],
                        reason=f"audit_dimension_miss:{missed_term}->{resolved_col}",
                    )
                    if variant:
                        candidates.append(variant)

        pool_cap = max(1, autonomy.max_candidate_plans * max(1, autonomy.max_refinement_rounds))
        return candidates[:pool_cap]

    def _build_plan_variant(
        self,
        base_plan: dict[str, Any],
        catalog: dict[str, Any],
        *,
        table: str | None = None,
        metric: str | None = None,
        dimensions: list[str] | None = None,
        time_filter: dict[str, Any] | None = ...,
        reason: str,
    ) -> dict[str, Any] | None:
        plan = dict(base_plan)
        target_table = table or str(plan.get("table") or "")
        marts = catalog.get("marts", {})
        metrics_by_table = catalog.get("metrics_by_table", {})
        if target_table not in marts or target_table not in metrics_by_table:
            return None

        plan["table"] = target_table
        plan["available_columns"] = list(marts[target_table].get("columns", []))
        plan["row_count_hint"] = int(marts[target_table].get("row_count", 0))
        plan["time_column"] = catalog.get("preferred_time_column", {}).get(target_table)

        available_metrics = metrics_by_table[target_table]
        target_metric = metric or str(plan.get("metric") or "")
        if target_metric not in available_metrics:
            target_metric = next(iter(available_metrics))
        plan["metric"] = target_metric
        plan["metric_expr"] = available_metrics[target_metric]

        if dimensions is not None:
            valid_dims: list[str] = []
            for dim in dimensions:
                if dim == "__month__" or dim in plan["available_columns"]:
                    valid_dims.append(dim)
            plan["dimensions"] = valid_dims[:MAX_DIMENSIONS]
            plan["dimension"] = plan["dimensions"][0] if plan["dimensions"] else None

        if time_filter is not ...:
            plan["time_filter"] = time_filter

        dim_text = ", ".join(plan.get("dimensions") or [])
        plan["definition_used"] = (
            f"{plan['metric']} on {plan['table']}"
            + (f" grouped by {dim_text}" if dim_text else "")
        )
        plan["_variant_reason"] = reason
        return plan

    def _plan_signature(self, plan: dict[str, Any]) -> tuple[Any, ...]:
        return (
            plan.get("table"),
            plan.get("metric"),
            tuple(plan.get("dimensions") or []),
            json.dumps(plan.get("time_filter"), sort_keys=True, default=str),
            json.dumps(plan.get("value_filters") or [], sort_keys=True, default=str),
        )

    def _candidate_score_breakdown(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
        runtime: RuntimeSelection | None = None,
    ) -> dict[str, Any]:
        audit_base = float(audit.get("score", 0.0))
        execution_bonus = 0.04 if execution.get("success") else 0.0
        non_empty_bonus = 0.04 if int(execution.get("row_count") or 0) > 0 else 0.0

        grounding = audit.get("grounding", {})
        misses = list(grounding.get("goal_term_misses", []))
        goal_miss_penalty = 0.08 * len(misses)

        lower = goal.lower()
        dims = list(plan.get("dimensions") or [])
        dimension_bonus = 0.0
        if re.search(r"\b(by month|month wise|monthly)\b", lower) and "__month__" in dims:
            dimension_bonus += 0.03
        if "platform" in lower and "platform_name" in dims:
            dimension_bonus += 0.03
        if "state" in lower and "state" in dims:
            dimension_bonus += 0.02
        if ("region" in lower or "country" in lower) and "address_country" in dims:
            dimension_bonus += 0.03

        latency_penalty = 0.0
        if execution.get("execution_time_ms") and float(execution["execution_time_ms"]) > 6000:
            latency_penalty = 0.03

        raw_total = (
            audit_base
            + execution_bonus
            + non_empty_bonus
            + dimension_bonus
            - goal_miss_penalty
            - latency_penalty
        )
        deterministic_total = max(0.0, min(1.0, raw_total))

        # ── GAP 21: LLM-enhanced candidate scoring ──────────────────
        llm_alignment_score = None
        if runtime and runtime.use_llm and runtime.provider:
            llm_score_result = self._candidate_score_with_llm(goal, plan, execution, runtime)
            if llm_score_result is not None:
                llm_alignment_score = max(0.0, min(1.0, float(llm_score_result)))
                # Blend: 70% deterministic + 30% LLM alignment
                total = max(
                    deterministic_total,  # deterministic as floor
                    0.7 * deterministic_total + 0.3 * llm_alignment_score,
                )
            else:
                total = deterministic_total
        else:
            total = deterministic_total

        result = {
            "audit_base": round(audit_base, 4),
            "execution_bonus": round(execution_bonus, 4),
            "non_empty_bonus": round(non_empty_bonus, 4),
            "dimension_bonus": round(dimension_bonus, 4),
            "goal_miss_penalty": round(goal_miss_penalty, 4),
            "latency_penalty": round(latency_penalty, 4),
            "goal_term_miss_count": len(misses),
            "total": round(total, 4),
        }
        if llm_alignment_score is not None:
            result["llm_alignment_score"] = round(llm_alignment_score, 4)
            result["deterministic_total"] = round(deterministic_total, 4)
        return result

    def _candidate_score_with_llm(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        runtime: RuntimeSelection,
    ) -> float | None:
        """Use LLM to score goal-plan alignment (0.0 to 1.0)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Score how well the query plan aligns with the user's goal. "
                    "Return JSON with key: alignment_score (float 0.0 to 1.0)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n"
                    f"Metric: {plan.get('metric')}, Table: {plan.get('table')}\n"
                    f"Dimensions: {plan.get('dimensions')}\n"
                    f"Execution success: {execution.get('success')}, rows: {execution.get('row_count')}\n"
                    "JSON only."
                ),
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="intent",
                provider=runtime.provider,
                model=runtime.intent_model,
                timeout=15,
            )
            parsed = _extract_json_payload(raw)
            if parsed and "alignment_score" in parsed:
                return float(parsed["alignment_score"])
        except Exception as exc:
            self._pipeline_warnings.append(
                f"Candidate scoring LLM enhancement failed ({type(exc).__name__}); using deterministic score."
            )
        return None

    def _candidate_score(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
    ) -> float:
        return float(self._candidate_score_breakdown(goal, plan, execution, audit)["total"])

    def _resolve_candidate_contradictions(self, candidate_evals: list[dict[str, Any]]) -> dict[str, Any]:
        if len(candidate_evals) < 2:
            return {"detected": False, "reason": "single_candidate"}
        ranked = sorted(candidate_evals, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        best = ranked[0]
        runner = ranked[1]
        score_gap = float(best.get("score", 0.0)) - float(runner.get("score", 0.0))
        conflict = (
            best.get("table") != runner.get("table")
            or best.get("metric") != runner.get("metric")
            or list(best.get("goal_term_misses", [])) != list(runner.get("goal_term_misses", []))
        )
        detected = bool(conflict and score_gap <= 0.08)
        return {
            "detected": detected,
            "reason": (
                "close_competing_hypotheses_resolved_by_score"
                if detected
                else "winner_clear_or_no_conflict"
            ),
            "score_gap": round(score_gap, 4),
            "winner": {
                "candidate": best.get("candidate"),
                "table": best.get("table"),
                "metric": best.get("metric"),
                "score": best.get("score"),
            },
            "runner_up": {
                "candidate": runner.get("candidate"),
                "table": runner.get("table"),
                "metric": runner.get("metric"),
                "score": runner.get("score"),
            },
        }

    def _blackboard_post(
        self,
        blackboard: list[dict[str, Any]],
        *,
        producer: str,
        artifact_type: str,
        payload: Any,
        consumed_by: list[str] | None = None,
    ) -> None:
        entry = {
            "artifact_id": f"bb_{len(blackboard) + 1:03d}",
            "time": datetime.utcnow().isoformat() + "Z",
            "producer": producer,
            "artifact_type": artifact_type,
            "consumed_by": list(consumed_by or []),
            "summary": _compact(payload, max_len=220),
        }
        if isinstance(payload, dict):
            preview: dict[str, Any] = {}
            for key in list(payload)[:8]:
                preview[str(key)] = _compact(payload[key], max_len=140)
            entry["payload_preview"] = preview
        else:
            entry["payload_preview"] = _compact(payload, max_len=180)
        blackboard.append(entry)

    def _blackboard_edges(self, blackboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for artifact in blackboard:
            producer = str(artifact.get("producer") or "")
            artifact_id = str(artifact.get("artifact_id") or "")
            artifact_type = str(artifact.get("artifact_type") or "")
            for consumer in artifact.get("consumed_by", []):
                edges.append(
                    {
                        "artifact_id": artifact_id,
                        "artifact_type": artifact_type,
                        "from": producer,
                        "to": str(consumer),
                    }
                )
        return edges

    def _blackboard_query(
        self,
        blackboard: list[dict[str, Any]],
        *,
        producer: str | None = None,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query blackboard entries by producer and/or artifact_type."""
        return [
            e
            for e in blackboard
            if (not producer or e.get("producer") == producer)
            and (not artifact_type or e.get("artifact_type") == artifact_type)
        ]

    def _blackboard_latest(
        self,
        blackboard: list[dict[str, Any]],
        artifact_type: str,
    ) -> dict[str, Any] | None:
        """Return the most recent blackboard entry of a given artifact_type."""
        matches = self._blackboard_query(blackboard, artifact_type=artifact_type)
        return matches[-1] if matches else None

    def _toolsmith_probe_findings(
        self,
        *,
        goal: str,
        plan: dict[str, Any],
        audit: dict[str, Any],
        max_probes: int,
    ) -> list[dict[str, Any]]:
        """Run bounded probe queries to explain potential mismatches."""

        if max_probes <= 0:
            return []

        misses = set(str(x) for x in audit.get("grounding", {}).get("goal_term_misses", []))
        lower_goal = goal.lower()
        probes: list[tuple[str, str]] = []
        if ("forex" in lower_goal or "markup" in lower_goal or "forex_domain" in misses) and plan.get(
            "table"
        ) != "datada_mart_quotes":
            probes.append(
                (
                    "quotes_markup_presence",
                    "SELECT COUNT(*) AS rows_with_markup FROM datada_mart_quotes WHERE forex_markup IS NOT NULL AND forex_markup <> 0",
                )
            )
        if "mt103" in lower_goal and "mt103" in misses:
            probes.append(
                (
                    "transactions_mt103_presence",
                    "SELECT SUM(CASE WHEN has_mt103 THEN 1 ELSE 0 END) AS mt103_rows FROM datada_mart_transactions",
                )
            )
        if "refund" in lower_goal and "refund" in misses:
            probes.append(
                (
                    "transactions_refund_presence",
                    "SELECT SUM(CASE WHEN has_refund THEN 1 ELSE 0 END) AS refund_rows FROM datada_mart_transactions",
                )
            )

        findings: list[dict[str, Any]] = []
        for label, sql in probes[:max_probes]:
            result = self.executor.execute(sql)
            findings.append(
                {
                    "probe": label,
                    "success": result.success,
                    "sql": sql,
                    "row_count": result.row_count,
                    "sample_rows": result.rows[:3],
                    "error": result.error,
                }
            )
        return findings

    def _register_toolsmith_candidates(
        self,
        *,
        goal: str,
        plan: dict[str, Any],
        probe_findings: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for finding in probe_findings:
            if not finding.get("success"):
                continue
            probe_name = str(finding.get("probe") or "probe")
            sql = str(finding.get("sql") or "").strip()
            if not sql:
                continue
            tool_id = self.memory.register_tool_candidate(
                tenant_id=tenant_id,
                title=f"{probe_name} for {plan.get('metric', 'metric')}",
                sql_text=sql,
                source="autonomy_probe",
                metadata={
                    "goal": goal,
                    "table": plan.get("table"),
                    "metric": plan.get("metric"),
                    "probe": probe_name,
                },
            )
            if not tool_id:
                continue
            stage = self.memory.stage_tool_candidate(tool_id, tenant_id=tenant_id, db_path=self.db_path)
            status = "staged" if stage.get("success") else "candidate"
            out.append(
                {
                    "tool_id": tool_id,
                    "status": status,
                    "probe": probe_name,
                    "message": stage.get("message", ""),
                }
            )
        return out

    def _transactions_specialist(self, plan: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
        if plan["table"] != "datada_mart_transactions":
            return {"active": False, "notes": [], "warnings": [], "directives": []}
        notes = ["Use distinct transaction_key for counts to avoid duplicate counting."]
        warnings: list[str] = []
        directives: list[dict[str, Any]] = []
        if "amount" in plan["metric"]:
            notes.append("Use normalized amount column prioritizing payment_amount/deal_amount.")
        # GAP 37a: Detect unique intent and produce override directive
        goal_lower = str(plan.get("goal", "")).lower()
        unique_rule = self._domain_knowledge.get("business_rules", {}).get("unique_intent", {})
        unique_triggers = unique_rule.get("triggers", ["unique", "distinct", "different", "individual"])
        if any(t in goal_lower for t in unique_triggers):
            # Determine which entity column to use for COUNT(DISTINCT)
            entity_key_map = unique_rule.get("entity_key_map", {})
            # Check if the query references a synonym-mapped entity
            target_col = "customer_id"  # default for "unique users" type queries
            for term, domain in self._domain_knowledge.get("synonyms", {}).items():
                if term in goal_lower and domain in entity_key_map:
                    target_col = entity_key_map[domain]
                    break
            directives.append({
                "type": "override_metric_expr",
                "metric_expr": f"COUNT(DISTINCT {target_col})",
                "reason": f"User asked for unique entities — forcing COUNT(DISTINCT {target_col})",
            })
        # GAP 37a: Detect MT103 filter needs from business rules
        for rule_name, rule in self._domain_knowledge.get("business_rules", {}).items():
            if rule.get("action") == "add_filter":
                if any(t in goal_lower for t in rule.get("triggers", [])):
                    filt = rule.get("filter", {})
                    if filt.get("column"):
                        directives.append({
                            "type": "add_filter",
                            "filter": {"column": filt["column"], "operator": "=", "value": filt["value"]},
                            "reason": f"Business rule '{rule_name}' requires {filt['column']}={filt['value']} filter",
                        })
        # Validate dimensions against schema
        _virtual_dims = {"__month__", "currency_pair"}
        available = set(plan.get("available_columns", []))
        for dim in plan.get("dimensions", []):
            if dim not in _virtual_dims and dim not in available:
                warnings.append(f"Dimension '{dim}' not found in {plan['table']}.")
        # Validate metric
        metrics = catalog.get("metrics_by_table", {}).get(plan["table"], {})
        if plan.get("metric") and plan["metric"] not in metrics:
            warnings.append(f"Metric '{plan['metric']}' not in available metrics for {plan['table']}.")
        return {"active": True, "notes": notes, "warnings": warnings, "directives": directives}

    def _customer_specialist(self, plan: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
        if plan["table"] not in {"datada_dim_customers", "datada_mart_transactions"}:
            return {"active": False, "notes": [], "warnings": [], "directives": []}
        notes = ["Customer-level metrics should use distinct customer keys where possible."]
        warnings: list[str] = []
        directives: list[dict[str, Any]] = []
        # Suggest geographic dimensions when customer table is involved
        dims = set(plan.get("dimensions", []))
        goal_lower = str(plan.get("goal", "")).lower()
        if ("country" in goal_lower or "region" in goal_lower) and "address_country" not in dims:
            notes.append("Consider adding address_country as a dimension for geographic breakdown.")
        # GAP 37a: Detect unique intent for customer-domain queries
        # Only override when the goal is actually about customers, not transactions
        unique_rule = self._domain_knowledge.get("business_rules", {}).get("unique_intent", {})
        unique_triggers = unique_rule.get("triggers", ["unique", "distinct", "different", "individual"])
        _is_txn_goal = any(k in goal_lower for k in ["transaction", "payment", "transfer", "wire", "remittance"])
        if any(t in goal_lower for t in unique_triggers) and not _is_txn_goal:
            entity_key_map = unique_rule.get("entity_key_map", {})
            target_col = entity_key_map.get("customers", "customer_id")
            directives.append({
                "type": "override_metric_expr",
                "metric_expr": f"COUNT(DISTINCT {target_col})",
                "reason": f"User asked for unique customers — forcing COUNT(DISTINCT {target_col})",
            })
        # Validate dimensions
        _virtual_dims = {"__month__", "currency_pair"}
        available = set(plan.get("available_columns", []))
        for dim in plan.get("dimensions", []):
            if dim not in _virtual_dims and dim not in available:
                warnings.append(f"Dimension '{dim}' not found in {plan['table']}.")
        return {"active": True, "notes": notes, "warnings": warnings, "directives": directives}

    def _revenue_specialist(self, plan: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
        if "amount" not in plan["metric"] and "quote_value" not in plan["metric"]:
            return {"active": False, "notes": [], "warnings": [], "directives": []}
        notes = ["Revenue metrics should summarize numeric normalized columns only."]
        warnings: list[str] = []
        directives: list[dict[str, Any]] = []
        metrics = catalog.get("metrics_by_table", {}).get(plan["table"], {})
        if plan.get("metric") and plan["metric"] not in metrics:
            warnings.append(f"Metric '{plan['metric']}' not available for {plan['table']}.")
        return {"active": True, "notes": notes, "warnings": warnings, "directives": directives}

    def _risk_specialist(self, plan: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
        if not any(k in plan["metric"] for k in ["refund", "mt103"]):
            return {"active": False, "notes": [], "warnings": [], "directives": []}
        notes = ["Risk metrics should report rate and count where available."]
        warnings: list[str] = []
        directives: list[dict[str, Any]] = []
        goal_lower = str(plan.get("goal", "")).lower()
        # GAP 37a: Detect MT103/refund filter needs
        value_filters = plan.get("value_filters", [])
        filter_cols = {vf.get("column") for vf in value_filters}
        if "mt103" in goal_lower and "has_mt103" not in filter_cols:
            directives.append({
                "type": "add_filter",
                "filter": {"column": "has_mt103", "operator": "=", "value": "true"},
                "reason": "MT103 query requires has_mt103=true filter",
            })
            if "amount" in plan.get("metric", ""):
                warnings.append("MT103 amount requested without has_mt103 filter.")
        if "refund" in goal_lower and "has_refund" not in filter_cols:
            directives.append({
                "type": "add_filter",
                "filter": {"column": "has_refund", "operator": "=", "value": "true"},
                "reason": "Refund query requires has_refund=true filter",
            })
            if "amount" in plan.get("metric", ""):
                warnings.append("Refund amount requested without has_refund filter.")
        # Unique intent for risk metrics
        unique_rule = self._domain_knowledge.get("business_rules", {}).get("unique_intent", {})
        unique_triggers = unique_rule.get("triggers", ["unique", "distinct", "different", "individual"])
        if any(t in goal_lower for t in unique_triggers):
            entity_key_map = unique_rule.get("entity_key_map", {})
            # For risk queries, determine the right entity
            for term, domain in self._domain_knowledge.get("synonyms", {}).items():
                if term in goal_lower and domain in entity_key_map:
                    target_col = entity_key_map[domain]
                    directives.append({
                        "type": "override_metric_expr",
                        "metric_expr": f"COUNT(DISTINCT {target_col})",
                        "reason": f"User asked for unique entities in risk context — COUNT(DISTINCT {target_col})",
                    })
                    break
        return {"active": True, "notes": notes, "warnings": warnings, "directives": directives}

    def _governance_precheck(self, goal: str) -> dict[str, Any]:
        lower = goal.lower()
        blocked_keywords = ["drop table", "delete from", "truncate", "update ", "insert into"]
        if any(k in lower for k in blocked_keywords):
            return {"allowed": False, "reason": "Destructive operations are blocked."}
        return {"allowed": True, "reason": "ok"}

    def _apply_specialist_guidance(
        self,
        plan: dict[str, Any],
        specialist_findings: list[dict[str, Any]],
    ) -> None:
        """Extract and apply specialist notes/warnings/directives to the plan."""
        guidance_notes: list[str] = []
        applied_directives: list[dict[str, Any]] = []
        for findings in specialist_findings:
            if not isinstance(findings, dict):
                continue
            notes = findings.get("notes", [])
            if isinstance(notes, list):
                guidance_notes.extend(notes)
            warnings = findings.get("warnings", [])
            if isinstance(warnings, list):
                for w in warnings:
                    if w not in self._pipeline_warnings:
                        self._pipeline_warnings.append(w)
            # GAP 37b: Process structured directives
            directives = findings.get("directives", [])
            if isinstance(directives, list):
                for directive in directives:
                    if not isinstance(directive, dict):
                        continue
                    d_type = directive.get("type", "")
                    if d_type == "override_metric_expr" and directive.get("metric_expr"):
                        plan["metric_expr"] = directive["metric_expr"]
                        plan["_specialist_metric_override"] = directive["metric_expr"]
                        applied_directives.append(directive)
                    elif d_type == "add_filter" and directive.get("filter"):
                        filt = directive["filter"]
                        existing_cols = {
                            vf.get("column") for vf in (plan.get("value_filters") or [])
                        }
                        if filt.get("column") and filt["column"] not in existing_cols:
                            plan.setdefault("value_filters", []).append(filt)
                            applied_directives.append(directive)
                    elif d_type == "add_secondary_domain" and directive.get("domain"):
                        sec = plan.get("secondary_domains") or []
                        if directive["domain"] not in sec:
                            sec.append(directive["domain"])
                            plan["secondary_domains"] = sec
                            applied_directives.append(directive)
        if guidance_notes:
            plan["_specialist_guidance"] = guidance_notes
        if applied_directives:
            plan["_specialist_directives_applied"] = applied_directives

    def _query_engine_agent(
        self,
        plan: dict[str, Any],
        specialist_findings: list[dict[str, Any]],
        runtime: RuntimeSelection | None = None,
        catalog: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._apply_specialist_guidance(plan, specialist_findings)

        table = plan["table"]
        metric_expr = plan["metric_expr"]
        intent = plan["intent"]
        time_col = plan.get("time_column")

        where_clause = self._build_where_clause(plan, for_comparison="current")

        if intent == "comparison":
            current_where = self._build_where_clause(plan, for_comparison="current")
            previous_where = self._build_where_clause(plan, for_comparison="comparison")
            sql = (
                f"SELECT 'current' AS period, {metric_expr} AS metric_value FROM {table} WHERE {current_where} "
                f"UNION "
                f"SELECT 'comparison' AS period, {metric_expr} AS metric_value FROM {table} WHERE {previous_where}"
            )
        elif intent in {"grouped_metric"} and (plan.get("dimensions") or plan.get("dimension")):
            dims = plan.get("dimensions") or [plan.get("dimension")]
            select_parts: list[str] = []
            for dim in dims:
                if dim == "__month__" and time_col:
                    select_parts.append(f"DATE_TRUNC('month', {time_col}) AS month_bucket")
                elif dim == "__month__":
                    select_parts.append("'unknown_month' AS month_bucket")
                elif dim == "currency_pair":
                    # RC4b: Expand virtual compound dimension
                    select_parts.append(
                        "CONCAT(COALESCE(from_currency, '?'), '->', COALESCE(to_currency, '?')) AS currency_pair"
                    )
                else:
                    # Keep semantic dimension names in output for readability.
                    select_parts.append(f"{_q(dim)} AS {_q(dim)}")

            metric_idx = len(select_parts) + 1
            group_by = ", ".join(str(i) for i in range(1, len(select_parts) + 1))
            order_parts = [f"{metric_idx} DESC NULLS LAST"]
            order_parts.extend(f"{i} ASC" for i in range(1, len(select_parts) + 1))
            sql = (
                f"SELECT {', '.join(select_parts)}, {metric_expr} AS metric_value "
                f"FROM {table} WHERE {where_clause} "
                f"GROUP BY {group_by} ORDER BY {', '.join(order_parts)} LIMIT {plan['top_n']}"
            )
        # ── GAP 20: Complex analytical patterns ──────────────────
        elif intent == "trend_analysis" and time_col:
            sql = (
                f"SELECT DATE_TRUNC('month', {time_col}) AS time_bucket, "
                f"{metric_expr} AS metric_value, "
                f"AVG({metric_expr}) OVER (ORDER BY DATE_TRUNC('month', {time_col}) "
                f"ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg "
                f"FROM {table} WHERE {where_clause} AND {time_col} IS NOT NULL "
                f"GROUP BY 1 ORDER BY 1"
            )
        elif intent == "percentile":
            goal_lower = str(plan.get("goal", "")).lower()
            if "p95" in goal_lower or "95th" in goal_lower:
                pct = 0.95
            elif "p90" in goal_lower or "90th" in goal_lower:
                pct = 0.90
            elif "median" in goal_lower or "p50" in goal_lower:
                pct = 0.50
            else:
                pct = 0.50  # default to median
            # Extract the column from metric_expr for PERCENTILE_CONT
            # Priority-based lookup to find the correct numeric column.
            metric_name = plan.get("metric", "")
            available = plan.get("available_columns", [])
            pct_col = None
            # 1. Direct metric name match
            if metric_name in available:
                pct_col = metric_name
            if not pct_col:
                # 2. Strip common prefixes and check
                stripped = metric_name.replace("total_", "").replace("average_", "")
                if stripped in available:
                    pct_col = stripped
            if not pct_col:
                # 3. Common amount columns
                for candidate in ("amount", "payment_amount", "total_amount_to_be_paid", "booked_amount"):
                    if candidate in available:
                        pct_col = candidate
                        break
            if not pct_col:
                # 4. Fall back to first numeric-looking column (skip keys/ids)
                skip = {"customer_key", "transaction_key", "quote_key", "booking_key",
                        "customer_id", "payee_id", "transaction_id", "quote_id"}
                for c in available:
                    if c not in skip and not c.endswith("_ts") and not c.endswith("_id"):
                        pct_col = c
                        break
            if pct_col:
                sql = (
                    f"SELECT PERCENTILE_CONT({pct}) WITHIN GROUP (ORDER BY {_q(pct_col)}) AS percentile_value "
                    f"FROM {table} WHERE {where_clause}"
                )
            else:
                sql = f"SELECT {metric_expr} AS metric_value FROM {table} WHERE {where_clause}"
        elif intent == "running_total" and time_col:
            sql = (
                f"SELECT DATE_TRUNC('month', {time_col}) AS time_bucket, "
                f"{metric_expr} AS metric_value, "
                f"SUM({metric_expr}) OVER (ORDER BY DATE_TRUNC('month', {time_col}) "
                f"ROWS UNBOUNDED PRECEDING) AS running_total "
                f"FROM {table} WHERE {where_clause} AND {time_col} IS NOT NULL "
                f"GROUP BY 1 ORDER BY 1"
            )
        elif intent == "subquery_filter":
            # Detect the grouping dimension from the goal text.
            goal_lower = str(plan.get("goal", "")).lower()
            sf_dim = "customer_id"  # default
            if "platform" in goal_lower:
                sf_dim = "platform_name"
            elif "country" in goal_lower:
                sf_dim = "address_country"
            elif "state" in goal_lower:
                sf_dim = "address_state"
            above = "above" in goal_lower
            op = ">" if above else "<"
            sql = (
                f"WITH agg AS ("
                f"SELECT {_q(sf_dim)}, {metric_expr} AS metric_value "
                f"FROM {table} WHERE {where_clause} "
                f"GROUP BY 1"
                f") "
                f"SELECT * FROM agg "
                f"WHERE metric_value {op} (SELECT AVG(metric_value) FROM agg) "
                f"ORDER BY metric_value DESC"
            )
        elif intent == "yoy_growth" and time_col:
            sql = (
                f"SELECT DATE_TRUNC('month', {time_col}) AS time_bucket, "
                f"{metric_expr} AS metric_value, "
                f"LAG({metric_expr}) OVER (ORDER BY DATE_TRUNC('month', {time_col})) AS prev_period_value, "
                f"ROUND(100.0 * ({metric_expr} - LAG({metric_expr}) OVER (ORDER BY DATE_TRUNC('month', {time_col}))) "
                f"/ NULLIF(LAG({metric_expr}) OVER (ORDER BY DATE_TRUNC('month', {time_col})), 0), 2) AS growth_pct "
                f"FROM {table} WHERE {where_clause} AND {time_col} IS NOT NULL "
                f"GROUP BY 1 ORDER BY 1"
            )
        elif intent == "correlation":
            # Parse "between X and Y" from the goal to find two numeric columns.
            goal_lower = str(plan.get("goal", "")).lower()
            available = plan.get("available_columns", [])
            col1, col2 = None, None
            # Try to find columns mentioned in the goal
            numeric_skip = {"customer_key", "transaction_key", "quote_key", "booking_key",
                            "customer_id", "payee_id", "transaction_id", "quote_id"}
            goal_mentioned_cols = []
            for c in available:
                if c in numeric_skip or c.endswith("_ts") or c.endswith("_id"):
                    continue
                # Check if column name (or a simplified version) appears in goal
                simplified = c.replace("_", " ")
                if simplified in goal_lower or c in goal_lower:
                    goal_mentioned_cols.append(c)
            # Also try partial matching for common terms
            if len(goal_mentioned_cols) < 2:
                for c in available:
                    if c in numeric_skip or c.endswith("_ts") or c.endswith("_id"):
                        continue
                    for part in c.split("_"):
                        if len(part) >= 4 and part in goal_lower and c not in goal_mentioned_cols:
                            goal_mentioned_cols.append(c)
                            break
            if len(goal_mentioned_cols) >= 2:
                col1, col2 = goal_mentioned_cols[0], goal_mentioned_cols[1]
            if col1 and col2:
                # Route to the table that has BOTH columns
                corr_table = table
                if col1 not in available or col2 not in available:
                    # Check other marts
                    for mart_name, mart_meta in (getattr(self.semantic, '_catalog', None) or {}).get("marts", {}).items():
                        mcols = mart_meta.get("columns", [])
                        if col1 in mcols and col2 in mcols:
                            corr_table = mart_name
                            break
                sql = (
                    f"SELECT CORR({_q(col1)}, {_q(col2)}) AS correlation_coefficient, "
                    f"COUNT(*) AS sample_size "
                    f"FROM {corr_table} WHERE {where_clause}"
                )
            else:
                sql = f"SELECT {metric_expr} AS metric_value FROM {table} WHERE {where_clause}"
        elif intent == "ranked_grouped":
            dims = plan.get("dimensions") or [plan.get("dimension")]
            dims = [d for d in dims if d]
            if dims:
                dim = dims[0]
                if dim == "__month__" and time_col:
                    dim_expr = f"DATE_TRUNC('month', {time_col})"
                    dim_alias = "month_bucket"
                else:
                    dim_expr = _q(dim)
                    dim_alias = dim
                sql = (
                    f"SELECT {dim_expr} AS {_q(dim_alias)}, {metric_expr} AS metric_value, "
                    f"RANK() OVER (ORDER BY {metric_expr} DESC) AS rank "
                    f"FROM {table} WHERE {where_clause} "
                    f"GROUP BY 1 ORDER BY 2 DESC LIMIT {plan['top_n']}"
                )
            else:
                sql = f"SELECT {metric_expr} AS metric_value FROM {table} WHERE {where_clause}"
        elif intent == "lookup":
            sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT {plan['top_n']}"
        else:
            sql = f"SELECT {metric_expr} AS metric_value FROM {table} WHERE {where_clause}"

        # ── GAP 14: Check for multi-table JOIN opportunity ──────────────────
        # Skip JOIN override for complex analytical intents that generate their own SQL.
        _no_join_override = {"comparison", "lookup", "subquery_filter", "running_total", "yoy_growth", "correlation"}
        secondary_domains = plan.get("secondary_domains", [])
        if secondary_domains and intent not in _no_join_override:
            for sec_domain in secondary_domains:
                sec_table = DOMAIN_TO_MART.get(sec_domain)
                if not sec_table:
                    continue
                join_key = (table, sec_table)
                join_path = JOIN_PATHS.get(join_key)
                if join_path:
                    join_sql = self._build_join_query(
                        plan, table, sec_table, join_path, metric_expr, where_clause,
                    )
                    if join_sql:
                        sql = join_sql
                        break

        # ── GAP 31: LLM SQL generation for complex intents ──────────────────
        if runtime and runtime.use_llm and runtime.provider and catalog:
            _llm_sql_intents = {"correlation", "subquery_filter", "running_total", "yoy_growth"}
            if plan.get("intent") in _llm_sql_intents or len(plan.get("secondary_domains", [])) > 1:
                llm_result = self._query_engine_with_llm(plan, runtime, catalog)
                if llm_result and llm_result.get("sql"):
                    validation = self.executor.validate(llm_result["sql"])
                    if validation.is_valid:
                        probe = self.executor.execute_probe(llm_result["sql"], limit=1)
                        if probe.success:
                            sql = llm_result["sql"]
                            plan["_llm_sql_used"] = True

        return {"sql": sql, "table": table}

    # ── GAP 31: LLM SQL Generation ──────────────────────────────
    def _query_engine_with_llm(
        self,
        plan: dict[str, Any],
        runtime: RuntimeSelection,
        catalog: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Use LLM to generate SQL for complex query patterns."""
        table = plan.get("table", "")
        # Build schema context
        schema_lines: list[str] = []
        for mart_name, mart_meta in catalog.get("marts", {}).items():
            cols = mart_meta.get("columns", [])
            schema_lines.append(f"  {mart_name}: {', '.join(cols[:20])}")
        schema_block = "\n".join(schema_lines) if schema_lines else "  (no schema)"

        plan_summary = {
            "goal": plan.get("goal", ""),
            "table": table,
            "intent": plan.get("intent", ""),
            "metric": plan.get("metric", ""),
            "metric_expr": plan.get("metric_expr", ""),
            "dimensions": plan.get("dimensions", []),
            "time_filter": plan.get("time_filter"),
            "value_filters": plan.get("value_filters", []),
            "time_column": plan.get("time_column"),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a DuckDB SQL expert. Generate a SQL query for the given analytics plan.\n\n"
                    f"SCHEMA:\n{schema_block}\n\n"
                    "DuckDB dialect notes:\n"
                    "- DATE_TRUNC('month', col) for time bucketing\n"
                    "- PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY col) for percentiles\n"
                    "- Window functions: SUM() OVER, LAG() OVER, RANK() OVER\n"
                    "- CTEs with WITH clause for subquery patterns\n"
                    "- CORR(col1, col2) for correlation\n\n"
                    "RULES:\n"
                    "- SELECT/WITH only — no INSERT, UPDATE, DELETE, DROP\n"
                    "- Alias all aggregated columns\n"
                    "- Use COALESCE for boolean filters\n"
                    "- Return ONLY JSON: {\"sql\": \"...\", \"reasoning\": \"...\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Plan: {json.dumps(plan_summary)}\n"
                    "Generate optimal DuckDB SQL. JSON only."
                ),
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="planner",
                provider=runtime.provider,
                timeout=30,
            )
            return _extract_json_payload(raw)
        except Exception as exc:
            self._pipeline_warnings.append(
                f"LLM SQL generation failed ({type(exc).__name__}); using template SQL."
            )
            return None

    @staticmethod
    def _qualify_metric_for_join(metric_expr: str, alias: str) -> str:
        """Qualify bare column references in metric expressions with a table alias.

        Prevents ambiguous column errors when a JOIN introduces columns with
        the same name in multiple tables (e.g. ``customer_id``).
        """
        if metric_expr == "COUNT(*)":
            return metric_expr
        # Simple column reference (no aggregate function).
        if "(" not in metric_expr:
            return f"{alias}.{metric_expr}"

        _SQL_WORDS = frozenset({
            "DISTINCT", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
            "AND", "OR", "NOT", "NULL", "TRUE", "FALSE", "ASC", "DESC",
            "OVER", "PARTITION", "BY", "ORDER", "ROWS", "BETWEEN",
            "PRECEDING", "FOLLOWING", "CURRENT", "ROW", "UNBOUNDED",
        })

        def _qualify(m: re.Match) -> str:
            word = m.group(0)
            if word.upper() in _SQL_WORDS:
                return word
            return f"{alias}.{word}"

        # Match identifiers not preceded by a dot (already qualified) and
        # not followed by '(' (function names).
        return re.sub(
            r"(?<!\.)(?<![a-zA-Z0-9_])\b([a-zA-Z_]\w*)\b(?!\s*\()",
            _qualify, metric_expr,
        )

    def _build_join_query(
        self,
        plan: dict[str, Any],
        primary_table: str,
        secondary_table: str,
        join_path: dict[str, Any],
        metric_expr: str,
        where_clause: str,
    ) -> str | None:
        """Generate a JOIN query between two tables using a known join path."""
        alias1, alias2 = join_path["aliases"]
        join_type = join_path["type"]
        on_clause = join_path["on"]

        dims = plan.get("dimensions") or []
        time_col = plan.get("time_column")
        intent = plan.get("intent", "metric")

        # Qualify bare column references in the metric expression so that
        # columns shared across joined tables (e.g. customer_id) are not
        # ambiguous.  (fixes GAP 37 COUNT(DISTINCT customer_id) issue)
        qualified_metric = self._qualify_metric_for_join(metric_expr, alias1)

        from_clause = f"{primary_table} {alias1} {join_type} {secondary_table} {alias2} ON {on_clause}"

        # Determine which columns belong to the primary table so we can
        # prefix secondary-table dimensions with the correct alias (GAP 14).
        primary_cols = set(plan.get("available_columns", []))

        if intent == "grouped_metric" and dims:
            select_parts: list[str] = []
            for dim in dims:
                if dim == "__month__" and time_col:
                    select_parts.append(f"DATE_TRUNC('month', {alias1}.{time_col}) AS month_bucket")
                elif dim == "__month__":
                    select_parts.append("'unknown_month' AS month_bucket")
                elif dim in primary_cols:
                    select_parts.append(f"{alias1}.{_q(dim)} AS {_q(dim)}")
                else:
                    # Dimension comes from the secondary (joined) table.
                    select_parts.append(f"{alias2}.{_q(dim)} AS {_q(dim)}")
            metric_idx = len(select_parts) + 1
            group_by = ", ".join(str(i) for i in range(1, len(select_parts) + 1))
            return (
                f"SELECT {', '.join(select_parts)}, {qualified_metric} AS metric_value "
                f"FROM {from_clause} WHERE {where_clause} "
                f"GROUP BY {group_by} ORDER BY {metric_idx} DESC NULLS LAST LIMIT {plan.get('top_n', 20)}"
            )
        else:
            return (
                f"SELECT {qualified_metric} AS metric_value "
                f"FROM {from_clause} WHERE {where_clause}"
            )

    def _build_where_clause(self, plan: dict[str, Any], for_comparison: str) -> str:
        clauses = ["1=1"]
        time_col = plan.get("time_column")
        time_filter = plan.get("time_filter")

        if time_col and time_filter:
            clauses.append(f"{time_col} IS NOT NULL")
            clauses.append(self._time_clause(time_col, time_filter, for_comparison))

        for vf in plan.get("value_filters", []):
            col = vf.get("column")
            value = vf.get("value", "")
            if not col:
                continue
            safe_value = str(value).replace("'", "''")
            clauses.append(
                f"LOWER(COALESCE(CAST({_q(col)} AS VARCHAR), '')) = LOWER('{safe_value}')"
            )

        return " AND ".join(clauses)

    def _time_clause(self, time_col: str, time_filter: dict[str, Any], for_comparison: str) -> str:
        kind = time_filter.get("kind")

        if kind == "month_year":
            month = int(time_filter.get("month"))
            year = time_filter.get("year")
            if for_comparison == "comparison":
                if year is None:
                    year = datetime.utcnow().year
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1
            if year is None:
                return f"EXTRACT(MONTH FROM {time_col}) = {month}"
            return f"EXTRACT(YEAR FROM {time_col}) = {year} AND EXTRACT(MONTH FROM {time_col}) = {month}"

        if kind == "year_only":
            year = int(time_filter.get("year"))
            if for_comparison == "comparison":
                year -= 1
            return f"EXTRACT(YEAR FROM {time_col}) = {year}"

        if kind == "relative":
            value = time_filter.get("value")
            if value == "this_month":
                if for_comparison == "comparison":
                    return f"DATE_TRUNC('month', {time_col}) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
                return f"DATE_TRUNC('month', {time_col}) = DATE_TRUNC('month', CURRENT_DATE)"
            if value == "last_month":
                if for_comparison == "comparison":
                    return f"DATE_TRUNC('month', {time_col}) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 month')"
                return f"DATE_TRUNC('month', {time_col}) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
            if value == "this_year":
                if for_comparison == "comparison":
                    return f"EXTRACT(YEAR FROM {time_col}) = EXTRACT(YEAR FROM CURRENT_DATE) - 1"
                return f"EXTRACT(YEAR FROM {time_col}) = EXTRACT(YEAR FROM CURRENT_DATE)"
            if value == "last_year":
                if for_comparison == "comparison":
                    return f"EXTRACT(YEAR FROM {time_col}) = EXTRACT(YEAR FROM CURRENT_DATE) - 2"
                return f"EXTRACT(YEAR FROM {time_col}) = EXTRACT(YEAR FROM CURRENT_DATE) - 1"

        # Fallback rolling windows for comparisons.
        if for_comparison == "comparison":
            return f"{time_col} >= CURRENT_DATE - INTERVAL '60 days' AND {time_col} < CURRENT_DATE - INTERVAL '30 days'"
        return f"{time_col} >= CURRENT_DATE - INTERVAL '30 days'"

    def _execution_agent(self, query_plan: dict[str, Any]) -> dict[str, Any]:
        res = self.executor.execute(query_plan["sql"])
        return {
            "success": res.success,
            "sql_executed": res.sql_executed,
            "error": res.error,
            "row_count": res.row_count,
            "columns": res.columns,
            "sample_rows": res.rows[:25],
            "execution_time_ms": res.execution_time_ms,
            "warnings": res.warnings,
        }

    # ── GAP 33: SQL Error Recovery ──────────────────────────────
    def _recover_failed_sql(
        self,
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        execution: dict[str, Any],
        runtime: RuntimeSelection,
    ) -> dict[str, Any] | None:
        """Use LLM to diagnose and fix a failed SQL query (single retry)."""
        failed_sql = query_plan.get("sql", "")
        error_msg = execution.get("error", "Unknown error")
        table = query_plan.get("table", "")
        goal_text = plan.get("goal", "")

        # Build schema context from stored catalog
        schema_info = ""
        _cat2 = getattr(self.semantic, '_catalog', None) or {}
        if _cat2:
            mart_meta = _cat2.get("marts", {}).get(table, {})
            cols = mart_meta.get("columns", [])
            if cols:
                schema_info = f"Table {table} columns: {', '.join(cols)}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a DuckDB SQL debugger. A query failed and you must fix it.\n\n"
                    "Common DuckDB fixes:\n"
                    "- Column not found → check the schema columns provided and use the correct name\n"
                    "- Type mismatch → use CAST(col AS type)\n"
                    "- Ambiguous column → use table alias prefix\n"
                    "- Syntax error → fix the SQL syntax\n\n"
                    "RULES:\n"
                    "- Return ONLY SELECT/WITH statements (no DML)\n"
                    "- Alias all aggregates\n"
                    "- Use COALESCE for booleans\n"
                    "- If the error is unfixable, return {\"sql\": null}\n\n"
                    f"Schema: {schema_info}\n\n"
                    "Return JSON: {\"sql\": \"...\", \"fix_description\": \"...\"} or {\"sql\": null}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question: {goal_text}\n"
                    f"Failed SQL: {failed_sql}\n"
                    f"Error: {error_msg}\n"
                    "Fix this query. JSON only."
                ),
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="planner",
                provider=runtime.provider,
                timeout=20,
            )
            return _extract_json_payload(raw)
        except Exception as exc:
            self._pipeline_warnings.append(
                f"SQL error recovery LLM call failed ({type(exc).__name__})"
            )
            return None

    def _audit_agent(
        self,
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        execution: dict[str, Any],
        runtime: RuntimeSelection | None = None,
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        goal_text = str(plan.get("goal", "")).lower()
        sql_lower = str(query_plan.get("sql", "")).lower()
        available_columns = [str(c).lower() for c in plan.get("available_columns", [])]
        execution_signature = _semantic_signature(
            {
                "sql": str(query_plan.get("sql") or "").strip(),
                "row_count": int(execution.get("row_count") or 0),
                "sample_row": (execution.get("sample_rows") or [None])[0],
            }
        )

        checks.append(
            {
                "name": "execution_success",
                "passed": execution["success"],
                "message": "Query executed" if execution["success"] else (execution.get("error") or "failed"),
            }
        )

        if execution["success"] and execution["row_count"] == 0:
            checks.append(
                {
                    "name": "non_empty_result",
                    "passed": False,
                    "message": "Query returned 0 rows",
                }
            )
            warnings.append("No rows returned; verify filters and period.")
        else:
            checks.append(
                {
                    "name": "non_empty_result",
                    "passed": True,
                    "message": f"Returned {execution['row_count']} rows",
                }
            )

        if plan.get("time_filter") and plan.get("time_column"):
            has_time_logic = plan["time_column"] in query_plan["sql"]
            checks.append(
                {
                    "name": "time_scope_applied",
                    "passed": has_time_logic,
                    "message": "Time logic present" if has_time_logic else "Time logic missing",
                }
            )
            if not has_time_logic:
                warnings.append("Expected time filter was not applied in SQL.")

        concept_expectations = {
            "markup": ["forex_markup"],
            "forex": ["forex_markup", "exchange_rate"],
            "charge": ["total_additional_charges", "swift_charges", "platform_charges"],
            "mt103": ["has_mt103"],
            "refund": ["has_refund"],
        }
        concept_misses: list[str] = []
        concept_hits: list[str] = []
        for concept, expected_hints in concept_expectations.items():
            if concept not in goal_text:
                continue
            hints = [h.lower() for h in expected_hints]
            supported = any(h in available_columns for h in hints)
            aligned = supported and (
                concept in str(plan.get("metric", "")).lower()
                or any(h in sql_lower for h in hints)
            )
            if aligned:
                concept_hits.append(concept)
            else:
                concept_misses.append(concept)

        if "forex" in goal_text and plan.get("table") != "datada_mart_quotes":
            concept_misses.append("forex_domain")
            warnings.append(
                f"Forex intent mapped to {plan.get('table')} instead of datada_mart_quotes."
            )

        concept_check_passed = len(concept_misses) == 0
        checks.append(
            {
                "name": "concept_alignment",
                "passed": concept_check_passed,
                "message": (
                    "Goal concepts aligned with selected metric/schema."
                    if concept_check_passed
                    else f"Concepts not fully aligned: {', '.join(concept_misses)}"
                ),
            }
        )
        concept_total = len(concept_hits) + len(concept_misses)
        concept_coverage = 1.0 if concept_total == 0 else len(concept_hits) / concept_total
        checks.append(
            {
                "name": "goal_term_coverage",
                "passed": concept_coverage >= 0.66,
                "message": f"Concept coverage {round(concept_coverage * 100, 1)}%",
            }
        )
        if concept_coverage < 1.0:
            warnings.append(
                f"Goal concept coverage below full match ({round(concept_coverage * 100, 1)}%)."
            )

        metric_expr = str(plan.get("metric_expr") or "").lower()
        metric_columns_detected = [
            col
            for col in available_columns
            if len(col) >= 3 and re.search(rf"\b{re.escape(col)}\b", metric_expr)
        ][:8]
        schema_grounded = bool(metric_columns_detected) or metric_expr.startswith("count(")
        checks.append(
            {
                "name": "schema_grounding",
                "passed": schema_grounded,
                "message": (
                    "Metric expression grounded in known table columns."
                    if schema_grounded
                    else "Metric expression could not be grounded to known columns."
                ),
            }
        )
        if not schema_grounded:
            warnings.append("Schema grounding check flagged metric expression.")

        replay_match: bool | None = None
        replay_signature: str | None = None
        if execution["success"] and query_plan.get("sql"):
            replay = self.executor.execute(query_plan["sql"])
            if replay.success:
                replay_match = replay.row_count == execution["row_count"]
                if replay_match and replay.rows and execution.get("sample_rows"):
                    replay_match = replay.rows[0] == execution["sample_rows"][0]
                replay_signature = _semantic_signature(
                    {
                        "sql": str(query_plan.get("sql") or "").strip(),
                        "row_count": int(replay.row_count or 0),
                        "sample_row": (replay.rows or [None])[0],
                    }
                )
                checks.append(
                    {
                        "name": "replay_consistency",
                        "passed": replay_match,
                        "message": (
                            "Re-execution produced matching top result."
                            if replay_match
                            else "Re-execution differed from first run."
                        ),
                    }
                )
                if replay_match is False:
                    warnings.append("Replay verification mismatch.")
            else:
                checks.append(
                    {
                        "name": "replay_consistency",
                        "passed": False,
                        "message": f"Replay failed: {replay.error}",
                    }
                )
                warnings.append("Replay verification failed.")

        if execution["execution_time_ms"] and execution["execution_time_ms"] > 4000:
            warnings.append("Execution latency above 4s.")

        failed_checks = sum(1 for c in checks if not c["passed"])
        score = 0.9
        if not execution["success"]:
            score = 0.18
        elif execution["row_count"] == 0:
            score -= 0.20
        score -= 0.10 * failed_checks
        score -= 0.10 * len(warnings)
        if not concept_check_passed:
            score = min(score, 0.45)
        score -= 0.08 * max(0.0, 1.0 - concept_coverage)
        if not schema_grounded:
            score -= 0.08

        # RC7: Semantic risk penalties
        # Penalize when domain was corrected by consistency enforcement
        if any("Domain corrected" in w for w in warnings):
            score -= 0.20
        # Penalize when specialist overrode the metric (signals ambiguity)
        if plan.get("_specialist_metric_override"):
            score -= 0.10

        score = max(0.0, min(1.0, score))

        # ── GAP 19: LLM-enhanced audit ──────────────────
        if runtime and runtime.use_llm and runtime.provider:
            llm_audit = self._audit_with_llm(plan, query_plan, execution, score, runtime)
            if llm_audit:
                adj = max(-0.3, min(0.2, float(llm_audit.get("score_adjustment", 0.0))))
                score = max(0.0, min(1.0, score + adj))
                extra_warnings = llm_audit.get("additional_warnings", [])
                if isinstance(extra_warnings, list):
                    warnings.extend(self._filter_noise_warnings(extra_warnings))

        query_columns_used = [c for c in available_columns if c in sql_lower]
        return {
            "checks": checks,
            "warnings": warnings,
            "score": score,
            "grounding": {
                "table": plan.get("table"),
                "metric": plan.get("metric"),
                "metric_expr": plan.get("metric_expr"),
                "goal_terms_detected": concept_hits + concept_misses,
                "goal_term_misses": concept_misses,
                "concept_coverage_pct": round(concept_coverage * 100, 1),
                "metric_columns_detected": metric_columns_detected,
                "query_columns_used": query_columns_used[:12],
                "replay_match": replay_match,
                "execution_signature": execution_signature,
                "replay_signature": replay_signature,
                "intent": plan.get("intent"),
                "dimensions": plan.get("dimensions", []),
                "time_filter": plan.get("time_filter"),
                "value_filters": plan.get("value_filters", []),
            },
        }

    def _audit_with_llm(
        self,
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        execution: dict[str, Any],
        deterministic_score: float,
        runtime: RuntimeSelection,
    ) -> dict[str, Any] | None:
        """Use LLM to evaluate audit quality and suggest score adjustments."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are AuditAgent evaluating whether a SQL query correctly answers the user's question. "
                    "Focus ONLY on data correctness:\n"
                    "1. Does the query retrieve the right data for the question?\n"
                    "2. Are the filters and aggregations correct?\n"
                    "3. Does the result answer what was asked?\n\n"
                    "Do NOT flag SQL style issues (WHERE 1=1, DISTINCT usage, LIMIT clauses, "
                    "code smells, unnecessary clauses). These are template artifacts, not bugs.\n"
                    "Do NOT warn about missing time filters unless the question explicitly asks about a time period.\n"
                    "Do NOT suggest performance optimizations.\n\n"
                    "Return JSON with keys: score_adjustment (float between -0.3 and 0.2), "
                    "additional_warnings (list — only genuine data correctness issues), reasoning (string)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {plan.get('goal', '')}\n"
                    f"SQL: {query_plan.get('sql', '')}\n"
                    f"Success: {execution.get('success')}, Rows: {execution.get('row_count')}\n"
                    f"Deterministic score: {deterministic_score:.2f}\n"
                    "JSON only."
                ),
            },
        ]
        try:
            raw = call_llm(
                messages,
                role="intent",
                provider=runtime.provider,
                model=runtime.intent_model,
                timeout=20,
            )
            return _extract_json_payload(raw)
        except Exception as exc:
            self._pipeline_warnings.append(
                f"Audit LLM enhancement failed ({type(exc).__name__}); using deterministic score."
            )
        return None

    # ── GAP 23: Warning noise filter ────────────────────
    _WARNING_NOISE_PATTERNS = re.compile(
        r"WHERE\s+1\s*=\s*1|"
        r"code\s*smell|"
        r"may\s+be\s+unnecessary|"
        r"consider\s+(removing|verifying|using)|"
        r"ensure\s+that|"
        r"DISTINCT.*may\s+be|"
        r"auto-generated|"
        r"templated\s+SQL|"
        r"performance\s+implications|"
        r"better\s+index\s+utilization|"
        r"not\s+a\s+valid\s+option|"
        r"NULLS\s+LAST.*may\s+not\s+be\s+supported|"
        r"redundant\s+and\s+suggests|"
        r"can\s+be\s+removed",
        re.IGNORECASE,
    )

    def _filter_noise_warnings(self, warnings: list[str]) -> list[str]:
        """Remove code-style and template-artifact warnings; keep data correctness issues."""
        return [w for w in warnings if not self._WARNING_NOISE_PATTERNS.search(w)]

    def _narrative_agent(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
        runtime: RuntimeSelection,
        storyteller_mode: bool,
        conversation_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = self._narrative_deterministic(goal, plan, execution, storyteller_mode)

        if not runtime.use_llm or not runtime.provider or not execution["success"]:
            fallback["suggested_questions"] = self._suggested_questions(plan)
            fallback["llm_narrative_used"] = False
            return fallback

        prompt = {
            "question": goal,
            "plan": {
                "table": plan["table"],
                "metric": plan["metric"],
                "intent": plan["intent"],
                "dimension": plan.get("dimension"),
                "dimensions": plan.get("dimensions", []),
            },
            "rows": execution["sample_rows"][:8],
            "row_count": execution["row_count"],
            "audit_warnings": audit.get("warnings", []),
            "conversation_context": conversation_context[-3:],
            "style": "storyteller" if storyteller_mode else "professional_conversational",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are NarrativeAgent — a data analyst writing answers for business users.\n\n"
                    "FORMAT RULES:\n"
                    "- Lead with the key number/metric in **bold** on the first line\n"
                    "- Use markdown formatting: **bold** for numbers, bullet points for breakdowns\n"
                    "- Keep it concise: max 80 words, max 3 bullet points\n"
                    "- For grouped data, use a markdown table or bullet list\n"
                    "- End with one actionable insight or observation\n\n"
                    "ACCURACY RULES:\n"
                    "- ONLY use numbers from the provided SQL result rows — never invent data\n"
                    "- If row_count is 0, say 'No data found' — do not speculate\n"
                    "- Include the actual values from the result, not approximations\n\n"
                    "Return JSON with keys: answer_markdown (string), suggested_questions (list of 2-3 strings)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, default=str),
            },
        ]

        try:
            raw = call_llm(
                messages,
                role="narrator",
                provider=runtime.provider,
                model=runtime.narrator_model,
                timeout=45,
            )
            parsed = _extract_json_payload(raw)
            if parsed and isinstance(parsed.get("answer_markdown"), str):
                cleaned = self._clean_llm_answer_markdown(parsed["answer_markdown"])
                if self._llm_narrative_contradicts_execution(cleaned, execution):
                    fallback["suggested_questions"] = self._suggested_questions(plan)
                    fallback["llm_narrative_used"] = False
                    return fallback
                cleaned = self._compact_answer_markdown(cleaned)
                return {
                    "answer_markdown": cleaned,
                    "headline_value": fallback.get("headline_value"),
                    "suggested_questions": self._normalize_suggested_questions(
                        parsed.get("suggested_questions"),
                        plan,
                    ),
                    "llm_narrative_used": True,
                }
            if isinstance(raw, str) and raw.strip():
                raw_text = raw.strip()
                # Avoid rendering malformed JSON blobs directly to end users.
                if raw_text.startswith("{") and "answer_markdown" in raw_text:
                    fallback["suggested_questions"] = self._suggested_questions(plan)
                    fallback["llm_narrative_used"] = False
                    return fallback
                cleaned = self._clean_llm_answer_markdown(raw_text)
                if self._llm_narrative_contradicts_execution(cleaned, execution):
                    fallback["suggested_questions"] = self._suggested_questions(plan)
                    fallback["llm_narrative_used"] = False
                    return fallback
                cleaned = self._compact_answer_markdown(cleaned)
                return {
                    "answer_markdown": cleaned,
                    "headline_value": fallback.get("headline_value"),
                    "suggested_questions": self._suggested_questions(plan),
                    "llm_narrative_used": True,
                }
        except Exception as exc:
            self._pipeline_warnings.append(
                f"LLM narrative generation failed ({type(exc).__name__}: {exc})"
            )

        fallback["suggested_questions"] = self._suggested_questions(plan)
        fallback["llm_narrative_used"] = False
        return fallback

    def _narrative_deterministic(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        storyteller_mode: bool = False,
    ) -> dict[str, Any]:
        if not execution["success"]:
            return {
                "answer_markdown": (
                    f"**Unable to answer:** {goal}\n\n"
                    f"Execution error: {execution.get('error', 'unknown error')}"
                ),
                "headline_value": None,
            }

        rows = execution["sample_rows"]
        if execution["row_count"] == 0:
            return {
                "answer_markdown": (
                    f"**No data found for:** {goal}\n\n"
                    "Try widening the time scope or using fewer filters."
                ),
                "headline_value": 0,
            }

        if plan["intent"] == "comparison" and len(rows) >= 2:
            current = _to_float(rows[0].get("metric_value")) or 0.0
            other = _to_float(rows[1].get("metric_value")) or 0.0
            delta = current - other
            delta_pct = (delta / other * 100.0) if other else None
            line = f"Difference: {_fmt_number(delta)}"
            if delta_pct is not None:
                line += f" ({delta_pct:+.2f}%)"
            intro = "Here is the story in two snapshots:" if storyteller_mode else ""
            return {
                "answer_markdown": (
                    f"**Comparison result for: {goal}**\n\n"
                    + (f"{intro}\n\n" if intro else "")
                    + f"- Current: {_fmt_number(current)}\n"
                    + f"- Comparison: {_fmt_number(other)}\n"
                    + f"- {line}"
                ),
                "headline_value": current,
            }

        # ── GAP 20: Narrative templates for complex analytical intents ──
        if plan["intent"] == "running_total" and rows:
            bullets = []
            for row in rows[:5]:
                bucket = row.get("time_bucket", "")
                val = _fmt_number(row.get("metric_value"))
                rt = _fmt_number(row.get("running_total"))
                bullets.append(f"- {bucket}: {val} (cumulative: {rt})")
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    + "\n".join(bullets)
                ),
                "headline_value": rows[-1].get("running_total") if rows else None,
            }

        if plan["intent"] == "yoy_growth" and rows:
            bullets = []
            for row in rows[:5]:
                bucket = row.get("time_bucket", "")
                val = _fmt_number(row.get("metric_value"))
                growth = row.get("growth_pct")
                growth_str = f"{growth}%" if growth is not None else "N/A"
                change = "increase" if growth is not None and float(growth) > 0 else "decrease" if growth is not None and float(growth) < 0 else "change"
                bullets.append(f"- {bucket}: {val} (growth: {growth_str}, {change})")
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    "Period-over-period growth analysis:\n"
                    + "\n".join(bullets)
                ),
                "headline_value": rows[-1].get("growth_pct") if rows else None,
            }

        if plan["intent"] == "correlation" and rows:
            row = rows[0]
            coeff = row.get("correlation_coefficient")
            sample_size = row.get("sample_size")
            coeff_str = f"{float(coeff):.4f}" if coeff is not None else "N/A"
            # Extract column names from goal for the narrative
            goal_lower = goal.lower()
            col_names = []
            for c in execution.get("columns", []):
                if c not in ("correlation_coefficient", "sample_size"):
                    col_names.append(c)
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    f"- **Correlation coefficient**: r={coeff_str}\n"
                    f"- **Sample size**: {sample_size}\n"
                    f"- The correlation between amount and exchange rate is {coeff_str}"
                ),
                "headline_value": coeff,
            }

        if plan["intent"] == "subquery_filter" and rows:
            preview = rows[:5]
            bullets = []
            for row in preview:
                keys = list(row.keys())
                if len(keys) == 2:
                    bullets.append(f"- {row[keys[0]]}: {_fmt_number(row[keys[1]])}")
                elif len(keys) >= 3:
                    left = " | ".join(str(row[k]) for k in keys[:-1])
                    bullets.append(f"- {left}: {_fmt_number(row[keys[-1]])}")
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    f"Found {len(rows)} entries matching the filter:\n"
                    + "\n".join(bullets)
                ),
                "headline_value": len(rows),
            }

        if len(rows) == 1 and len(execution["columns"]) == 1:
            key = execution["columns"][0]
            value = rows[0].get(key)
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    f"- **{key}**: {_fmt_number(value)}"
                ),
                "headline_value": value,
            }

        if plan["intent"] == "grouped_metric" and rows:
            preview = rows[:5]
            bullets = []
            for row in preview:
                keys = list(row.keys())
                if len(keys) == 2:
                    bullets.append(f"- {row[keys[0]]}: {_fmt_number(row[keys[1]])}")
                elif len(keys) >= 3:
                    left = " | ".join(str(row[k]) for k in keys[:-1])
                    bullets.append(f"- {left}: {_fmt_number(row[keys[-1]])}")
            intro = (
                "I looked at the pattern across groups and here are the leaders:"
                if storyteller_mode
                else f"Top {len(preview)} groups:"
            )
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    f"{intro}\n" + "\n".join(bullets)
                ),
                "headline_value": preview[0].get(list(preview[0].keys())[-1]) if preview else None,
            }

        return {
            "answer_markdown": (
                f"**{goal}**\n\n"
                f"Returned {execution['row_count']} rows. Showing sample in result preview."
            ),
            "headline_value": execution["row_count"],
        }

    def _viz_agent(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        if not execution["success"] or execution["row_count"] == 0:
            return {"type": "none", "reason": "no_data"}

        cols = execution["columns"]
        if plan["intent"] == "comparison" and set(cols) >= {"period", "metric_value"}:
            return {
                "type": "bar",
                "x": "period",
                "y": "metric_value",
                "title": "Current vs Comparison",
            }

        if plan["intent"] == "grouped_metric" and len(cols) >= 2:
            dims = plan.get("dimensions") or [plan.get("dimension")]
            chart_type = "line" if "__month__" in dims else "bar"
            x_col = cols[0]
            y_col = cols[-1]
            series_col = cols[1] if len(cols) >= 3 else None
            return {
                "type": chart_type,
                "x": x_col,
                "y": y_col,
                "series": series_col,
                "title": f"{plan['metric']} by {', '.join([d for d in dims if d]) or x_col}",
            }

        if len(cols) == 1 and execution["row_count"] == 1:
            return {
                "type": "indicator",
                "metric": cols[0],
                "title": plan["metric"],
            }

        return {
            "type": "table",
            "columns": cols,
        }

    def _suggested_questions(self, plan: dict[str, Any]) -> list[str]:
        metric = plan["metric"]
        table = plan["table"]
        if table == "datada_mart_transactions":
            return [
                "Show this metric by platform",
                "Compare this month vs last month",
                "What is the refund rate by state?",
            ]
        if table == "datada_mart_quotes":
            return [
                "Break down quote value by from_currency",
                "Show total forex markup revenue by month",
                "Show total charges by currency",
                "Show quote volume by status",
                "Compare this year vs last year quotes",
            ]
        if table == "datada_dim_customers":
            return [
                "Show customer distribution by country",
                "How many university payees are active?",
                "Trend customer creation by month",
            ]
        if table == "datada_document_chunks":
            return [
                "Show more snippets from the top matching source",
                "List document titles related to this topic",
                "Summarize key clauses with citations",
            ]
        return [
            f"Show {metric} by status",
            "Compare this month vs last month",
            "List top 10 entities by value",
        ]

    def _sanitize_intake_payload(self, parsed: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        if not isinstance(parsed, dict):
            return cleaned

        for key in ["intent", "domain", "metric", "dimension"]:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()

        raw_dims = parsed.get("dimensions")
        dims: list[str] = []
        if isinstance(raw_dims, list):
            for item in raw_dims:
                if isinstance(item, str) and item.strip():
                    dims.append(item.strip())
        if dims:
            cleaned["dimensions"] = dims[:MAX_DIMENSIONS]
            if "dimension" not in cleaned:
                cleaned["dimension"] = dims[0]

        top_n = parsed.get("top_n")
        if top_n is not None:
            try:
                cleaned["top_n"] = max(1, min(100, int(top_n)))
            except Exception:
                pass

        time_filter = parsed.get("time_filter")
        if isinstance(time_filter, dict):
            cleaned["time_filter"] = time_filter

        raw_filters = parsed.get("value_filters")
        value_filters: list[dict[str, str]] = []
        if isinstance(raw_filters, list):
            for item in raw_filters:
                if not isinstance(item, dict):
                    continue
                col = item.get("column")
                val = item.get("value")
                # Convert non-string values (e.g. JSON boolean true → "true")
                if val is not None and not isinstance(val, str):
                    val = str(val).lower()
                if isinstance(col, str) and col.strip() and isinstance(val, str) and val.strip():
                    value_filters.append({"column": col.strip(), "value": val.strip()})
        if value_filters:
            cleaned["value_filters"] = value_filters
        return cleaned

    def _normalize_suggested_questions(self, value: Any, plan: dict[str, Any]) -> list[str]:
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                    continue
                if isinstance(item, dict):
                    for k in ["question", "text", "title", "prompt"]:
                        candidate = item.get(k)
                        if isinstance(candidate, str) and candidate.strip():
                            out.append(candidate.strip())
                            break
            if out:
                return out[:5]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return self._suggested_questions(plan)

    def _clean_llm_answer_markdown(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\**\s*answer markdown\s*\**\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
        fenced = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        cleaned = cleaned.replace("\r\n", "\n").strip()
        return cleaned or text.strip()

    def _llm_narrative_contradicts_execution(
        self,
        text: str,
        execution: dict[str, Any],
    ) -> bool:
        if execution.get("row_count", 0) <= 0:
            return False
        lower = text.lower()
        contradiction_patterns = [
            "don't have any information",
            "do not have any information",
            "no information about",
            "dataset only includes",
            "need to gather financial data",
            "data is not available",
        ]
        return any(p in lower for p in contradiction_patterns)

    def _compact_answer_markdown(
        self,
        text: str,
        *,
        max_lines: int = 8,
        max_chars: int = 720,
    ) -> str:
        lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        out = "\n".join(lines).strip()
        if len(out) > max_chars:
            out = out[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
        return out or text.strip()

    def _to_confidence(self, score: float) -> ConfidenceLevel:
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        if score >= 0.6:
            return ConfidenceLevel.MEDIUM
        if score >= 0.4:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN

    def _memory_write_turn(
        self,
        *,
        trace: list[dict[str, Any]],
        trace_id: str,
        goal: str,
        resolved_goal: str,
        tenant_id: str,
        runtime: RuntimeSelection,
        success: bool,
        confidence_score: float,
        row_count: int | None,
        plan: dict[str, Any],
        sql: str | None,
        audit_warnings: list[str],
        correction_applied: bool,
        correction_reason: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        start = time.perf_counter()
        try:
            self.memory.store_turn(
                tenant_id=tenant_id,
                trace_id=trace_id,
                goal=goal,
                resolved_goal=resolved_goal,
                runtime_mode=runtime.mode,
                provider=runtime.provider,
                success=success,
                confidence_score=confidence_score,
                row_count=row_count,
                plan=plan,
                sql=sql,
                audit_warnings=audit_warnings,
                correction_applied=correction_applied,
                correction_reason=correction_reason,
                metadata=metadata,
            )
            trace.append(
                {
                    "agent": "MemoryAgent",
                    "role": "episodic_write",
                    "status": "success",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": "turn persisted",
                }
            )
        except Exception as exc:
            trace.append(
                {
                    "agent": "MemoryAgent",
                    "role": "episodic_write",
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": f"memory write skipped: {exc}",
                }
            )

    def _memory_learn_from_success(
        self,
        trace: list[dict[str, Any]],
        *,
        tenant_id: str,
        goal: str,
        plan: dict[str, Any],
        score: float,
    ) -> None:
        start = time.perf_counter()
        try:
            correction_id = self.memory.learn_from_success(
                tenant_id=tenant_id,
                goal=goal,
                plan=plan,
                score=score,
            )
            trace.append(
                {
                    "agent": "MemoryAgent",
                    "role": "autonomous_learning",
                    "status": "success",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": correction_id or "no-new-rule",
                }
            )
        except Exception as exc:
            trace.append(
                {
                    "agent": "MemoryAgent",
                    "role": "autonomous_learning",
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": str(exc),
                }
            )

    def _runtime_payload(
        self,
        runtime: RuntimeSelection,
        *,
        llm_intake_used: bool = False,
        llm_narrative_used: bool = False,
        autonomy: AutonomyConfig | None = None,
        correction_applied: bool = False,
    ) -> dict[str, Any]:
        autonomy_cfg = autonomy or AutonomyConfig()
        return {
            "requested_mode": runtime.requested_mode,
            "mode": runtime.mode,
            "provider": runtime.provider,
            "use_llm": runtime.use_llm,
            "reason": runtime.reason,
            "intent_model": runtime.intent_model,
            "narrator_model": runtime.narrator_model,
            "llm_intake_used": llm_intake_used,
            "llm_narrative_used": llm_narrative_used,
            "llm_effective": llm_intake_used or llm_narrative_used,
            "autonomy_mode": autonomy_cfg.mode,
            "auto_correction": autonomy_cfg.auto_correction,
            "strict_truth": autonomy_cfg.strict_truth,
            "max_refinement_rounds": autonomy_cfg.max_refinement_rounds,
            "max_candidate_plans": autonomy_cfg.max_candidate_plans,
            "correction_applied": correction_applied,
            "memory_db_path": str(self.memory_db_path),
        }


def load_dotenv_file(path: Path | str = ".env") -> None:
    """Best-effort env loader to activate API keys in local runs."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
