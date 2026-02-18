"""Agentic analytics team runtime for dataDa.

This module implements a hierarchical, role-based multi-agent workflow that
simulates a compact analytics and data-engineering team.
"""

from __future__ import annotations

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
from haikugraph.llm.router import call_llm
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


@dataclass
class RuntimeSelection:
    requested_mode: str
    mode: str
    use_llm: bool
    provider: str | None
    reason: str
    intent_model: str | None = None
    narrator_model: str | None = None


class SemanticLayerManager:
    """Builds and maintains typed semantic marts for agent use."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._prepared_mtime: float | None = None
        self._catalog: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")

            mtime = self.db_path.stat().st_mtime
            if not force and self._catalog is not None and self._prepared_mtime == mtime:
                return self._catalog

            conn = duckdb.connect(str(self.db_path), read_only=False)
            try:
                source_tables = self._detect_source_tables(conn)
                self._build_transactions_view(conn, source_tables.get("transactions"))
                self._build_quotes_view(conn, source_tables.get("quotes"))
                self._build_customers_view(conn, source_tables.get("customers"))
                self._build_bookings_view(conn, source_tables.get("bookings"))
                catalog = self._build_catalog(conn, source_tables)
            finally:
                conn.close()

            self._catalog = catalog
            self._prepared_mtime = mtime
            return catalog

    def _detect_source_tables(self, conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
        tables = [
            t
            for (t,) in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
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
        return f"NULL AS {_q(out_alias)}"

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
                CREATE OR REPLACE VIEW datada_mart_transactions AS
                SELECT
                    ''::VARCHAR AS transaction_key,
                    NULL::VARCHAR AS transaction_id,
                    NULL::VARCHAR AS customer_id,
                    NULL::VARCHAR AS payee_id,
                    NULL::VARCHAR AS platform_name,
                    NULL::VARCHAR AS state,
                    NULL::VARCHAR AS txn_flow,
                    NULL::VARCHAR AS payment_status,
                    NULL::VARCHAR AS account_details_status,
                    NULL::VARCHAR AS deal_details_status,
                    NULL::VARCHAR AS quote_status,
                    NULL::TIMESTAMP AS event_ts,
                    NULL::TIMESTAMP AS mt103_created_ts,
                    NULL::TIMESTAMP AS payment_created_ts,
                    NULL::TIMESTAMP AS created_ts,
                    NULL::TIMESTAMP AS updated_ts,
                    0.0::DOUBLE AS payment_amount,
                    0.0::DOUBLE AS deal_amount,
                    0.0::DOUBLE AS amount_collected,
                    0.0::DOUBLE AS amount,
                    FALSE::BOOLEAN AS has_mt103,
                    FALSE::BOOLEAN AS has_refund
                WHERE FALSE
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
            COALESCE(transaction_id, CONCAT('row_', CAST(ROW_NUMBER() OVER () AS VARCHAR))) AS transaction_key,
            transaction_id,
            customer_id,
            payee_id,
            platform_name,
            state,
            txn_flow,
            payment_status,
            account_details_status,
            deal_details_status,
            quote_status,
            COALESCE(payment_created_ts, created_ts, updated_ts) AS event_ts,
            mt103_created_ts,
            payment_created_ts,
            created_ts,
            updated_ts,
            COALESCE(payment_amount_num, 0.0) AS payment_amount,
            COALESCE(deal_amount_num, 0.0) AS deal_amount,
            COALESCE(amount_collected_num, 0.0) AS amount_collected,
            COALESCE(payment_amount_num, deal_amount_num, amount_collected_num, 0.0) AS amount,
            CASE WHEN mt103_created_ts IS NULL THEN FALSE ELSE TRUE END AS has_mt103,
            CASE WHEN refund_id_raw IS NULL THEN FALSE ELSE TRUE END AS has_refund
        FROM base
        """
        conn.execute(sql)

    def _build_quotes_view(self, conn: duckdb.DuckDBPyConnection, table: str | None) -> None:
        if not table:
            conn.execute(
                """
                CREATE OR REPLACE VIEW datada_mart_quotes AS
                SELECT
                    ''::VARCHAR AS quote_key,
                    NULL::VARCHAR AS quote_id,
                    NULL::VARCHAR AS ref_id,
                    NULL::VARCHAR AS status,
                    NULL::VARCHAR AS from_currency,
                    NULL::VARCHAR AS to_currency,
                    NULL::VARCHAR AS purpose_code,
                    NULL::VARCHAR AS transaction_range,
                    NULL::TIMESTAMP AS created_ts,
                    NULL::TIMESTAMP AS updated_ts,
                    0.0::DOUBLE AS amount_at_source,
                    0.0::DOUBLE AS amount_at_destination,
                    0.0::DOUBLE AS total_amount_to_be_paid,
                    0.0::DOUBLE AS exchange_rate,
                    0.0::DOUBLE AS total_additional_charges,
                    0.0::DOUBLE AS forex_markup,
                    0.0::DOUBLE AS amount_without_markup,
                    0.0::DOUBLE AS swift_charges,
                    0.0::DOUBLE AS platform_charges
                WHERE FALSE
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
                CREATE OR REPLACE VIEW datada_dim_customers AS
                SELECT
                    ''::VARCHAR AS customer_key,
                    ''::VARCHAR AS payee_key,
                    NULL::VARCHAR AS customer_id,
                    NULL::VARCHAR AS payee_id,
                    NULL::BOOLEAN AS is_university,
                    NULL::VARCHAR AS type,
                    NULL::VARCHAR AS address_country,
                    NULL::VARCHAR AS address_state,
                    NULL::VARCHAR AS status,
                    NULL::TIMESTAMP AS created_ts,
                    NULL::TIMESTAMP AS updated_ts
                WHERE FALSE
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
                CREATE OR REPLACE VIEW datada_mart_bookings AS
                SELECT
                    ''::VARCHAR AS booking_key,
                    NULL::TIMESTAMP AS booked_ts,
                    NULL::DATE AS value_date,
                    NULL::VARCHAR AS currency,
                    NULL::VARCHAR AS deal_type,
                    NULL::VARCHAR AS status,
                    NULL::VARCHAR AS linked_txn_status,
                    0.0::DOUBLE AS booked_amount,
                    0.0::DOUBLE AS available_balance,
                    0.0::DOUBLE AS amount_on_hold,
                    0.0::DOUBLE AS linked_txn_amount,
                    0.0::DOUBLE AS rate
                WHERE FALSE
                """
            )
            return

        cols = self._table_cols(conn, table)
        sql = f"""
        CREATE OR REPLACE VIEW datada_mart_bookings AS
        WITH base AS (
            SELECT
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
            COALESCE(rate_num, 0.0) AS rate
        FROM base
        """
        conn.execute(sql)

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
            "metrics_by_table": {
                "datada_mart_transactions": {
                    "transaction_count": "COUNT(DISTINCT transaction_key)",
                    "unique_customers": "COUNT(DISTINCT customer_id)",
                    "total_amount": "SUM(amount)",
                    "avg_amount": "AVG(amount)",
                    "refund_count": "SUM(CASE WHEN has_refund THEN 1 ELSE 0 END)",
                    "refund_rate": "AVG(CASE WHEN has_refund THEN 1.0 ELSE 0.0 END)",
                    "mt103_count": "SUM(CASE WHEN has_mt103 THEN 1 ELSE 0 END)",
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
                    "booking_count": "COUNT(DISTINCT booking_key)",
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
            "quality": {},
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
            "datada_mart_transactions": ["platform_name", "state", "payment_status", "txn_flow"],
            "datada_mart_quotes": ["status", "from_currency", "to_currency", "purpose_code"],
            "datada_dim_customers": ["type", "address_country", "address_state", "status"],
            "datada_mart_bookings": ["currency", "deal_type", "status", "linked_txn_status"],
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
        catalog["quality"] = tx_quality
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

    def close(self) -> None:
        self.executor.close()

    def __enter__(self) -> "AgenticAnalyticsTeam":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def record_feedback(
        self,
        *,
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

    def run(
        self,
        goal: str,
        runtime: RuntimeSelection,
        *,
        conversation_context: list[dict[str, Any]] | None = None,
        storyteller_mode: bool = False,
        autonomy: AutonomyConfig | None = None,
    ) -> AssistantQueryResponse:
        trace_id = str(uuid.uuid4())
        trace: list[dict[str, Any]] = []
        started = time.perf_counter()
        history = conversation_context or []
        autonomy_cfg = autonomy or AutonomyConfig()

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
            catalog = self._run_agent(trace, "DataEngineeringTeam", "semantic_layer", self.semantic.prepare)
            mission = self._run_agent(
                trace,
                "ChiefAnalystAgent",
                "supervision",
                self._chief_analyst,
                effective_goal,
                runtime,
                catalog,
            )
            memory_hints = self._run_agent(
                trace,
                "MemoryAgent",
                "episodic_recall",
                self.memory.recall,
                effective_goal,
            )
            learned_corrections = self._run_agent(
                trace,
                "MemoryAgent",
                "correction_rule_recall",
                self.memory.get_matching_corrections,
                effective_goal,
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

            pre_gov = self._run_agent(
                trace,
                "GovernanceAgent",
                "policy_gate",
                self._governance_precheck,
                effective_goal,
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
                )

            if intake.get("intent") == "data_overview":
                overview = self._run_agent(
                    trace,
                    "CatalogExplainerAgent",
                    "dataset_overview",
                    self._data_overview_agent,
                    goal,
                    catalog,
                    storyteller_mode,
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
                        )
                    ],
                    trace_id=trace_id,
                    runtime=self._runtime_payload(runtime, autonomy=autonomy_cfg),
                    agent_trace=trace,
                    chart_spec={
                        "type": "table",
                        "columns": ["mart", "row_count"],
                        "title": "Available semantic marts",
                    },
                    data_quality=catalog.get("quality", {}),
                    sample_rows=overview.get("sample_rows", []),
                    columns=["mart", "row_count"],
                    suggested_questions=overview.get("suggested_questions", []),
                )

            retrieval = self._run_agent(
                trace,
                "SemanticRetrievalAgent",
                "domain_mapping",
                self._semantic_retrieval_agent,
                intake,
                catalog,
            )
            plan = self._run_agent(
                trace,
                "PlanningAgent",
                "task_graph",
                self._planning_agent,
                intake,
                retrieval,
                catalog,
            )

            tx_findings = self._run_agent(
                trace,
                "TransactionsSpecialistAgent",
                "domain_analysis",
                self._transactions_specialist,
                plan,
            )
            customer_findings = self._run_agent(
                trace,
                "CustomerSpecialistAgent",
                "domain_analysis",
                self._customer_specialist,
                plan,
            )
            revenue_findings = self._run_agent(
                trace,
                "RevenueSpecialistAgent",
                "domain_analysis",
                self._revenue_specialist,
                plan,
            )
            risk_findings = self._run_agent(
                trace,
                "RiskSpecialistAgent",
                "domain_analysis",
                self._risk_specialist,
                plan,
            )

            query_plan = self._run_agent(
                trace,
                "QueryEngineerAgent",
                "sql_compilation",
                self._query_engine_agent,
                plan,
                [tx_findings, customer_findings, revenue_findings, risk_findings],
            )
            execution = self._run_agent(
                trace,
                "ExecutionAgent",
                "safe_execution",
                self._execution_agent,
                query_plan,
            )
            audit = self._run_agent(
                trace,
                "AuditAgent",
                "consistency_validation",
                self._audit_agent,
                plan,
                query_plan,
                execution,
            )
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
            chart_spec = self._run_agent(
                trace,
                "VisualizationAgent",
                "chart_suggestion",
                self._viz_agent,
                plan,
                execution,
            )

            confidence_score = max(0.0, min(1.0, float(audit.get("score", 0.6))))
            confidence = self._to_confidence(confidence_score)

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
                    "probe_findings": autonomy_result.get("probe_findings", []),
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
                },
            )

            if autonomy_result.get("correction_applied") and confidence_score >= 0.78:
                self._memory_learn_from_success(
                    trace,
                    goal=effective_goal,
                    plan=plan,
                    score=confidence_score,
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
                ),
                agent_trace=trace,
                chart_spec=chart_spec,
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
                    },
                },
                suggested_questions=narration.get("suggested_questions", []),
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
            )

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
            trace.append(
                {
                    "agent": agent,
                    "role": role,
                    "status": "success",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "summary": _compact(out),
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
                }
            )
            raise

    def _chief_analyst(
        self,
        goal: str,
        runtime: RuntimeSelection,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "mission": goal,
            "runtime_mode": runtime.mode,
            "specialists": [
                "transactions",
                "customers",
                "revenue",
                "risk",
            ],
            "available_marts": list(catalog.get("marts", {}).keys()),
        }

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

    def _data_overview_agent(
        self,
        goal: str,
        catalog: dict[str, Any],
        storyteller_mode: bool,
    ) -> dict[str, Any]:
        marts = catalog.get("marts", {})
        rows = [
            {"mart": mart, "row_count": meta.get("row_count", 0)}
            for mart, meta in marts.items()
        ]
        rows.sort(key=lambda x: x["row_count"], reverse=True)

        bullets = []
        for row in rows:
            bullets.append(f"- `{row['mart']}`: {_fmt_number(row['row_count'])} rows")

        intro = "Here is the map of your data." if storyteller_mode else "Here is what data you currently have."
        answer = (
            f"**{goal}**\n\n"
            f"{intro}\n\n"
            + "\n".join(bullets)
            + "\n\nI can answer questions on transactions, quotes, customers, and bookings."
        )
        if storyteller_mode:
            answer += (
                "\n\nThink of this as four chapters: payments, quotes, customer profiles, and booking activity."
            )

        return {
            "answer_markdown": answer,
            "sample_rows": rows,
            "suggested_questions": [
                "Show total transaction amount by month",
                "Show MT103 count by month and platform",
                "Which countries have the most customers?",
            ],
        }

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
            llm_parsed = self._intake_with_llm(goal, runtime, parsed)
            if llm_parsed:
                cleaned = self._sanitize_intake_payload(llm_parsed)
                if not explicit_top:
                    cleaned.pop("top_n", None)
                if cleaned:
                    merged = {
                        **parsed,
                        **{k: v for k, v in cleaned.items() if v not in (None, "")},
                    }
                    merged = self._enforce_intake_consistency(goal, deterministic, merged)
                    parsed.update(merged)
                    parsed["_llm_intake_used"] = True
        parsed = self._enforce_intake_consistency(goal, deterministic, parsed)
        if memory_hints:
            parsed = self._apply_memory_hints(goal, parsed, memory_hints)
        return parsed

    def _intake_with_llm(
        self,
        goal: str,
        runtime: RuntimeSelection,
        fallback: dict[str, Any],
    ) -> dict[str, Any] | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are IntakeAgent in an analytics team. "
                    "Return only JSON with keys: intent, domain, metric, dimension, dimensions, "
                    "top_n, time_filter, and value_filters."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {goal}\n"
                    f"Fallback parse: {json.dumps(fallback)}\n"
                    "Refine this parse. Output JSON only."
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
        except Exception:
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

    def _intake_deterministic(self, goal: str, catalog: dict[str, Any]) -> dict[str, Any]:
        g = goal.strip()
        lower = g.lower()

        intent = "metric"
        if any(
            k in lower
            for k in [
                "what kind of data",
                "what kinda data",
                "what data do i have",
                "what can i ask",
                "show available data",
                "describe my data",
                "what tables do i have",
            ]
        ):
            intent = "data_overview"
        elif any(k in lower for k in [" vs ", " versus ", "compare", "compared"]):
            intent = "comparison"
        elif any(
            k in lower
            for k in [
                "top ",
                "split by",
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

        domain = "transactions"
        if any(
            k in lower
            for k in [
                "quote",
                "forex",
                "exchange rate",
                "amount to be paid",
                "markup",
                "charge",
                "spread",
            ]
        ):
            domain = "quotes"
        elif any(k in lower for k in ["booking", "booked", "deal type", "value date"]):
            domain = "bookings"
        elif any(k in lower for k in ["payee", "university", "address", "beneficiary"]):
            domain = "customers"
        elif "customer" in lower and "transaction" not in lower and "amount" not in lower:
            domain = "customers"

        metric = "transaction_count"
        has_count_words = self._goal_has_count_intent(lower)
        has_amount_words = self._goal_has_amount_intent(lower)
        has_mt103 = "mt103" in lower
        has_refund = "refund" in lower
        if domain == "transactions":
            if "refund rate" in lower:
                metric = "refund_rate"
            elif has_refund and has_count_words:
                metric = "refund_count"
            elif has_refund and has_amount_words:
                metric = "total_amount"
            elif has_refund:
                metric = "refund_count"
            elif "mt103" in lower and any(k in lower for k in ["rate", "coverage", "%"]):
                metric = "mt103_rate"
            elif has_mt103 and has_count_words:
                metric = "mt103_count"
            elif has_mt103 and has_amount_words:
                metric = "total_amount"
            elif has_mt103:
                metric = "mt103_count"
            elif any(k in lower for k in ["unique customer", "distinct customer"]):
                metric = "unique_customers"
            elif any(k in lower for k in ["average", "avg"]) and any(
                k in lower for k in ["amount", "revenue", "payment"]
            ):
                metric = "avg_amount"
            elif has_count_words:
                metric = "transaction_count"
            elif has_amount_words:
                metric = "total_amount"
            else:
                metric = "transaction_count"
        elif domain == "quotes":
            if "markup" in lower and any(k in lower for k in ["average", "avg", "mean"]):
                metric = "avg_forex_markup"
            elif "markup" in lower and any(k in lower for k in ["revenue", "amount", "sum", "value"]):
                metric = "forex_markup_revenue"
            elif any(k in lower for k in ["charge", "charges", "fee", "fees"]):
                metric = "total_charges"
            elif any(k in lower for k in ["average", "avg"]):
                metric = "avg_quote_value"
            elif has_count_words:
                metric = "quote_count"
            elif any(k in lower for k in ["amount", "value", "sum"]):
                metric = "total_quote_value"
            else:
                metric = "quote_count"
        elif domain == "bookings":
            if any(k in lower for k in ["rate", "fx rate", "exchange rate"]):
                metric = "avg_rate"
            elif has_count_words:
                metric = "booking_count"
            elif any(k in lower for k in ["amount", "sum", "booked"]):
                metric = "total_booked_amount"
            else:
                metric = "booking_count"
        elif domain == "customers":
            if "payee" in lower:
                metric = "payee_count"
            elif "university" in lower:
                metric = "university_count"
            else:
                metric = "customer_count"

        dim_candidates = {
            "transactions": {
                "platform": "platform_name",
                "state": "state",
                "status": "payment_status",
                "flow": "txn_flow",
                "customer": "customer_id",
                "month": "__month__",
            },
            "quotes": {
                "from currency": "from_currency",
                "to currency": "to_currency",
                "currency": "from_currency",
                "status": "status",
                "purpose": "purpose_code",
                "month": "__month__",
            },
            "customers": {
                "country": "address_country",
                "state": "address_state",
                "status": "status",
                "type": "type",
                "university": "is_university",
                "month": "__month__",
            },
            "bookings": {
                "currency": "currency",
                "status": "status",
                "deal": "deal_type",
                "linked": "linked_txn_status",
                "month": "__month__",
            },
        }

        dimensions: list[str] = []
        if re.search(r"\b(by month|monthly|month[\s-]?wise|trend)\b", lower):
            dimensions.append("__month__")

        dim_signal = any(
            t in lower for t in [" by ", "split", "breakdown", "top", "wise", "per ", "month wise"]
        )
        for key, col in dim_candidates.get(domain, {}).items():
            if key == "month":
                continue
            if key in lower and dim_signal and col not in dimensions:
                dimensions.append(col)

        if intent == "grouped_metric" and not dimensions and domain in dim_candidates:
            # Fallback grouped intent: prefer first stable dimension for readable grouping.
            default_col = next(iter(dim_candidates[domain].values()))
            if default_col != "__month__":
                dimensions.append(default_col)

        if len(dimensions) > 2:
            dimensions = dimensions[:2]
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
        domain_table = {
            "transactions": "datada_mart_transactions",
            "quotes": "datada_mart_quotes",
            "customers": "datada_dim_customers",
            "bookings": "datada_mart_bookings",
        }[domain]

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

        if domain == "transactions":
            if "without mt103" in lower or "excluding mt103" in lower:
                add_value_filter("has_mt103", "false")
            elif has_mt103 and (
                "with mt103" in lower
                or "mt103 transactions" in lower
                or "mt103 only" in lower
                or "only mt103" in lower
                or has_amount_words
            ):
                add_value_filter("has_mt103", "true")

            if "without refund" in lower or "excluding refund" in lower:
                add_value_filter("has_refund", "false")
            elif has_refund and ("with refund" in lower or "refund only" in lower or has_amount_words):
                add_value_filter("has_refund", "true")

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
        }

    def _enforce_intake_consistency(
        self,
        goal: str,
        deterministic: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(parsed)
        lower = goal.lower()

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

        if out.get("domain") == "transactions":
            if "with mt103" in lower or ("mt103" in lower and any(t in lower for t in amount_terms)):
                add_filter("has_mt103", "true")
                if any(t in lower for t in amount_terms) and not has_count_words:
                    out["metric"] = "total_amount"
            if "with refund" in lower or ("refund" in lower and any(t in lower for t in amount_terms)):
                add_filter("has_refund", "true")
                if any(t in lower for t in amount_terms) and not has_count_words:
                    out["metric"] = "total_amount"

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
        if has_explicit_subject:
            return parsed

        merged = dict(parsed)
        past_metric = str(best.get("metric") or "").strip()
        past_dims = list(best.get("dimensions") or [])
        if past_metric and merged.get("metric") == "transaction_count":
            merged["metric"] = past_metric
        if past_dims and not merged.get("dimensions"):
            merged["dimensions"] = past_dims[:2]
            merged["dimension"] = merged["dimensions"][0]
        if not merged.get("time_filter") and isinstance(best.get("time_filter"), dict):
            merged["time_filter"] = best["time_filter"]
        if not merged.get("value_filters") and isinstance(best.get("value_filters"), list):
            merged["value_filters"] = best["value_filters"][:4]
        return merged

    def _semantic_retrieval_agent(
        self,
        intake: dict[str, Any],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        domain_to_table = {
            "transactions": "datada_mart_transactions",
            "quotes": "datada_mart_quotes",
            "customers": "datada_dim_customers",
            "bookings": "datada_mart_bookings",
        }
        table = domain_to_table.get(intake["domain"], "datada_mart_transactions")
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
    ) -> dict[str, Any]:
        metric_expr = retrieval["metrics"].get(intake["metric"])
        if metric_expr is None:
            fallback_metric, metric_expr = next(iter(retrieval["metrics"].items()))
            intake["metric"] = fallback_metric

        raw_dims = intake.get("dimensions")
        if not isinstance(raw_dims, list):
            raw_dims = [intake.get("dimension")] if intake.get("dimension") else []
        dimensions: list[str] = []
        for dim in raw_dims:
            if not isinstance(dim, str):
                continue
            if dim == "__month__":
                dimensions.append(dim)
                continue
            if dim in retrieval["columns"]:
                dimensions.append(dim)
        if len(dimensions) > 2:
            dimensions = dimensions[:2]
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

        return {
            "goal": intake.get("goal", ""),
            "intent": intake["intent"],
            "table": retrieval["table"],
            "metric": intake["metric"],
            "metric_expr": metric_expr,
            "dimension": dimension,
            "dimensions": dimensions,
            "available_columns": retrieval.get("columns", []),
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
    ) -> dict[str, Any]:
        """Evaluate candidate plans and autonomously switch when evidence improves."""

        base_score = self._candidate_score(goal, base_plan, base_execution, base_audit)
        selected = {
            "plan": base_plan,
            "query_plan": base_query_plan,
            "execution": base_execution,
            "audit": base_audit,
            "score": base_score,
            "reason": "base_plan",
        }
        candidate_evals: list[dict[str, Any]] = [
            {
                "candidate": "base_plan",
                "table": base_plan.get("table"),
                "metric": base_plan.get("metric"),
                "score": round(base_score, 4),
                "row_count": int(base_execution.get("row_count") or 0),
                "goal_term_misses": list(base_audit.get("grounding", {}).get("goal_term_misses", [])),
            }
        ]
        correction_applied = False
        correction_reason = ""

        if not autonomy.auto_correction:
            return {
                **selected,
                "evaluated_candidates": candidate_evals,
                "correction_applied": False,
                "correction_reason": "",
                "probe_findings": [],
            }

        candidates = self._generate_candidate_plans(
            goal=goal,
            base_plan=base_plan,
            base_execution=base_execution,
            catalog=catalog,
            memory_hints=memory_hints or [],
            learned_corrections=learned_corrections or [],
            autonomy=autonomy,
        )

        seen = {self._plan_signature(base_plan)}
        for candidate in candidates[: max(0, autonomy.max_candidate_plans - 1)]:
            signature = self._plan_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            query_plan = self._query_engine_agent(candidate, [])
            execution = self._execution_agent(query_plan)
            audit = self._audit_agent(candidate, query_plan, execution)
            score = self._candidate_score(goal, candidate, execution, audit)
            candidate_evals.append(
                {
                    "candidate": candidate.get("_variant_reason", "variant"),
                    "table": candidate.get("table"),
                    "metric": candidate.get("metric"),
                    "score": round(score, 4),
                    "row_count": int(execution.get("row_count") or 0),
                    "goal_term_misses": list(audit.get("grounding", {}).get("goal_term_misses", [])),
                }
            )
            if score > selected["score"]:
                selected = {
                    "plan": candidate,
                    "query_plan": query_plan,
                    "execution": execution,
                    "audit": audit,
                    "score": score,
                    "reason": candidate.get("_variant_reason", "higher_score"),
                }

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
                "reason": "base_plan",
            }

        probe_findings = self._toolsmith_probe_findings(
            goal=goal,
            plan=selected["plan"],
            audit=selected["audit"],
            max_probes=max(0, autonomy.max_probe_queries),
        )
        return {
            "plan": selected["plan"],
            "query_plan": selected["query_plan"],
            "execution": selected["execution"],
            "audit": selected["audit"],
            "score": selected["score"],
            "evaluated_candidates": candidate_evals,
            "correction_applied": correction_applied,
            "correction_reason": correction_reason,
            "probe_findings": probe_findings,
        }

    def _generate_candidate_plans(
        self,
        *,
        goal: str,
        base_plan: dict[str, Any],
        base_execution: dict[str, Any],
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

        if any(k in lower for k in ["customer", "payee", "university", "beneficiary"]):
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
        if re.search(r"\b(by month|month wise|monthly|trend)\b", lower) and "__month__" not in existing_dims:
            enrich_dims.append("__month__")
        if "platform" in lower and "platform_name" not in existing_dims:
            enrich_dims.append("platform_name")
        if "state" in lower and "state" not in existing_dims:
            enrich_dims.append("state")
        if enrich_dims:
            merged_dims = (existing_dims + enrich_dims)[:2]
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
            plan["dimensions"] = valid_dims[:2]
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

    def _candidate_score(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
    ) -> float:
        score = float(audit.get("score", 0.0))
        if execution.get("success"):
            score += 0.04
        if int(execution.get("row_count") or 0) > 0:
            score += 0.04

        grounding = audit.get("grounding", {})
        misses = list(grounding.get("goal_term_misses", []))
        score -= 0.08 * len(misses)

        lower = goal.lower()
        dims = list(plan.get("dimensions") or [])
        if re.search(r"\b(by month|month wise|monthly|trend)\b", lower) and "__month__" in dims:
            score += 0.03
        if "platform" in lower and "platform_name" in dims:
            score += 0.03
        if "state" in lower and "state" in dims:
            score += 0.02
        if execution.get("execution_time_ms") and float(execution["execution_time_ms"]) > 6000:
            score -= 0.03

        return max(0.0, min(1.0, score))

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

    def _transactions_specialist(self, plan: dict[str, Any]) -> dict[str, Any]:
        if plan["table"] != "datada_mart_transactions":
            return {"active": False, "notes": []}
        notes = ["Use distinct transaction_key for counts to avoid duplicate counting."]
        if "amount" in plan["metric"]:
            notes.append("Use normalized amount column prioritizing payment_amount/deal_amount.")
        return {"active": True, "notes": notes}

    def _customer_specialist(self, plan: dict[str, Any]) -> dict[str, Any]:
        if plan["table"] not in {"datada_dim_customers", "datada_mart_transactions"}:
            return {"active": False, "notes": []}
        return {
            "active": True,
            "notes": ["Customer-level metrics should use distinct customer keys where possible."],
        }

    def _revenue_specialist(self, plan: dict[str, Any]) -> dict[str, Any]:
        if "amount" not in plan["metric"] and "quote_value" not in plan["metric"]:
            return {"active": False, "notes": []}
        return {
            "active": True,
            "notes": [
                "Revenue metrics should summarize numeric normalized columns only.",
            ],
        }

    def _risk_specialist(self, plan: dict[str, Any]) -> dict[str, Any]:
        if not any(k in plan["metric"] for k in ["refund", "mt103"]):
            return {"active": False, "notes": []}
        return {
            "active": True,
            "notes": [
                "Risk metrics should report rate and count where available.",
            ],
        }

    def _governance_precheck(self, goal: str) -> dict[str, Any]:
        lower = goal.lower()
        blocked_keywords = ["drop table", "delete from", "truncate", "update ", "insert into"]
        if any(k in lower for k in blocked_keywords):
            return {"allowed": False, "reason": "Destructive operations are blocked."}
        return {"allowed": True, "reason": "ok"}

    def _query_engine_agent(
        self,
        plan: dict[str, Any],
        specialist_findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del specialist_findings  # reserved for richer planner integration.

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
        elif intent == "lookup":
            sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT {plan['top_n']}"
        else:
            sql = f"SELECT {metric_expr} AS metric_value FROM {table} WHERE {where_clause}"

        return {"sql": sql, "table": table}

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
            safe_value = value.replace("'", "''")
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

    def _audit_agent(
        self,
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        execution: dict[str, Any],
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        goal_text = str(plan.get("goal", "")).lower()
        sql_lower = str(query_plan.get("sql", "")).lower()
        available_columns = [str(c).lower() for c in plan.get("available_columns", [])]

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

        replay_match: bool | None = None
        if execution["success"] and query_plan.get("sql"):
            replay = self.executor.execute(query_plan["sql"])
            if replay.success:
                replay_match = replay.row_count == execution["row_count"]
                if replay_match and replay.rows and execution.get("sample_rows"):
                    replay_match = replay.rows[0] == execution["sample_rows"][0]
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
        score = max(0.0, min(1.0, score))

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
                "query_columns_used": query_columns_used[:12],
                "replay_match": replay_match,
                "intent": plan.get("intent"),
                "dimensions": plan.get("dimensions", []),
                "time_filter": plan.get("time_filter"),
                "value_filters": plan.get("value_filters", []),
            },
        }

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
                    "You are NarrativeAgent in a data analytics team. "
                    "Write clear, friendly answers. Avoid jargon. "
                    "If style is storyteller, explain insights as a short story with context and next action. "
                    "Ground every statement in provided SQL result rows only; do not claim missing data unless row_count is 0. "
                    "Keep it succinct: max 90 words and no more than 3 bullets. "
                    "Return JSON only with keys: answer_markdown and suggested_questions."
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
        except Exception:
            pass

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
            cleaned["dimensions"] = dims[:2]
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
        goal: str,
        plan: dict[str, Any],
        score: float,
    ) -> None:
        start = time.perf_counter()
        try:
            correction_id = self.memory.learn_from_success(
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
