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
            payment_created_ts,
            created_ts,
            updated_ts,
            COALESCE(payment_amount_num, 0.0) AS payment_amount,
            COALESCE(deal_amount_num, 0.0) AS deal_amount,
            COALESCE(amount_collected_num, 0.0) AS amount_collected,
            COALESCE(payment_amount_num, deal_amount_num, amount_collected_num, 0.0) AS amount,
            CASE WHEN mt103_created_at_raw IS NULL THEN FALSE ELSE TRUE END AS has_mt103,
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
                    0.0::DOUBLE AS total_additional_charges
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
                {self._num(cols, 'total_additional_charges', 'total_additional_charges_num')}
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
            COALESCE(total_additional_charges_num, 0.0) AS total_additional_charges
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
                    "quote_count": "COUNT(DISTINCT quote_key)",
                    "total_quote_value": "SUM(total_amount_to_be_paid)",
                    "avg_quote_value": "AVG(total_amount_to_be_paid)",
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
        # Use a writable-mode connection for compatibility with semantic layer refresh.
        # SQL safety is still enforced by SafeSQLExecutor guardrails.
        self.executor = SafeSQLExecutor(self.db_path, read_only=False)

    def close(self) -> None:
        self.executor.close()

    def __enter__(self) -> "AgenticAnalyticsTeam":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def run(
        self,
        goal: str,
        runtime: RuntimeSelection,
        *,
        conversation_context: list[dict[str, Any]] | None = None,
        storyteller_mode: bool = False,
    ) -> AssistantQueryResponse:
        trace_id = str(uuid.uuid4())
        trace: list[dict[str, Any]] = []
        started = time.perf_counter()
        history = conversation_context or []

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
            intake = self._run_agent(
                trace,
                "IntakeAgent",
                "goal_structuring",
                self._intake_agent,
                effective_goal,
                runtime,
                mission,
                catalog,
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
                    runtime=self._runtime_payload(runtime),
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
                )
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
                ),
                agent_trace=trace,
                chart_spec=chart_spec,
                evidence_packets=evidence_packets,
                data_quality={
                    **catalog.get("quality", {}),
                    "audit_score": confidence_score,
                    "audit_warnings": audit.get("warnings", []),
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
                runtime=self._runtime_payload(runtime),
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
        followup = (
            len(lower) < 48
            or lower.startswith(("and ", "also ", "what about", "now ", "for that"))
            or any(token in lower for token in ["same", "that", "those", "it", "them"])
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
    ) -> dict[str, Any]:
        parsed = self._intake_deterministic(goal, catalog)
        parsed["_llm_intake_used"] = False
        explicit_top = bool(re.search(r"\btop\s+\d+\b", goal.lower()))
        if runtime.use_llm and runtime.provider:
            llm_parsed = self._intake_with_llm(goal, runtime, parsed)
            if llm_parsed:
                cleaned = self._sanitize_intake_payload(llm_parsed)
                if not explicit_top:
                    cleaned.pop("top_n", None)
                if cleaned:
                    parsed.update({k: v for k, v in cleaned.items() if v not in (None, "")})
                    parsed["_llm_intake_used"] = True
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
        if any(k in lower for k in ["quote", "forex", "exchange rate", "amount to be paid"]):
            domain = "quotes"
        elif any(k in lower for k in ["booking", "booked", "deal type", "value date"]):
            domain = "bookings"
        elif any(k in lower for k in ["payee", "university", "address", "beneficiary"]):
            domain = "customers"
        elif "customer" in lower and "transaction" not in lower and "amount" not in lower:
            domain = "customers"

        metric = "transaction_count"
        if domain == "transactions":
            if "refund rate" in lower:
                metric = "refund_rate"
            elif "refund" in lower:
                metric = "refund_count"
            elif "mt103" in lower and any(k in lower for k in ["rate", "coverage", "%"]):
                metric = "mt103_rate"
            elif "mt103" in lower:
                metric = "mt103_count"
            elif any(k in lower for k in ["unique customer", "distinct customer"]):
                metric = "unique_customers"
            elif any(k in lower for k in ["average", "avg"]) and any(
                k in lower for k in ["amount", "revenue", "payment"]
            ):
                metric = "avg_amount"
            elif any(k in lower for k in ["total", "sum", "revenue", "amount", "value"]):
                metric = "total_amount"
            else:
                metric = "transaction_count"
        elif domain == "quotes":
            if any(k in lower for k in ["average", "avg"]):
                metric = "avg_quote_value"
            elif any(k in lower for k in ["amount", "value", "total", "sum"]):
                metric = "total_quote_value"
            else:
                metric = "quote_count"
        elif domain == "bookings":
            if any(k in lower for k in ["rate", "fx rate", "exchange rate"]):
                metric = "avg_rate"
            elif any(k in lower for k in ["amount", "total", "sum", "booked"]):
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

        return {
            "intent": intake["intent"],
            "table": retrieval["table"],
            "metric": intake["metric"],
            "metric_expr": metric_expr,
            "dimension": dimension,
            "dimensions": dimensions,
            "time_column": retrieval.get("preferred_time_column"),
            "time_filter": intake.get("time_filter"),
            "value_filters": intake.get("value_filters", []),
            "top_n": max(1, min(100, int(intake.get("top_n", 20)))),
            "definition_used": (
                f"{intake['metric']} on {retrieval['table']}"
                + (f" grouped by {', '.join(dimensions)}" if dimensions else "")
            ),
            "row_count_hint": retrieval["row_count"],
        }

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

        if execution["execution_time_ms"] and execution["execution_time_ms"] > 4000:
            warnings.append("Execution latency above 4s.")

        score = 0.82
        if not execution["success"]:
            score = 0.18
        if execution["row_count"] == 0:
            score -= 0.20
        score -= 0.10 * len(warnings)
        score = max(0.0, min(1.0, score))

        return {
            "checks": checks,
            "warnings": warnings,
            "score": score,
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
                return {
                    "answer_markdown": self._clean_llm_answer_markdown(parsed["answer_markdown"]),
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
                return {
                    "answer_markdown": self._clean_llm_answer_markdown(raw_text),
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

    def _to_confidence(self, score: float) -> ConfidenceLevel:
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        if score >= 0.6:
            return ConfidenceLevel.MEDIUM
        if score >= 0.4:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN

    def _runtime_payload(
        self,
        runtime: RuntimeSelection,
        *,
        llm_intake_used: bool = False,
        llm_narrative_used: bool = False,
    ) -> dict[str, Any]:
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
