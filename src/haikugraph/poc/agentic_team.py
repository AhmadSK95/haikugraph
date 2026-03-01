"""Agentic analytics team runtime for dataDa.

This module implements a hierarchical, role-based multi-agent workflow that
simulates a compact analytics and data-engineering team.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import tempfile
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
from haikugraph.analytics.advanced_packs import run_advanced_packs
from haikugraph.llm.router import (
    call_llm,
    get_llm_metrics,
    reset_llm_metrics,
    summarize_llm_metrics,
)
import yaml

from haikugraph.poc.autonomy import AutonomyConfig, AgentMemoryStore
from haikugraph.contracts.analysis_contract import (
    AnalysisContract,
    ContractValidationResult,
    build_contract_from_pipeline,
    validate_sql_against_contract,
)
from haikugraph.safety.policy_gate import (
    run_all_policy_gates,
    get_blocking_verdict,
    format_refusal_response,
)
from haikugraph.scoring.calibrated_confidence import (
    compute_calibrated_confidence,
)
from haikugraph.explain.explain_yourself import (
    build_explain_yourself,
)
from haikugraph.poc.explainability_payload import (
    build_dual_level_explainability_payload,
)
from haikugraph.poc.kpi_decomposition import (
    build_kpi_decomposition as build_kpi_decomposition_payload,
)
from haikugraph.poc.contradiction_resolution import (
    resolve_candidate_contradictions,
)
from haikugraph.poc.visualization_report import (
    build_visualization_spec,
)
from haikugraph.poc.recommendations import (
    build_action_recommendations,
    render_recommendations_markdown,
)
from haikugraph.poc.root_cause import (
    build_root_cause_hypotheses,
    render_root_cause_markdown,
)
from haikugraph.poc.blackboard import (
    append_blackboard_artifact,
    blackboard_edges,
    latest_blackboard,
    query_blackboard,
)
from haikugraph.poc.decision_spine import (
    deterministic_runtime_snapshot,
    grouped_signal as grouped_signal_spine,
    metric_family,
)
from haikugraph.poc.skill_runtime import SkillRuntime
from haikugraph.sql.safe_executor import SafeSQLExecutor


MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
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
        "cross_domain_data_first": {
            "triggers": ["compare", "compared", "versus", "vs", "split by month", "month wise"],
            "action": "data_first_cross_domain",
            "policy": {
                "prefer_multi_domain_context": True,
                "prefer_time_grain": "__month__",
            },
        },
        "transaction_validity_guard": {
            "triggers": [
                "valid transaction",
                "valid transactions",
                "customer spend on transaction",
                "transaction spend",
                "transactions",
            ],
            "action": "enforce_valid_transactions_mt103",
            "filter": {"column": "has_mt103", "value": "true"},
            "apply_when": {
                "domain": "transactions",
                "metric_types": ["count", "amount", "revenue"],
            },
        },
        "narrator_data_presenter": {
            "triggers": ["insight", "insights", "analysis", "compare", "trend"],
            "action": "narrative_policy",
            "policy": {
                "mode": "data_presenter_plus_insights",
                "require_numeric_evidence": True,
            },
        },
    },
    "business_dictionary": {
        "global_column_definitions": {
            "created_ts": "Business event creation time used for period reporting and trend alignment.",
            "updated_ts": "Last update time; useful for operational freshness checks, not primary KPI slicing.",
            "status": "Lifecycle outcome used to split success/failure pipeline quality.",
            "from_currency": "Source currency the customer pays in.",
            "to_currency": "Destination currency delivered to the beneficiary.",
        },
        "tables": {
            "datada_mart_transactions": {
                "business_purpose": (
                    "Core payment execution ledger: each row represents a transfer journey with amount, "
                    "state transitions, and MT103/refund indicators."
                ),
                "analyst_questions": [
                    "How many valid transfers did we complete this month?",
                    "Which platforms drive transaction volume and amount?",
                    "Where are payment failures or refunds clustering?",
                ],
                "column_definitions": {
                    "transaction_key": "Stable transaction grain key used for deduplicated transfer counting.",
                    "transaction_id": "External/business transaction identifier seen by operations and support teams.",
                    "has_mt103": "True when SWIFT MT103 settlement proof exists; use for valid-transaction reporting.",
                    "has_refund": "True when the transaction entered refund flow; use for risk/ops quality analysis.",
                    "payment_amount": "Amount charged to the customer at payment execution.",
                    "amount": "Canonical transaction amount used for top-line transfer value reporting.",
                    "platform_name": "Channel where the transfer was initiated (app/web/offer channel).",
                    "payment_status": "Payment engine result (success/failed/cancelled) for funnel health analysis.",
                    "state": "Detailed workflow state in the transfer lifecycle.",
                    "mt103_created_ts": "Timestamp when MT103 proof was generated.",
                },
                "metric_definitions": {
                    "transaction_count": "Unique transfers in the selected slice.",
                    "valid_transaction_count": "Unique transfers with MT103 proof (business-valid transfers).",
                    "total_amount": "Total transfer value across selected rows.",
                    "customer_spend": "Customer payment outflow for selected rows.",
                    "valid_customer_spend": "Customer payment outflow restricted to MT103-valid transfers.",
                    "refund_count": "Unique refunded transfers in scope.",
                    "mt103_count": "Unique transfers carrying MT103 proof in scope.",
                },
            },
            "datada_mart_quotes": {
                "business_purpose": (
                    "Pre-execution FX pricing stream: quote requests, rates, fees, and markup economics "
                    "before final booking/payment."
                ),
                "analyst_questions": [
                    "Which currency pairs drive quote demand?",
                    "How much forex markup revenue are we generating?",
                    "How do rates and charges shift over time?",
                ],
                "column_definitions": {
                    "quote_key": "Unique quote grain key.",
                    "quote_id": "Business quote identifier.",
                    "exchange_rate": "Quoted FX conversion rate at request time.",
                    "forex_markup": "Markup component captured over base FX amount.",
                    "total_additional_charges": "All non-rate quote charges combined.",
                    "total_amount_to_be_paid": "Customer payable total including charges and markup.",
                    "amount_without_markup": "Reference amount excluding markup for margin analysis.",
                    "platform_charges": "Platform fee component in the quote.",
                    "swift_charges": "SWIFT/network charge component in the quote.",
                },
                "metric_definitions": {
                    "quote_count": "Number of quotes produced in scope.",
                    "forex_markup_revenue": "Total FX markup captured in scope.",
                    "avg_forex_markup": "Average FX markup captured per quote in scope.",
                    "avg_exchange_rate": "Average quoted rate in scope.",
                    "total_quote_value": "Total payable quote value in scope.",
                    "total_charges": "Total additional quote charges in scope.",
                },
            },
            "datada_dim_customers": {
                "business_purpose": (
                    "Customer and payee profile dimension used for segmentation, geography mix, "
                    "and institution-type analysis."
                ),
                "analyst_questions": [
                    "What is our country and customer-type mix?",
                    "How many university customers do we serve?",
                    "Which segments contribute most to activity?",
                ],
                "column_definitions": {
                    "customer_key": "Stable customer grain key for deduplicated customer analytics.",
                    "payee_key": "Stable payee grain key for deduplicated beneficiary analytics.",
                    "is_university": "True when customer belongs to university/institution segment.",
                    "type": "Customer segment classification (for portfolio mix analysis).",
                    "address_country": "Customer geography (country) for regional reporting.",
                    "address_state": "Customer geography (state/city granularity) for deeper regional drilldown.",
                },
                "metric_definitions": {
                    "customer_count": "Unique customers in scope.",
                    "payee_count": "Unique beneficiaries/payees in scope.",
                    "university_count": "University-segment customer count in scope.",
                },
            },
            "datada_mart_bookings": {
                "business_purpose": (
                    "Deal booking stream after quote acceptance: booked amount, deal type, and linked execution state."
                ),
                "analyst_questions": [
                    "Which deal types are growing?",
                    "How much value is booked per month?",
                    "How does booking quality vary by linked transaction status?",
                ],
                "column_definitions": {
                    "booking_key": "Unique booking grain key.",
                    "deal_type": "FX deal type bucket (spot/tom/cash) used for product-mix analysis.",
                    "booked_amount": "Booked value for the deal record.",
                    "linked_txn_status": "Execution status of downstream linked transaction.",
                    "rate": "Applied booking rate for conversion.",
                    "amount_on_hold": "Amount reserved/held during booking lifecycle.",
                },
                "metric_definitions": {
                    "booking_count": "Unique bookings in scope.",
                    "total_booked_amount": "Total booked value in scope.",
                    "avg_rate": "Average booking conversion rate in scope.",
                },
            },
        },
    },
    "agent_governance_rules": [
        "The more data the better - any agent identfies a domain - get the data first and make the decision later finally when finally churning out the  output",
        "transactions are valid when mt103 is tagged",
        "Narrators need to be more data presenters plus insights supported by analytics",
        "I need an extra organizational knowledge agent that keeps, maintains and grows the domain knowledge expertise and can maintain it's own views and tables for this which the agent has autonomy over and a strong integration into our agent pipeline",
    ],
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


def _rows_semantically_equal(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-6,
) -> bool:
    """Compare result rows with numeric tolerance to avoid float replay false negatives."""

    if left is None or right is None:
        return left is right
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        lv = left.get(key)
        rv = right.get(key)
        if lv is None or rv is None:
            if lv is not rv:
                return False
            continue
        lnum = _to_float(lv)
        rnum = _to_float(rv)
        if lnum is not None and rnum is not None:
            if not math.isclose(lnum, rnum, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
            continue
        if str(lv) != str(rv):
            return False
    return True


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
        ),
        base_with_row AS (
            SELECT
                *,
                ROW_NUMBER() OVER () AS _src_rownum
            FROM base
        ),
        base_ranked AS (
            SELECT
                *,
                COALESCE(transaction_id, CONCAT('row_', CAST(_src_rownum AS VARCHAR))) AS _tx_partition_key,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(transaction_id, CONCAT('row_', CAST(_src_rownum AS VARCHAR)))
                    ORDER BY COALESCE(updated_ts, created_ts, payment_created_ts, mt103_created_ts) DESC NULLS LAST, _src_rownum DESC
                ) AS _dedup_rn,
                MAX(CASE WHEN mt103_created_ts IS NULL THEN 0 ELSE 1 END) OVER (
                    PARTITION BY COALESCE(transaction_id, CONCAT('row_', CAST(_src_rownum AS VARCHAR)))
                ) AS _any_mt103,
                MAX(CASE WHEN refund_id_raw IS NULL THEN 0 ELSE 1 END) OVER (
                    PARTITION BY COALESCE(transaction_id, CONCAT('row_', CAST(_src_rownum AS VARCHAR)))
                ) AS _any_refund
            FROM base_with_row
        )
        SELECT
            b._tx_partition_key AS transaction_key,
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
            CASE WHEN b._any_mt103 = 1 THEN TRUE ELSE FALSE END AS has_mt103,
            CASE WHEN b._any_refund = 1 THEN TRUE ELSE FALSE END AS has_refund,
            c.address_country,
            c.address_state,
            c.type AS customer_type,
            c.is_university
        FROM base_ranked b
        LEFT JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY customer_key) AS _dedup_rn
            FROM datada_dim_customers
        ) c ON b.customer_id = c.customer_id AND c._dedup_rn = 1
        WHERE b._dedup_rn = 1
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
        ),
        base_with_row AS (
            SELECT
                *,
                ROW_NUMBER() OVER () AS _src_rownum
            FROM base
        ),
        base_ranked AS (
            SELECT
                *,
                COALESCE(quote_id, ref_id, CONCAT('quote_', CAST(_src_rownum AS VARCHAR))) AS _quote_partition_key,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(quote_id, ref_id, CONCAT('quote_', CAST(_src_rownum AS VARCHAR)))
                    ORDER BY COALESCE(updated_ts, created_ts) DESC NULLS LAST, _src_rownum DESC
                ) AS _dedup_rn
            FROM base_with_row
        )
        SELECT
            _quote_partition_key AS quote_key,
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
        FROM base_ranked
        WHERE _dedup_rn = 1
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
        ),
        base_with_row AS (
            SELECT
                *,
                ROW_NUMBER() OVER () AS _src_rownum
            FROM base
        ),
        base_ranked AS (
            SELECT
                *,
                COALESCE(customer_id, payee_id, CONCAT('cust_', CAST(_src_rownum AS VARCHAR))) AS _cust_partition_key,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(customer_id, payee_id, CONCAT('cust_', CAST(_src_rownum AS VARCHAR)))
                    ORDER BY COALESCE(updated_ts, created_ts) DESC NULLS LAST, _src_rownum DESC
                ) AS _dedup_rn
            FROM base_with_row
        )
        SELECT
            _cust_partition_key AS customer_key,
            COALESCE(payee_id, CONCAT('payee_', CAST(_src_rownum AS VARCHAR))) AS payee_key,
            customer_id,
            payee_id,
            is_university,
            type,
            address_country,
            address_state,
            status,
            created_ts,
            updated_ts
        FROM base_ranked
        WHERE _dedup_rn = 1
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
        ),
        base_with_row AS (
            SELECT
                *,
                ROW_NUMBER() OVER () AS _src_rownum
            FROM base
        ),
        base_ranked AS (
            SELECT
                *,
                COALESCE(deal_id, CONCAT('booking_', CAST(_src_rownum AS VARCHAR))) AS _booking_partition_key,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(deal_id, CONCAT('booking_', CAST(_src_rownum AS VARCHAR)))
                    ORDER BY COALESCE(booked_ts) DESC NULLS LAST, _src_rownum DESC
                ) AS _dedup_rn
            FROM base_with_row
        )
        SELECT
            _booking_partition_key AS booking_key,
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
        FROM base_ranked
        WHERE _dedup_rn = 1
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
                    "transaction_id_count": "COUNT(DISTINCT transaction_id)",
                    "valid_transaction_count": "COUNT(DISTINCT CASE WHEN has_mt103 THEN transaction_key END)",
                    "unique_customers": "COUNT(DISTINCT customer_id)",
                    "total_amount": "SUM(amount)",
                    "customer_spend": "SUM(payment_amount)",
                    "valid_customer_spend": "SUM(CASE WHEN has_mt103 THEN payment_amount ELSE 0 END)",
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
                    "avg_exchange_rate": "AVG(exchange_rate)",
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
                "datada_mart_transactions": "created_ts",
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
        self.skill_runtime = SkillRuntime.discover(db_path=self.db_path)
        default_memory_db = self.db_path.with_name(f"{self.db_path.stem}_agent_memory.duckdb")
        explicit_memory_db = os.environ.get("HG_MEMORY_DB_PATH")
        self.memory_db_path = Path(explicit_memory_db or str(default_memory_db)).expanduser()
        try:
            self.memory = AgentMemoryStore(self.memory_db_path)
        except Exception as exc:
            msg = str(exc)
            lock_conflict = "Could not set lock on file" in msg or "Conflicting lock" in msg
            if explicit_memory_db or not lock_conflict:
                raise
            fallback_memory_db = Path(tempfile.gettempdir()) / f"{self.db_path.stem}_agent_memory_{os.getpid()}.duckdb"
            self.memory_db_path = fallback_memory_db
            self.memory = AgentMemoryStore(self.memory_db_path)
        # Use a writable-mode connection for compatibility with semantic layer refresh.
        # SQL safety is still enforced by SafeSQLExecutor guardrails.
        self.executor = SafeSQLExecutor(self.db_path, read_only=False)
        # GAP 35: Load domain expertise knowledge base
        self._domain_knowledge = _load_domain_knowledge()
        try:
            seeded = self.memory.bootstrap_glossary_from_business_dictionary(
                business_dictionary=self._domain_knowledge.get("business_dictionary"),
                tenant_id="public",
                contributed_by="domain_knowledge_bootstrap",
            )
            self._seeded_glossary_stats = seeded
        except Exception:
            self._seeded_glossary_stats = {"inserted": 0, "updated": 0, "total": 0}

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

    def create_business_rule(
        self,
        *,
        tenant_id: str = "public",
        domain: str,
        name: str,
        rule_type: str,
        triggers: list[str],
        action_payload: dict[str, Any],
        notes: str = "",
        priority: float = 1.0,
        status: str = "draft",
        source: str = "admin_ui",
        created_by: str = "admin",
        approved_by: str = "",
    ) -> str:
        return self.memory.create_business_rule(
            tenant_id=tenant_id,
            domain=domain,
            name=name,
            rule_type=rule_type,
            triggers=triggers,
            action_payload=action_payload,
            notes=notes,
            priority=priority,
            status=status,
            source=source,
            created_by=created_by,
            approved_by=approved_by,
        )

    def list_business_rules(
        self,
        *,
        tenant_id: str = "public",
        status: str | None = None,
        domain: str | None = None,
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        return self.memory.list_business_rules(
            tenant_id=tenant_id,
            status=status,
            domain=domain,
            limit=limit,
        )

    def upsert_org_knowledge_view(
        self,
        *,
        tenant_id: str = "public",
        view_name: str,
        payload: dict[str, Any],
        goal_signature: str = "",
        source_agent: str = "OrganizationalKnowledgeAgent",
    ) -> dict[str, Any]:
        return self.memory.upsert_org_knowledge_view(
            tenant_id=tenant_id,
            view_name=view_name,
            payload=payload,
            goal_signature=goal_signature,
            source_agent=source_agent,
        )

    def list_org_knowledge_views(
        self,
        *,
        tenant_id: str = "public",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.memory.list_org_knowledge_views(tenant_id=tenant_id, limit=limit)

    def set_business_rule_status(
        self,
        rule_id: str,
        *,
        tenant_id: str = "public",
        status: str,
        actor: str = "admin",
        note: str = "",
    ) -> dict[str, Any]:
        return self.memory.set_business_rule_status(
            rule_id,
            tenant_id=tenant_id,
            status=status,
            actor=actor,
            note=note,
        )

    def update_business_rule(
        self,
        rule_id: str,
        *,
        tenant_id: str = "public",
        domain: str | None = None,
        name: str | None = None,
        rule_type: str | None = None,
        triggers: list[str] | None = None,
        action_payload: dict[str, Any] | None = None,
        notes: str | None = None,
        priority: float | None = None,
        status: str | None = None,
        actor: str = "admin",
        note: str = "",
    ) -> dict[str, Any]:
        return self.memory.update_business_rule(
            rule_id,
            tenant_id=tenant_id,
            domain=domain,
            name=name,
            rule_type=rule_type,
            triggers=triggers,
            action_payload=action_payload,
            notes=notes,
            priority=priority,
            status=status,
            actor=actor,
            note=note,
        )

    def rollback_business_rule(
        self,
        rule_id: str,
        *,
        tenant_id: str = "public",
        actor: str = "admin",
    ) -> dict[str, Any]:
        return self.memory.rollback_business_rule(
            rule_id,
            tenant_id=tenant_id,
            actor=actor,
        )

    def record_fix(
        self,
        *,
        tenant_id: str = "public",
        trace_id: str | None,
        session_id: str | None,
        goal: str | None,
        issue: str,
        keyword: str,
        domain: str,
        target_table: str,
        target_metric: str,
        target_dimensions: list[str] | None = None,
        notes: str = "",
        actor: str = "admin",
    ) -> dict[str, Any]:
        """Persist a user fix, plus an immediately-active admin rule and correction."""

        feedback = self.record_feedback(
            tenant_id=tenant_id,
            trace_id=trace_id,
            session_id=session_id,
            goal=goal,
            issue=issue,
            suggested_fix=notes or issue,
            severity="medium",
            keyword=keyword,
            target_table=target_table,
            target_metric=target_metric,
            target_dimensions=target_dimensions or [],
        )
        payload = {
            "target_table": target_table,
            "target_metric": target_metric,
            "target_dimensions": list(target_dimensions or []),
        }
        rule_id = self.create_business_rule(
            tenant_id=tenant_id,
            domain=domain or "general",
            name=f"fix:{keyword.strip().lower() or 'query'}",
            rule_type="plan_override",
            triggers=[keyword],
            action_payload=payload,
            notes=notes or issue,
            priority=2.2,
            status="active",
            source="fix_button",
            created_by=actor,
            approved_by=actor,
        )
        return {
            "feedback_id": feedback.get("feedback_id", ""),
            "correction_id": feedback.get("correction_id", ""),
            "rule_id": rule_id,
        }

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
        scenario_context: dict[str, Any] | None = None,
    ) -> AssistantQueryResponse:
        trace_id = str(uuid.uuid4())
        trace: list[dict[str, Any]] = []
        blackboard: list[dict[str, Any]] = []
        started = time.perf_counter()
        history = conversation_context or []
        autonomy_cfg = autonomy or AutonomyConfig()
        self._pipeline_warnings: list[str] = []
        self._skill_contract_warned_agents: set[str] = set()
        self._skill_contract_warning_emitted = False
        self._org_knowledge_context: dict[str, Any] = {}
        reset_llm_metrics()
        if runtime.fallback_warning:
            self._pipeline_warnings.append(runtime.fallback_warning)
        _contract_spec: dict[str, Any] = {}
        _contract_validation: dict[str, Any] = {}

        # ── HARD GATE: Run policy gates on ORIGINAL goal before any rewriting ──
        _early_verdicts = run_all_policy_gates(goal)
        _early_blocking = get_blocking_verdict(_early_verdicts)
        if _early_blocking:
            trace.append({
                "agent": "GovernanceAgent",
                "role": "early_policy_gate",
                "status": "blocked",
                "duration_ms": 0.0,
                "summary": f"Policy gate [{_early_blocking.gate}] blocked: {_early_blocking.reason}",
            })
            return AssistantQueryResponse(
                success=False,
                answer_markdown=format_refusal_response(_early_blocking),
                # Policy refusal can be high-confidence if the gate is deterministic.
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.90,
                definition_used=goal,
                evidence=[],
                sanity_checks=[
                    SanityCheck(
                        check_name="early_policy_gate",
                        passed=False,
                        message=_early_blocking.reason,
                    )
                ],
                trace_id=trace_id,
                runtime=self._runtime_payload(
                    runtime,
                    llm_intake_used=False,
                    autonomy=autonomy_cfg,
                ),
                agent_trace=trace,
                suggested_questions=["Ask about transactions, quotes, customers, or bookings instead."],
                warnings=[f"Blocked by policy gate: {_early_blocking.gate}"],
            )

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
            business_rule_matches = self._run_agent(
                trace,
                "DomainKnowledgeAgent",
                "rule_matching",
                self.memory.get_matching_business_rules,
                effective_goal,
                tenant_id=tenant_id,
                limit=8,
            )
            self._blackboard_post(
                blackboard,
                producer="DomainKnowledgeAgent",
                artifact_type="business_rules",
                payload={"count": len(business_rule_matches), "top": (business_rule_matches or [])[:3]},
                consumed_by=["PlanningAgent", "SpecialistAgents", "AutonomyAgent"],
            )
            org_knowledge = self._run_agent(
                trace,
                "OrganizationalKnowledgeAgent",
                "knowledge_curation",
                self._organizational_knowledge_agent,
                effective_goal,
                catalog,
                business_rule_matches,
                tenant_id,
            )
            self._org_knowledge_context = dict(org_knowledge or {})
            self._blackboard_post(
                blackboard,
                producer="OrganizationalKnowledgeAgent",
                artifact_type="org_knowledge",
                payload={
                    "required_domains": org_knowledge.get("required_domains", []),
                    "prefer_data_first": bool(org_knowledge.get("prefer_data_first")),
                    "enforce_mt103_validity": bool(org_knowledge.get("enforce_mt103_validity")),
                    "narrative_data_presenter": bool(org_knowledge.get("narrative_data_presenter")),
                    "knowledge_views": org_knowledge.get("knowledge_views", []),
                },
                consumed_by=[
                    "ChiefAnalystAgent",
                    "IntakeAgent",
                    "PlanningAgent",
                    "DomainKnowledgeAgent",
                    "AutonomyAgent",
                    "NarrativeAgent",
                ],
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
                history,
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
            sub_questions = self._detect_multi_part(effective_goal, runtime, catalog)
            if sub_questions and len(sub_questions) >= 2:
                    sub_results = []
                    for sq in sub_questions:
                        sr = self._run_sub_query(
                            sq,
                            runtime,
                            catalog,
                            mission,
                            memory_hints or [],
                            tenant_id=tenant_id,
                            storyteller_mode=storyteller_mode,
                        )
                        sub_results.append(sr)
                    if any(r["success"] for r in sub_results):
                        plan, query_plan, execution, audit = self._merge_multi_part_results(
                            effective_goal, sub_results, runtime,
                        )
                        _contract_spec = self._build_contract_spec(plan)
                        _contract_validation = self._validate_contract_against_sql(
                            _contract_spec, query_plan.get("sql", ""), plan,
                        )
                        # Build multi-part narrative from sub-results
                        narr_parts: list[str] = []
                        for r in sub_results:
                            prebuilt_narrative = str(r.get("narrative") or "").strip()
                            if prebuilt_narrative:
                                narr_parts.append(prebuilt_narrative)
                            else:
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
                        # ── G3: Calibrated confidence ──────────────────────────
                        try:
                            _cal = compute_calibrated_confidence(
                                contract_validation=_contract_validation,
                                audit_result=audit,
                                intake_confidence=float(intake.get("confidence", 0.5)),
                                goal_text=effective_goal,
                                sql_text=query_plan.get("sql", ""),
                                fallback_used=bool(plan.get("_fallback_used")),
                                row_count=execution.get("row_count", 0),
                            )
                            confidence_score = _cal.overall
                        except Exception as _cal_exc:
                            self._pipeline_warnings.append(
                                f"Calibrated confidence failed ({type(_cal_exc).__name__}); "
                                f"capping raw score at 0.6 as safety measure."
                            )
                            confidence_score = min(confidence_score, 0.6)
                        confidence = self._to_confidence(confidence_score)
                        multi_autonomy = {
                            "confidence_decomposition": [
                                {"factor": "multi_part_merge", "weight": 0.2, "score": confidence_score}
                            ],
                            "narrative": combined_answer,
                            "answer_summary": f"Merged {len(sub_results)} sub-queries.",
                        }
                        multi_decision_flow = self._build_decision_flow(
                            effective_goal,
                            intake,
                            _contract_spec,
                            _contract_validation,
                            plan,
                            query_plan,
                            execution,
                            audit,
                            multi_autonomy,
                        )
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
                        multi_runtime_payload = self._runtime_payload(
                            runtime,
                            llm_intake_used=True,
                            autonomy=autonomy_cfg,
                        ) | {"blackboard_entries": len(blackboard)}
                        multi_data_quality = {
                            **catalog.get("quality", {}),
                            "audit_score": confidence_score,
                            "multi_part": True,
                            "kpi_decomposition": self._build_kpi_decomposition(
                                plan=plan,
                                execution=execution,
                            ),
                            "blackboard": {"artifact_count": len(blackboard), "edges": self._blackboard_edges(blackboard)},
                        }
                        multi_explainability = build_dual_level_explainability_payload(
                            goal=effective_goal,
                            answer_markdown=combined_answer,
                            confidence_score=confidence_score,
                            intake=intake,
                            contract=_contract_spec,
                            contract_validation=_contract_validation,
                            decision_flow=multi_decision_flow,
                            trace=trace,
                            sql=query_plan.get("sql", ""),
                            runtime_payload=multi_runtime_payload,
                            data_quality=multi_data_quality,
                            warnings=list(getattr(self, "_pipeline_warnings", [])),
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
                            runtime=multi_runtime_payload,
                            agent_trace=trace,
                            chart_spec=None,
                            evidence_packets=[
                                {"agent": "MultiPartDecomposer", "sub_queries": len(sub_results)},
                                {"agent": "Blackboard", "artifact_count": len(blackboard),
                                 "artifacts": blackboard, "edges": self._blackboard_edges(blackboard)},
                            ],
                            data_quality=multi_data_quality,
                            suggested_questions=[
                                f"Tell me more about: {r['sub_goal']}" for r in sub_results[:3]
                            ],
                            warnings=self._pipeline_warnings,
                            contract_spec=_contract_spec,
                            contract_validation=_contract_validation,
                            decision_flow=multi_decision_flow,
                            explainability=multi_explainability,
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
                catalog,
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
                    # Governance refusal is deterministic and should report high confidence.
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.90,
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
                    tenant_id,
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
                            description=(
                                "Schema dictionary for all marts"
                                if schema_result.get("table") == "__all__"
                                else f"Schema for {schema_result.get('table', '')}"
                            ),
                            value=(
                                f"{schema_result.get('row_count', 0):,} rows, {len(schema_result.get('columns', []))} fields"
                            ),
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

            # ── QA-R4 Fix 5: Insight synthesis fast-path ────────────────
            if intake.get("intent") == "insight_synthesis":
                insight_result = self._run_agent(
                    trace,
                    "InsightSynthesisAgent",
                    "multi_metric_synthesis",
                    self._insight_synthesis_agent,
                    effective_goal,
                    catalog,
                    runtime,
                    storyteller_mode,
                )
                self._blackboard_post(
                    blackboard,
                    producer="InsightSynthesisAgent",
                    artifact_type="insight_synthesis",
                    payload=insight_result,
                    consumed_by=["ChiefAnalystAgent"],
                )
                total_ms = (time.perf_counter() - started) * 1000
                trace.append({
                    "agent": "ChiefAnalystAgent",
                    "role": "finalize_response",
                    "status": "success",
                    "duration_ms": round(total_ms, 2),
                    "summary": "insight synthesis response assembled",
                })
                _insight_score = insight_result.get("confidence_score", 0.72)
                self._memory_write_turn(
                    trace=trace, trace_id=trace_id, goal=goal,
                    resolved_goal=effective_goal, tenant_id=tenant_id,
                    runtime=runtime, success=True,
                    confidence_score=_insight_score,
                    row_count=insight_result.get("query_count", 0),
                    plan={"table": "", "metric": "", "dimensions": [],
                          "time_filter": None, "value_filters": []},
                    sql=insight_result.get("sql_used"),
                    audit_warnings=[],
                    correction_applied=False, correction_reason="",
                    metadata={"autonomy_mode": autonomy_cfg.mode,
                              "insight_synthesis": True},
                )
                return AssistantQueryResponse(
                    success=True,
                    answer_markdown=insight_result.get("answer_markdown", ""),
                    confidence=self._to_confidence(_insight_score),
                    confidence_score=_insight_score,
                    definition_used="multi-metric insight synthesis",
                    evidence=insight_result.get("evidence", []),
                    sanity_checks=[
                        SanityCheck(
                            check_name="insight_queries_executed",
                            passed=bool(insight_result.get("query_count", 0)),
                            message=f"{insight_result.get('query_count', 0)} analytical queries executed.",
                        )
                    ],
                    sql=insight_result.get("sql_used"),
                    row_count=insight_result.get("query_count", 0),
                    columns=[],
                    sample_rows=[],
                    execution_time_ms=total_ms,
                    trace_id=trace_id,
                    runtime=self._runtime_payload(runtime, autonomy=autonomy_cfg)
                    | {"blackboard_entries": len(blackboard)},
                    agent_trace=trace,
                    chart_spec=None,
                    evidence_packets=[
                        {"agent": "InsightSynthesisAgent",
                         "queries": insight_result.get("query_count", 0)},
                        {"agent": "Blackboard",
                         "artifact_count": len(blackboard),
                         "artifacts": blackboard,
                         "edges": self._blackboard_edges(blackboard)},
                    ],
                    data_quality={
                        **catalog.get("quality", {}),
                        "insight_synthesis": True,
                        "blackboard": {"artifact_count": len(blackboard),
                                       "edges": self._blackboard_edges(blackboard)},
                    },
                    suggested_questions=insight_result.get("suggested_questions", []),
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
            if isinstance(scenario_context, dict):
                assumptions = scenario_context.get("assumptions")
                if isinstance(assumptions, list) and assumptions:
                    clean_assumptions = [str(item).strip() for item in assumptions if str(item).strip()]
                    if clean_assumptions:
                        plan["scenario_assumptions"] = clean_assumptions[:12]
                        plan["scenario_assumption_set_id"] = str(scenario_context.get("scenario_set_id") or "")
                        plan["scenario_assumption_set_name"] = str(scenario_context.get("name") or "")
                        self._pipeline_warnings.append(
                            "Scenario assumption set applied to planning and statistical packs."
                        )
            business_rule_apply = self._run_agent(
                trace,
                "DomainKnowledgeAgent",
                "rule_application",
                self._apply_business_rule_matches,
                plan,
                business_rule_matches,
                catalog,
            )
            self._blackboard_post(
                blackboard,
                producer="DomainKnowledgeAgent",
                artifact_type="rule_application",
                payload=business_rule_apply,
                consumed_by=["SpecialistAgents", "QueryEngineerAgent", "AutonomyAgent"],
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
            query_plan, _contract_spec_pre, _contract_validation_pre = self._enforce_contract_guard(
                plan=plan,
                query_plan=query_plan,
                specialist_findings=[tx_findings, customer_findings, revenue_findings, risk_findings],
                runtime=runtime,
                catalog=catalog,
                trace=trace,
                stage="pre_execution",
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
                    "objective_coverage": autonomy_result.get("objective_coverage", {}),
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
            _sql_before_guard = str(query_plan.get("sql") or "")
            query_plan, _contract_spec, _contract_validation = self._enforce_contract_guard(
                plan=plan,
                query_plan=query_plan,
                specialist_findings=all_specialist_findings,
                runtime=runtime,
                catalog=catalog,
                trace=trace,
                stage="post_autonomy",
            )
            if str(query_plan.get("sql") or "") != _sql_before_guard:
                execution = self._run_agent(
                    trace,
                    "ExecutionAgent",
                    "contract_guard_execution",
                    self._execution_agent,
                    query_plan,
                )
                audit = self._run_agent(
                    trace,
                    "AuditAgent",
                    "contract_guard_audit",
                    self._audit_agent,
                    plan,
                    query_plan,
                    execution,
                    runtime,
                )
                self._pipeline_warnings.extend(audit.get("warnings", []))

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
            recommendation_items = build_action_recommendations(
                goal=effective_goal,
                plan=plan,
                execution=execution,
                audit=audit,
                advanced_packs=None,
            )
            recommendation_md = render_recommendations_markdown(recommendation_items)
            if recommendation_md:
                answer_text = str(narration.get("answer_markdown") or "").strip()
                if "**recommended actions**" not in answer_text.lower():
                    narration["answer_markdown"] = f"{answer_text}\n\n{recommendation_md}".strip()
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
            # ── G3: Calibrated confidence ──────────────────────────
            try:
                _cal = compute_calibrated_confidence(
                    contract_validation=_contract_validation,
                    audit_result=audit,
                    intake_confidence=float(intake.get("confidence", 0.5)),
                    goal_text=effective_goal,
                    sql_text=query_plan.get("sql", ""),
                    fallback_used=bool(plan.get("_fallback_used")),
                    row_count=execution.get("row_count", 0),
                )
                confidence_score = _cal.overall
            except Exception as _cal_exc:
                self._pipeline_warnings.append(
                    f"Calibrated confidence failed ({type(_cal_exc).__name__}); "
                    f"capping raw score at 0.6 as safety measure."
                )
                confidence_score = min(confidence_score, 0.6)
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
            objective_cov = autonomy_result.get("objective_coverage", {}) or {}
            objective_cov_pct = float(objective_cov.get("coverage_pct") or 0.0)
            objective_failures = list(objective_cov.get("failed_objectives") or [])
            # BRD-FR7: Rich confidence decomposition
            _conf_factors: list[str] = []
            _conf_factors.append(f"base_score=0.82")
            if not execution["success"]:
                _conf_factors.append("execution_failed=-0.64")
            elif execution["row_count"] == 0:
                _conf_factors.append("zero_rows=-0.25")
            if audit_warnings:
                _conf_factors.append(f"audit_warnings({len(audit_warnings)})=-{0.10*len(audit_warnings):.2f}")
            if misses:
                _conf_factors.append(f"concept_misses={misses}")
            _pw_all = getattr(self, "_pipeline_warnings", [])
            if any("Metric unrecognized" in w for w in _pw_all):
                _conf_factors.append("unrecognized_metric=-0.30,cap=0.49")
            _tf_check = plan.get("time_filter")
            if isinstance(_tf_check, dict) and _tf_check.get("kind") == "future_blocked":
                _conf_factors.append(f"future_blocked({_tf_check.get('year')}),cap=0.25")
            confidence_reasoning = (
                f"Score {confidence_score:.2f} ({confidence.value}). "
                f"Concept coverage {coverage_pct}%. "
                f"Objective coverage {objective_cov_pct:.1f}%. "
                f"Factors: [{', '.join(_conf_factors)}]"
                + (f". {len(audit_warnings)} audit warning(s)" if audit_warnings else "")
                + (f". Objective misses: {objective_failures}" if objective_failures else "")
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
                    "canonical_metric_id": plan.get("canonical_metric_id"),
                    "canonical_metric": plan.get("canonical_metric", {}),
                    "secondary_canonical_metric_id": plan.get("secondary_canonical_metric_id"),
                    "secondary_canonical_metric": plan.get("secondary_canonical_metric", {}),
                },
                {
                    "agent": "QueryEngineerAgent",
                    "table": plan["table"],
                    "metric": plan["metric"],
                    "sql": query_plan["sql"],
                    "row_count": execution["row_count"],
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "canonical_metric_id": plan.get("canonical_metric_id"),
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
                    "objective_coverage": autonomy_result.get("objective_coverage", {}),
                    "evaluated_candidates": autonomy_result.get("evaluated_candidates", []),
                    "confidence_decomposition": autonomy_result.get("confidence_decomposition", []),
                    "contradiction_resolution": autonomy_result.get("contradiction_resolution", {}),
                    "hard_gates": autonomy_result.get("hard_gates", {}),
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
                {
                    "agent": "DomainKnowledgeAgent",
                    "matched_rules": plan.get("_business_rules_matched", []),
                    "applied_rules": plan.get("_business_rules_applied", []),
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
                    "scenario_assumption_set_id": str(plan.get("scenario_assumption_set_id") or ""),
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
            advanced_packs_payload: dict[str, Any] | None = None
            root_cause_payload: dict[str, Any] = {}
            rows_for_stats = execution.get("analysis_rows") or execution.get("sample_rows") or []
            if execution["success"] and rows_for_stats:
                try:
                    import pandas as _pd

                    stats_df = _pd.DataFrame(rows_for_stats)
                    if not stats_df.empty:
                        stats_result = run_stats_analysis(stats_df)
                        stats_dict = stats_result.to_dict()
                        forecast_enabled = str(
                            os.environ.get("HG_ADVANCED_FORECAST_ENABLED", "false")
                        ).strip().lower() in {"1", "true", "yes", "on"}
                        advanced_pack_options: dict[str, Any] = {
                            "outlier_z_threshold": float(
                                os.environ.get("HG_OUTLIER_Z_THRESHOLD", "1.5")
                            ),
                            "outlier_alert_pct_threshold": float(
                                os.environ.get("HG_OUTLIER_ALERT_PCT_THRESHOLD", "5.0")
                            ),
                            "variance_baseline_mode": str(
                                os.environ.get("HG_VARIANCE_BASELINE_MODE", "first")
                            ),
                            "forecast_horizon_steps": int(
                                os.environ.get("HG_FORECAST_HORIZON_STEPS", "3")
                            ),
                        }
                        scenario_assumptions = plan.get("scenario_assumptions")
                        if isinstance(scenario_assumptions, list) and scenario_assumptions:
                            advanced_pack_options["scenario_assumptions"] = scenario_assumptions
                        advanced_packs = run_advanced_packs(
                            stats_df,
                            goal_text=effective_goal,
                            forecast_enabled=forecast_enabled,
                            options=advanced_pack_options,
                        )
                        if isinstance(advanced_packs, dict):
                            advanced_packs_payload = dict(advanced_packs)
                            stats_dict["advanced_packs"] = advanced_packs
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
            if advanced_packs_payload:
                enriched_recs = build_action_recommendations(
                    goal=effective_goal,
                    plan=plan,
                    execution=execution,
                    audit=audit,
                    advanced_packs=advanced_packs_payload,
                )
                if enriched_recs:
                    merged_recs: list[dict[str, Any]] = []
                    seen_actions = set()
                    for rec in list(recommendation_items) + list(enriched_recs):
                        action_key = str(rec.get("action") or "").strip().lower()
                        if not action_key or action_key in seen_actions:
                            continue
                        seen_actions.add(action_key)
                        merged_recs.append(rec)
                    recommendation_items = merged_recs[:3]
                    recommendation_md = render_recommendations_markdown(recommendation_items)
                    if recommendation_md:
                        answer_text = str(narration.get("answer_markdown") or "").strip()
                        if "**recommended actions**" not in answer_text.lower():
                            narration["answer_markdown"] = (
                                f"{answer_text}\n\n{recommendation_md}".strip()
                            )
            root_cause_payload = build_root_cause_hypotheses(
                goal=effective_goal,
                plan=plan,
                execution=execution,
                audit=audit,
            )
            root_cause_md = render_root_cause_markdown(root_cause_payload)
            if root_cause_md:
                answer_text = str(narration.get("answer_markdown") or "").strip()
                if "**root-cause hypotheses**" not in answer_text.lower():
                    narration["answer_markdown"] = f"{answer_text}\n\n{root_cause_md}".strip()

            # ── BRD WP-1/WP-4: Build decision flow from contract + execution ──
            _decision_flow = self._build_decision_flow(
                effective_goal, intake, _contract_spec, _contract_validation,
                plan, query_plan, execution, audit, autonomy_result,
            )
            answer_markdown = self._prepend_governance_disclosure(
                str(narration.get("answer_markdown") or ""),
                plan,
            )
            contradiction_resolution = autonomy_result.get("contradiction_resolution", {}) or {}
            if bool(contradiction_resolution.get("needs_clarification")):
                clarification_prompt = str(contradiction_resolution.get("clarification_prompt") or "").strip()
                if clarification_prompt:
                    answer_markdown = (
                        f"{answer_markdown.rstrip()}\n\n"
                        f"**Clarification needed:** {clarification_prompt}"
                    )
                    self._pipeline_warnings.append(
                        "Detected near-tie contradictory candidate interpretations; clarification requested."
                    )
            raw_pipeline_warnings = [
                str(w).strip()
                for w in list(dict.fromkeys(getattr(self, "_pipeline_warnings", [])))
                if str(w).strip()
            ]
            final_warnings = self._finalize_user_warnings(raw_pipeline_warnings)
            response_runtime_payload = self._runtime_payload(
                runtime,
                llm_intake_used=bool(intake.get("_llm_intake_used", False)),
                llm_narrative_used=bool(narration.get("llm_narrative_used", False)),
                autonomy=autonomy_cfg,
                correction_applied=bool(autonomy_result.get("correction_applied", False)),
            ) | {
                "blackboard_entries": len(blackboard),
                "warning_hygiene": {
                    "user_visible_count": len(final_warnings),
                    "internal_count": len(raw_pipeline_warnings),
                },
            }
            response_data_quality = {
                **catalog.get("quality", {}),
                "audit_score": confidence_score,
                "audit_warnings": audit.get("warnings", []),
                "grounding": audit.get("grounding", {}),
                "autonomy": {
                    "correction_applied": bool(autonomy_result.get("correction_applied", False)),
                    "correction_reason": autonomy_result.get("correction_reason"),
                    "objective_coverage": autonomy_result.get("objective_coverage", {}),
                    "evaluated_candidates": autonomy_result.get("evaluated_candidates", []),
                    "confidence_decomposition": autonomy_result.get("confidence_decomposition", []),
                    "contradiction_resolution": contradiction_resolution,
                    "hard_gates": autonomy_result.get("hard_gates", {}),
                    "refinement_rounds": autonomy_result.get("refinement_rounds", []),
                    "toolsmith_candidates": autonomy_result.get("toolsmith_candidates", []),
                },
                "business_rules": {
                    "matched": plan.get("_business_rules_matched", []),
                    "applied": plan.get("_business_rules_applied", []),
                    "governance_applied": plan.get("_governance_applied", []),
                    "governance_notes": plan.get("_governance_notes", []),
                },
                "kpi_decomposition": self._build_kpi_decomposition(
                    plan=plan,
                    execution=execution,
                ),
                "canonical_metric_contract": {
                    "metric_id": plan.get("canonical_metric_id"),
                    "secondary_metric_id": plan.get("secondary_canonical_metric_id"),
                },
                "scenario": {
                    "assumption_set_id": plan.get("scenario_assumption_set_id"),
                    "assumption_set_name": plan.get("scenario_assumption_set_name"),
                    "assumption_count": len(plan.get("scenario_assumptions") or []),
                },
                "recommendations": recommendation_items,
                "root_cause": root_cause_payload,
                "blackboard": {"artifact_count": len(blackboard), "edges": self._blackboard_edges(blackboard)},
            }
            explainability_payload = build_dual_level_explainability_payload(
                goal=effective_goal,
                answer_markdown=answer_markdown,
                confidence_score=confidence_score,
                intake=intake,
                contract=_contract_spec,
                contract_validation=_contract_validation,
                decision_flow=_decision_flow,
                trace=trace,
                sql=query_plan.get("sql", ""),
                runtime_payload=response_runtime_payload,
                data_quality=response_data_quality,
                contribution_map=contribution_map,
                warnings=final_warnings,
            )

            return AssistantQueryResponse(
                success=execution["success"],
                answer_markdown=answer_markdown,
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
                runtime=response_runtime_payload,
                agent_trace=trace,
                chart_spec=chart_spec,
                contribution_map=contribution_map,
                confidence_reasoning=confidence_reasoning,
                evidence_packets=evidence_packets,
                data_quality=response_data_quality,
                stats_analysis=stats_dict,
                suggested_questions=narration.get("suggested_questions", []),
                warnings=final_warnings,
                contract_spec=_contract_spec,
                contract_validation=_contract_validation,
                decision_flow=_decision_flow,
                explainability=explainability_payload,
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

    def _finalize_user_warnings(self, warnings: list[str], *, max_items: int = 1) -> list[str]:
        """Reduce warning noise to actionable user-facing items."""
        if not warnings:
            return []

        internal_prefixes = (
            "[SkillContract]",
            "[Contract]",
            "[ContractGuard:",
            "Handoff contract warning",
        )
        internal_phrases = (
            "dynamically resolved to column",
            "specialists will apply count(distinct)",
            "details available in explainability trace",
        )

        out: list[str] = []
        seen: set[str] = set()
        for raw in warnings:
            message = str(raw or "").strip()
            if not message:
                continue
            message_l = message.lower()
            if message.startswith(internal_prefixes):
                continue
            if any(phrase in message_l for phrase in internal_phrases):
                continue
            key = message_l
            if key in seen:
                continue
            seen.add(key)
            out.append(message)
            if len(out) >= max(1, int(max_items)):
                break
        return out

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
        elif agent == "DomainKnowledgeAgent":
            if isinstance(out, list):
                contribution = f"matched_rules={len(out)}"
                if out:
                    reasoning = f"top_rule={out[0].get('name', '')}, hits={out[0].get('trigger_hits', [])}"
                else:
                    reasoning = "no matching business rules"
            elif isinstance(out, dict):
                contribution = f"matched={out.get('matched', 0)}, applied={out.get('applied', 0)}"
                reasoning = "applied admin business rules to planning stage" if out.get("applied", 0) else "no business rules applied"
            else:
                contribution = "business_rules=none"
                reasoning = "no business rules applied"
        elif agent == "OrganizationalKnowledgeAgent":
            required_domains = out.get("required_domains", []) if isinstance(out, dict) else []
            view_count = len(out.get("knowledge_views", [])) if isinstance(out, dict) else 0
            enforce_mt103 = bool(out.get("enforce_mt103_validity")) if isinstance(out, dict) else False
            data_first = bool(out.get("prefer_data_first")) if isinstance(out, dict) else False
            contribution = (
                f"required_domains={required_domains}; views={view_count}; "
                f"data_first={data_first}; mt103_validity={enforce_mt103}"
            )
            reasoning = (
                "curated organization rulebook + domain overview; "
                "published planning controls and validity policies"
            )
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
        skill_selection = self.skill_runtime.resolve(agent)
        try:
            out = fn(*args, **kwargs)
            skill_eval = self.skill_runtime.evaluate_contract(agent, out)
            if skill_eval.enabled and not skill_eval.passed:
                self._skill_contract_warned_agents.add(agent)
                if not getattr(self, "_skill_contract_warning_emitted", False):
                    self._pipeline_warnings.append(
                        "[SkillContract] One or more agent outputs missed internal contract checks "
                        "(details available in explainability trace)."
                    )
                    self._skill_contract_warning_emitted = True
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
                    "selected_skills": skill_selection.selected_skills,
                    "skill_policy_reason": skill_selection.skill_policy_reason,
                    "skill_contract_file": skill_selection.skill_contract_file,
                    "skill_layer_file": skill_selection.skill_layer_file,
                    "skill_contract_enforced": skill_eval.enabled,
                    "skill_contract_passed": skill_eval.passed,
                    "skill_contract_checks": skill_eval.checks,
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
                    "selected_skills": skill_selection.selected_skills,
                    "skill_policy_reason": skill_selection.skill_policy_reason,
                    "skill_contract_file": skill_selection.skill_contract_file,
                    "skill_layer_file": skill_selection.skill_layer_file,
                    "skill_contract_enforced": False,
                    "skill_contract_passed": False,
                    "skill_contract_checks": [],
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

        # Walk backward through context to find the last user goal
        last_goal = ""
        for turn in reversed(conversation_context):
            _g = turn.get("goal") or turn.get("user_goal") or ""
            if not _g and turn.get("role") == "user":
                _g = turn.get("content", "")
            _g = str(_g).strip()
            if _g:
                last_goal = _g
                break
        if not last_goal:
            return goal

        preserve_followup_surface = any(
            phrase in lower
            for phrase in [
                "switch metric",
                "same slice",
                "same scope",
                "keep that scope",
                "keep same",
                "add total amount",
                "add amount too",
                "same grouped output",
                "same grouped",
                "include booked amount",
                "keep top",
                "top currencies",
                "top platforms",
            ]
        )

        if runtime.use_llm and runtime.provider and runtime.provider != "ollama" and not preserve_followup_surface:
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

        # Deterministic fallback should preserve follow-up context without
        # polluting metric/domain intent extraction with raw prompt stuffing.
        if has_explicit_subject and not has_reference_pronoun and not has_reference_phrase:
            return goal
        return f"{last_goal}. Follow-up: {goal}"

    def _organizational_knowledge_agent(
        self,
        goal: str,
        catalog: dict[str, Any],
        business_rule_matches: list[dict[str, Any]],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Curate organization-level knowledge views and governance context."""
        lower = goal.lower()
        governance_rules = list(self._domain_knowledge.get("agent_governance_rules") or [])

        domain_keywords = {
            "transactions": ["transaction", "transactions", "payment", "transfer", "mt103", "refund", "spend"],
            "quotes": ["quote", "quotes", "forex", "fx", "exchange", "markup", "charge"],
            "customers": ["customer", "customers", "client", "clients", "payee", "beneficiary"],
            "bookings": ["booking", "bookings", "deal", "value date"],
        }
        for term, target in self._domain_knowledge.get("synonyms", {}).items():
            if target in domain_keywords and term not in domain_keywords[target]:
                domain_keywords[target].append(term)

        required_domains: list[str] = []
        for domain, tokens in domain_keywords.items():
            if any(tok in lower for tok in tokens):
                required_domains.append(domain)
        if not required_domains:
            required_domains = ["transactions"]

        prefer_data_first = any("more data the better" in str(r).lower() for r in governance_rules)
        enforce_mt103_validity = any("mt103 is tagged" in str(r).lower() for r in governance_rules)
        narrative_data_presenter = any(
            "narrators need to be more data presenters" in str(r).lower()
            for r in governance_rules
        )

        metrics_by_table = catalog.get("metrics_by_table", {})
        domain_profiles: list[dict[str, Any]] = []
        for domain in required_domains:
            table = DOMAIN_TO_MART.get(domain, "")
            mart = (catalog.get("marts") or {}).get(table, {})
            domain_profiles.append(
                {
                    "domain": domain,
                    "table": table,
                    "row_count": int(mart.get("row_count") or 0),
                    "column_count": len(mart.get("columns") or []),
                    "metric_count": len((metrics_by_table.get(table) or {}).keys()),
                    "sample_metrics": list((metrics_by_table.get(table) or {}).keys())[:6],
                }
            )

        goal_signature = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:16]
        view_ops: list[dict[str, Any]] = []
        view_ops.append(
            self.upsert_org_knowledge_view(
                tenant_id=tenant_id,
                view_name="domain_overview",
                payload={"domains": domain_profiles, "required_domains": required_domains},
                goal_signature=goal_signature,
            )
        )
        view_ops.append(
            self.upsert_org_knowledge_view(
                tenant_id=tenant_id,
                view_name="governance_rulebook",
                payload={
                    "agent_governance_rules": governance_rules,
                    "matched_business_rules": [
                        {
                            "rule_id": str(r.get("rule_id") or ""),
                            "name": str(r.get("name") or ""),
                            "rule_type": str(r.get("rule_type") or ""),
                        }
                        for r in (business_rule_matches or [])
                    ],
                },
                goal_signature=goal_signature,
            )
        )
        view_ops.append(
            self.upsert_org_knowledge_view(
                tenant_id=tenant_id,
                view_name="goal_context",
                payload={
                    "goal": goal,
                    "required_domains": required_domains,
                    "prefer_data_first": prefer_data_first,
                    "enforce_mt103_validity": enforce_mt103_validity,
                    "narrative_data_presenter": narrative_data_presenter,
                },
                goal_signature=goal_signature,
            )
        )

        return {
            "goal_signature": goal_signature,
            "required_domains": required_domains,
            "prefer_data_first": prefer_data_first,
            "enforce_mt103_validity": enforce_mt103_validity,
            "narrative_data_presenter": narrative_data_presenter,
            "domain_profiles": domain_profiles,
            "agent_governance_rules": governance_rules,
            "knowledge_views": view_ops,
        }

    def _apply_organizational_controls(
        self,
        goal: str,
        intake: dict[str, Any],
        plan: dict[str, Any],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply organization-level governance directives to the current plan."""
        ctx = getattr(self, "_org_knowledge_context", {}) or {}
        if not isinstance(ctx, dict) or not ctx:
            return plan

        lower = goal.lower()
        adjusted = dict(plan)
        required_domains = list(ctx.get("required_domains") or [])
        prefer_data_first = bool(ctx.get("prefer_data_first"))
        enforce_mt103_validity = bool(ctx.get("enforce_mt103_validity"))
        governance_applied = list(adjusted.get("_governance_applied") or [])
        governance_notes = list(adjusted.get("_governance_notes") or [])

        if prefer_data_first and required_domains:
            primary_domain = intake.get("domain")
            secondary_domains = list(adjusted.get("secondary_domains") or [])
            added_domains: list[str] = []
            for d in required_domains:
                if d != primary_domain and d not in secondary_domains:
                    secondary_domains.append(d)
                    added_domains.append(d)
            adjusted["secondary_domains"] = secondary_domains
            if added_domains:
                governance_applied.append("prefer_data_first:add_secondary_domains")
                governance_notes.append(
                    f"Added secondary domains for richer context: {', '.join(added_domains)}."
                )
            if any(k in lower for k in ["split", "by month", "month wise", "monthly", "trend"]) and "__month__" not in adjusted.get("dimensions", []):
                adjusted.setdefault("dimensions", []).append("__month__")
                adjusted["dimension"] = adjusted["dimensions"][0]
                governance_applied.append("prefer_data_first:add_month_grain")
                governance_notes.append(
                    "Added month grain for data-first comparison/trend coverage."
                )

        is_markup_vs_spend = (
            "markup" in lower
            and "spend" in lower
            and "transaction" in lower
            and "quotes" in [intake.get("domain")] + list(adjusted.get("secondary_domains") or [])
            and "transactions" in [intake.get("domain")] + list(adjusted.get("secondary_domains") or [])
        )

        tx_in_scope = adjusted.get("table") == "datada_mart_transactions" or is_markup_vs_spend
        if enforce_mt103_validity and tx_in_scope:
            spend_on_txn_variant = bool(
                re.search(
                    r"\b(spend|amount|revenue|value)\b.*\btransaction(s)?\b|\btransaction(s)?\b.*\b(spend|amount|revenue|value)\b",
                    lower,
                )
            )
            goal_mentions_validity = any(
                phrase in lower
                for phrase in ["valid transaction", "valid transactions"]
            )
            goal_mentions_validity = goal_mentions_validity or "mt103" in lower or spend_on_txn_variant
            if goal_mentions_validity or is_markup_vs_spend:
                existing = {
                    str(vf.get("column") or "").strip().lower(): str(vf.get("value") or "").strip().lower()
                    for vf in (adjusted.get("value_filters") or [])
                    if isinstance(vf, dict)
                }
                if existing.get("has_mt103") not in {"true", "1", "yes"}:
                    adjusted.setdefault("value_filters", []).append(
                        {"column": "has_mt103", "operator": "=", "value": "true"}
                    )
                    self._pipeline_warnings.append(
                        "Applied governance rule: transactions validity requires has_mt103=true."
                    )
                    governance_applied.append("transaction_validity_guard:has_mt103=true")
                    governance_notes.append(
                        "Applied domain validity rule: transactions are valid only when has_mt103=true."
                    )

        if is_markup_vs_spend:
            adjusted["intent"] = "cross_domain_compare"
            adjusted["compare_table"] = "datada_mart_transactions"
            adjusted["compare_metric"] = "valid_customer_spend" if enforce_mt103_validity else "customer_spend"
            tx_metrics = (catalog.get("metrics_by_table") or {}).get("datada_mart_transactions", {})
            adjusted["compare_metric_expr"] = str(
                tx_metrics.get(adjusted["compare_metric"]) or "SUM(payment_amount)"
            )
            if "__month__" not in adjusted.get("dimensions", []):
                adjusted.setdefault("dimensions", []).insert(0, "__month__")
                adjusted["dimension"] = adjusted["dimensions"][0]
            adjusted["definition_used"] = (
                "forex_markup_revenue vs "
                f"{adjusted['compare_metric']} by month"
            )
            governance_applied.append("cross_domain_compare:markup_vs_spend")
            governance_notes.append(
                "Built cross-domain compare: forex markup (quotes) vs customer spend (transactions)."
            )

        # Canonical MT103 metric binding when MT103 intent is explicit.
        if adjusted.get("table") == "datada_mart_transactions" and "mt103" in lower:
            tx_metrics = (catalog.get("metrics_by_table") or {}).get("datada_mart_transactions", {})
            metric = str(adjusted.get("metric") or "")
            if metric in {"transaction_count", "valid_transaction_count"} and "mt103_count" in tx_metrics:
                adjusted["metric"] = "mt103_count"
                adjusted["metric_expr"] = str(tx_metrics["mt103_count"])
                governance_applied.append("mt103_metric_binding:mt103_count")
                governance_notes.append(
                    "Bound MT103 query to canonical metric `mt103_count`."
                )
            preferred_time = (
                (catalog.get("preferred_time_column_by_metric") or {})
                .get("datada_mart_transactions", {})
                .get(str(adjusted.get("metric") or ""))
            )
            if preferred_time and adjusted.get("time_column") != preferred_time:
                adjusted["time_column"] = preferred_time
                governance_applied.append(f"mt103_time_binding:{preferred_time}")
                governance_notes.append(
                    f"Bound MT103 time scope to canonical column `{preferred_time}`."
                )

        if governance_applied:
            adjusted["_governance_applied"] = list(dict.fromkeys(governance_applied))
        if governance_notes:
            adjusted["_governance_notes"] = list(dict.fromkeys(governance_notes))

        return adjusted

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

    # ── QA-R4 Fix 5: Insight synthesis agent ────────────────────────
    def _insight_synthesis_agent(
        self,
        goal: str,
        catalog: dict[str, Any],
        runtime: RuntimeSelection,
        storyteller_mode: bool = False,
    ) -> dict[str, Any]:
        """Run multiple pre-defined analytical queries across available marts
        and synthesize numbered business insights with evidence."""
        marts = catalog.get("marts", {})
        metrics_by_table = catalog.get("metrics_by_table", {})

        # Define a set of analytical probes across all available marts.
        probes: list[dict[str, str]] = []
        for mart_name, mart_meta in marts.items():
            metrics = metrics_by_table.get(mart_name, {})
            cols = mart_meta.get("columns", [])
            time_col = None
            for c in cols:
                if c.endswith("_ts") or c == "event_ts" or c == "created_at":
                    time_col = c
                    break
            # Total count probe
            first_metric = next(iter(metrics), None)
            if first_metric:
                metric_expr = metrics[first_metric]
                probes.append({
                    "label": f"Total {first_metric} in {mart_name}",
                    "sql": f"SELECT {metric_expr} AS metric_value FROM {mart_name} WHERE 1=1",
                    "mart": mart_name,
                    "metric": first_metric,
                })
                # Time-based trend probe (last 3 months)
                if time_col:
                    probes.append({
                        "label": f"{first_metric} trend (last 3 months) in {mart_name}",
                        "sql": (
                            f"SELECT DATE_TRUNC('month', {time_col}) AS month_bucket, "
                            f"{metric_expr} AS metric_value "
                            f"FROM {mart_name} "
                            f"WHERE {time_col} >= CURRENT_DATE - INTERVAL '90 days' "
                            f"AND {time_col} IS NOT NULL "
                            f"GROUP BY 1 ORDER BY 1"
                        ),
                        "mart": mart_name,
                        "metric": first_metric,
                    })
            # Top dimension probe (first non-key string-like column)
            skip_cols = {"customer_key", "transaction_key", "quote_key", "booking_key"}
            dim_col = None
            for c in cols:
                if c not in skip_cols and not c.endswith("_ts") and not c.endswith("_key") and not c.endswith("_id"):
                    dim_col = c
                    break
            if dim_col and first_metric:
                probes.append({
                    "label": f"{first_metric} by {dim_col} in {mart_name}",
                    "sql": (
                        f"SELECT {dim_col}, {metrics[first_metric]} AS metric_value "
                        f"FROM {mart_name} WHERE 1=1 "
                        f"GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
                    ),
                    "mart": mart_name,
                    "metric": first_metric,
                })

        # Execute probes and collect results
        results: list[dict[str, Any]] = []
        evidence: list[EvidenceItem] = []
        sqls: list[str] = []
        for probe in probes[:12]:  # Cap at 12 probes for performance
            res = self.executor.execute(probe["sql"])
            if res.success and res.row_count > 0:
                results.append({
                    "label": probe["label"],
                    "rows": res.rows[:10],
                    "columns": res.columns,
                    "row_count": res.row_count,
                    "mart": probe["mart"],
                    "metric": probe["metric"],
                })
                evidence.append(EvidenceItem(
                    description=probe["label"],
                    value=str(res.rows[0] if res.rows else "N/A"),
                    source=probe["mart"],
                    sql_reference=probe["sql"],
                ))
                sqls.append(probe["sql"])

        # Synthesize insights from collected data
        lines: list[str] = ["## Business Insights\n"]
        insight_num = 0

        # Extract key findings
        for r in results:
            if insight_num >= 5:
                break
            if r["row_count"] == 1 and len(r["columns"]) <= 2:
                # Scalar metric
                val = r["rows"][0] if r["rows"] else {}
                metric_val = val.get("metric_value", val.get(r["columns"][-1], "N/A")) if isinstance(val, dict) else val
                insight_num += 1
                lines.append(
                    f"**{insight_num}. {r['label']}:** {_fmt_number(metric_val) if isinstance(metric_val, (int, float)) else metric_val}"
                )
            elif r["row_count"] > 1:
                # Multi-row: summarize top entries
                insight_num += 1
                top_entries = []
                for row in r["rows"][:3]:
                    if isinstance(row, dict):
                        parts = [f"{k}={v}" for k, v in row.items() if k != "metric_value"]
                        val = row.get("metric_value", "")
                        top_entries.append(f"{', '.join(parts)} → {_fmt_number(val) if isinstance(val, (int, float)) else val}")
                    else:
                        top_entries.append(str(row))
                lines.append(f"**{insight_num}. {r['label']}:**")
                for entry in top_entries:
                    lines.append(f"   - {entry}")

        if insight_num == 0:
            lines.append("No significant data patterns found across available marts.")

        lines.append("\n---\n*Insights synthesized from multiple analytical queries across available data marts.*")

        # If LLM is available, let it refine the narrative
        answer_md = "\n".join(lines)
        if runtime and runtime.use_llm and runtime.provider and results:
            try:
                data_summary = "\n".join(
                    f"- {r['label']}: {r['rows'][:3]}" for r in results[:8]
                )
                messages = [
                    {"role": "system", "content": (
                        "You are a business analyst synthesizing data into actionable insights. "
                        "Given the following analytical results, produce 3-5 numbered insights "
                        "with specific numbers and actionable recommendations. "
                        "Format as markdown with bold insight titles."
                    )},
                    {"role": "user", "content": (
                        f"Original question: {goal}\n\n"
                        f"Analytical results:\n{data_summary}\n\n"
                        "Provide concise, data-backed insights."
                    )},
                ]
                llm_response = call_llm(
                    messages,
                    role="narrative",
                    provider=runtime.provider,
                    model=getattr(runtime, "narrative_model", None) or getattr(runtime, "intent_model", None),
                    timeout=20,
                )
                if llm_response:
                    answer_md = llm_response
            except Exception:
                pass  # Fall back to deterministic insights

        suggested = [
            "What are the key trends over the last quarter?",
            "Which segment is growing fastest?",
            "Show me a breakdown by platform",
        ]

        return {
            "answer_markdown": answer_md,
            "query_count": len(results),
            "evidence": evidence,
            "sql_used": "; ".join(sqls[:3]) if sqls else None,
            "confidence_score": min(0.78, 0.50 + 0.07 * len(results)),
            "suggested_questions": suggested,
        }

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
    def _intent_marker_profile(
        self,
        text: str,
        catalog: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        lower = str(text or "").lower()
        token_set = set(re.findall(r"[a-z0-9_]+", lower))
        safe_catalog = catalog if isinstance(catalog, dict) else {}

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
        document_tokens = [
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
        metric_tokens = [
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
            "compare",
            "versus",
            " vs ",
            "trend",
            "over time",
            "split",
            "breakdown",
            "group by",
            "top ",
            "rank",
        ]

        metric_hit = False
        for tok in metric_tokens:
            if " " in tok:
                if tok in lower:
                    metric_hit = True
                    break
            elif tok in token_set:
                metric_hit = True
                break

        return {
            "data_overview": any(re.search(p, lower) for p in data_overview_patterns),
            "schema_exploration": self._detect_schema_exploration(lower, safe_catalog) is not None,
            "document_qa": any(tok in lower for tok in document_tokens),
            "metric_analytics": metric_hit
            or self._goal_has_count_intent(lower)
            or self._goal_has_amount_intent(lower),
        }

    def _has_intent_collision(self, profile: dict[str, bool]) -> bool:
        active = [name for name, enabled in (profile or {}).items() if enabled]
        return len(active) >= 2

    def _detect_multi_part(
        self,
        goal: str,
        runtime: RuntimeSelection,
        catalog: dict[str, Any] | None = None,
    ) -> list[str] | None:
        """Detect and decompose multi-part questions into sub-questions.

        Uses a deterministic pre-check first, then LLM decomposition if
        multi-part signals are found.  Returns None for single-part questions.
        """
        lower_goal = goal.lower()

        def _deterministic_split(text: str) -> list[str]:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if not cleaned:
                return []

            # Numbered tasks: "(1) ... (2) ..." / "1) ... 2) ..."
            numbered = re.split(r"\(\s*\d+\s*\)|\b\d+\)\s*", cleaned)
            numbered = [c.strip(" .,!;:-") for c in numbered if c.strip(" .,!;:-")]
            if numbered and len(numbered) >= 3 and any(
                k in numbered[0].lower() for k in ["both tasks", "both sections", "one response", "do both"]
            ):
                numbered = numbered[1:]
            if len(numbered) >= 2:
                return numbered[:4]

            # Protect common single-query grouped phrasing.
            if (
                "?" not in cleaned
                and ";" not in cleaned
                and "\n" not in text
                and re.search(r"\bby\s+\w+(\s+\w+)?\s+and\s+\w+(\s+\w+)?\b", cleaned.lower())
                and not re.search(r"\b(?:also|then|next|separately)\b", cleaned.lower())
            ):
                return []

            parts: list[str] = []
            if cleaned.count("?") >= 2:
                for chunk in re.split(r"\?\s*", cleaned):
                    c = chunk.strip(" .,!;")
                    if c:
                        parts.append(c + "?")
                return parts[:4]

            for chunk in re.split(r"[;\n]+", cleaned):
                c = chunk.strip(" .,!;")
                if c:
                    parts.append(c)
            if len(parts) >= 2:
                return parts[:4]

            split_markers = r"\b(?:also|then|next|in addition)\b"
            if re.search(split_markers, cleaned.lower()):
                parts = [c.strip(" .,!;") for c in re.split(split_markers, cleaned, flags=re.IGNORECASE) if c.strip(" .,!;")]
                if len(parts) >= 2:
                    return parts[:4]
            return []

        # Keep coupled count+amount asks in one plan instead of splitting.
        if (
            self._goal_has_count_intent(lower_goal)
            and self._goal_has_amount_intent(lower_goal)
            and any(
                k in lower_goal
                for k in [
                    "both tasks",
                    "both sections",
                    "two asks together",
                    "one response",
                    "same platform",
                    "same grouped",
                    "together",
                ]
            )
        ):
            return None

        # Keep follow-up metric switches in a single scoped plan.
        add_amount_followup = bool(
            re.search(r"\badd\b.*\b(amount|value|revenue)\b.*\b(too|also)?\b", lower_goal)
            or "switch metric" in lower_goal
            or "add total amount" in lower_goal
            or "add amount too" in lower_goal
        )
        if add_amount_followup:
            if any(k in lower_goal for k in ["same slice", "same scope", "keep that", "keep same"]):
                return None

        det_parts = _deterministic_split(goal)
        if len(det_parts) >= 2:
            return det_parts

        intent_profile = self._intent_marker_profile(goal, catalog)
        explicit_sequence = bool(
            re.search(r"[?\n;]|(?:\bthen\b|\balso\b|\bnext\b|\bin addition\b)", lower_goal)
        )
        mixed_metric_non_metric = bool(
            intent_profile.get("metric_analytics")
            and (
                intent_profile.get("data_overview")
                or intent_profile.get("schema_exploration")
                or intent_profile.get("document_qa")
            )
        )

        # Mixed analytic + non-analytic asks in one sentence require either
        # explicit sequencing or clarification (handled by ClarificationAgent).
        if mixed_metric_non_metric and not explicit_sequence:
            return None

        # For non-metric mixed intents (for example overview + schema), allow a
        # deterministic split on conjunctions when parts carry distinct intents.
        if self._has_intent_collision(intent_profile) and not mixed_metric_non_metric:
            candidate_parts = [
                c.strip(" .,!;:-")
                for c in re.split(r"\b(?:and then|then|also|and)\b", goal, flags=re.IGNORECASE)
                if c and c.strip(" .,!;:-")
            ]
            if len(candidate_parts) >= 2:
                labels: set[str] = set()
                for part in candidate_parts:
                    part_profile = self._intent_marker_profile(part, catalog)
                    if part_profile.get("schema_exploration"):
                        labels.add("schema")
                    elif part_profile.get("data_overview"):
                        labels.add("overview")
                    elif part_profile.get("document_qa"):
                        labels.add("documents")
                    elif part_profile.get("metric_analytics"):
                        labels.add("metric")
                if len(labels) >= 2:
                    return candidate_parts[:4]

        # Deterministic pre-check: skip LLM if clearly single-part
        multi_signals = [
            " and what", " and how", " and which", " and show",
            ", what", ", how", ", which", "? also", "? and ",
        ]
        if not any(s in lower_goal for s in multi_signals) or len(goal) < 40:
            return None
        if not (runtime.use_llm and runtime.provider):
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
        tenant_id: str = "public",
        storyteller_mode: bool = False,
    ) -> dict[str, Any]:
        """Run a single sub-question through the intake→plan→query→exec→audit pipeline."""
        try:
            intake = self._intake_agent(sub_goal, runtime, mission, catalog, memory_hints)
            sub_intent = str(intake.get("intent") or "")

            if sub_intent == "data_overview":
                discovery_plan = self._data_overview_plan_agent(sub_goal, catalog)
                overview_profile = self._data_overview_profile_agent(discovery_plan, catalog)
                overview = self._data_overview_agent(
                    sub_goal,
                    catalog,
                    storyteller_mode,
                    runtime,
                    overview_profile,
                )
                sample_rows = list(overview.get("sample_rows") or [])[:25]
                columns = list(overview.get("columns") or ["mart", "row_count"])
                return {
                    "sub_goal": sub_goal,
                    "plan": {
                        "goal": sub_goal,
                        "intent": "data_overview",
                        "table": "semantic_catalog",
                        "metric": "catalog_overview",
                        "metric_expr": "",
                        "dimensions": [],
                        "definition_used": "semantic catalog overview",
                        "top_n": 20,
                        "value_filters": [],
                        "time_filter": None,
                    },
                    "query_plan": {"sql": "-- data_overview_profile", "table": "semantic_catalog"},
                    "execution": {
                        "success": True,
                        "error": None,
                        "row_count": len(sample_rows),
                        "columns": columns,
                        "sample_rows": sample_rows,
                        "execution_time_ms": 0.0,
                        "sql_executed": "-- data_overview_profile",
                        "warnings": [],
                    },
                    "audit": {
                        "score": 0.95,
                        "warnings": [],
                        "checks": [{"name": "overview_ready", "passed": True, "message": "Overview generated."}],
                        "grounding": {"intent": "data_overview", "table": "semantic_catalog"},
                    },
                    "success": True,
                    "narrative": str(overview.get("answer_markdown") or ""),
                }

            if sub_intent == "schema_exploration":
                schema_domain = str(intake.get("domain") or "__all__")
                schema_result = self._schema_exploration_agent(
                    sub_goal,
                    schema_domain,
                    catalog,
                    tenant_id,
                )
                return {
                    "sub_goal": sub_goal,
                    "plan": {
                        "goal": sub_goal,
                        "intent": "schema_exploration",
                        "table": str(schema_result.get("table") or "__all__"),
                        "metric": "schema_dictionary",
                        "metric_expr": "",
                        "dimensions": [],
                        "definition_used": f"schema exploration for {schema_domain}",
                        "top_n": 20,
                        "value_filters": [],
                        "time_filter": None,
                    },
                    "query_plan": {"sql": "-- schema_dictionary", "table": str(schema_result.get("table") or "__all__")},
                    "execution": {
                        "success": True,
                        "error": None,
                        "row_count": int(schema_result.get("row_count") or 0),
                        "columns": list(schema_result.get("columns") or []),
                        "sample_rows": [],
                        "execution_time_ms": 0.0,
                        "sql_executed": "-- schema_dictionary",
                        "warnings": [],
                    },
                    "audit": {
                        "score": 0.92,
                        "warnings": [],
                        "checks": [{"name": "schema_ready", "passed": True, "message": "Schema dictionary generated."}],
                        "grounding": {"intent": "schema_exploration", "table": str(schema_result.get("table") or "__all__")},
                    },
                    "success": True,
                    "narrative": str(schema_result.get("answer_markdown") or ""),
                }

            retrieval = self._semantic_retrieval_agent(intake, catalog)
            plan = self._planning_agent(intake, retrieval, catalog, runtime)
            query_plan = self._query_engine_agent(plan, [], runtime=runtime, catalog=catalog)
            query_plan, _sub_contract, _sub_validation = self._enforce_contract_guard(
                plan=plan,
                query_plan=query_plan,
                specialist_findings=[],
                runtime=runtime,
                catalog=catalog,
                stage="sub_query_pre_execution",
            )
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
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        parsed = self._intake_deterministic(goal, catalog)
        parsed["_llm_intake_used"] = False
        deterministic = dict(parsed)
        explicit_top = bool(re.search(r"\btop\s+\d+\b", goal.lower()))
        if runtime.use_llm and runtime.provider and runtime.provider != "ollama":
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
        parsed = self._apply_conversation_scope_hints(
            goal,
            parsed,
            conversation_context or [],
            catalog,
        )
        if memory_hints:
            parsed = self._apply_memory_hints(goal, parsed, memory_hints)
        # Only treat hard metric mapping failures as clarification blockers.
        # Soft concept mismatch warnings should degrade confidence, not stop execution.
        parsed["_metric_unrecognized"] = any(
            "Metric unrecognized: no catalog metric matched" in str(w)
            for w in getattr(self, "_pipeline_warnings", [])
        )
        parsed = self._bind_canonical_metric_contracts(parsed, catalog)
        return parsed

    def _clarification_agent(
        self,
        goal: str,
        intake: dict[str, Any],
        conversation_context: list[dict[str, Any]],
        memory_hints: list[dict[str, Any]] | None = None,
        catalog: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect ambiguous goals and request clarification before execution."""

        lower = goal.lower().strip()
        tokens = re.findall(r"[a-z0-9_]+", lower)
        has_grouped_signal = grouped_signal_spine(lower) or any(
            marker in lower for marker in [" by ", "trend"]
        )
        denominator_per_metric = bool(
            re.search(r"\bper\s+(transaction(?:s)?|quote(?:s)?|booking(?:s)?|order(?:s)?)\b", lower)
        )
        if denominator_per_metric and any(k in lower for k in ["avg", "average", "mean"]):
            has_grouped_signal = False
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
            "client",
            "clients",
            "order",
            "product",
            "region",
            "country",
            "countries",
            "refund",
            "mt103",
            "markup",
            "forex",
            "charge",
            "payee",
            "university",
            "universities",
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
            "unique",
            "distinct",
        ]
        has_domain_term = any(t in lower for t in domain_terms)
        has_metric_term = any(t in lower for t in metric_terms)
        explicit_time = bool(intake.get("time_filter"))
        intent = str(intake.get("intent") or "")
        metric = str(intake.get("metric") or "")
        intent_profile = self._intent_marker_profile(goal, catalog)
        explicit_sequence = bool(
            re.search(r"[?\n;]|(?:\bthen\b|\balso\b|\bnext\b|\bin addition\b)", lower)
        )
        mixed_metric_non_metric = bool(
            intent_profile.get("metric_analytics")
            and (
                intent_profile.get("data_overview")
                or intent_profile.get("schema_exploration")
                or intent_profile.get("document_qa")
            )
        )

        reasons: list[str] = []
        if intent in {"data_overview", "document_qa", "schema_exploration"} and not mixed_metric_non_metric:
            return {
                "needs_clarification": False,
                "reason": "",
                "questions": [],
                "suggested_questions": [],
                "ambiguity_score": 0.0,
            }

        # Follow-ups WITH conversation history have context-enriched goals —
        # skip brief-goal checks since ContextAgent already resolved them.
        _has_history = bool(conversation_context)
        if followup_signal and not _has_history and not (has_domain_term or has_metric_term):
            reasons.append("follow_up_without_history")
        if not _has_history:
            if (
                has_grouped_signal
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
        if has_grouped_signal and not list(intake.get("dimensions") or []):
            reasons.append("grouping_without_dimension_resolution")
        if bool(intake.get("_metric_unrecognized")):
            analytic_request = bool(
                has_metric_term
                or has_grouped_signal
                or str(intake.get("intent") or "") in {"grouped_metric", "comparison", "trend_analysis"}
            )
            if analytic_request:
                reasons.append("metric_not_mapped_to_catalog")
            else:
                self._pipeline_warnings.append(
                    "Metric was not mapped to catalog for open-ended discovery request; proceeding without clarification."
                )
        if self._goal_has_count_intent(lower) and self._goal_has_amount_intent(lower):
            if not str(intake.get("secondary_metric") or "").strip():
                reasons.append("dual_metric_unresolved")
        if mixed_metric_non_metric and not explicit_sequence:
            reasons.append("intent_collision_unresolved")
        has_compare_intent = any(term in lower for term in ["compare", "versus", "vs"])
        has_markup_signal = any(term in lower for term in ["markup", "forex markup", "fx charge", "forex charge"])
        has_spend_signal = any(
            term in lower for term in ["spend", "spent", "customer spend", "transaction spend", "spend on transaction"]
        )
        detected_domains = {
            str(intake.get("domain") or "").strip(),
            *[str(d).strip() for d in (intake.get("secondary_domains") or [])],
            *[str(d).strip() for d in (intake.get("domains_detected") or [])],
        }
        cross_domain_ready = {"quotes", "transactions"}.issubset({d for d in detected_domains if d})
        if (
            has_compare_intent
            and has_markup_signal
            and has_spend_signal
            and str(intake.get("intent") or "") != "cross_domain_compare"
            and not cross_domain_ready
        ):
            reasons.append("cross_domain_compare_not_resolved")

        # Canonical grounding clash guard:
        # For amount-like transaction questions with explicit month+year and
        # no time-column qualifier, ask a follow-up so the user chooses which
        # timeline to use (created_ts vs event_ts).
        time_filter = intake.get("time_filter") if isinstance(intake.get("time_filter"), dict) else {}
        time_kind = str((time_filter or {}).get("kind") or "")
        explicit_time_column = any(
            k in lower
            for k in [
                "created",
                "creation",
                "event",
                "value date",
                "booked date",
                "mt103 created",
                "posted date",
                "occurred",
            ]
        )
        has_month_token = any(m in lower for m in MONTH_MAP.keys())
        has_year_token = bool(re.search(r"\b20\d{2}\b", lower))
        has_abs_month_year = (time_kind == "month_year") and has_month_token and has_year_token
        has_trend_intent = any(term in lower for term in ["trend", "over time", "month over month", "month-on-month"])
        has_risk_filter_signal = any(term in lower for term in ["mt103", "refund"])
        has_transaction_amount_phrase = bool(re.search(r"\btransaction\s+amount\b", lower))
        has_aggregate_transaction_amount_phrase = bool(
            re.search(r"\bamount\s+of\s+transactions?\b", lower)
            or re.search(r"\bpayment\s+amount\s+for\s+transactions?\b", lower)
        )
        requires_explicit_time_column_choice = bool(
            has_transaction_amount_phrase and not has_aggregate_transaction_amount_phrase
        )
        amount_like_metric = str(intake.get("metric") or "") in {
            "total_amount",
            "avg_amount",
            "customer_spend",
            "valid_customer_spend",
            "forex_markup_revenue",
            "total_quote_value",
            "total_booked_amount",
        }
        domain = str(intake.get("domain") or "")
        if (
            has_abs_month_year
            and domain == "transactions"
            and amount_like_metric
            and str(intake.get("intent") or "") in {"metric", "comparison", "grouped_metric", "trend_analysis"}
            and (has_compare_intent or has_trend_intent or requires_explicit_time_column_choice)
            and not has_risk_filter_signal
            and not explicit_time_column
            and str(intake.get("intent") or "") != "cross_domain_compare"
        ):
            tx_table = DOMAIN_TO_MART.get("transactions", "datada_mart_transactions")
            tx_cols = set(
                ((catalog or {}).get("marts", {}).get(tx_table, {}) or {}).get("columns", [])
            )
            if {"created_ts", "event_ts"}.issubset(tx_cols):
                reasons.append("time_column_ambiguous_transactions")
        # Detect when value filters reference terms not in the schema
        vf_terms = [str(vf.get("value", "")).lower() for vf in (intake.get("value_filters") or [])]
        if not vf_terms and has_domain_term:
            # Check if goal mentions filtering concepts that couldn't be resolved
            filter_hints = ["only", "just", "specific", "particular", "where"]
            active_filter_hints = [h for h in filter_hints if h in lower]
            # "only" around temporal refs is time scoping (e.g. "Dec-2025 only")
            if "only" in active_filter_hints:
                if re.search(
                    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|20\d{2}|this month|last month|this year|last year)\b(?:\s*[-/]\s*\d{2,4})?\s+only\b",
                    lower,
                ) or re.search(
                    r"\bonly\s+(?:for\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|20\d{2}|this month|last month|this year|last year)\b",
                    lower,
                ):
                    active_filter_hints = [h for h in active_filter_hints if h != "only"]

            # Grouped prompts with clear domain should proceed; specialists/audit will refine.
            if active_filter_hints and has_grouped_signal and has_domain_term:
                active_filter_hints = []

            if active_filter_hints and not any(t in lower for t in metric_terms):
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

        # Follow-up time narrowing prompts should inherit prior scope/metric.
        # Example: "Just December" after "Total transaction amount by month".
        if _has_history and "metric_not_mapped_to_catalog" in reasons:
            is_time_narrow_followup = bool(
                re.search(
                    r"\b(just|only|for)\s+"
                    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
                    r"january|february|march|april|june|july|august|september|october|november|december|"
                    r"this month|last month|this year|last year)\b",
                    lower,
                )
            ) or lower in {"just december", "only december", "this month", "last month"}
            if followup_signal or is_time_narrow_followup:
                reasons.remove("metric_not_mapped_to_catalog")

        needs_clarification = bool(reasons)
        questions: list[str] = []
        if needs_clarification:
            if not has_domain_term:
                questions.append(
                    "Which domain should I analyze: transactions, quotes, customers, bookings, or documents?"
                )
            if has_grouped_signal:
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
            if "metric_not_mapped_to_catalog" in reasons:
                questions.append(
                    "Which exact metric do you want (count, total amount, average, rate, or revenue)?"
                )
            if "dual_metric_unresolved" in reasons:
                questions.append(
                    "You asked for both count and amount. Which amount metric should I pair with the count?"
                )
            if "intent_collision_unresolved" in reasons:
                questions.append(
                    "Your request mixes data-discovery and analytics in one step. Should I do discovery first, then run metrics?"
                )
            if "grouping_without_dimension_resolution" in reasons:
                questions.append(
                    "What should I group by (month, platform, country, state, or customer type)?"
                )
            if "cross_domain_compare_not_resolved" in reasons:
                questions.append(
                    "Do you want a month-by-month comparison of forex markup (quotes) vs customer spend (transactions)?"
                )
            if "time_column_ambiguous_transactions" in reasons:
                questions.append(
                    "For the month/year filter, should I use transaction `created_ts` or `event_ts`?"
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
                    "'correlation', 'cross_domain_compare', 'data_overview', 'schema_exploration'\n"
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
            # Boolean dimensions (for example has_mt103, is_university) are
            # valid split/group dimensions for analyst-style asks.
            if col.startswith("has_") or col.startswith("is_"):
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

        # Domain-specific aliases with strong intent mapping.
        if domain == "quotes":
            if "to_currency" in all_columns:
                mapping["destination currency"] = "to_currency"
                mapping["destination currencies"] = "to_currency"
                mapping["dest currency"] = "to_currency"
                mapping["to currency"] = "to_currency"
            if "from_currency" in all_columns:
                mapping["source currency"] = "from_currency"
                mapping["from currency"] = "from_currency"
            mapping["currency pair"] = "currency_pair"
        elif domain == "customers":
            if "is_university" in all_columns:
                mapping["university"] = "is_university"
                mapping["universities"] = "is_university"
                mapping["non-university"] = "is_university"
                mapping["non-universities"] = "is_university"
            if "address_country" in all_columns:
                mapping["country"] = "address_country"
                mapping["countries"] = "address_country"
                mapping["united kingdom"] = "address_country"
            mapping["client"] = "customer_id"
            mapping["clients"] = "customer_id"

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
            "transaction id": ["transaction_id_count"],
            "transaction ids": ["transaction_id_count"],
            "refund": ["refund_count", "refund_rate"],
            "mt103": ["mt103_count", "mt103_rate"],
            "markup": ["forex_markup_revenue", "avg_forex_markup"],
            "spend": ["customer_spend", "valid_customer_spend", "total_amount"],
            "spent": ["customer_spend", "valid_customer_spend", "total_amount"],
            "payee": ["payee_count"],
            "university": ["university_count"],
            "universities": ["university_count"],
            "client": ["customer_count", "active_customer_count"],
            "clients": ["customer_count", "active_customer_count"],
            "charge": ["total_charges"],
            "fee": ["total_charges"],
        }

        has_count_words = self._goal_has_count_intent(lower)
        has_amount_words = self._goal_has_amount_intent(lower)
        has_avg = any(k in lower for k in ["avg", "average", "mean"])

        # Strong domain-specific semantic mappings (avoid score ties/fallback drift).
        if domain == "quotes":
            if any(k in lower for k in ["exchange rate", "fx rate", "forex rate"]):
                if "avg_exchange_rate" in metrics:
                    return "avg_exchange_rate"
            if any(k in lower for k in ["average forex markup", "avg forex markup", "average markup", "avg markup"]):
                if "avg_forex_markup" in metrics:
                    return "avg_forex_markup"
            if has_amount_words and any(
                k in lower for k in ["currency exchange", "exchange", "quote amount", "quote value"]
            ):
                if "total_quote_value" in metrics:
                    return "total_quote_value"
            if any(k in lower for k in ["fx charge", "fx charges", "forex charge", "forex charges"]):
                if any(
                    k in lower
                    for k in [
                        "additional charge",
                        "additional charges",
                        "total charges",
                        "service charge",
                        "service charges",
                        "platform charge",
                        "platform charges",
                        "swift charge",
                        "swift charges",
                    ]
                ):
                    if "total_charges" in metrics:
                        return "total_charges"
                if "forex_markup_revenue" in metrics:
                    return "forex_markup_revenue"
        if domain == "transactions":
            if has_count_words and any(
                k in lower for k in ["transaction id", "transaction ids", "distinct transaction id", "distinct transaction ids"]
            ):
                if "transaction_id_count" in metrics:
                    return "transaction_id_count"
            if any(k in lower for k in ["customer spend", "spend on transaction", "transaction spend", "spent"]):
                if "customer_spend" in metrics:
                    return "customer_spend"

        metric_from_registry = self._resolve_metric_from_registry_alias(lower, domain, catalog)
        if metric_from_registry:
            return metric_from_registry

        # Business semantic dictionary override: map natural-language metric
        # descriptions to canonical metric IDs when lexical scoring is weak.
        table_dict = self._table_business_dictionary(mart)
        metric_defs = table_dict.get("metric_definitions") if isinstance(table_dict, dict) else {}
        if isinstance(metric_defs, dict):
            goal_tokens = {t for t in re.findall(r"[a-z0-9_]+", lower) if len(t) >= 3}
            semantic_best = ""
            semantic_score = 0.0
            for metric_name in metrics.keys():
                definition = str(metric_defs.get(metric_name) or "").lower()
                if not definition:
                    continue
                def_tokens = {t for t in re.findall(r"[a-z0-9_]+", definition) if len(t) >= 3}
                if not def_tokens:
                    continue
                overlap = len(goal_tokens & def_tokens) / max(1, len(def_tokens))
                if overlap > semantic_score:
                    semantic_score = overlap
                    semantic_best = str(metric_name)
            if semantic_best and semantic_score >= 0.45:
                return semantic_best

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
            if has_count_words and not has_amount_words:
                if metric_name.endswith("_count"):
                    score += 5
                elif any(t in metric_name for t in ["total_", "avg_", "revenue", "amount", "value", "charges"]):
                    score -= 2
            if has_amount_words and any(
                t in metric_name for t in ["amount", "value", "revenue", "charges", "booked"]
            ):
                score += 1
            if has_avg and metric_name.startswith("avg_"):
                score += 2
            if has_avg and "exchange rate" in lower and metric_name == "avg_exchange_rate":
                score += 4

            if score > best_score:
                best_score = score
                best_metric = metric_name

        # ── QA-R4 Fix 3: Flag unrecognized metrics ──────────────────────
        if best_score == 0:
            self._pipeline_warnings.append(
                f"Metric unrecognized: no catalog metric matched the goal "
                f"(domain='{domain}'). Falling back to default '{best_metric}'. "
                "The answer may not reflect what was asked."
            )
        else:
            # ── QA-R5 Fix 4: Detect unmatched concept tokens ─────────────
            # If the best metric matched on a common token (e.g. "customer")
            # but goal contains concept words absent from ANY catalog metric,
            # flag it — the selected metric may not reflect the user's intent.
            _stopwords = {
                "the", "a", "an", "of", "for", "in", "on", "to", "and", "or",
                "is", "are", "was", "were", "be", "been", "with", "at", "from",
                "by", "how", "what", "which", "that", "this", "it", "do", "does",
                "i", "we", "my", "our", "me", "us", "can", "will", "would",
                "should", "could", "has", "have", "had", "all", "no", "not",
                "get", "give", "show", "list", "return", "find", "display",
                "there", "many", "much", "some", "any", "available", "about",
                "also", "just", "only", "per", "each", "every", "more", "most",
                "than", "these", "those", "its", "their", "them", "up", "down",
                "plausible", "please", "tell", "compute", "calculate",
                "insight", "insights",
            }
            _metric_kw = {
                "count", "total", "sum", "average", "avg", "mean", "rate",
                "revenue", "amount", "value", "volume", "number", "top",
                "bottom", "rank", "unique", "distinct",
                "spend",
            }
            _time_kw = {
                "month", "year", "quarter", "week", "day", "weekly", "monthly",
                "yearly", "quarterly", "daily", "trend", "over", "time",
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                "sep", "sept", "oct", "nov", "dec",
                "january", "february", "march", "april", "june", "july",
                "august", "september", "october", "november", "december",
            }
            _domain_kw = {
                "transaction", "transactions", "quote", "quotes",
                "booking", "bookings", "customer", "customers",
                "document", "documents", "pdf",
                "refund", "refunds", "refunded", "mt103",
                "forex", "markup", "payee", "payees", "university",
            }
            # Known dimension/column terms that are not alien concepts
            _dim_kw = {
                "platform", "country", "state", "status", "type", "flow",
                "currency", "name", "source", "target", "deal", "payment",
                "destination", "dest",
                "split", "breakdown", "grouped", "group", "aggregated",
                "comparison", "compare", "compared", "delta", "difference", "versus",
                "previous", "context", "question", "was",
            }
            _noise = _stopwords | _metric_kw | _time_kw | _domain_kw | _dim_kw
            goal_tokens = set(re.findall(r"[a-z]+", lower))
            goal_tokens -= _noise
            # Remove tokens that appear in the selected metric name
            metric_tokens = set(re.split(r"[_\s]+", best_metric.lower()))
            goal_tokens -= metric_tokens
            # Also remove any single-char tokens and numeric-like tokens
            goal_tokens = {t for t in goal_tokens if len(t) > 1}
            # Strong concept tokens: longer words (>3 chars) that aren't filler
            _strong = {t for t in goal_tokens if len(t) > 3}
            if len(goal_tokens) >= 2 or len(_strong) >= 1:
                self._pipeline_warnings.append(
                    f"Metric unrecognized: goal concepts {goal_tokens} not found in "
                    f"selected metric '{best_metric}'. The answer may not reflect "
                    "what was asked."
                )

        return best_metric

    def _business_dictionary_view(self) -> dict[str, Any]:
        data = self._domain_knowledge.get("business_dictionary")
        return data if isinstance(data, dict) else {}

    def _table_business_dictionary(self, table: str) -> dict[str, Any]:
        if not table:
            return {}
        tables = self._business_dictionary_view().get("tables")
        if not isinstance(tables, dict):
            return {}
        item = tables.get(table)
        return item if isinstance(item, dict) else {}

    def _infer_metric_unit(self, metric_name: str, metric_expr: str) -> str:
        low_name = str(metric_name or "").lower()
        low_expr = str(metric_expr or "").lower()
        if low_name.endswith("_count") or "count(" in low_expr:
            return "count"
        if low_name.endswith("_rate") or "avg(" in low_expr or low_name.startswith("avg_"):
            return "ratio"
        if any(tok in low_name for tok in ("amount", "value", "revenue", "spend", "charges", "markup", "rate")):
            return "amount"
        if "sum(" in low_expr:
            return "amount"
        return "number"

    def _canonical_metric_aliases(
        self,
        *,
        domain: str,
        metric_name: str,
        business_definition: str,
    ) -> list[str]:
        aliases: list[str] = [metric_name, metric_name.replace("_", " ")]
        low_metric = str(metric_name or "").lower()
        if domain == "transactions":
            domain_aliases = {
                "transaction_count": ["how many transactions", "number of transactions", "transfer count"],
                "transaction_id_count": ["distinct transaction ids", "unique transaction ids"],
                "valid_transaction_count": ["valid transactions", "mt103 valid transactions"],
                "total_amount": ["transaction amount", "total transaction amount", "payment amount"],
                "customer_spend": ["customer spend", "transaction spend", "amount spent"],
                "valid_customer_spend": ["valid customer spend", "mt103 spend"],
                "refund_count": ["refunded transactions", "refund volume"],
                "mt103_count": ["mt103 transactions", "mt103 volume"],
                "mt103_rate": ["mt103 rate", "mt103 coverage"],
            }
            aliases.extend(domain_aliases.get(low_metric, []))
        elif domain == "quotes":
            domain_aliases = {
                "quote_count": ["how many quotes", "number of quotes", "quote volume"],
                "total_quote_value": ["quote amount", "total quote amount", "quote value"],
                "avg_quote_value": ["average quote value"],
                "avg_exchange_rate": ["exchange rate", "fx rate", "forex rate"],
                "forex_markup_revenue": ["fx charges", "forex charges", "markup revenue"],
                "avg_forex_markup": ["average markup", "avg markup"],
                "total_charges": ["total charges", "additional charges", "service charges"],
            }
            aliases.extend(domain_aliases.get(low_metric, []))
        elif domain == "customers":
            domain_aliases = {
                "customer_count": ["how many customers", "customer volume"],
                "payee_count": ["how many payees", "payee volume"],
                "university_count": ["university customers", "university count"],
            }
            aliases.extend(domain_aliases.get(low_metric, []))
        elif domain == "bookings":
            domain_aliases = {
                "booking_count": ["how many bookings", "booking volume"],
                "total_booked_amount": ["booked amount", "total booked amount"],
                "avg_rate": ["average booking rate", "booking conversion rate"],
            }
            aliases.extend(domain_aliases.get(low_metric, []))

        out: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            a = str(alias or "").strip().lower()
            if len(a) < 4:
                continue
            if a in seen:
                continue
            seen.add(a)
            out.append(a)
        return out

    def _canonical_metric_registry(self, catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
        cached = catalog.get("_canonical_metric_registry")
        if isinstance(cached, dict) and cached:
            return cached

        metrics_by_table = catalog.get("metrics_by_table") if isinstance(catalog, dict) else {}
        preferred_time = catalog.get("preferred_time_column") if isinstance(catalog, dict) else {}
        preferred_time_by_metric = catalog.get("preferred_time_column_by_metric") if isinstance(catalog, dict) else {}
        table_to_domain = {table: domain for domain, table in DOMAIN_TO_MART.items()}

        registry: dict[str, dict[str, Any]] = {}
        for table, metric_map in (metrics_by_table or {}).items():
            if not isinstance(metric_map, dict):
                continue
            domain = table_to_domain.get(str(table), "unknown")
            for metric_name, metric_expr in metric_map.items():
                metric_name_s = str(metric_name or "").strip()
                metric_expr_s = str(metric_expr or "").strip()
                if not metric_name_s:
                    continue
                canonical_id = f"{domain}.{metric_name_s}" if domain != "unknown" else f"{table}.{metric_name_s}"
                metric_def = self._metric_business_definition(
                    table=str(table),
                    metric_name=metric_name_s,
                    metric_expr=metric_expr_s,
                )
                registry[canonical_id] = {
                    "canonical_metric_id": canonical_id,
                    "domain": domain,
                    "table": str(table),
                    "metric_name": metric_name_s,
                    "metric_expr": metric_expr_s,
                    "unit": self._infer_metric_unit(metric_name_s, metric_expr_s),
                    "default_grain": ["month"] if metric_name_s.endswith("_rate") else [],
                    "preferred_time_column": (
                        ((preferred_time_by_metric or {}).get(str(table), {}) or {}).get(metric_name_s)
                        or (preferred_time or {}).get(str(table))
                    ),
                    "business_definition": metric_def,
                    "aliases": self._canonical_metric_aliases(
                        domain=domain,
                        metric_name=metric_name_s,
                        business_definition=metric_def,
                    ),
                }

        if isinstance(catalog, dict):
            catalog["_canonical_metric_registry"] = registry
        return registry

    def _resolve_metric_from_registry_alias(
        self,
        lower: str,
        domain: str,
        catalog: dict[str, Any],
    ) -> str | None:
        registry = self._canonical_metric_registry(catalog)
        if not registry:
            return None

        scored: list[tuple[int, str]] = []
        for entry in registry.values():
            if str(entry.get("domain") or "") != domain:
                continue
            metric_name = str(entry.get("metric_name") or "").strip()
            if not metric_name:
                continue
            score = 0
            name_phrase = metric_name.replace("_", " ")
            if name_phrase and name_phrase in lower:
                score += 4
            for alias in entry.get("aliases") or []:
                alias_s = str(alias or "").strip().lower()
                if len(alias_s) < 4:
                    continue
                if alias_s in lower:
                    score += 3 if " " in alias_s else 1
            if score > 0:
                scored.append((score, metric_name))

        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_metric = scored[0]
        if best_score < 4:
            return None
        return best_metric

    def _canonical_metric_contract(
        self,
        *,
        domain: str,
        metric_name: str,
        table: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any] | None:
        metric_name_s = str(metric_name or "").strip()
        if not metric_name_s:
            return None
        registry = self._canonical_metric_registry(catalog)
        if not registry:
            return None

        domain_s = str(domain or "").strip()
        table_s = str(table or "").strip()
        canonical_id = ""
        if domain_s:
            canonical_id = f"{domain_s}.{metric_name_s}"
            if canonical_id in registry:
                return dict(registry[canonical_id])

        for entry in registry.values():
            if str(entry.get("metric_name") or "") != metric_name_s:
                continue
            if table_s and str(entry.get("table") or "") != table_s:
                continue
            return dict(entry)
        return None

    def _bind_canonical_metric_contracts(
        self,
        payload: dict[str, Any],
        catalog: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(catalog, dict):
            return payload
        out = dict(payload)
        table_to_domain = {table: domain for domain, table in DOMAIN_TO_MART.items()}
        out.pop("canonical_metric_id", None)
        out.pop("canonical_metric", None)
        out.pop("secondary_canonical_metric_id", None)
        out.pop("secondary_canonical_metric", None)

        domain = str(out.get("domain") or "").strip()
        table = str(out.get("table") or DOMAIN_TO_MART.get(domain, "")).strip()
        metric_name = str(out.get("metric") or "").strip()
        primary = self._canonical_metric_contract(
            domain=domain or table_to_domain.get(table, ""),
            metric_name=metric_name,
            table=table,
            catalog=catalog,
        )
        if primary:
            out["canonical_metric_id"] = primary.get("canonical_metric_id")
            out["canonical_metric"] = primary

        secondary_metric_name = str(out.get("secondary_metric") or out.get("compare_metric") or "").strip()
        secondary_table = str(out.get("compare_table") or table).strip()
        secondary_domain = str(
            out.get("secondary_domain")
            or table_to_domain.get(secondary_table, domain)
            or ""
        ).strip()
        secondary = self._canonical_metric_contract(
            domain=secondary_domain,
            metric_name=secondary_metric_name,
            table=secondary_table,
            catalog=catalog,
        )
        if secondary:
            out["secondary_canonical_metric_id"] = secondary.get("canonical_metric_id")
            out["secondary_canonical_metric"] = secondary
        return out

    def _metric_business_definition(
        self,
        *,
        table: str,
        metric_name: str,
        metric_expr: str,
    ) -> str:
        table_dict = self._table_business_dictionary(table)
        metric_defs = table_dict.get("metric_definitions")
        if isinstance(metric_defs, dict):
            text = str(metric_defs.get(metric_name) or "").strip()
            if text:
                return text
        low_name = str(metric_name or "").lower()
        low_expr = str(metric_expr or "").lower()
        if "count" in low_name or "count(" in low_expr:
            return "Volume KPI showing how many records/events occurred in scope."
        if "avg" in low_name or "avg(" in low_expr:
            return "Average KPI used to track normalized unit-level behavior."
        if "sum" in low_expr or any(tok in low_name for tok in ("total", "amount", "revenue", "spend")):
            return "Value KPI showing total economic volume in scope."
        if "rate" in low_name:
            return "Rate KPI used to compare quality or conversion across slices."
        return "Derived KPI available for grouped analysis and trend tracking."

    def _detect_schema_exploration(self, lower: str, catalog: dict[str, Any]) -> str | None:
        """Return schema target domain if the goal is schema/glossary exploration."""
        del catalog  # Reserved for future catalog-aware routing.
        overview_only_signals = [
            "what kind of data do i have",
            "what data do i have",
            "what do we have",
            "what is available",
            "show available data",
            "data overview",
        ]
        schema_specific_tokens = [
            "schema",
            "dictionary",
            "glossary",
            "field",
            "fields",
            "column",
            "columns",
            "attribute",
            "attributes",
            "table",
            "tables",
            "definition",
            "definitions",
        ]
        if any(sig in lower for sig in overview_only_signals) and not any(tok in lower for tok in schema_specific_tokens):
            return None

        schema_signals = [
            r"\bwhat\s+(?:do\s+we|does\s+it|do\s+i)\s+(?:capture|track|store|have|record)\b",
            r"\bwhat\s+(?:fields?|columns?|attributes?|demographics?|properties|info|information)\b",
            r"\bdescribe\s+the\b",
            r"\bwhat\s+are\s+the\b.*\b(?:fields?|columns?|attributes?|demographics?)\b",
            r"\bwhat\s+information\b",
            r"\bschema\b",
            r"\bstructure\s+of\b",
            r"\bwhat\s+(?:does|is\s+in)\s+the\s+\w+\s+table\b",
            r"\bwhat\s+(?:data|details?)\s+(?:do|does|is)\b.*\b(?:contain|include|store)\b",
            r"\bglossary\b",
            r"\bdictionary\b",
            r"\bdata\s+dictionary\b",
        ]
        if not any(re.search(p, lower) for p in schema_signals):
            return None

        domain_keywords = {
            "customer": "customers",
            "customers": "customers",
            "transaction": "transactions",
            "transactions": "transactions",
            "quote": "quotes",
            "quotes": "quotes",
            "booking": "bookings",
            "bookings": "bookings",
        }
        for keyword, domain in domain_keywords.items():
            if keyword in lower:
                return domain

        all_schema_signals = [
            "all tables",
            "entire schema",
            "whole schema",
            "every table",
            "each table",
            "each field",
            "every field",
            "full schema",
            "glossary",
            "dictionary",
        ]
        if any(sig in lower for sig in all_schema_signals):
            return "__all__"

        # Generic schema requests should default to the full dictionary.
        return "__all__"

    def _column_dictionary_definition(
        self,
        *,
        table: str,
        column: str,
        glossary: list[dict[str, Any]],
    ) -> str:
        col = str(column or "").strip()
        lower_col = col.lower()
        lower_table = str(table or "").lower()

        table_dictionary = self._table_business_dictionary(table)
        table_column_defs = table_dictionary.get("column_definitions")
        if isinstance(table_column_defs, dict):
            text = str(table_column_defs.get(lower_col) or table_column_defs.get(col) or "").strip()
            if text:
                return text
        global_defs = self._business_dictionary_view().get("global_column_definitions")
        if isinstance(global_defs, dict):
            text = str(global_defs.get(lower_col) or global_defs.get(col) or "").strip()
            if text:
                return text

        for entry in glossary:
            term = str(entry.get("term") or "").strip().lower()
            target_table = str(entry.get("target_table") or "").strip().lower()
            target_column = str(entry.get("target_column") or "").strip().lower()
            if target_table and target_table != lower_table:
                continue
            if target_column and target_column != lower_col:
                continue
            if term and term not in {lower_col, lower_col.replace("_", " ")} and not target_column:
                continue
            definition = str(entry.get("definition") or "").strip()
            if definition:
                return definition

        if lower_col.endswith("_key"):
            return "Surrogate key used for deduplication and stable grouping."
        if lower_col.endswith("_id"):
            return "Business/source identifier used to join with related entities."
        if lower_col.startswith(("has_", "is_")):
            return "Boolean status flag for filtering valid subsets."
        if lower_col.endswith(("_ts", "_at")) or "date" in lower_col or "time" in lower_col:
            return "Timestamp/date field used for time-scoping and trend analysis."
        if any(tok in lower_col for tok in ("amount", "value", "revenue", "charges", "markup", "rate", "spend")):
            return "Numeric business metric used in aggregations."
        if "currency" in lower_col:
            return "Currency code/value used for FX and pair analysis."
        if "status" in lower_col:
            return "Lifecycle or processing state used in status breakdowns."
        if any(tok in lower_col for tok in ("platform", "country", "state", "type", "flow")):
            return "Categorical dimension used for grouped analysis."
        return "General field available for filtering, grouping, or drill-down analysis."

    def _schema_exploration_agent(
        self,
        goal: str,
        domain: str,
        catalog: dict[str, Any],
        tenant_id: str = "public",
    ) -> dict[str, Any]:
        """Return schema dictionary views (single domain or all marts)."""
        del goal
        marts = catalog.get("marts", {})
        glossary = self.list_glossary(tenant_id=tenant_id)

        def _render_table_dictionary(mart_name: str) -> tuple[list[str], dict[str, Any]]:
            mart_meta = marts.get(mart_name, {})
            columns = list(mart_meta.get("columns") or [])
            row_count = int(mart_meta.get("row_count") or 0)
            metrics = catalog.get("metrics_by_table", {}).get(mart_name, {})
            dim_values = catalog.get("dimension_values", {}).get(mart_name, {})
            time_col = catalog.get("preferred_time_column", {}).get(mart_name, "")
            table_dictionary = self._table_business_dictionary(mart_name)
            business_purpose = str(table_dictionary.get("business_purpose") or "").strip()
            analyst_questions = [
                str(q).strip()
                for q in (table_dictionary.get("analyst_questions") or [])
                if str(q).strip()
            ]

            lines: list[str] = []
            lines.append(f"### `{mart_name}`")
            lines.append(f"- Rows: **{row_count:,}**")
            lines.append(f"- Columns: **{len(columns)}**")
            if business_purpose:
                lines.append(f"- Business purpose: {business_purpose}")
            if analyst_questions:
                lines.append("- Typical business asks:")
                lines.extend(f"  - {q}" for q in analyst_questions[:3])
            lines.append("")
            lines.append("| Field | Meaning | Notes |")
            lines.append("| --- | --- | --- |")
            for col in columns:
                meaning = self._column_dictionary_definition(
                    table=mart_name,
                    column=col,
                    glossary=glossary,
                )
                notes: list[str] = []
                if col == time_col:
                    notes.append("preferred time")
                if col in dim_values:
                    sample = ", ".join(list(dim_values.get(col) or [])[:3])
                    if sample:
                        notes.append(f"sample: {sample}")
                lines.append(
                    f"| `{col}` | {meaning} | {'; '.join(notes) if notes else '-'} |"
                )

            if metrics:
                lines.append("")
                lines.append("| Metric | SQL expression | Business meaning |")
                lines.append("| --- | --- | --- |")
                for name, expr in metrics.items():
                    metric_meaning = self._metric_business_definition(
                        table=mart_name,
                        metric_name=str(name),
                        metric_expr=str(expr),
                    )
                    lines.append(f"| `{name}` | `{expr}` | {metric_meaning} |")
            lines.append("")

            return lines, {
                "table": mart_name,
                "columns": columns,
                "row_count": row_count,
                "business_purpose": business_purpose,
            }

        if domain == "__all__":
            all_lines: list[str] = []
            all_lines.append("## Full Schema Dictionary")
            all_lines.append(
                "This is the full table/field glossary across all semantic marts."
            )
            all_lines.append("")
            table_summaries: list[dict[str, Any]] = []
            for mart_name in sorted(marts.keys()):
                rendered_lines, summary = _render_table_dictionary(mart_name)
                all_lines.extend(rendered_lines)
                table_summaries.append(summary)
            suggested = [
                "Show monthly transaction count and total amount by platform",
                "Which quote currency pairs drive forex markup revenue?",
                "Give me customer mix by country and type",
                "How do booking values trend month over month?",
            ]
            return {
                "answer_markdown": "\n".join(all_lines),
                "domain": "__all__",
                "table": "__all__",
                "columns": [
                    f"{item['table']}.{col}"
                    for item in table_summaries
                    for col in item["columns"]
                ],
                "row_count": sum(int(item["row_count"]) for item in table_summaries),
                "table_summaries": table_summaries,
                "suggested_questions": suggested,
            }

        mart = DOMAIN_TO_MART.get(domain, "")
        rendered_lines, summary = _render_table_dictionary(mart)
        suggested: list[str] = [
            f"How many {domain} do we have?",
            f"Show {domain} metrics by month",
            f"What are top dimensions for {domain}?",
            f"Give me quality checks for {domain} data",
        ]
        return {
            "answer_markdown": "\n".join(
                [f"## {domain.title()} Schema Dictionary"] + rendered_lines
            ),
            "domain": domain,
            "table": mart,
            "columns": summary["columns"],
            "row_count": summary["row_count"],
            "suggested_questions": suggested,
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
        # ── QA-R4 Fix 5 / QA-R5 Fix 5: Insight synthesis intent ─────────
        elif any(
            k in lower
            for k in [
                "actionable insight",
                "business insight",
                "key insight",
                "top insight",
                "key takeaway",
                "actionable takeaway",
                "executive summary",
                "what should we focus on",
                "what stands out",
                "notable pattern",
                "key finding",
                "key findings",
                "decision-grade insight",
                "decision grade insight",
                "strategic insight",
            ]
        ) or re.search(r"\b(?:insights|findings)\b", lower):
            intent = "insight_synthesis"
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
                "each ",
                "month wise",
                "platform wise",
                "country wise",
            ]
        ) or bool(re.search(r"\bper\s+(platform|country|region|state|currency|deal\s+type|customer\s+type|month|flow|status|type)\b", lower)):
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
                "currency exchange",
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
        elif any(
            k in lower
            for k in [
                "payee",
                "university",
                "universities",
                "address",
                "beneficiary",
                "client",
                "clients",
                "country",
                "countries",
            ]
        ):
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
            "quotes": [
                "quote",
                "forex",
                "currency exchange",
                "exchange rate",
                "amount to be paid",
                "markup",
                "charge",
                "spread",
                "currency pair",
                "currency combination",
            ],
            "bookings": ["booking", "booked", "deal type", "value date"],
            "customers": [
                "customer",
                "payee",
                "university",
                "universities",
                "address",
                "beneficiary",
                "client",
                "clients",
                "country",
                "countries",
            ],
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
        secondary_metric: str | None = None
        metric_bundle: str | None = None

        if intent == "data_overview":
            metric = "catalog_overview"
        elif intent == "schema_exploration":
            metric = "schema_dictionary"
        elif domain == "documents" or intent == "document_qa":
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

            # When users ask for both count and amount, preserve a dual-metric
            # contract instead of collapsing to one.
            if has_count_words and has_amount_words:
                if domain == "transactions":
                    metric = "mt103_count" if has_mt103 else ("refund_count" if has_refund else "transaction_count")
                    secondary_metric = "total_amount"
                    metric_bundle = "count_plus_amount"
                elif domain == "quotes":
                    metric = "quote_count"
                    if any(k in lower for k in ["markup", "forex markup", "fx markup"]):
                        secondary_metric = "forex_markup_revenue"
                    else:
                        secondary_metric = "total_quote_value"
                    metric_bundle = "count_plus_amount"
                elif domain == "bookings":
                    metric = "booking_count"
                    secondary_metric = "total_booked_amount"
                    metric_bundle = "count_plus_amount"

        # RC4a: Detect "currency pair" as compound dimension for quotes domain
        _currency_pair_detected = False
        if domain == "quotes" and any(kw in lower for kw in ["currency pair", "currency combination", "most common currency"]):
            _currency_pair_detected = True
            if any(kw in lower for kw in ["most common", "top", "most popular", "most frequent", "frequent", "which"]):
                intent = "grouped_metric"
                if metric == "quote_count" or not metric:
                    metric = "quote_count"

        dim_candidates = self._build_dim_candidates_from_catalog(domain, catalog)
        mart_name = DOMAIN_TO_MART.get(domain, "")
        dim_value_cols = set(catalog.get("dimension_values", {}).get(mart_name, {}).keys())
        mart_cols = set(catalog.get("marts", {}).get(mart_name, {}).get("columns", []))
        for sec_d in domains_detected:
            if sec_d != domain:
                sec_mart = DOMAIN_TO_MART.get(sec_d, "")
                sec_dim_vals = set(catalog.get("dimension_values", {}).get(sec_mart, {}).keys())
                sec_cols = set(catalog.get("marts", {}).get(sec_mart, {}).get("columns", []))
                dim_value_cols |= sec_dim_vals
                mart_cols |= sec_cols
        available_dim_cols = dim_value_cols | mart_cols

        dimensions: list[str] = []
        has_per_dimension = bool(
            re.search(r"\bper\s+(platform|country|region|state|currency|deal\s+type|customer\s+type|month|flow|status|type)\b", lower)
        )
        dim_signal = has_per_dimension or any(
            t in lower
            for t in [" by ", "split", "breakdown", "top", "wise", "month wise", "grouped", "group by"]
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
            if domain == "quotes" and key == "currency":
                if any(kw in lower for kw in ["destination currency", "dest currency", "to currency"]):
                    continue
                if any(kw in lower for kw in ["source currency", "from currency"]):
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
            per_match_single = re.findall(r"\bper\s+(\w+)", lower)
            per_match_double = re.findall(r"\bper\s+(\w+\s+\w+)", lower)
            split_match = re.findall(r"\bsplit\s+(?:by\s+)?(\w+)", lower)
            # Prioritize longer (multi-word) matches over single-word
            candidate_words_ordered: list[str] = (
                list(by_match_double)
                + list(per_match_double)
                + list(by_match_single)
                + list(per_match_single)
                + list(split_match)
            )
            noise = {
                "the", "a", "an", "and", "or", "each", "every",
                "month", "monthly", "compare", "compared", "versus", "vs",
            }
            metric_phrase_tokens = {
                "count",
                "counts",
                "amount",
                "amounts",
                "total",
                "avg",
                "average",
                "sum",
                "value",
                "values",
                "revenue",
                "markup",
                "forex",
                "charge",
                "charges",
                "rate",
            }
            seen: set[str] = set()
            candidate_words: list[str] = []
            for w in candidate_words_ordered:
                w_clean = w.strip()
                if any(tok in w_clean for tok in ["compare", "compared", "versus"]):
                    continue
                parts = [p for p in re.findall(r"[a-z]+", w_clean.lower()) if p]
                if parts and any(p in metric_phrase_tokens for p in parts):
                    continue
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
            explicit_phrase_map = {
                "destination currency": "to_currency",
                "dest currency": "to_currency",
                "to currency": "to_currency",
                "source currency": "from_currency",
                "from currency": "from_currency",
                "customer type": "customer_type",
                "customer country": "address_country",
                "customer state": "address_state",
            }
            for word in candidate_words:
                if word in known_keys:
                    continue
                if domain == "quotes":
                    if word == "destination" and any(
                        kw in lower for kw in ["destination currency", "dest currency", "to currency"]
                    ):
                        continue
                    if word == "source" and any(
                        kw in lower for kw in ["source currency", "from currency"]
                    ):
                        continue
                # Skip single words already consumed by a successful multi-word match
                if word in consumed_words:
                    continue
                # Try substring match: "country" -> "address_country"
                resolved = None
                explicit_col = explicit_phrase_map.get(word)
                if explicit_col and explicit_col in available_dim_cols:
                    resolved = explicit_col
                if not resolved and word == "customer type":
                    for candidate in ("customer_type", "type"):
                        if candidate in available_dim_cols:
                            resolved = candidate
                            break
                if not resolved and word in {"customer country", "region"}:
                    for candidate in ("region", "address_country", "country"):
                        if candidate in available_dim_cols:
                            resolved = candidate
                            break
                if not resolved and word in {"customer state", "state"}:
                    for candidate in ("address_state", "state"):
                        if candidate in available_dim_cols:
                            resolved = candidate
                            break
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
                    word_parts = [p for p in re.findall(r"[a-z]+", str(word).lower()) if p]
                    if word_parts and any(p in metric_phrase_tokens for p in word_parts):
                        continue
                    self._pipeline_warnings.append(
                        f"Dimension keyword '{word}' was not recognized for the "
                        f"'{domain}' domain and was ignored."
                    )

        if domain == "quotes":
            has_source_currency_phrase = any(kw in lower for kw in ["source currency", "source currencies", "from currency"])
            has_destination_currency_phrase = any(kw in lower for kw in ["destination currency", "destination currencies", "dest currency", "to currency"])
            if has_destination_currency_phrase or ("top currencies" in lower and not has_source_currency_phrase):
                dimensions = [d for d in dimensions if d not in {"from_currency", "amount_at_destination", "amount_at_source"}]
                if "to_currency" not in dimensions:
                    dimensions.insert(0, "to_currency")
            elif has_source_currency_phrase:
                dimensions = [d for d in dimensions if d not in {"to_currency", "amount_at_destination", "amount_at_source"}]
                if "from_currency" not in dimensions:
                    dimensions.insert(0, "from_currency")

        # "trend" implies time-series but should not override explicit dimensions.
        # Only add __month__ when "trend" appears and no month signal already added it.
        if has_trend_signal and domain != "documents" and "__month__" not in dimensions:
            dimensions.append("__month__")

        # Superlative/grouped questions may omit explicit "by/split".
        if intent == "grouped_metric":
            if "platform" in lower and "platform_name" not in dimensions and "platform_name" in available_dim_cols:
                dimensions.insert(0, "platform_name")
            if any(k in lower for k in ["destination currency", "dest currency", "to currency"]) and "to_currency" in available_dim_cols:
                dimensions = [d for d in dimensions if d != "from_currency"]
                if "to_currency" not in dimensions:
                    dimensions.insert(0, "to_currency")
            if "country" in lower or "region" in lower:
                for candidate in ("region", "address_country", "country"):
                    if candidate in available_dim_cols and candidate not in dimensions:
                        dimensions.insert(0, candidate)
                        break
            if "customer type" in lower or re.search(r"\bper\s+type\b", lower):
                for candidate in ("customer_type", "type"):
                    if candidate in available_dim_cols and candidate not in dimensions:
                        dimensions.insert(0, candidate)
                        break
            if "deal type" in lower and "deal_type" in available_dim_cols and "deal_type" not in dimensions:
                dimensions.insert(0, "deal_type")

        if intent == "grouped_metric" and not dimensions and dim_candidates:
            preferred_default = {
                "transactions": "platform_name",
                "quotes": "to_currency",
                "customers": "address_country",
                "bookings": "deal_type",
            }.get(domain)
            if preferred_default and preferred_default in available_dim_cols:
                dimensions.append(preferred_default)
            else:
                default_col = next(iter(dim_candidates.values()))
                if default_col != "__month__":
                    dimensions.append(default_col)

        # Customer split semantics: "universities vs non-universities"
        # should reliably bind to boolean university segmentation.
        if domain == "customers" and (dim_signal or intent == "grouped_metric"):
            has_uni_split = any(
                k in lower for k in ["university", "universities", "non-university", "non-universities"]
            )
            if has_uni_split and "is_university" in available_dim_cols and "is_university" not in dimensions:
                dimensions.append("is_university")

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
        # ── QA-R4 Fix 2: Extract ALL months for comparison intent ────────
        all_months: list[int] = []
        for name, number in MONTH_MAP.items():
            if re.search(rf"\b{name}\b", lower):
                all_months.append(number)
        month = all_months[0] if all_months else None

        years = re.findall(r"\b(20\d{2})\b", lower)
        year = int(years[0]) if years else None
        quarter: int | None = None
        qnum_match = re.search(r"\bq([1-4])\b", lower)
        if qnum_match:
            quarter = int(qnum_match.group(1))
        else:
            qword_map = {
                "first quarter": 1,
                "second quarter": 2,
                "third quarter": 3,
                "fourth quarter": 4,
            }
            for phrase, qval in qword_map.items():
                if phrase in lower:
                    quarter = qval
                    break

        # Explicit phrase support: "last month of 2025" => Dec 2025
        last_month_of_year = re.search(r"\blast\s+month\s+of\s+(20\d{2})\b", lower)
        if last_month_of_year:
            time_filter = {"kind": "month_year", "month": 12, "year": int(last_month_of_year.group(1))}
        # When two distinct months are mentioned in a comparison query,
        # create an explicit_comparison filter so _time_clause can map
        # current → later month, comparison → earlier month with no subtraction.
        elif len(all_months) >= 2 and intent == "comparison":
            unique_months = list(dict.fromkeys(all_months))  # preserve order, dedupe
            if len(unique_months) >= 2:
                time_filter = {
                    "kind": "explicit_comparison",
                    "month_a": unique_months[0],
                    "month_b": unique_months[1],
                    "year": year,
                }
            else:
                time_filter = {"kind": "month_year", "month": month, "year": year}
        elif quarter is not None:
            quarter_months = {
                1: [1, 2, 3],
                2: [4, 5, 6],
                3: [7, 8, 9],
                4: [10, 11, 12],
            }[quarter]
            time_filter = {"kind": "month_list", "months": quarter_months, "year": year}
        elif month is not None:
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
                value_l = value.lower()
                if value_l in {"true", "false"}:
                    continue
                if value_l in {"mt103", "refund", "university", "universities"} and col not in {"has_mt103", "has_refund", "is_university"}:
                    continue
                if value_l in lower:
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

        if intent in {"data_overview", "schema_exploration", "document_qa"}:
            domains_detected = [domain]
            secondary_domains = []

        return {
            "goal": g,
            "intent": intent,
            "domain": domain,
            "metric": metric,
            "secondary_metric": secondary_metric,
            "metric_bundle": metric_bundle,
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
        goal_for_rules = " ".join(
            s for s in [
                str(goal or "").strip(),
                str(parsed.get("goal") or "").strip(),
                str(deterministic.get("goal") or "").strip(),
            ] if s
        )
        lower = goal_for_rules.lower()
        origin_lower = str(goal or "").lower()

        # ── QA-R4 Fix 1: Time-filter specificity guard ──────────────────
        # If deterministic parsing found a more specific time filter (e.g.
        # month_year) but the LLM downgraded it (e.g. to year_only), keep
        # the deterministic version.
        _TIME_SPECIFICITY = {"explicit_comparison": 4, "month_year": 3, "month_list": 3, "relative": 2, "year_only": 1}
        det_tf = deterministic.get("time_filter")
        out_tf = out.get("time_filter")
        if det_tf and out_tf:
            det_spec = _TIME_SPECIFICITY.get(det_tf.get("kind", ""), 0)
            out_spec = _TIME_SPECIFICITY.get(out_tf.get("kind", ""), 0)
            if det_spec > out_spec:
                self._pipeline_warnings.append(
                    f"Time filter restored from '{out_tf.get('kind')}' to "
                    f"'{det_tf.get('kind')}' — deterministic parser was more specific."
                )
                out["time_filter"] = det_tf
            explicit_month_year = bool(
                re.search(
                    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
                    r"january|february|march|april|june|july|august|september|october|november|december)\b",
                    origin_lower,
                )
                and re.search(r"\b20\d{2}\b", origin_lower)
            )
            if (
                explicit_month_year
                and det_tf.get("kind") in {"month_year", "explicit_comparison", "month_list"}
                and out_tf != det_tf
            ):
                self._pipeline_warnings.append(
                    "Time filter locked to deterministic month/year parse to prevent LLM month drift."
                )
                out["time_filter"] = det_tf
        elif det_tf and not out_tf:
            explicit_month_year = bool(
                re.search(
                    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
                    r"january|february|march|april|june|july|august|september|october|november|december)\b",
                    origin_lower,
                )
                and re.search(r"\b20\d{2}\b", origin_lower)
            )
            if explicit_month_year and det_tf.get("kind") in {"month_year", "explicit_comparison", "month_list"}:
                out["time_filter"] = det_tf

        # ── QA-R4 Fix 1b / Fix 6 / QA-R5 Fix 2: Future-date guard ───────
        _current_year = datetime.utcnow().year
        _tf = out.get("time_filter")
        if _tf and isinstance(_tf, dict):
            _tf_year = _tf.get("year")
            if _tf_year is not None and int(_tf_year) > _current_year:
                self._pipeline_warnings.append(
                    f"Future year {_tf_year} detected (current year is {_current_year}). "
                    "No data can exist for future dates."
                )
                out["time_filter"] = {"kind": "future_blocked", "year": int(_tf_year)}

        # RC6: Hard domain guard — explicit entity nouns override LLM drift
        if (
            "transaction" in lower
            and not any(k in lower for k in ["booking", "booked", "deal type", "value date"])
            and not any(k in lower for k in ["forex", "markup", "quote", "exchange rate"])
        ):
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
        _hard_route_intents = {"schema_exploration", "data_overview", "document_qa"}
        if det_intent in _hard_route_intents:
            out["intent"] = det_intent
            out["domain"] = deterministic.get("domain", out.get("domain"))
            out["metric"] = deterministic.get("metric", out.get("metric"))
            out["dimensions"] = []
            out["dimension"] = None
            out["value_filters"] = []
        if det_intent in _complex_intents and not is_llm_mode:
            out["intent"] = det_intent

        compare_terms = [" vs ", " versus ", "compare", "compared"]
        if out.get("intent") == "comparison" and not any(t in lower for t in compare_terms):
            out["intent"] = deterministic.get("intent", out.get("intent"))

        amount_terms = ["amount", "revenue", "value", "sum", "booked", "markup"]
        has_count_words = self._goal_has_count_intent(origin_lower)
        if any(t in origin_lower for t in amount_terms):
            if str(out.get("metric", "")).endswith("_count") and str(
                deterministic.get("metric", "")
            ) in {"total_amount", "total_quote_value", "total_booked_amount", "forex_markup_revenue"}:
                out["metric"] = deterministic.get("metric")

        # Keep dual-metric intent stable when the question explicitly asks for
        # both count and amount (skill: query-expert).
        if self._goal_has_count_intent(origin_lower) and self._goal_has_amount_intent(origin_lower):
            det_primary = str(deterministic.get("metric") or "").strip()
            det_secondary = str(deterministic.get("secondary_metric") or "").strip()
            if det_primary:
                out["metric"] = det_primary
            if det_secondary:
                out["secondary_metric"] = det_secondary
                out["metric_bundle"] = str(deterministic.get("metric_bundle") or "count_plus_amount")

        if any(k in origin_lower for k in ["same slice", "same scope", "keep that", "keep same", "both", "together"]):
            det_primary = str(deterministic.get("metric") or "").strip()
            det_secondary = str(deterministic.get("secondary_metric") or "").strip()
            if det_primary and det_secondary:
                out["metric"] = det_primary
                out["secondary_metric"] = det_secondary
                out["metric_bundle"] = str(deterministic.get("metric_bundle") or "count_plus_amount")

        # Keep deterministic semantic grounding for high-signal metric intents.
        if any(k in lower for k in ["exchange rate", "fx rate", "forex rate"]):
            if str(deterministic.get("metric") or "") == "avg_exchange_rate":
                out["metric"] = "avg_exchange_rate"
        if any(k in lower for k in ["average forex markup", "avg forex markup", "average markup", "avg markup"]):
            if str(deterministic.get("metric") or "") in {"avg_forex_markup", "forex_markup_revenue"}:
                out["metric"] = "avg_forex_markup"
        if any(k in lower for k in ["fx charge", "fx charges", "forex charge", "forex charges"]):
            explicit_additional_charge = any(
                k in lower
                for k in [
                    "additional charge",
                    "additional charges",
                    "total charges",
                    "service charge",
                    "service charges",
                    "platform charge",
                    "platform charges",
                    "swift charge",
                    "swift charges",
                ]
            )
            if not explicit_additional_charge and str(deterministic.get("metric") or "") == "forex_markup_revenue":
                out["metric"] = "forex_markup_revenue"

        quote_count_phrases = (
            "quote count",
            "quotes count",
            "count of quote",
            "count of quotes",
            "number of quotes",
            "how many quotes",
        )
        explicit_quote_count = any(p in lower for p in quote_count_phrases)
        if explicit_quote_count or (
            ("quote" in lower and has_count_words)
            and not self._goal_has_amount_intent(origin_lower)
        ):
            out["domain"] = "quotes"
            out["metric"] = "quote_count"
            if not (self._goal_has_count_intent(origin_lower) and self._goal_has_amount_intent(origin_lower)):
                out["secondary_metric"] = None
                out["metric_bundle"] = None

        value_filters = list(out.get("value_filters") or [])
        det_filters = list(deterministic.get("value_filters") or [])

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
                if any(t in origin_lower for t in amount_terms) and not has_count_words:
                    out["metric"] = "total_amount"
        if (
            "mt103" in origin_lower
            and any(t in origin_lower for t in amount_terms)
            and not self._goal_has_count_intent(origin_lower)
        ):
            out["metric"] = "total_amount"
            out["secondary_metric"] = None
            out["metric_bundle"] = None

        # Preserve deterministic categorical filters when LLM output drops them.
        for dvf in det_filters:
            col = str(dvf.get("column") or "").strip()
            val = str(dvf.get("value") or "").strip()
            if col and val:
                add_filter(col, val)

        if "split" in lower and str(out.get("intent") or "") == "metric":
            out["intent"] = "grouped_metric"

        grouped_signal = grouped_signal_spine(lower)
        ranking_signal = any(
            token in origin_lower
            for token in [
                "top ",
                "highest",
                "lowest",
                "most ",
                "which platform",
                "which country",
                "which state",
                "which currency",
            ]
        )
        if ranking_signal:
            det_dims = list(deterministic.get("dimensions") or [])
            if det_dims:
                out["dimensions"] = det_dims[:MAX_DIMENSIONS]
                out["dimension"] = out["dimensions"][0] if out["dimensions"] else None
                if out.get("intent") in {"metric", "lookup"}:
                    out["intent"] = "grouped_metric"

        raw_dims = list(out.get("dimensions") or [])
        dim0 = out.get("dimension")
        if not raw_dims and isinstance(dim0, str) and dim0.strip():
            raw_dims = [dim0.strip()]
        if grouped_signal and not raw_dims:
            raw_dims = list(deterministic.get("dimensions") or [])

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
        has_per_dimension = bool(
            re.search(r"\bper\s+(platform|country|region|state|currency|deal\s+type|customer\s+type|month|flow|status|type)\b", lower)
        )
        dim_signal = has_per_dimension or any(
            token in lower for token in [" by ", "split", "breakdown", "wise", "grouped", "group by"]
        )
        if "month" in lower and dim_signal and "__month__" not in dims:
            dims.append("__month__")
        if out.get("domain") == "transactions":
            if "platform" in lower and dim_signal and "platform_name" not in dims:
                dims.append("platform_name")
            if "state" in lower and dim_signal and "state" not in dims:
                dims.append("state")
            if ("region" in lower or "country" in lower) and dim_signal:
                for candidate in ("region", "address_country", "country"):
                    if candidate in available_cols and candidate not in dims:
                        dims.append(candidate)
                        break
            if ("customer type" in lower or re.search(r"\bper\s+type\b", lower)) and dim_signal:
                for candidate in ("customer_type", "type"):
                    if candidate in available_cols and candidate not in dims:
                        dims.append(candidate)
                        break
        if len(dims) > MAX_DIMENSIONS:
            dropped = dims[MAX_DIMENSIONS:]
            dims = dims[:MAX_DIMENSIONS]
            self._pipeline_warnings.append(
                f"Requested {len(dims) + len(dropped)} dimensions but only "
                f"{MAX_DIMENSIONS} are supported. Dropped: {', '.join(dropped)}."
            )

        # Guardrail: avoid accidental high-cardinality grouping unless explicit.
        _high_card_dims = {
            "transaction_id",
            "quote_id",
            "booking_id",
            "customer_id",
            "payee_id",
            "transaction_key",
            "quote_key",
            "booking_key",
            "customer_key",
            "payee_key",
        }
        def _explicit_dim_reference(dim: str) -> bool:
            normalized = dim.replace("_", " ")
            return (
                dim in lower
                or normalized in lower
                or any(tok in lower for tok in [f"by {normalized}", f"split by {normalized}", f"per {normalized}"])
            )

        pruned_dims: list[str] = []
        dropped_high_card: list[str] = []
        for dim in dims:
            if dim in _high_card_dims and not _explicit_dim_reference(dim):
                dropped_high_card.append(dim)
                continue
            pruned_dims.append(dim)
        if dropped_high_card:
            self._pipeline_warnings.append(
                "Dropped high-cardinality dimensions not explicitly requested: "
                + ", ".join(dropped_high_card)
            )
        dims = pruned_dims

        # Pin common business dimensions from phrasing to reduce drift.
        if out.get("intent") == "grouped_metric":
            domain = str(out.get("domain") or "")
            if "platform" in lower and "platform_name" in available_cols and "platform_name" not in dims:
                dims.insert(0, "platform_name")
            has_source_currency_phrase = any(k in lower for k in ["source currency", "source currencies", "from currency"])
            has_destination_currency_phrase = any(k in lower for k in ["destination currency", "destination currencies", "dest currency", "to currency"])
            if (has_destination_currency_phrase or ("top currencies" in lower and not has_source_currency_phrase)) and "to_currency" in available_cols:
                dims = [d for d in dims if d not in {"from_currency", "__month__"}]
                if "to_currency" not in dims:
                    dims.insert(0, "to_currency")
            if has_source_currency_phrase and "from_currency" in available_cols:
                dims = [d for d in dims if d not in {"to_currency", "__month__"}]
                if "from_currency" not in dims:
                    dims.insert(0, "from_currency")
            if "country" in lower or "region" in lower:
                for candidate in ("region", "address_country", "country"):
                    if candidate in available_cols and candidate not in dims:
                        dims.insert(0, candidate)
                        break
            if "customer type" in lower or re.search(r"\bper\s+type\b", lower):
                for candidate in ("customer_type", "type"):
                    if candidate in available_cols and candidate not in dims:
                        dims.insert(0, candidate)
                        break
            if "deal type" in lower and "deal_type" in available_cols and "deal_type" not in dims:
                dims.insert(0, "deal_type")
            if not dims:
                default_dim = {
                    "transactions": "platform_name",
                    "quotes": "to_currency",
                    "customers": "address_country",
                    "bookings": "deal_type",
                }.get(domain, "")
                if default_dim and default_dim in available_cols:
                    dims = [default_dim]
        out["dimensions"] = dims
        out["dimension"] = dims[0] if dims else None
        if grouped_signal and out.get("intent") in {"metric", "lookup"}:
            out["intent"] = "grouped_metric"

        if (
            any(kw in origin_lower for kw in ["mt103", "mt 103"])
            and not any(kw in origin_lower for kw in ["transaction flow", "txn flow", "flow"])
        ):
            value_filters = [
                vf
                for vf in value_filters
                if not (
                    str(vf.get("column", "")).lower() == "txn_flow"
                    and "mt103" in str(vf.get("value", "")).lower()
                )
            ]
        out["value_filters"] = value_filters
        return out

    def _build_contract_spec(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Build a semantic contract specification from the execution plan.

        The contract captures the binding between the user's intent and the
        SQL execution plan — metric, domain, dimensions, time_scope, filters,
        and exclusions. This is used for pre-execution validation (FR-1) and
        post-execution explainability (FR-6).
        """
        table = str(plan.get("table") or "").strip()
        domain = str(plan.get("domain") or "").strip()
        if not domain and table:
            domain = next((d for d, mart in DOMAIN_TO_MART.items() if mart == table), "")
        catalog = getattr(self.semantic, "_catalog", None)
        canonical_bound = self._bind_canonical_metric_contracts(plan, catalog) if isinstance(catalog, dict) else dict(plan)
        if canonical_bound.get("canonical_metric_id"):
            plan["canonical_metric_id"] = canonical_bound.get("canonical_metric_id")
            plan["canonical_metric"] = canonical_bound.get("canonical_metric")
        if canonical_bound.get("secondary_canonical_metric_id"):
            plan["secondary_canonical_metric_id"] = canonical_bound.get("secondary_canonical_metric_id")
            plan["secondary_canonical_metric"] = canonical_bound.get("secondary_canonical_metric")

        # ── G1: Delegate to contract-first module for structured contract ────
        try:
            structured = build_contract_from_pipeline(plan)
            structured_dict = structured.to_dict()
            if (not structured_dict.get("domain") or structured_dict.get("domain") == "unknown") and domain:
                structured_dict["domain"] = domain
            if (not structured_dict.get("metric") or structured_dict.get("metric") == "unknown") and plan.get("metric_expr"):
                structured_dict["metric"] = str(plan.get("metric_expr"))
            if not structured_dict.get("grain"):
                dims = list(plan.get("dimensions") or [])
                structured_dict["grain"] = [
                    ("month" if d == "__month__" else str(d))
                    for d in dims
                    if isinstance(d, str)
                ]
            plan["_structured_contract"] = structured_dict
        except Exception as _sc_exc:
            self._pipeline_warnings.append(
                f"Structured contract build failed ({type(_sc_exc).__name__}: {_sc_exc}); "
                f"falling back to plan-based contract."
            )
            plan["_structured_contract"] = {}

        time_filter = plan.get("time_filter")
        time_scope = "none"
        if time_filter:
            kind = time_filter.get("kind", "")
            if kind == "future_blocked":
                time_scope = f"future_blocked:{time_filter.get('year', '?')}"
            elif kind == "month_year":
                time_scope = f"month_year:{time_filter.get('year', '?')}-{time_filter.get('month', '?')}"
            elif kind == "year_only":
                time_scope = f"year:{time_filter.get('year', '?')}"
            elif kind == "month_list":
                months = ",".join(str(m) for m in (time_filter.get("months") or []))
                time_scope = f"month_list:{time_filter.get('year', '?')}[{months}]"
            elif kind == "explicit_comparison":
                time_scope = f"comparison:{time_filter.get('month_a', '?')}v{time_filter.get('month_b', '?')}@{time_filter.get('year', '?')}"
            elif kind == "relative":
                time_scope = f"relative:{time_filter.get('value', '?')}"
            else:
                time_scope = f"unknown:{kind}"

        return {
            "metric": plan.get("metric", ""),
            "metric_expr": plan.get("metric_expr", ""),
            "secondary_metric": plan.get("secondary_metric", ""),
            "secondary_metric_expr": plan.get("secondary_metric_expr", ""),
            "domain": domain,
            "table": table,
            "dimensions": list(plan.get("dimensions") or []),
            "time_scope": time_scope,
            "time_filter": time_filter,
            "filters": list(plan.get("value_filters") or []),
            "exclusions": list(plan.get("_exclusions") or []),
            "intent": plan.get("intent", ""),
            "definition_used": plan.get("definition_used", ""),
            "canonical_metric_id": plan.get("canonical_metric_id", ""),
            "canonical_metric": dict(plan.get("canonical_metric") or {}),
            "secondary_canonical_metric_id": plan.get("secondary_canonical_metric_id", ""),
            "secondary_canonical_metric": dict(plan.get("secondary_canonical_metric") or {}),
        }

    def _validate_contract_against_sql(
        self, contract: dict[str, Any], sql: str, plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate that generated SQL honors the semantic contract.

        Checks:
        1. Time scope: If contract specifies month/year, SQL must contain
           EXTRACT(MONTH/YEAR) or DATE_TRUNC predicates.
        2. Grouping fidelity: If contract has dimensions, SQL must have GROUP BY.
        3. Metric grounding: metric_expr tokens must appear in SQL.
        4. Future blocking: future_blocked contracts must produce FALSE in WHERE.

        Returns dict with 'valid': bool, 'violations': list[str], 'checks': list[dict].
        """
        # ── G1: Run structured contract validation ────────────────────
        structured = plan.get("_structured_contract")
        if structured and structured.get("metric"):
            try:
                sc = AnalysisContract(**structured)
                result = validate_sql_against_contract(sql, sc)
                if not result.valid:
                    for v in result.violations:
                        if v not in self._pipeline_warnings:
                            self._pipeline_warnings.append(f"[Contract] {v}")
            except Exception as _cv_exc:
                self._pipeline_warnings.append(
                    f"Structured contract validation failed ({type(_cv_exc).__name__}); "
                    f"falling back to heuristic checks."
                )

        sql_upper = sql.upper()
        sql_lower = sql.lower()
        violations: list[str] = []
        checks: list[dict[str, Any]] = []

        # Check 1: Time scope enforcement
        ts = contract.get("time_scope", "none")
        if ts.startswith("month_year:"):
            parts = ts.split(":")[1].split("-")
            if len(parts) == 2:
                year_str, month_str = parts
                has_year = f"= {year_str}" in sql or f"={year_str}" in sql
                has_month = f"= {month_str}" in sql or f"={month_str}" in sql
                if not has_year:
                    violations.append(f"Contract requires year={year_str} but SQL missing year predicate")
                if not has_month:
                    violations.append(f"Contract requires month={month_str} but SQL missing month predicate")
                checks.append({"name": "time_year", "passed": has_year, "detail": f"year={year_str}"})
                checks.append({"name": "time_month", "passed": has_month, "detail": f"month={month_str}"})
        elif ts.startswith("year:"):
            year_str = ts.split(":")[1]
            has_year = f"= {year_str}" in sql or f"={year_str}" in sql
            if not has_year:
                violations.append(f"Contract requires year={year_str} but SQL missing year predicate")
            checks.append({"name": "time_year", "passed": has_year, "detail": f"year={year_str}"})
        elif ts.startswith("future_blocked:"):
            if "FALSE" not in sql_upper:
                violations.append("Contract is future_blocked but SQL does not contain FALSE predicate")
            checks.append({"name": "future_block", "passed": "FALSE" in sql_upper, "detail": ts})

        # Check 2: Grouping fidelity
        dims = contract.get("dimensions", [])
        if dims and contract.get("intent") in ("grouped_metric", "trend_analysis"):
            has_group = "GROUP BY" in sql_upper
            if not has_group:
                violations.append(f"Contract specifies dimensions {dims} but SQL has no GROUP BY")
            checks.append({"name": "grouping", "passed": has_group, "detail": f"dims={dims}"})

        # Check 3: Metric grounding
        metric_expr = contract.get("metric_expr", "")
        if metric_expr and metric_expr != "N/A":
            # Extract key function/column from metric_expr
            grounded = any(
                tok.lower() in sql_lower
                for tok in re.findall(r'[a-zA-Z_]\w+', metric_expr)
                if len(tok) > 2 and tok.upper() not in (
                    "COUNT", "SUM", "AVG", "MIN", "MAX", "CASE", "WHEN",
                    "THEN", "ELSE", "END", "DISTINCT", "CAST", "VARCHAR",
                    "COALESCE", "TRUE", "FALSE",
                )
            )
            checks.append({"name": "metric_grounding", "passed": grounded, "detail": metric_expr})
            if not grounded:
                violations.append(f"Metric expression '{metric_expr}' not grounded in SQL")

        secondary_metric_expr = contract.get("secondary_metric_expr", "")
        if secondary_metric_expr and secondary_metric_expr != "N/A":
            secondary_grounded = any(
                tok.lower() in sql_lower
                for tok in re.findall(r"[a-zA-Z_]\w+", str(secondary_metric_expr))
                if len(tok) > 2 and tok.upper() not in (
                    "COUNT", "SUM", "AVG", "MIN", "MAX", "CASE", "WHEN",
                    "THEN", "ELSE", "END", "DISTINCT", "CAST", "VARCHAR",
                    "COALESCE", "TRUE", "FALSE",
                )
            )
            checks.append(
                {
                    "name": "secondary_metric_grounding",
                    "passed": secondary_grounded,
                    "detail": secondary_metric_expr,
                }
            )
            if not secondary_grounded:
                violations.append(
                    f"Secondary metric expression '{secondary_metric_expr}' not grounded in SQL"
                )

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "checks": checks,
            "contract": contract,
        }

    def _deterministic_runtime(self, runtime: RuntimeSelection | None) -> RuntimeSelection:
        snapshot = deterministic_runtime_snapshot(
            requested_mode=runtime.requested_mode if runtime else "deterministic",
            mode=runtime.mode if runtime else "deterministic",
            provider=runtime.provider if runtime else None,
        )
        return RuntimeSelection(
            requested_mode=snapshot.requested_mode,
            mode=snapshot.mode,
            use_llm=snapshot.use_llm,
            provider=snapshot.provider,
            reason=snapshot.reason,
            intent_model=None,
            narrator_model=None,
        )

    def _enforce_contract_guard(
        self,
        *,
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        specialist_findings: list[dict[str, Any]],
        runtime: RuntimeSelection | None,
        catalog: dict[str, Any] | None,
        trace: list[dict[str, Any]] | None = None,
        stage: str = "pre_execution",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Validate contract pre-execution and deterministically recompile on drift."""
        contract = self._build_contract_spec(plan)
        validation = self._validate_contract_against_sql(contract, str(query_plan.get("sql") or ""), plan)
        if validation.get("valid"):
            return query_plan, contract, validation

        violation_text = "; ".join(validation.get("violations") or []) or "contract drift detected"
        self._pipeline_warnings.append(f"[ContractGuard:{stage}] {violation_text}")

        det_runtime = self._deterministic_runtime(runtime)
        recomp_plan = self._query_engine_agent(plan, specialist_findings, det_runtime, catalog)
        recomp_validation = self._validate_contract_against_sql(
            contract, str(recomp_plan.get("sql") or ""), plan
        )
        if recomp_validation.get("valid"):
            if trace is not None:
                trace.append(
                    {
                        "agent": "ContractGuardAgent",
                        "role": stage,
                        "status": "success",
                        "duration_ms": 0.0,
                        "summary": "deterministic SQL recompile replaced contract-drifting SQL",
                    }
                )
            self._pipeline_warnings.append(
                f"[ContractGuard:{stage}] deterministic recompile applied to enforce contract."
            )
            return recomp_plan, contract, recomp_validation

        self._pipeline_warnings.append(
            f"[ContractGuard:{stage}] deterministic recompile still violated contract; continuing with original SQL."
        )
        if trace is not None:
            trace.append(
                {
                    "agent": "ContractGuardAgent",
                    "role": stage,
                    "status": "failed",
                    "duration_ms": 0.0,
                    "summary": "contract drift unresolved after deterministic recompile",
                }
            )
        return query_plan, contract, validation

    def _build_decision_flow(
        self,
        goal: str,
        intake: dict[str, Any],
        contract: dict[str, Any],
        contract_validation: dict[str, Any],
        plan: dict[str, Any],
        query_plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
        autonomy_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the 'Explain yourself' decision-flow timeline (FR-6).

        Returns a list of decision steps, each containing:
        - step: step name
        - description: what happened
        - details: relevant data
        """
        flow: list[dict[str, Any]] = []
        canonical_id = str(contract.get("canonical_metric_id") or "").strip()
        binding_description = (
            f"Bound to contract: {contract.get('metric')} on {contract.get('table')} [{contract.get('time_scope')}]"
        )
        if canonical_id:
            binding_description += f" [canonical={canonical_id}]"

        # Step 1: Question understanding
        flow.append({
            "step": "question_understanding",
            "description": f"Interpreted: intent={intake.get('intent')}, domain={intake.get('domain')}, metric={intake.get('metric')}",
            "details": {
                "original_goal": goal,
                "resolved_intent": intake.get("intent"),
                "resolved_domain": intake.get("domain"),
                "resolved_metric": intake.get("metric"),
                "time_filter": intake.get("time_filter"),
                "dimensions": intake.get("dimensions", []),
                "canonical_metric_id": intake.get("canonical_metric_id"),
                "secondary_canonical_metric_id": intake.get("secondary_canonical_metric_id"),
            },
        })

        # Step 2: Contract binding
        flow.append({
            "step": "contract_binding",
            "description": binding_description,
            "details": contract,
        })

        # Step 3: Contract validation
        cv_status = "PASSED" if contract_validation.get("valid") else f"FAILED: {contract_validation.get('violations', [])}"
        flow.append({
            "step": "contract_validation",
            "description": f"SQL contract check: {cv_status}",
            "details": contract_validation,
        })

        # Step 4: SQL generation
        flow.append({
            "step": "sql_generation",
            "description": f"Generated SQL against {plan.get('table', '?')}",
            "details": {
                "sql": query_plan.get("sql", ""),
                "table": plan.get("table"),
                "metric_expr": plan.get("metric_expr"),
            },
        })

        # Step 5: Execution result
        flow.append({
            "step": "execution",
            "description": f"{'Success' if execution.get('success') else 'Failed'}: {execution.get('row_count', 0)} rows in {execution.get('execution_time_ms', 0):.0f}ms",
            "details": {
                "success": execution.get("success"),
                "row_count": execution.get("row_count", 0),
                "execution_time_ms": execution.get("execution_time_ms", 0),
                "error": execution.get("error"),
            },
        })

        # Step 6: Audit checks
        audit_checks = audit.get("checks", [])
        passed = sum(1 for c in audit_checks if c.get("passed"))
        total = len(audit_checks)
        flow.append({
            "step": "audit_checks",
            "description": f"Audit: {passed}/{total} checks passed, score={audit.get('score', 0):.2f}",
            "details": {
                "checks": audit_checks,
                "score": audit.get("score"),
                "warnings": audit.get("warnings", []),
                "grounding": audit.get("grounding", {}),
            },
        })

        # Step 7: Confidence decomposition
        decomp = autonomy_result.get("confidence_decomposition", [])
        flow.append({
            "step": "confidence_decomposition",
            "description": f"Final confidence: {audit.get('score', 0):.2f} with {len(decomp)} factors",
            "details": {
                "score": audit.get("score"),
                "decomposition": decomp,
                "pipeline_warnings": list(getattr(self, "_pipeline_warnings", [])),
            },
        })

        # ── G4: Attach structured explain-yourself panel ──────────────
        try:
            panel = build_explain_yourself(
                goal=goal,
                intent={"type": intake.get("intent", "unknown"), "confidence": 0.8, "rationale": "Pipeline classification"},
                contract=contract,
                contract_validation=contract_validation,
                sql=query_plan.get("sql", ""),
                audit=audit,
                confidence=autonomy_result.get("confidence_decomposition_detail"),
                narrative=autonomy_result.get("narrative", ""),
                answer_summary=autonomy_result.get("answer_summary", ""),
            )
            flow.append({
                "step": "explain_yourself_panel",
                "description": f"Explainability panel completeness: {panel.completeness:.0%}",
                "details": panel.to_dict(),
            })
        except Exception as _ep_exc:
            self._pipeline_warnings.append(
                f"Explain-yourself panel failed ({type(_ep_exc).__name__}); "
                f"panel omitted from decision flow."
            )

        return flow

    def _build_kpi_decomposition(
        self,
        *,
        plan: dict[str, Any],
        execution: dict[str, Any],
    ) -> dict[str, Any]:
        """Build KPI decomposition tree with owner/target enrichment."""
        return build_kpi_decomposition_payload(plan=plan, execution=execution)

    def _parse_scope_from_sql(self, sql: str) -> dict[str, Any]:
        text = str(sql or "").strip()
        if not text:
            return {}

        table = ""
        match_table = re.search(r"\bFROM\s+([a-zA-Z_][\w]*)\b", text, re.IGNORECASE)
        if match_table:
            table = match_table.group(1)
        domain = next((d for d, mart in DOMAIN_TO_MART.items() if mart == table), "")

        years = [int(y) for y in re.findall(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+[^)]+\)\s*=\s*(20\d{2})", text, re.IGNORECASE)]
        year = years[0] if years else None
        time_filter: dict[str, Any] | None = None
        month_match = re.search(
            r"EXTRACT\s*\(\s*MONTH\s+FROM\s+[^)]+\)\s*=\s*([0-9]{1,2})",
            text,
            re.IGNORECASE,
        )
        month_in_match = re.search(
            r"EXTRACT\s*\(\s*MONTH\s+FROM\s+[^)]+\)\s+IN\s*\(([^)]+)\)",
            text,
            re.IGNORECASE,
        )
        if month_match:
            month = int(month_match.group(1))
            if 1 <= month <= 12:
                time_filter = {"kind": "month_year", "month": month, "year": year}
        elif month_in_match and year is not None:
            months = [
                int(tok.strip())
                for tok in month_in_match.group(1).split(",")
                if tok.strip().isdigit()
            ]
            months = [m for m in months if 1 <= m <= 12]
            if len(months) >= 3:
                time_filter = {
                    "kind": "month_list",
                    "months": sorted(set(months)),
                    "year": int(year),
                }
            elif len(months) >= 2:
                time_filter = {
                    "kind": "explicit_comparison",
                    "month_a": min(months),
                    "month_b": max(months),
                    "year": int(year),
                }
            elif len(months) == 1:
                time_filter = {"kind": "month_year", "month": months[0], "year": int(year)}
        elif year is not None:
            time_filter = {"kind": "year_only", "year": int(year)}

        value_filters: list[dict[str, str]] = []
        for col in ("has_mt103", "has_refund", "is_university"):
            pat = rf"{col}[^\\n]*LOWER\\('([^']+)'\\)"
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                value_filters.append({"column": col, "value": str(m.group(1)).strip().lower()})

        dimensions: list[str] = []
        select_match = re.search(r"\bSELECT\b(.+?)\bFROM\b", text, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_part = select_match.group(1)
            aliases = re.findall(r'AS\s+"([^"]+)"', select_part, flags=re.IGNORECASE)
            aliases += re.findall(r"\bAS\s+([a-zA-Z_][\w]*)", select_part, flags=re.IGNORECASE)
            for alias in aliases:
                clean = str(alias).strip()
                if clean in {"metric_value", "secondary_metric_value", "period", "primary_metric", "secondary_metric"}:
                    continue
                if clean in {"month_bucket", "time_bucket"}:
                    clean = "__month__"
                if clean and clean not in dimensions:
                    dimensions.append(clean)

        top_n = None
        limit_match = re.search(r"\bLIMIT\s+([0-9]+)", text, re.IGNORECASE)
        if limit_match:
            top_n = max(1, min(100, int(limit_match.group(1))))

        return {
            "domain": domain,
            "table": table,
            "time_filter": time_filter,
            "value_filters": value_filters,
            "dimensions": dimensions,
            "top_n": top_n,
            "has_secondary_metric": ("secondary_metric_value" in text.lower()),
        }

    def _apply_conversation_scope_hints(
        self,
        goal: str,
        parsed: dict[str, Any],
        conversation_context: list[dict[str, Any]],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        if not conversation_context:
            return parsed

        lower = goal.lower().strip()
        tokens = re.findall(r"[a-z0-9_]+", lower)
        explicit_subject_terms = [
            "transaction",
            "quote",
            "booking",
            "customer",
            "mt103",
            "refund",
            "markup",
            "forex",
            "currency",
            "platform",
            "deal type",
            "country",
            "state",
        ]
        has_explicit_subject = any(term in lower for term in explicit_subject_terms)
        has_followup_phrase = any(
            marker in lower
            for marker in [
                "now",
                "also",
                "same",
                "keep",
                "that",
                "those",
                "include",
                "switch",
                "instead",
                "continue",
            ]
        )
        followup = bool(
            has_followup_phrase
            or lower.startswith(("now ", "also ", "and ", "then ", "for that "))
            or (len(tokens) <= 10 and not has_explicit_subject)
        )
        if not followup:
            return parsed

        last_sql = ""
        for turn in reversed(conversation_context):
            candidate = str(turn.get("sql") or "").strip()
            if candidate:
                last_sql = candidate
                break
        if not last_sql:
            return parsed

        scope = self._parse_scope_from_sql(last_sql)
        if not scope:
            return parsed

        merged = dict(parsed)
        if not has_explicit_subject and scope.get("domain"):
            merged["domain"] = str(scope["domain"])
            merged["domains_detected"] = [str(scope["domain"])]
            merged["secondary_domains"] = []

        sticky_scope = any(
            marker in lower
            for marker in [
                "same filter",
                "same filters",
                "same slice",
                "same scope",
                "keep same",
                "keep that",
                "keep the same",
                "that same",
            ]
        ) or lower.startswith(("now ", "also ", "then "))

        if sticky_scope:
            has_explicit_time_in_goal = bool(
                re.search(
                    r"\b(20\d{2}|q[1-4]|quarter|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|month|year|this month|last month|this year|last year)\b",
                    lower,
                )
            )
            if scope.get("time_filter") and (not has_explicit_time_in_goal or not merged.get("time_filter")):
                merged["time_filter"] = scope.get("time_filter")
            if not merged.get("value_filters") and scope.get("value_filters"):
                merged["value_filters"] = list(scope.get("value_filters") or [])
            if scope.get("dimensions"):
                domain_cols = set(
                    catalog.get("marts", {}).get(DOMAIN_TO_MART.get(merged.get("domain", ""), ""), {}).get("columns", [])
                )
                scope_dims = [
                    d for d in (scope.get("dimensions") or [])
                    if d in {"__month__", "currency_pair"} or d in domain_cols
                ]
                if scope_dims:
                    merged_dims = list(merged.get("dimensions") or [])
                    preserve_dimension_intent = any(
                        marker in lower
                        for marker in [
                            "keep top",
                            "same split",
                            "same breakdown",
                            "top currencies",
                            "top platforms",
                            "same grouping",
                        ]
                    )
                    if preserve_dimension_intent and scope_dims:
                        merged["dimensions"] = scope_dims[:MAX_DIMENSIONS]
                        merged["dimension"] = merged["dimensions"][0] if merged["dimensions"] else None
                        if merged.get("intent") in {"metric", "lookup"} and merged.get("dimensions"):
                            merged["intent"] = "grouped_metric"
                    elif not merged_dims:
                        for d in scope_dims:
                            if d not in merged_dims:
                                merged_dims.append(d)
                        merged["dimensions"] = merged_dims[:MAX_DIMENSIONS]
                        merged["dimension"] = merged["dimensions"][0] if merged["dimensions"] else None
                        if merged.get("intent") in {"metric", "lookup"} and merged.get("dimensions"):
                            merged["intent"] = "grouped_metric"
            if not merged.get("top_n") and scope.get("top_n"):
                merged["top_n"] = int(scope["top_n"])

        wants_amount = self._goal_has_amount_intent(lower)
        wants_count = self._goal_has_count_intent(lower)
        if ("include" in lower or "also" in lower or "too" in lower) and wants_amount:
            domain = str(merged.get("domain") or "")
            if domain == "transactions":
                if not wants_count and str(merged.get("metric") or "").endswith("_count"):
                    wants_count = True
                if wants_count and not str(merged.get("secondary_metric") or "").strip():
                    if "mt103" in lower and not any(
                        str(vf.get("column") or "") == "has_mt103"
                        for vf in (merged.get("value_filters") or [])
                    ):
                        merged.setdefault("value_filters", []).append({"column": "has_mt103", "value": "true"})
                    merged["secondary_metric"] = "total_amount"
                    merged["metric_bundle"] = "count_plus_amount"
            elif domain == "quotes" and wants_count and not str(merged.get("secondary_metric") or "").strip():
                merged["secondary_metric"] = "total_quote_value"
                merged["metric_bundle"] = "count_plus_amount"
            elif domain == "bookings" and wants_count and not str(merged.get("secondary_metric") or "").strip():
                merged["secondary_metric"] = "total_booked_amount"
                merged["metric_bundle"] = "count_plus_amount"

        return merged

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
        has_explicit_time_scope = bool(
            re.search(
                r"\b(this month|last month|this year|last year|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|20\d{2})\b",
                lower,
            )
        )
        has_explicit_grouping = any(token in lower for token in [" by ", "split", "breakdown", "wise", "per ", "group"])
        has_explicit_filtering = any(token in lower for token in [" only ", " where ", "status", "platform", "state", "country"])
        best = memory_hints[0]
        if float(best.get("similarity", 0.0)) < 0.55:
            return parsed

        # Learned corrections are only safe to replay when the current goal is
        # sparse and does not carry explicit scope/group/filter constraints.
        replay_safe = not (has_explicit_time_scope or has_explicit_grouping or has_explicit_filtering)

        # GAP 41a: For explicit queries, still apply learned corrections if available
        if has_explicit_subject:
            if (
                not replay_safe
                or not best.get("correction_applied")
                or float(best.get("similarity", 0)) < 0.82
            ):
                return parsed
            if parsed.get("dimensions") or parsed.get("time_filter") or parsed.get("value_filters"):
                return parsed
            # Apply learned correction from a previous corrected run
            merged = dict(parsed)
            past_metric = str(best.get("metric") or "").strip()
            if past_metric and str(merged.get("metric") or "").strip() in {"", "transaction_count"}:
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
        if past_metric and merged.get("metric") == "transaction_count" and replay_safe:
            merged["metric"] = past_metric
        if past_dims and not merged.get("dimensions") and replay_safe:
            merged["dimensions"] = past_dims[:MAX_DIMENSIONS]
            merged["dimension"] = merged["dimensions"][0]
        if replay_safe and not merged.get("time_filter") and isinstance(best.get("time_filter"), dict):
            merged["time_filter"] = best["time_filter"]
        if replay_safe and not merged.get("value_filters") and isinstance(best.get("value_filters"), list):
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
        secondary = []
        for domain in intake.get("secondary_domains", []):
            sec_table = DOMAIN_TO_MART.get(domain)
            if not sec_table:
                continue
            sec_meta = (catalog.get("marts") or {}).get(sec_table, {})
            secondary.append(
                {
                    "domain": domain,
                    "table": sec_table,
                    "row_count": int(sec_meta.get("row_count") or 0),
                    "columns": list(sec_meta.get("columns") or []),
                    "metrics": catalog.get("metrics_by_table", {}).get(sec_table, {}),
                }
            )
        return {
            "domain": intake["domain"],
            "table": table,
            "columns": table_meta["columns"],
            "row_count": table_meta["row_count"],
            "metrics": catalog["metrics_by_table"][table],
            "preferred_time_column": catalog["preferred_time_column"].get(table),
            "preferred_time_by_metric": catalog.get("preferred_time_column_by_metric", {}).get(table, {}),
            "secondary": secondary,
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
        secondary_metric = str(intake.get("secondary_metric") or "").strip() or None
        secondary_metric_expr: str | None = None
        if secondary_metric:
            secondary_metric_expr = retrieval["metrics"].get(secondary_metric)
            if secondary_metric_expr is None:
                self._pipeline_warnings.append(
                    f"Secondary metric '{secondary_metric}' is not available in {retrieval['table']} and was ignored."
                )
                secondary_metric = None

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

            has_refund_bool = any(
                str(vf.get("column") or "") == "has_refund"
                and str(vf.get("value") or "").lower() == "true"
                for vf in value_filters
            )
            if has_refund_bool:
                value_filters = [
                    vf
                    for vf in value_filters
                    if not (
                        str(vf.get("column") or "") == "payment_status"
                        and "refund" in str(vf.get("value") or "").lower()
                    )
                ]

        deterministic_plan = {
            "goal": intake.get("goal", ""),
            "intent": intake["intent"],
            "table": retrieval["table"],
            "metric": intake["metric"],
            "metric_expr": metric_expr,
            "secondary_metric": secondary_metric,
            "secondary_metric_expr": secondary_metric_expr,
            "metric_bundle": intake.get("metric_bundle"),
            "dimension": dimension,
            "dimensions": dimensions,
            "available_columns": list(
                dict.fromkeys(
                    list(retrieval.get("columns", []))
                    + [
                        col
                        for sec in (retrieval.get("secondary") or [])
                        for col in (sec.get("columns") or [])
                    ]
                )
            ),
            "available_columns_by_table": {
                retrieval.get("table"): list(retrieval.get("columns", [])),
                **{
                    sec.get("table"): list(sec.get("columns") or [])
                    for sec in (retrieval.get("secondary") or [])
                    if sec.get("table")
                },
            },
            "secondary_domains": intake.get("secondary_domains", []),
            "time_column": retrieval.get("preferred_time_by_metric", {}).get(
                intake["metric"],
                retrieval.get("preferred_time_column"),
            ),
            "time_filter": intake.get("time_filter"),
            "value_filters": value_filters,
            "top_n": max(1, min(100, int(intake.get("top_n", 20)))),
            "definition_used": (
                (
                    f"{intake['metric']}"
                    + (
                        f" + {secondary_metric}"
                        if secondary_metric
                        else ""
                    )
                )
                + f" on {retrieval['table']}"
                + (f" grouped by {', '.join(dimensions)}" if dimensions else "")
            ),
            "row_count_hint": retrieval["row_count"],
        }
        deterministic_plan = self._apply_organizational_controls(
            intake.get("goal", ""),
            intake,
            deterministic_plan,
            catalog,
        )

        if runtime and runtime.use_llm and runtime.provider and not intake.get("_llm_intake_used"):
            llm_refinement = self._planning_agent_with_llm(
                deterministic_plan, retrieval, catalog, runtime
            )
            if llm_refinement:
                deterministic_plan["_llm_planning_reasoning"] = llm_refinement.get("reasoning", "")
                # Apply LLM suggestions only if they reference valid catalog entries
                dual_metric_locked = bool(deterministic_plan.get("secondary_metric"))
                if (
                    not dual_metric_locked
                    and llm_refinement.get("metric")
                    and llm_refinement["metric"] in retrieval["metrics"]
                ):
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

        # Keep currency-pair semantics stable across providers/modes.
        goal_lower = str(intake.get("goal") or "").lower()
        if intake.get("domain") == "quotes" and any(
            term in goal_lower for term in ("currency pair", "currency combination", "most frequent pair")
        ):
            deterministic_plan["intent"] = "grouped_metric"
            deterministic_plan["dimensions"] = ["currency_pair"]
            deterministic_plan["dimension"] = "currency_pair"
            if "quote_count" in retrieval.get("metrics", {}):
                deterministic_plan["metric"] = "quote_count"
                deterministic_plan["metric_expr"] = retrieval["metrics"]["quote_count"]

        deterministic_plan = self._apply_organizational_controls(
            intake.get("goal", ""),
            intake,
            deterministic_plan,
            catalog,
        )
        deterministic_plan = self._bind_canonical_metric_contracts(deterministic_plan, catalog)
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
                "objective_coverage_pct": float((base_breakdown.get("objective_coverage") or {}).get("coverage_pct", 0.0)),
                "objective_failures": list((base_breakdown.get("objective_coverage") or {}).get("failed_objectives", [])),
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
                "hard_gates": {
                    "objective_coverage_gate": True,
                    "contradiction_gate": True,
                    "metric_family_gate": True,
                    "base_metric_family": metric_family(str(base_plan.get("metric") or "")),
                    "selected_metric_family": metric_family(str(base_plan.get("metric") or "")),
                },
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
                query_plan = self._query_engine_agent(
                    candidate,
                    specialist_findings or [],
                    catalog=catalog,
                )
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
                        "objective_coverage_pct": float((breakdown.get("objective_coverage") or {}).get("coverage_pct", 0.0)),
                        "objective_failures": list((breakdown.get("objective_coverage") or {}).get("failed_objectives", [])),
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
        goal_lower = str(goal or "").lower()
        amount_only_intent = self._goal_has_amount_intent(goal_lower) and not self._goal_has_count_intent(goal_lower)
        dual_metric_intent = self._goal_has_amount_intent(goal_lower) and self._goal_has_count_intent(goal_lower)
        base_metric_name = str(base_plan.get("metric") or "")
        selected_metric_name = str((selected.get("plan") or {}).get("metric") or "")
        selected_secondary_metric = str((selected.get("plan") or {}).get("secondary_metric") or "").strip()
        base_requires_dual = dual_metric_intent and str(base_plan.get("metric_bundle") or "") == "count_plus_amount"
        contradiction_resolution = self._resolve_candidate_contradictions(candidate_evals)
        selected_objective = (selected.get("score_breakdown") or {}).get("objective_coverage", {})
        selected_coverage_pct = float(selected_objective.get("coverage_pct") or 0.0)
        selected_failed_objectives = int(selected_objective.get("failed_count") or 0)
        objective_gate_pass = selected_coverage_pct >= 70.0 and selected_failed_objectives <= 2
        contradiction_gate_pass = not bool(contradiction_resolution.get("needs_clarification"))
        base_family = metric_family(base_metric_name)
        selected_family = metric_family(selected_metric_name)
        explicit_family_expected = (
            "count" if self._goal_has_count_intent(goal_lower) and not self._goal_has_amount_intent(goal_lower) else (
                "amount" if self._goal_has_amount_intent(goal_lower) and not self._goal_has_count_intent(goal_lower) else ""
            )
        )
        metric_family_gate_pass = True
        if explicit_family_expected and selected_family != explicit_family_expected:
            metric_family_gate_pass = False
        if base_family != "unknown" and selected_family != "unknown":
            if base_family != selected_family and not dual_metric_intent and not explicit_family_expected:
                metric_family_gate_pass = False

        hard_reject_autonomy_switch = False
        if amount_only_intent and not base_metric_name.endswith("_count") and selected_metric_name.endswith("_count"):
            hard_reject_autonomy_switch = True
            self._pipeline_warnings.append(
                "Autonomy switch rejected: amount-intent query cannot downgrade to a count metric."
            )
        if base_requires_dual and not selected_secondary_metric:
            hard_reject_autonomy_switch = True
            self._pipeline_warnings.append(
                "Autonomy switch rejected: dual-metric query cannot drop secondary metric."
            )
        if not objective_gate_pass:
            hard_reject_autonomy_switch = True
            self._pipeline_warnings.append(
                "Autonomy switch rejected: candidate objective coverage did not pass hard gate."
            )
        if not contradiction_gate_pass:
            hard_reject_autonomy_switch = True
            self._pipeline_warnings.append(
                "Autonomy switch rejected: contradiction severity requires clarification."
            )
        if not metric_family_gate_pass:
            hard_reject_autonomy_switch = True
            self._pipeline_warnings.append(
                "Autonomy switch rejected: metric-family lock prevented unsafe metric-type drift."
            )

        if (
            selected is not None
            and not hard_reject_autonomy_switch
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
            "objective_coverage": selected.get("score_breakdown", {}).get("objective_coverage", {}),
            "evaluated_candidates": candidate_evals,
            "confidence_decomposition": candidate_evals,
            "contradiction_resolution": contradiction_resolution,
            "hard_gates": {
                "objective_coverage_gate": objective_gate_pass,
                "contradiction_gate": contradiction_gate_pass,
                "metric_family_gate": metric_family_gate_pass,
                "base_metric_family": base_family,
                "selected_metric_family": selected_family,
            },
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
        base_metric = str(base_plan.get("metric") or "").strip()
        explicit_metric_intent = (
            self._goal_has_count_intent(lower)
            or self._goal_has_amount_intent(lower)
            or any(
                k in lower
                for k in [
                    "avg",
                    "average",
                    "mean",
                    "rate",
                    "ratio",
                    "exchange rate",
                    "markup",
                    "charge",
                    "charges",
                    "fee",
                    "fees",
                ]
            )
        )
        explicit_dimension_intent = bool(
            re.search(r"\b(by|split|breakdown|group(?:ed)?|top)\b", lower)
            and any(
                k in lower
                for k in [
                    "platform",
                    "currency",
                    "currencies",
                    "country",
                    "state",
                    "deal type",
                    "type",
                ]
            )
        )
        explicit_time_intent = bool(
            re.search(
                r"\b(20\d{2}|q[1-4]|quarter|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|this month|last month|this year|last year)\b",
                lower,
            )
        )
        explicit_dual_metric_intent = self._goal_has_count_intent(lower) and self._goal_has_amount_intent(lower)
        dual_metric_locked = (
            explicit_dual_metric_intent
            and str(base_plan.get("metric_bundle") or "") == "count_plus_amount"
        )
        base_dims = [d for d in (base_plan.get("dimensions") or []) if isinstance(d, str)]
        base_dim_set = set(base_dims)
        generic_rule_keywords = {
            "quote",
            "quotes",
            "transaction",
            "transactions",
            "customer",
            "customers",
            "booking",
            "bookings",
            "data",
        }

        for rule in learned_corrections[:3]:
            match_score = float(rule.get("match_score", 0.0) or 0.0)
            keyword = str(rule.get("keyword") or "").strip().lower()
            # Prevent unrelated learned rules from hijacking explicit goals.
            if match_score < 0.55 and (not keyword or keyword not in lower):
                continue
            target_table = str(rule.get("target_table") or "").strip()
            target_metric = str(rule.get("target_metric") or "").strip()
            target_dims = list(rule.get("target_dimensions") or [])
            if not target_table or not target_metric:
                continue
            if dual_metric_locked and target_metric != base_metric:
                continue
            if not explicit_dimension_intent and not base_dim_set and target_dims:
                continue
            if explicit_dimension_intent and base_dim_set and target_dims:
                target_dim_set = {str(d) for d in target_dims if str(d)}
                if target_dim_set and target_dim_set != base_dim_set:
                    continue
            keyword_tokens = [t for t in re.findall(r"[a-z0-9_]+", keyword) if t]
            keyword_has_specificity = len([t for t in keyword_tokens if len(t) >= 4 and t not in generic_rule_keywords]) >= 1
            if keyword in generic_rule_keywords and target_metric != base_metric:
                continue
            # Explicit metric requests should not be overridden by weak/generic replay rules.
            if (
                explicit_metric_intent
                and base_metric
                and target_metric != base_metric
                and (match_score < 0.92 or not keyword_has_specificity)
            ):
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
            if explicit_metric_intent and base_metric and metric and metric != base_metric:
                continue
            dims = list(hint.get("dimensions") or [])
            if not explicit_dimension_intent and not base_dim_set and dims:
                continue
            if explicit_dimension_intent and base_dim_set and dims:
                hint_dim_set = {str(d) for d in dims if str(d)}
                if hint_dim_set and hint_dim_set != base_dim_set:
                    continue
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

        quote_signal = any(k in lower for k in ["quote", "quotes"])
        explicit_quote_count_intent = quote_signal and self._goal_has_count_intent(lower) and not self._goal_has_amount_intent(lower)
        quote_value_signal = any(k in lower for k in ["forex", "markup", "charges", "fx"])
        if quote_value_signal or (quote_signal and not explicit_quote_count_intent):
            if explicit_dual_metric_intent and str(base_plan.get("metric_bundle") or "") == "count_plus_amount":
                pass
            else:
                metric = "forex_markup_revenue"
                if any(k in lower for k in ["avg", "average", "mean"]):
                    metric = "avg_forex_markup"
                if any(
                    k in lower
                    for k in [
                        "additional charge",
                        "additional charges",
                        "total charges",
                        "service charge",
                        "service charges",
                        "swift charge",
                        "swift charges",
                        "platform charge",
                        "platform charges",
                    ]
                ):
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
            has_dual_booking_intent = self._goal_has_count_intent(lower) and self._goal_has_amount_intent(lower)
            if has_dual_booking_intent:
                metric = "booking_count"
            else:
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

        _base_tf = base_plan.get("time_filter")
        _is_future_blocked = isinstance(_base_tf, dict) and _base_tf.get("kind") == "future_blocked"
        if (
            int(base_execution.get("row_count") or 0) == 0
            and _base_tf
            and not _is_future_blocked
            and not explicit_time_intent
        ):
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
        by_table = dict(plan.get("available_columns_by_table") or {})
        by_table[target_table] = list(marts[target_table].get("columns", []))
        plan["available_columns_by_table"] = by_table
        plan["row_count_hint"] = int(marts[target_table].get("row_count", 0))
        plan["time_column"] = catalog.get("preferred_time_column", {}).get(target_table)

        available_metrics = metrics_by_table[target_table]
        target_metric = metric or str(plan.get("metric") or "")
        if target_metric not in available_metrics:
            target_metric = next(iter(available_metrics))
        plan["metric"] = target_metric
        plan["metric_expr"] = available_metrics[target_metric]
        target_secondary_metric = str(plan.get("secondary_metric") or "").strip()
        if target_secondary_metric == target_metric:
            target_secondary_metric = ""
        if target_secondary_metric and target_secondary_metric in available_metrics:
            plan["secondary_metric"] = target_secondary_metric
            plan["secondary_metric_expr"] = available_metrics[target_secondary_metric]
            plan["metric_bundle"] = str(plan.get("metric_bundle") or "count_plus_amount")
        else:
            plan["secondary_metric"] = None
            plan["secondary_metric_expr"] = None
            plan["metric_bundle"] = None

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
        metric_label = str(plan["metric"])
        if plan.get("secondary_metric"):
            metric_label += f" + {plan['secondary_metric']}"
        plan["definition_used"] = (
            f"{metric_label} on {plan['table']}"
            + (f" grouped by {dim_text}" if dim_text else "")
        )
        plan["_variant_reason"] = reason
        return plan

    def _plan_signature(self, plan: dict[str, Any]) -> tuple[Any, ...]:
        return (
            plan.get("table"),
            plan.get("metric"),
            plan.get("secondary_metric"),
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
        objective_coverage = self._objective_coverage_matrix(goal, plan, execution, audit)
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
        if (
            "markup" in lower
            and "spend" in lower
            and "transaction" in lower
            and plan.get("intent") == "cross_domain_compare"
            and "__month__" in dims
        ):
            dimension_bonus += 0.08

        latency_penalty = 0.0
        if execution.get("execution_time_ms") and float(execution["execution_time_ms"]) > 6000:
            latency_penalty = 0.03

        governance_penalty = 0.0
        org_ctx = getattr(self, "_org_knowledge_context", {}) or {}
        if bool(org_ctx.get("enforce_mt103_validity")):
            tx_like = plan.get("table") == "datada_mart_transactions" or plan.get("compare_table") == "datada_mart_transactions"
            if tx_like:
                has_mt103_filter = any(
                    str(vf.get("column") or "").lower() == "has_mt103"
                    and str(vf.get("value") or "").lower() in {"true", "1", "yes"}
                    for vf in (plan.get("value_filters") or [])
                    if isinstance(vf, dict)
                )
                if not has_mt103_filter:
                    governance_penalty = 0.1

        objective_failed = int(objective_coverage.get("failed_count") or 0)
        objective_bonus = 0.06 * max(
            0.0,
            min(1.0, float(objective_coverage.get("coverage_pct", 0.0)) / 100.0),
        )
        objective_penalty = 0.03 * objective_failed

        raw_total = (
            audit_base
            + execution_bonus
            + non_empty_bonus
            + dimension_bonus
            + objective_bonus
            - goal_miss_penalty
            - latency_penalty
            - governance_penalty
            - objective_penalty
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
            "governance_penalty": round(governance_penalty, 4),
            "objective_bonus": round(objective_bonus, 4),
            "objective_penalty": round(objective_penalty, 4),
            "goal_term_miss_count": len(misses),
            "objective_coverage": objective_coverage,
            "total": round(total, 4),
        }
        if llm_alignment_score is not None:
            result["llm_alignment_score"] = round(llm_alignment_score, 4)
            result["deterministic_total"] = round(deterministic_total, 4)
        return result

    def _objective_coverage_matrix(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        lower = str(goal or "").lower()
        intent = str(plan.get("intent") or "").lower()
        metric = str(plan.get("metric") or "").lower()
        metric_expr = str(plan.get("metric_expr") or "").lower()
        dims = [str(d).lower() for d in (plan.get("dimensions") or []) if str(d).strip()]
        value_filters = [
            vf for vf in (plan.get("value_filters") or [])
            if isinstance(vf, dict)
        ]
        grounding = (audit.get("grounding") or {}) if isinstance(audit, dict) else {}
        checks = list(audit.get("checks") or []) if isinstance(audit, dict) else []

        has_time_words = bool(
            re.search(
                r"\b(20\d{2}|q[1-4]|quarter|month|monthly|year|weekly|daily|"
                r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
                r"this month|last month|this year|last year)\b",
                lower,
            )
        )
        requires_group = bool(
            re.search(r"\b(by|split|breakdown|group(?:ed)?|per)\b", lower)
        )
        requires_count = self._goal_has_count_intent(lower)
        requires_amount = self._goal_has_amount_intent(lower)
        requires_compare = any(tok in lower for tok in ["compare", "compared", "versus", " vs "])
        requires_cross_domain = bool(
            ("markup" in lower or "forex" in lower)
            and ("spend" in lower or "transaction" in lower)
            and requires_compare
        )
        requires_validity_guard = (
            "mt103" in lower
            or "valid transaction" in lower
            or "valid transactions" in lower
            or "customer spend on transaction" in lower
        )

        objective_rows: list[dict[str, Any]] = []

        def add_obj(name: str, required: bool, passed: bool, detail: str) -> None:
            objective_rows.append(
                {
                    "name": name,
                    "required": bool(required),
                    "passed": bool(passed) if required else None,
                    "detail": detail,
                }
            )

        metric_tokens = {metric, metric_expr}
        add_obj(
            "metric_alignment",
            True,
            (
                (requires_count and ("count" in metric or "count(" in metric_expr))
                or (requires_amount and any(tok in metric for tok in ["amount", "revenue", "spend", "value"]))
                or (
                    not requires_count
                    and not requires_amount
                    and bool(metric.strip())
                )
            ),
            f"metric={metric or 'n/a'}, expr={metric_expr or 'n/a'}",
        )
        add_obj(
            "dimension_alignment",
            requires_group,
            bool(dims),
            f"dimensions={dims}",
        )
        add_obj(
            "time_scope_alignment",
            has_time_words,
            bool(plan.get("time_filter")) or "__month__" in dims,
            f"time_filter={plan.get('time_filter')}",
        )
        add_obj(
            "comparison_alignment",
            requires_compare,
            intent in {"comparison", "cross_domain_compare"},
            f"intent={intent}",
        )
        add_obj(
            "cross_domain_alignment",
            requires_cross_domain,
            intent == "cross_domain_compare",
            f"intent={intent}",
        )
        has_mt103_filter = any(
            str(vf.get("column") or "").lower() == "has_mt103"
            and str(vf.get("value") or "").lower() in {"true", "1", "yes"}
            for vf in value_filters
        )
        add_obj(
            "validity_guard_alignment",
            requires_validity_guard,
            has_mt103_filter,
            f"filters={value_filters}",
        )
        add_obj(
            "execution_success",
            True,
            bool(execution.get("success")),
            f"success={execution.get('success')}",
        )
        add_obj(
            "result_non_empty",
            True,
            int(execution.get("row_count") or 0) > 0,
            f"row_count={execution.get('row_count')}",
        )
        concept_cov = float(grounding.get("concept_coverage_pct") or 0.0)
        schema_grounding_pass = any(
            str(c.get("name") or "") == "schema_grounding" and bool(c.get("passed"))
            for c in checks
            if isinstance(c, dict)
        )
        add_obj(
            "grounding_quality",
            True,
            concept_cov >= 66.0 and schema_grounding_pass,
            f"concept_coverage={concept_cov}, schema_grounding={schema_grounding_pass}",
        )

        required_rows = [r for r in objective_rows if r.get("required")]
        passed_rows = [r for r in required_rows if r.get("passed")]
        failed_rows = [r for r in required_rows if not r.get("passed")]
        coverage_pct = 100.0 if not required_rows else (len(passed_rows) / len(required_rows)) * 100.0

        return {
            "required_count": len(required_rows),
            "passed_count": len(passed_rows),
            "failed_count": len(failed_rows),
            "coverage_pct": round(coverage_pct, 1),
            "failed_objectives": [str(r.get("name")) for r in failed_rows],
            "objectives": objective_rows,
        }

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
        return resolve_candidate_contradictions(candidate_evals)

    def _blackboard_post(
        self,
        blackboard: list[dict[str, Any]],
        *,
        producer: str,
        artifact_type: str,
        payload: Any,
        consumed_by: list[str] | None = None,
    ) -> None:
        entry = append_blackboard_artifact(
            blackboard,
            producer=producer,
            artifact_type=artifact_type,
            payload=payload,
            consumed_by=consumed_by,
            compact=lambda value, max_len: _compact(value, max_len=max_len),
        )
        reason_codes = list((entry.get("handoff_contract") or {}).get("reason_codes") or [])
        if reason_codes:
            self._pipeline_warnings.append(
                f"Handoff contract warning for {artifact_type}: {', '.join(str(code) for code in reason_codes)}"
            )

    def _blackboard_edges(self, blackboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return blackboard_edges(blackboard)

    def _blackboard_query(
        self,
        blackboard: list[dict[str, Any]],
        *,
        producer: str | None = None,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query blackboard entries by producer and/or artifact_type."""
        return query_blackboard(blackboard, producer=producer, artifact_type=artifact_type)

    def _blackboard_latest(
        self,
        blackboard: list[dict[str, Any]],
        artifact_type: str,
    ) -> dict[str, Any] | None:
        """Return the most recent blackboard entry of a given artifact_type."""
        return latest_blackboard(blackboard, artifact_type=artifact_type)

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
            if plan.get("metric") == "transaction_id_count" or any(
                phrase in goal_lower
                for phrase in ["transaction id", "transaction ids", "distinct transaction id", "distinct transaction ids"]
            ):
                return {"active": True, "notes": notes, "warnings": warnings, "directives": directives}
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
        """Policy gate: block destructive ops, fabrication requests, and manipulation.

        FR-3: Fabrication resistance must be hard-gated.
        FR-4: Future-time requests handled downstream (future_blocked).
        """
        # ── G2: Policy-safe autonomy gates ─────────────────────────
        verdicts = run_all_policy_gates(goal)
        blocking = get_blocking_verdict(verdicts)
        if blocking:
            return {
                "allowed": False,
                "reason": blocking.reason,
                "policy": blocking.gate,
                "verdicts": [v.to_dict() for v in verdicts],
            }

        lower = goal.lower().strip()

        # Keep one explicit local gate for destructive SQL phrasing.
        blocked_keywords = ["drop table", "delete from", "truncate", "update ", "insert into", "alter table"]
        if any(k in lower for k in blocked_keywords):
            return {
                "allowed": False,
                "reason": "Destructive operations are blocked.",
                "policy": "destructive_sql",
                "verdicts": [v.to_dict() for v in verdicts],
            }

        return {"allowed": True, "reason": "ok", "policy": "none", "verdicts": [v.to_dict() for v in verdicts]}

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

    def _apply_business_rule_matches(
        self,
        plan: dict[str, Any],
        matches: list[dict[str, Any]] | None,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply matched admin business rules to the compiled plan."""
        if not matches:
            plan["_business_rules_matched"] = []
            plan["_business_rules_applied"] = []
            return {"matched": 0, "applied": 0, "rules": []}

        virtual_dims = {"__month__", "currency_pair"}
        marts = set((catalog.get("marts") or {}).keys())
        metrics_by_table = catalog.get("metrics_by_table") or {}
        applied: list[dict[str, Any]] = []
        matched: list[dict[str, Any]] = []

        for rule in matches:
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("rule_id") or "")
            rname = str(rule.get("name") or "")
            rtype = str(rule.get("rule_type") or "").strip().lower()
            payload = rule.get("action_payload") if isinstance(rule.get("action_payload"), dict) else {}
            trigger_hits = list(rule.get("trigger_hits") or [])
            matched.append(
                {
                    "rule_id": rid,
                    "name": rname,
                    "rule_type": rtype,
                    "trigger_hits": trigger_hits,
                    "match_score": float(rule.get("match_score") or 0.0),
                }
            )
            if not payload:
                continue

            applied_change: dict[str, Any] = {"rule_id": rid, "name": rname, "rule_type": rtype}

            if rtype in {"plan_override", "route_override"}:
                target_table = str(payload.get("target_table") or "").strip()
                if target_table and target_table in marts and target_table != plan.get("table"):
                    plan["table"] = target_table
                    applied_change["table"] = target_table

                target_metric = str(payload.get("target_metric") or "").strip()
                table_metrics = metrics_by_table.get(plan.get("table"), {})
                if target_metric and target_metric in table_metrics and target_metric != plan.get("metric"):
                    plan["metric"] = target_metric
                    plan["metric_expr"] = str(table_metrics[target_metric])
                    applied_change["metric"] = target_metric
                    applied_change["metric_expr"] = str(table_metrics[target_metric])

                target_dims_raw = payload.get("target_dimensions")
                if isinstance(target_dims_raw, list) and target_dims_raw:
                    available = set(plan.get("available_columns", []))
                    valid_dims = [
                        str(d).strip()
                        for d in target_dims_raw
                        if str(d).strip() and (str(d).strip() in available or str(d).strip() in virtual_dims)
                    ]
                    if valid_dims:
                        plan["dimensions"] = valid_dims[:MAX_DIMENSIONS]
                        applied_change["dimensions"] = list(plan["dimensions"])

            elif rtype == "add_filter":
                col = str(payload.get("column") or "").strip()
                value = payload.get("value")
                op = str(payload.get("operator") or "=").strip() or "="
                if col:
                    existing_cols = {str(vf.get("column") or "").strip() for vf in (plan.get("value_filters") or [])}
                    if col not in existing_cols:
                        filt = {"column": col, "operator": op, "value": str(value) if value is not None else ""}
                        plan.setdefault("value_filters", []).append(filt)
                        applied_change["filter"] = filt

            elif rtype == "override_metric_expr":
                metric_expr = str(payload.get("metric_expr") or "").strip()
                if metric_expr:
                    plan["metric_expr"] = metric_expr
                    applied_change["metric_expr"] = metric_expr

            if len(applied_change.keys()) > 3:
                applied_change["reason"] = f"Matched triggers: {', '.join(trigger_hits[:3])}"
                applied.append(applied_change)

        plan["_business_rules_matched"] = matched
        plan["_business_rules_applied"] = applied
        return {"matched": len(matched), "applied": len(applied), "rules": applied}

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
        secondary_metric_expr = str(plan.get("secondary_metric_expr") or "").strip()
        intent = plan["intent"]
        time_col = plan.get("time_column")

        where_clause = self._build_where_clause(plan, for_comparison="current")

        if intent == "cross_domain_compare":
            compare_table = str(plan.get("compare_table") or "datada_mart_transactions")
            compare_metric_expr = str(plan.get("compare_metric_expr") or "SUM(payment_amount)")
            compare_metric = str(plan.get("compare_metric") or "customer_spend")
            top_n = max(1, min(240, int(plan.get("top_n") or 120)))
            primary_alias = "q"
            compare_alias = "t"
            primary_time_col = time_col or "created_ts"
            compare_time_col = "created_ts"
            if catalog:
                compare_time_col = (
                    catalog.get("preferred_time_column_by_metric", {})
                    .get(compare_table, {})
                    .get(compare_metric)
                    or catalog.get("preferred_time_column", {}).get(compare_table)
                    or compare_time_col
                )

            def _table_where(
                table_name: str,
                alias: str,
                table_time_col: str | None,
            ) -> str:
                clauses = ["1=1"]
                if table_time_col and plan.get("time_filter"):
                    clauses.append(f"{alias}.{table_time_col} IS NOT NULL")
                    clauses.append(self._time_clause(f"{alias}.{table_time_col}", plan["time_filter"], "current"))
                filters = list(plan.get("value_filters") or [])
                table_cols = set()
                if catalog:
                    table_cols = set((catalog.get("marts", {}).get(table_name, {}) or {}).get("columns", []))
                if not table_cols:
                    table_cols = set((plan.get("available_columns_by_table") or {}).get(table_name, []) or [])
                for vf in filters:
                    if not isinstance(vf, dict):
                        continue
                    col = str(vf.get("column") or "").strip()
                    if not col:
                        continue
                    if table_cols and col not in table_cols:
                        continue
                    safe_value = str(vf.get("value") or "").replace("'", "''")
                    clauses.append(
                        f"LOWER(COALESCE(CAST({alias}.{_q(col)} AS VARCHAR), '')) = LOWER('{safe_value}')"
                    )
                return " AND ".join(clauses)

            primary_where = _table_where(table, primary_alias, primary_time_col)
            compare_where = _table_where(compare_table, compare_alias, compare_time_col)
            primary_metric_expr_q = self._qualify_metric_for_join(metric_expr, primary_alias)
            compare_metric_expr_q = self._qualify_metric_for_join(compare_metric_expr, compare_alias)

            sql = (
                "WITH primary_agg AS ("
                f" SELECT DATE_TRUNC('month', {primary_alias}.{primary_time_col}) AS month_bucket, "
                f"{primary_metric_expr_q} AS primary_metric "
                f"FROM {table} {primary_alias} WHERE {primary_where} GROUP BY 1"
                "), secondary_agg AS ("
                f" SELECT DATE_TRUNC('month', {compare_alias}.{compare_time_col}) AS month_bucket, "
                f"{compare_metric_expr_q} AS secondary_metric "
                f"FROM {compare_table} {compare_alias} WHERE {compare_where} GROUP BY 1"
                ") "
                "SELECT COALESCE(p.month_bucket, s.month_bucket) AS month_bucket, "
                "COALESCE(p.primary_metric, 0.0) AS forex_markup_revenue, "
                "COALESCE(s.secondary_metric, 0.0) AS customer_spend, "
                "CASE WHEN COALESCE(s.secondary_metric, 0.0)=0 THEN NULL "
                "ELSE COALESCE(p.primary_metric, 0.0)/NULLIF(s.secondary_metric, 0.0) END AS metric_value "
                "FROM primary_agg p FULL OUTER JOIN secondary_agg s USING(month_bucket) "
                "ORDER BY 1 ASC "
                f"LIMIT {top_n}"
            )
            return {"sql": sql, "table": table}

        if intent == "comparison":
            current_where = self._build_where_clause(plan, for_comparison="current")
            previous_where = self._build_where_clause(plan, for_comparison="comparison")
            if secondary_metric_expr:
                sql = (
                    f"SELECT 'current' AS period, {metric_expr} AS metric_value, "
                    f"{secondary_metric_expr} AS secondary_metric_value "
                    f"FROM {table} WHERE {current_where} "
                    f"UNION "
                    f"SELECT 'comparison' AS period, {metric_expr} AS metric_value, "
                    f"{secondary_metric_expr} AS secondary_metric_value "
                    f"FROM {table} WHERE {previous_where}"
                )
            else:
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
            metric_select = f"{metric_expr} AS metric_value"
            if secondary_metric_expr:
                metric_select += f", {secondary_metric_expr} AS secondary_metric_value"
            sql = (
                f"SELECT {', '.join(select_parts)}, {metric_select} "
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
                metric_select = f"{metric_expr} AS metric_value"
                if secondary_metric_expr:
                    metric_select += f", {secondary_metric_expr} AS secondary_metric_value"
                sql = f"SELECT {metric_select} FROM {table} WHERE {where_clause}"
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
                metric_select = f"{metric_expr} AS metric_value"
                if secondary_metric_expr:
                    metric_select += f", {secondary_metric_expr} AS secondary_metric_value"
                sql = f"SELECT {metric_select} FROM {table} WHERE {where_clause}"
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
                metric_select = f"{metric_expr} AS metric_value"
                if secondary_metric_expr:
                    metric_select += f", {secondary_metric_expr} AS secondary_metric_value"
                sql = f"SELECT {metric_select} FROM {table} WHERE {where_clause}"
        elif intent == "lookup":
            sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT {plan['top_n']}"
        else:
            metric_select = f"{metric_expr} AS metric_value"
            if secondary_metric_expr:
                metric_select += f", {secondary_metric_expr} AS secondary_metric_value"
            sql = f"SELECT {metric_select} FROM {table} WHERE {where_clause}"

        # ── GAP 14: Check for multi-table JOIN opportunity ──────────────────
        # Skip JOIN override for complex analytical intents that generate their own SQL.
        _no_join_override = {"comparison", "lookup", "subquery_filter", "running_total", "yoy_growth", "correlation"}
        secondary_domains = plan.get("secondary_domains", [])
        primary_cols = set((plan.get("available_columns_by_table") or {}).get(table, []) or plan.get("available_columns", []))
        virtual_dims = {"__month__", "currency_pair"}
        dims_for_join = [d for d in (plan.get("dimensions") or []) if isinstance(d, str)]
        filters_for_join = [vf for vf in (plan.get("value_filters") or []) if isinstance(vf, dict)]
        needs_secondary_join = any(d not in primary_cols and d not in virtual_dims for d in dims_for_join) or any(
            str(vf.get("column") or "").strip() not in primary_cols for vf in filters_for_join if vf.get("column")
        )
        if secondary_domains and needs_secondary_join and intent not in _no_join_override:
            for sec_domain in secondary_domains:
                sec_table = DOMAIN_TO_MART.get(sec_domain)
                if not sec_table:
                    continue
                join_key = (table, sec_table)
                join_path = JOIN_PATHS.get(join_key)
                if join_path:
                    by_table = plan.get("available_columns_by_table") or {}
                    left_cols = set((by_table.get(table) or []))
                    right_cols = set((by_table.get(sec_table) or []))
                    alias_left, alias_right = join_path.get("aliases", ("", ""))
                    alias_cols = {alias_left: left_cols, alias_right: right_cols}
                    join_terms = re.findall(r"([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)", str(join_path.get("on") or ""))
                    if join_terms and any(col not in alias_cols.get(alias, set()) for alias, col in join_terms):
                        continue
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

        # ── SPEC-1 / Phase B: SQL post-compile time scope enforcement ──────
        # Safety net: if the intake/plan specified month_year but the compiled
        # SQL only has YEAR (not MONTH), patch the SQL to include the month
        # predicate.  This catches any upstream downgrade from month_year to
        # year_only that may occur in the pipeline.
        _tf = plan.get("time_filter")
        _tc = plan.get("time_column")
        if (
            _tf
            and isinstance(_tf, dict)
            and _tf.get("kind") == "month_year"
            and _tf.get("month")
            and _tc
            and "EXTRACT(MONTH" not in sql.upper()
            and "EXTRACT(YEAR" in sql.upper()
        ):
            _patch_month = int(_tf["month"])
            _year_pattern = f"EXTRACT(YEAR FROM {_tc})"
            # Insert month predicate right after the year predicate
            _year_pred_re = re.compile(
                rf"(EXTRACT\(YEAR\s+FROM\s+{re.escape(_tc)}\)\s*=\s*\d+)",
                re.IGNORECASE,
            )
            _replacement = rf"\1 AND EXTRACT(MONTH FROM {_tc}) = {_patch_month}"
            patched_sql = _year_pred_re.sub(_replacement, sql, count=0)
            if patched_sql != sql:
                sql = patched_sql
                self._pipeline_warnings.append(
                    f"[Contract] Month predicate (={_patch_month}) injected into SQL "
                    f"— plan specified month_year but compiled SQL was missing MONTH."
                )

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
        by_table = plan.get("available_columns_by_table") or {}
        primary_cols = set(by_table.get(primary_table) or [])
        if not primary_cols:
            primary_cols = set(plan.get("available_columns", []))
        clauses = ["1=1"]
        if time_col and plan.get("time_filter"):
            clauses.append(f"{alias1}.{time_col} IS NOT NULL")
            clauses.append(self._time_clause(f"{alias1}.{time_col}", plan["time_filter"], "current"))
        for vf in plan.get("value_filters", []):
            if not isinstance(vf, dict):
                continue
            col = str(vf.get("column") or "").strip()
            if not col:
                continue
            alias = alias1 if col in primary_cols else alias2
            safe_value = str(vf.get("value", "")).replace("'", "''")
            clauses.append(
                f"LOWER(COALESCE(CAST({alias}.{_q(col)} AS VARCHAR), '')) = LOWER('{safe_value}')"
            )
        qualified_where_clause = " AND ".join(clauses)

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
                f"FROM {from_clause} WHERE {qualified_where_clause} "
                f"GROUP BY {group_by} ORDER BY {metric_idx} DESC NULLS LAST LIMIT {plan.get('top_n', 20)}"
            )
        else:
            return (
                f"SELECT {qualified_metric} AS metric_value "
                f"FROM {from_clause} WHERE {qualified_where_clause}"
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

        # ── QA-R5 Fix 2: future_blocked kind → zero rows ────────────────
        if kind == "future_blocked":
            return "FALSE"

        # ── QA-R4 Fix 6: Future-year safety net in SQL ───────────────────
        _current_year = datetime.utcnow().year
        _tf_year = time_filter.get("year")
        if _tf_year is not None and int(_tf_year) > _current_year:
            return "FALSE"

        # ── QA-R4 Fix 2b: Explicit comparison with two named months ──────
        if kind == "explicit_comparison":
            month_a = int(time_filter.get("month_a"))
            month_b = int(time_filter.get("month_b"))
            year = time_filter.get("year")
            if year is None:
                year = _current_year
            year = int(year)
            if for_comparison == "comparison":
                # Earlier month
                m = min(month_a, month_b)
            else:
                # Later month (current period)
                m = max(month_a, month_b)
            return f"EXTRACT(YEAR FROM {time_col}) = {year} AND EXTRACT(MONTH FROM {time_col}) = {m}"

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

        if kind == "month_list":
            months = [
                int(m)
                for m in (time_filter.get("months") or [])
                if str(m).isdigit() and 1 <= int(m) <= 12
            ]
            if not months:
                return "TRUE"
            year = time_filter.get("year")
            month_sql = ",".join(str(m) for m in sorted(set(months)))
            if year is None:
                return f"EXTRACT(MONTH FROM {time_col}) IN ({month_sql})"
            return (
                f"EXTRACT(YEAR FROM {time_col}) = {int(year)} "
                f"AND EXTRACT(MONTH FROM {time_col}) IN ({month_sql})"
            )

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

        # Unknown/unsupported time filter kinds should not silently force
        # rolling windows (which can fabricate scope). Fall back to no-op.
        return "1=1"

    def _execution_agent(self, query_plan: dict[str, Any]) -> dict[str, Any]:
        res = self.executor.execute(query_plan["sql"])
        analysis_rows = res.rows[:300]
        return {
            "success": res.success,
            "sql_executed": res.sql_executed,
            "error": res.error,
            "row_count": res.row_count,
            "columns": res.columns,
            "sample_rows": analysis_rows[:25],
            "analysis_rows": analysis_rows,
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

        forex_signal = "forex" in goal_text or "fx" in goal_text
        concept_expectations = {
            "markup": ["forex_markup"],
            "forex": ["forex_markup", "exchange_rate"],
            "spend": ["payment_amount", "amount", "customer_spend"],
            "charge": (
                ["total_additional_charges", "swift_charges", "platform_charges", "forex_markup"]
                if forex_signal
                else ["total_additional_charges", "swift_charges", "platform_charges"]
            ),
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
        if "markup" in goal_text and "spend" in goal_text and "transaction" in goal_text:
            if plan.get("intent") != "cross_domain_compare":
                concept_misses.append("cross_domain_compare")
                warnings.append(
                    "Goal asks for markup compared against transaction spend, "
                    "but plan is not using cross-domain comparison."
                )
            if "__month__" not in list(plan.get("dimensions") or []):
                concept_misses.append("month_split")
                warnings.append("Goal asks for month split but plan has no __month__ dimension.")

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
                    replay_match = _rows_semantically_equal(
                        replay.rows[0], execution["sample_rows"][0]
                    )
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
        # ── QA-R4 Fix 4: Recalibrated confidence baseline ───────────────
        score = 0.82
        if not execution["success"]:
            score = 0.18
        elif execution["row_count"] == 0:
            score -= 0.25
        score -= 0.10 * failed_checks
        score -= 0.10 * len(warnings)
        if not concept_check_passed:
            score = min(score, 0.40)
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

        # ── QA-R4 Fix 4b: Goal mentions time but no time filter → -0.25 ─
        _time_words = ["month", "year", "quarter", "week", "day",
                       "january", "jan", "february", "feb", "march", "mar",
                       "april", "apr", "may", "june", "jun", "july", "jul",
                       "august", "aug", "september", "sep", "sept",
                       "october", "oct", "november", "nov", "december", "dec",
                       "2024", "2025", "2026", "2027", "2028", "2029", "2030",
                       "2031", "2032", "2033", "2034", "2035",
                       "this month", "last month", "this year", "last year"]
        if any(tw in goal_text for tw in _time_words) and not plan.get("time_filter"):
            dims = list(plan.get("dimensions") or [])
            has_time_dimension = "__month__" in dims or plan.get("dimension") == "__month__"
            is_time_series_intent = plan.get("intent") in {"trend_analysis", "cross_domain_compare"}
            if not (has_time_dimension or is_time_series_intent):
                score -= 0.25
                warnings.append("Goal mentions a time period but no time filter was applied.")

        # BRD-FR4: Future-blocked queries get hard score cap
        _tf = plan.get("time_filter")
        if isinstance(_tf, dict) and _tf.get("kind") == "future_blocked":
            score = min(score, 0.25)
            warnings.append(f"Future year {_tf.get('year')} — no data exists for future dates.")

        # ── QA-R4 Fix 3b / Fix 4c / BRD-FR7: Unrecognized metric → hard cap ─
        _pw = getattr(self, "_pipeline_warnings", [])
        if any("Metric unrecognized" in w for w in _pw):
            score -= 0.30
            score = min(score, 0.49)  # FR-7: Never high-confidence on unrecognized metric
            warnings.append("Metric was not recognized in catalog; result may be a fallback.")

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
        fallback["answer_markdown"] = self._enrich_narrative_with_analyst_sections(
            str(fallback.get("answer_markdown") or ""),
            goal=goal,
            plan=plan,
            execution=execution,
            audit=audit,
        )
        org_ctx = getattr(self, "_org_knowledge_context", {}) or {}
        narrative_data_presenter = bool(org_ctx.get("narrative_data_presenter"))

        if not runtime.use_llm or not runtime.provider or not execution["success"]:
            fallback["suggested_questions"] = self._suggested_questions(plan)
            fallback["llm_narrative_used"] = False
            return fallback

        prompt = {
            "question": goal,
            "plan": {
                "table": plan["table"],
                "metric": plan["metric"],
                "canonical_metric_id": plan.get("canonical_metric_id"),
                "canonical_metric_definition": (
                    (plan.get("canonical_metric") or {}).get("business_definition")
                    if isinstance(plan.get("canonical_metric"), dict)
                    else None
                ),
                "intent": plan["intent"],
                "dimension": plan.get("dimension"),
                "dimensions": plan.get("dimensions", []),
            },
            "rows": execution["sample_rows"][:8],
            "row_count": execution["row_count"],
            "audit_warnings": audit.get("warnings", []),
            "conversation_context": conversation_context[-3:],
            "style": "storyteller" if storyteller_mode else "professional_conversational",
            "narrative_policy": (
                "data_presenter_plus_insights" if narrative_data_presenter else "default"
            ),
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
                    "PRESENTATION RULES:\n"
                    "- Always present the factual data first, then insights derived from that data\n"
                    "- Insights must cite numeric evidence from rows (no generic claims)\n"
                    "- If policy is data_presenter_plus_insights, include one pattern statement "
                    "about trend/ratio/rank backed by values\n\n"
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
                cleaned = self._enforce_narrative_goal_anchors(goal, plan, execution, cleaned)
                cleaned = self._inject_share_insight(goal, plan, execution, cleaned)
                cleaned = self._enrich_narrative_with_analyst_sections(
                    cleaned,
                    goal=goal,
                    plan=plan,
                    execution=execution,
                    audit=audit,
                )
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
                cleaned = self._enforce_narrative_goal_anchors(goal, plan, execution, cleaned)
                cleaned = self._inject_share_insight(goal, plan, execution, cleaned)
                cleaned = self._enrich_narrative_with_analyst_sections(
                    cleaned,
                    goal=goal,
                    plan=plan,
                    execution=execution,
                    audit=audit,
                )
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

    def _build_analyst_sections(
        self,
        *,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
    ) -> str:
        rows = list(execution.get("sample_rows") or [])
        dims = [str(d) for d in (plan.get("dimensions") or []) if str(d).strip()]
        metric_name = str(plan.get("metric") or "metric_value")
        intent = str(plan.get("intent") or "")
        warnings = [str(w) for w in (audit.get("warnings") or []) if str(w).strip()]
        canonical_metric_id = str(plan.get("canonical_metric_id") or "").strip()
        canonical_metric = plan.get("canonical_metric") if isinstance(plan.get("canonical_metric"), dict) else {}
        canonical_definition = str(canonical_metric.get("business_definition") or "").strip()

        driver_line = "Primary metric was selected from semantic contracts and query intent."
        if intent == "grouped_metric" and rows:
            dim_keys = [k for k in rows[0].keys() if k not in {"metric_value", "secondary_metric_value"}]
            if dim_keys:
                top_dim = " | ".join(str(rows[0].get(k)) for k in dim_keys if rows[0].get(k) is not None)
                driver_line = (
                    f"Top observed segment is **{top_dim or 'overall'}** for `{metric_name}` "
                    f"using grouping on {', '.join(dims) if dims else 'detected dimensions'}."
                )
        elif intent in {"comparison", "cross_domain_compare"}:
            driver_line = "Comparison intent drove side-by-side metric computation across aligned periods/domains."
        elif intent in {"trend_analysis", "running_total", "yoy_growth"}:
            driver_line = "Time-series intent drove chronological aggregation before narrative synthesis."

        evidence_lines: list[str] = []
        for row in rows[:3]:
            keys = list(row.keys())
            if not keys:
                continue
            if {"metric_value", "secondary_metric_value"}.issubset(set(keys)):
                dim_keys = [k for k in keys if k not in {"metric_value", "secondary_metric_value"}]
                label = " | ".join(str(row.get(k)) for k in dim_keys if row.get(k) is not None) or "overall"
                evidence_lines.append(
                    f"{label}: {metric_name}={_fmt_number(row.get('metric_value'))}, "
                    f"{plan.get('secondary_metric', 'secondary_metric')}={_fmt_number(row.get('secondary_metric_value'))}"
                )
            elif "metric_value" in keys:
                dim_keys = [k for k in keys if k != "metric_value"]
                label = " | ".join(str(row.get(k)) for k in dim_keys if row.get(k) is not None) or "overall"
                evidence_lines.append(f"{label}: {_fmt_number(row.get('metric_value'))}")
            elif len(keys) == 2:
                evidence_lines.append(f"{row.get(keys[0])}: {_fmt_number(row.get(keys[1]))}")
            else:
                pair = []
                for k in keys[:3]:
                    pair.append(f"{k}={row.get(k)}")
                evidence_lines.append(", ".join(pair))
        if not evidence_lines:
            evidence_lines.append(f"row_count={execution.get('row_count', 0)} for table `{plan.get('table', '')}`")

        caveat = warnings[0] if warnings else "No major audit warnings; still validate with source-truth checks for critical decisions."

        lines = [
            "**What Drove This**",
            f"- {driver_line}",
        ]
        if canonical_metric_id:
            contract_line = f"Metric contract: `{canonical_metric_id}`."
            if canonical_definition:
                contract_line += f" {canonical_definition}"
            lines.append(f"- {contract_line}")
        lines.extend(
            [
                "",
                "**Evidence**",
            ]
        )
        lines.extend(f"- {line}" for line in evidence_lines[:3])
        lines.extend(
            [
                "",
                "**Caveat**",
                f"- {caveat}",
            ]
        )
        return "\n".join(lines)

    def _enrich_narrative_with_analyst_sections(
        self,
        answer_markdown: str,
        *,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        audit: dict[str, Any],
    ) -> str:
        text = str(answer_markdown or "").strip()
        if not text:
            return text
        if not execution.get("success"):
            return text
        if int(execution.get("row_count") or 0) <= 0:
            return text
        lower = text.lower()
        if "**what drove this**" in lower and "**evidence**" in lower and "**caveat**" in lower:
            return text
        sections = self._build_analyst_sections(
            goal=goal,
            plan=plan,
            execution=execution,
            audit=audit,
        )
        return f"{text}\n\n{sections}"

    def _enforce_narrative_goal_anchors(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        answer_markdown: str,
    ) -> str:
        text = str(answer_markdown or "").strip()
        if not text:
            return text

        lower_goal = goal.lower()
        lower_text = text.lower()
        anchors: list[str] = []

        month_names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        month_mentions = [m for m in month_names if m in lower_goal]
        year_mentions = re.findall(r"\b20\d{2}\b", goal)
        if month_mentions:
            missing_months = [m for m in month_mentions if m not in lower_text]
            if missing_months:
                title_case = ", ".join(m.title() for m in month_mentions)
                years = ", ".join(sorted(set(year_mentions)))
                if years:
                    anchors.append(f"Compared periods: **{title_case} {years}**.")
                else:
                    anchors.append(f"Compared periods: **{title_case}**.")

        needs_comparison_language = any(
            tok in lower_goal
            for tok in ("compare", "compared", "versus", "vs", "higher", "lower", "difference")
        )
        has_analysis_language = any(
            tok in lower_text
            for tok in ("higher", "lower", "increase", "decrease", "difference", "improved", "declined", "trend")
        )
        if needs_comparison_language and not has_analysis_language:
            rows = list(execution.get("sample_rows") or [])
            if len(rows) >= 2:
                first = rows[0]
                second = rows[1]
                v1 = _to_float(first.get("metric_value"))
                v2 = _to_float(second.get("metric_value"))
                if v1 is not None and v2 is not None:
                    diff = v1 - v2
                    label_col = None
                    for key in first.keys():
                        if key not in {"metric_value", "secondary_metric_value"}:
                            label_col = key
                            break
                    left_label = str(first.get(label_col) if label_col else "first period")
                    right_label = str(second.get(label_col) if label_col else "second period")
                    direction = "higher" if diff >= 0 else "lower"
                    anchors.append(
                        f"Comparison insight: **{left_label}** is {direction} than **{right_label}** by **{_fmt_number(abs(diff))}**."
                    )

        if not anchors:
            return text
        return f"{text}\n\n" + "\n".join(f"- {line}" for line in anchors)

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

        if plan["intent"] == "cross_domain_compare" and rows:
            cleaned = [r for r in rows if r.get("month_bucket") is not None]
            if not cleaned:
                cleaned = rows
            top_ratio_row = None
            top_markup_row = None
            for row in cleaned:
                ratio = _to_float(row.get("metric_value"))
                markup = _to_float(row.get("forex_markup_revenue"))
                if ratio is not None and (top_ratio_row is None or ratio > (_to_float(top_ratio_row.get("metric_value")) or -1)):
                    top_ratio_row = row
                if markup is not None and (top_markup_row is None or markup > (_to_float(top_markup_row.get("forex_markup_revenue")) or -1)):
                    top_markup_row = row
            preview = cleaned[:6]
            bullets = []
            for row in preview:
                m = str(row.get("month_bucket", ""))
                mk = _fmt_number(row.get("forex_markup_revenue"))
                sp = _fmt_number(row.get("customer_spend"))
                rt = _to_float(row.get("metric_value"))
                ratio_txt = "n/a" if rt is None else f"{rt * 100:.2f}%"
                bullets.append(f"- {m}: markup={mk}, spend={sp}, markup/spend={ratio_txt}")
            insight_lines = []
            if top_ratio_row is not None:
                insight_lines.append(
                    f"Highest markup-to-spend ratio: **{top_ratio_row.get('month_bucket')}** "
                    f"at **{((_to_float(top_ratio_row.get('metric_value')) or 0.0) * 100):.2f}%**."
                )
            if top_markup_row is not None:
                insight_lines.append(
                    f"Peak absolute markup month: **{top_markup_row.get('month_bucket')}** "
                    f"with **{_fmt_number(top_markup_row.get('forex_markup_revenue'))}**."
                )
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    "Data split (monthly):\n"
                    + "\n".join(bullets)
                    + ("\n\nInsights:\n- " + "\n- ".join(insight_lines) if insight_lines else "")
                ),
                "headline_value": _to_float(cleaned[-1].get("metric_value")) if cleaned else None,
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

        if len(rows) == 1 and {"metric_value", "secondary_metric_value"}.issubset(set(execution["columns"])):
            primary_value = rows[0].get("metric_value")
            secondary_value = rows[0].get("secondary_metric_value")
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    f"- **{plan.get('metric', 'primary_metric')}**: {_fmt_number(primary_value)}\n"
                    f"- **{plan.get('secondary_metric', 'secondary_metric')}**: {_fmt_number(secondary_value)}"
                ),
                "headline_value": primary_value,
            }

        if plan["intent"] == "grouped_metric" and rows:
            preview = rows[:5]
            bullets = []
            dual_metric = {"metric_value", "secondary_metric_value"}.issubset(set(execution.get("columns", [])))
            for row in preview:
                keys = list(row.keys())
                if dual_metric:
                    dim_keys = [k for k in keys if k not in {"metric_value", "secondary_metric_value"}]
                    dim_label = " | ".join(str(row.get(k)) for k in dim_keys) if dim_keys else "overall"
                    bullets.append(
                        f"- {dim_label}: "
                        f"{plan.get('metric', 'count')}={_fmt_number(row.get('metric_value'))}, "
                        f"{plan.get('secondary_metric', 'amount')}={_fmt_number(row.get('secondary_metric_value'))}"
                    )
                elif len(keys) == 2:
                    bullets.append(f"- {row[keys[0]]}: {_fmt_number(row[keys[1]])}")
                elif len(keys) >= 3:
                    left = " | ".join(str(row[k]) for k in keys[:-1])
                    bullets.append(f"- {left}: {_fmt_number(row[keys[-1]])}")
            intro = (
                "I looked at the pattern across groups and here are the leaders:"
                if storyteller_mode
                else f"Top {len(preview)} groups:"
            )
            insight_line = ""
            if dual_metric and self.skill_runtime.has_skill("NarrativeAgent", "verification-before-completion"):
                first = preview[0] if preview else {}
                count_val = _to_float(first.get("metric_value")) or 0.0
                amt_val = _to_float(first.get("secondary_metric_value")) or 0.0
                avg_ticket = (amt_val / count_val) if count_val else None
                if avg_ticket is not None:
                    insight_line = (
                        f"\n\nInsight: top group avg amount per counted transaction is "
                        f"**{_fmt_number(avg_ticket)}**."
                    )
            return {
                "answer_markdown": (
                    f"**{goal}**\n\n"
                    f"{intro}\n" + "\n".join(bullets) + insight_line
                ),
                "headline_value": (
                    preview[0].get("metric_value")
                    if (dual_metric and preview)
                    else (preview[0].get(list(preview[0].keys())[-1]) if preview else None)
                ),
            }

        return {
            "answer_markdown": (
                f"**{goal}**\n\n"
                f"Returned {execution['row_count']} rows. Showing sample in result preview."
            ),
            "headline_value": execution["row_count"],
        }

    def _prepend_governance_disclosure(self, answer_markdown: str, plan: dict[str, Any]) -> str:
        notes = [str(n).strip() for n in (plan.get("_governance_notes") or []) if str(n).strip()]
        if not notes:
            return answer_markdown
        top = notes[:2]
        note_block = "\n".join(f"- {n}" for n in top)
        return (
            "**Policy note applied before analysis:**\n"
            f"{note_block}\n\n"
            f"{answer_markdown}"
        )

    def _viz_agent(self, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        return build_visualization_spec(plan, execution)

    def _suggested_questions(self, plan: dict[str, Any]) -> list[str]:
        metric = plan["metric"]
        table = plan["table"]
        if plan.get("secondary_metric"):
            return [
                "Compare these two metrics month over month",
                "Show the same split by state",
                "Highlight top segments where amount and count diverge",
            ]
        if plan.get("intent") == "cross_domain_compare":
            return [
                "Show the same markup vs spend comparison by platform",
                "Which month has the highest markup-to-spend ratio?",
                "Compare this ratio for MT103-only vs all transactions",
            ]
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

        for key in ["intent", "domain", "metric", "secondary_metric", "metric_bundle", "dimension"]:
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

    def _inject_share_insight(
        self,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        answer_markdown: str,
    ) -> str:
        lower_goal = str(goal or "").lower()
        lower_answer = str(answer_markdown or "").lower()
        if not any(k in lower_goal for k in ["share", "concentration"]):
            return answer_markdown
        if "share" in lower_answer or "concentration" in lower_answer:
            return answer_markdown
        if str(plan.get("intent") or "") != "grouped_metric":
            return answer_markdown
        rows = execution.get("sample_rows") or []
        if not isinstance(rows, list) or not rows:
            return answer_markdown
        values: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = _to_float(row.get("metric_value"))
            if val is not None and val >= 0:
                values.append(float(val))
        if len(values) < 1:
            return answer_markdown
        total = sum(values)
        if total <= 0:
            return answer_markdown
        top_share = (max(values) / total) * 100.0
        return (
            f"{answer_markdown.rstrip()}\n"
            f"- Concentration share: top segment contributes **{top_share:.1f}%** of the shown total."
        )

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
        llm_metrics = get_llm_metrics()
        llm_errors = [str(item.get("error")) for item in llm_metrics if item.get("error")]
        llm_success_calls = any(not item.get("error") for item in llm_metrics)
        llm_effective = bool(llm_intake_used or llm_narrative_used or llm_success_calls)
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
            "llm_effective": llm_effective,
            "autonomy_mode": autonomy_cfg.mode,
            "auto_correction": autonomy_cfg.auto_correction,
            "strict_truth": autonomy_cfg.strict_truth,
            "max_refinement_rounds": autonomy_cfg.max_refinement_rounds,
            "max_candidate_plans": autonomy_cfg.max_candidate_plans,
            "correction_applied": correction_applied,
            "memory_db_path": str(self.memory_db_path),
            "glossary_seed_stats": dict(getattr(self, "_seeded_glossary_stats", {}) or {}),
            "skills_runtime": self.skill_runtime.summary(),
            "llm_metrics": llm_metrics,
            "llm_metrics_summary": summarize_llm_metrics(llm_metrics),
            "llm_error_count": len(llm_errors),
            "llm_last_error": llm_errors[-1] if llm_errors else "",
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
