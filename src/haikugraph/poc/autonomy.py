"""Autonomy primitives for dataDa agent runtime.

This module provides:
- bounded-autonomy configuration
- persistent agent memory
- correction rule registry fed by autonomous loops + user feedback
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _similarity(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    inter = left & right
    union = left | right
    return len(inter) / len(union)


_SEMANTIC_SYNONYMS: dict[str, set[str]] = {
    "transaction": {"transactions", "transfer", "payment", "wire", "remittance"},
    "transactions": {"transaction", "transfer", "payment", "wire", "remittance"},
    "quote": {"quotes", "fx", "forex", "exchange"},
    "quotes": {"quote", "fx", "forex", "exchange"},
    "markup": {"charge", "charges", "spread", "fee", "fees"},
    "charges": {"charge", "markup", "fees"},
    "surcharge": {"charge", "charges", "markup", "fee", "fees"},
    "fx": {"forex", "exchange", "quote"},
    "forex": {"fx", "exchange", "quote"},
    "customer": {"customers", "client", "clients", "user", "users"},
    "customers": {"customer", "client", "clients", "user", "users"},
    "booking": {"bookings", "deal", "deals"},
    "bookings": {"booking", "deal", "deals"},
    "month": {"monthly", "monthwise"},
    "country": {"countries", "region"},
    "region": {"country", "territory"},
    "spend": {"spending", "amount", "value"},
    "spending": {"spend", "amount", "value"},
    "revenue": {"amount", "value", "earnings"},
    "count": {"volume", "number"},
}

_PHRASE_SYNONYMS: dict[str, set[str]] = {
    "foreign exchange": {"forex", "fx"},
    "foreign exchange fee": {"forex", "markup", "charges"},
    "foreign exchange surcharge": {"forex", "markup", "charges"},
    "service charge": {"charges", "fee", "markup"},
    "service charges": {"charges", "fees", "markup"},
    "payment rail": {"platform", "flow"},
    "volume of transactions": {"transaction", "count"},
}

_MEMORY_NOISE_TOKENS = {
    "show",
    "tell",
    "about",
    "please",
    "need",
    "insight",
    "analysis",
    "for",
    "the",
    "a",
    "an",
    "this",
    "that",
}


def _expand_tokens(text: str) -> set[str]:
    lower = str(text or "").lower()
    base = _tokens(lower)
    expanded = set(base)
    for phrase, mapped in _PHRASE_SYNONYMS.items():
        if phrase in lower:
            expanded.update(mapped)
    for tok in list(base):
        expanded.update(_SEMANTIC_SYNONYMS.get(tok, set()))
    return expanded


def _semantic_similarity(a: str, b: str) -> float:
    a_exp = _expand_tokens(a)
    b_exp = _expand_tokens(b)
    if not a_exp or not b_exp:
        return 0.0
    jaccard = len(a_exp & b_exp) / max(1, len(a_exp | b_exp))
    # Trigger-oriented coverage: if the goal covers all trigger concepts (possibly via
    # synonyms), we should treat it as a strong semantic match even when the goal is longer.
    trigger_coverage = len(a_exp & b_exp) / max(1, len(b_exp))
    goal_coverage = len(a_exp & b_exp) / max(1, len(a_exp))
    ratio = SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    return max(jaccard, trigger_coverage, goal_coverage * 0.75, ratio * 0.8)


def _focus_tokens(text: str) -> set[str]:
    toks = _expand_tokens(text)
    return {t for t in toks if len(t) >= 4 and t not in _MEMORY_NOISE_TOKENS}


_GENERIC_CORRECTION_TOKENS = {
    "data",
    "query",
    "queries",
    "metric",
    "metrics",
    "quote",
    "quotes",
    "transaction",
    "transactions",
    "customer",
    "customers",
    "booking",
    "bookings",
}


def _is_low_specificity_keyword(keyword: str) -> bool:
    tokens = [t for t in re.findall(r"[a-z0-9_]+", keyword.lower()) if t]
    if not tokens:
        return True
    informative = [t for t in tokens if len(t) >= 4 and t not in _GENERIC_CORRECTION_TOKENS]
    return len(informative) == 0


def _clean_tenant(tenant_id: str | None) -> str:
    return (tenant_id or "public").strip() or "public"


@dataclass
class AutonomyConfig:
    """Runtime controls for bounded autonomy.

    Bounds are applied to side-effects and resource usage, not agent reasoning.
    """

    mode: str = "bounded"
    auto_correction: bool = True
    strict_truth: bool = True
    max_refinement_rounds: int = 2
    max_candidate_plans: int = 5
    min_score_delta_for_switch: float = 0.03
    max_probe_queries: int = 2


class AgentMemoryStore:
    """Persistent memory + correction registry in DuckDB."""

    _EXPECTED_SCHEMAS: dict[str, list[tuple[str, str]]] = {
        "datada_agent_memory": [
            ("memory_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("trace_id", "VARCHAR"),
            ("goal", "VARCHAR"),
            ("resolved_goal", "VARCHAR"),
            ("runtime_mode", "VARCHAR"),
            ("provider", "VARCHAR"),
            ("success", "BOOLEAN"),
            ("confidence_score", "DOUBLE"),
            ("row_count", "BIGINT"),
            ("table_name", "VARCHAR"),
            ("metric", "VARCHAR"),
            ("dimensions_json", "VARCHAR"),
            ("time_filter_json", "VARCHAR"),
            ("value_filters_json", "VARCHAR"),
            ("sql_text", "VARCHAR"),
            ("audit_warnings_json", "VARCHAR"),
            ("correction_applied", "BOOLEAN"),
            ("correction_reason", "VARCHAR"),
            ("metadata_json", "VARCHAR"),
        ],
        "datada_agent_feedback": [
            ("feedback_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("trace_id", "VARCHAR"),
            ("session_id", "VARCHAR"),
            ("goal", "VARCHAR"),
            ("issue", "VARCHAR"),
            ("suggested_fix", "VARCHAR"),
            ("severity", "VARCHAR"),
            ("metadata_json", "VARCHAR"),
        ],
        "datada_agent_corrections": [
            ("correction_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("source", "VARCHAR"),
            ("keyword", "VARCHAR"),
            ("target_table", "VARCHAR"),
            ("target_metric", "VARCHAR"),
            ("target_dimensions_json", "VARCHAR"),
            ("notes", "VARCHAR"),
            ("weight", "DOUBLE"),
            ("enabled", "BOOLEAN"),
        ],
        "datada_agent_correction_events": [
            ("event_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("correction_id", "VARCHAR"),
            ("action", "VARCHAR"),
            ("enabled_before", "BOOLEAN"),
            ("enabled_after", "BOOLEAN"),
            ("note", "VARCHAR"),
        ],
        "datada_agent_toolsmith": [
            ("tool_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("status", "VARCHAR"),
            ("source", "VARCHAR"),
            ("title", "VARCHAR"),
            ("sql_text", "VARCHAR"),
            ("test_sql_text", "VARCHAR"),
            ("test_success", "BOOLEAN"),
            ("test_message", "VARCHAR"),
            ("metadata_json", "VARCHAR"),
        ],
        "datada_business_rules": [
            ("rule_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("domain", "VARCHAR"),
            ("name", "VARCHAR"),
            ("rule_type", "VARCHAR"),
            ("triggers_json", "VARCHAR"),
            ("action_payload_json", "VARCHAR"),
            ("notes", "VARCHAR"),
            ("priority", "DOUBLE"),
            ("status", "VARCHAR"),
            ("source", "VARCHAR"),
            ("created_by", "VARCHAR"),
            ("approved_by", "VARCHAR"),
            ("version", "BIGINT"),
        ],
        "datada_business_rule_events": [
            ("event_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("rule_id", "VARCHAR"),
            ("action", "VARCHAR"),
            ("before_json", "VARCHAR"),
            ("after_json", "VARCHAR"),
            ("note", "VARCHAR"),
            ("actor", "VARCHAR"),
        ],
        "datada_org_knowledge_views": [
            ("view_id", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
            ("tenant_id", "VARCHAR"),
            ("view_name", "VARCHAR"),
            ("goal_signature", "VARCHAR"),
            ("payload_json", "VARCHAR"),
            ("source_agent", "VARCHAR"),
            ("version", "BIGINT"),
        ],
    }

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._ensure_tables()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.db_path), read_only=False)

    def _ensure_tables(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_agent_memory (
                        memory_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR,
                        trace_id VARCHAR,
                        goal VARCHAR,
                        resolved_goal VARCHAR,
                        runtime_mode VARCHAR,
                        provider VARCHAR,
                        success BOOLEAN,
                        confidence_score DOUBLE,
                        row_count BIGINT,
                        table_name VARCHAR,
                        metric VARCHAR,
                        dimensions_json VARCHAR,
                        time_filter_json VARCHAR,
                        value_filters_json VARCHAR,
                        sql_text VARCHAR,
                        audit_warnings_json VARCHAR,
                        correction_applied BOOLEAN,
                        correction_reason VARCHAR,
                        metadata_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_agent_feedback (
                        feedback_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR,
                        trace_id VARCHAR,
                        session_id VARCHAR,
                        goal VARCHAR,
                        issue VARCHAR,
                        suggested_fix VARCHAR,
                        severity VARCHAR,
                        metadata_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_agent_corrections (
                        correction_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR,
                        source VARCHAR,
                        keyword VARCHAR,
                        target_table VARCHAR,
                        target_metric VARCHAR,
                        target_dimensions_json VARCHAR,
                        notes VARCHAR,
                        weight DOUBLE,
                        enabled BOOLEAN
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_agent_correction_events (
                        event_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR,
                        correction_id VARCHAR,
                        action VARCHAR,
                        enabled_before BOOLEAN,
                        enabled_after BOOLEAN,
                        note VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_agent_toolsmith (
                        tool_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR,
                        status VARCHAR,
                        source VARCHAR,
                        title VARCHAR,
                        sql_text VARCHAR,
                        test_sql_text VARCHAR,
                        test_success BOOLEAN,
                        test_message VARCHAR,
                        metadata_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_glossary (
                        term_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR DEFAULT 'public',
                        term VARCHAR,
                        definition VARCHAR,
                        sql_expression VARCHAR,
                        target_table VARCHAR,
                        target_column VARCHAR,
                        examples_json VARCHAR,
                        contributed_by VARCHAR,
                        version INTEGER DEFAULT 1,
                        deprecated BOOLEAN DEFAULT FALSE,
                        effectiveness_score DOUBLE DEFAULT 0.5,
                        use_count INTEGER DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_teachings (
                        teaching_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR DEFAULT 'public',
                        expert_name VARCHAR,
                        teaching_text VARCHAR,
                        parsed_keyword VARCHAR,
                        parsed_table VARCHAR,
                        parsed_metric VARCHAR,
                        parsed_dimensions_json VARCHAR,
                        confidence DOUBLE DEFAULT 0.5,
                        times_applied INTEGER DEFAULT 0,
                        times_correct INTEGER DEFAULT 0,
                        active BOOLEAN DEFAULT TRUE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_business_rules (
                        rule_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR,
                        domain VARCHAR,
                        name VARCHAR,
                        rule_type VARCHAR,
                        triggers_json VARCHAR,
                        action_payload_json VARCHAR,
                        notes VARCHAR,
                        priority DOUBLE,
                        status VARCHAR,
                        source VARCHAR,
                        created_by VARCHAR,
                        approved_by VARCHAR,
                        version BIGINT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_business_rule_events (
                        event_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR,
                        rule_id VARCHAR,
                        action VARCHAR,
                        before_json VARCHAR,
                        after_json VARCHAR,
                        note VARCHAR,
                        actor VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_org_knowledge_views (
                        view_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR,
                        view_name VARCHAR,
                        goal_signature VARCHAR,
                        payload_json VARCHAR,
                        source_agent VARCHAR,
                        version BIGINT
                    )
                    """
                )
                self._ensure_column(conn, "datada_agent_memory", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_agent_feedback", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_agent_corrections", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_agent_correction_events", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_agent_toolsmith", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_business_rules", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_business_rule_events", "tenant_id VARCHAR")
                self._ensure_column(conn, "datada_org_knowledge_views", "tenant_id VARCHAR")
                self._ensure_schema_compatibility(conn)
                conn.execute(
                    """
                    UPDATE datada_agent_memory SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_agent_feedback SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_agent_corrections SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_agent_correction_events SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_agent_toolsmith SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_business_rules SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_business_rule_events SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
                conn.execute(
                    """
                    UPDATE datada_org_knowledge_views SET tenant_id = 'public'
                    WHERE tenant_id IS NULL OR TRIM(COALESCE(tenant_id, '')) = ''
                    """
                )
            finally:
                conn.close()

    def _has_column(self, conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            """,
            [table, column],
        ).fetchone()
        return bool(row and int(row[0] or 0) > 0)

    def _ensure_column(self, conn: duckdb.DuckDBPyConnection, table: str, column_def: str) -> None:
        col_name = column_def.split()[0].strip().lower()
        if self._has_column(conn, table, col_name):
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    def _normalize_type(self, sql_type: str) -> str:
        text = str(sql_type or "").strip().upper()
        aliases = {
            "BOOL": "BOOLEAN",
            "INTEGER": "BIGINT",
            "INT64": "BIGINT",
            "DOUBLE PRECISION": "DOUBLE",
            "STRING": "VARCHAR",
        }
        return aliases.get(text, text)

    def _current_schema(self, conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
        rows = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = ?
            """,
            [table],
        ).fetchall()
        return {str(name).lower(): self._normalize_type(dtype) for name, dtype in rows}

    def _cast_expr(self, column: str, target_type: str) -> str:
        quoted = f'"{column}"'
        t = self._normalize_type(target_type)
        if t == "VARCHAR":
            return f"CAST({quoted} AS VARCHAR)"
        if t == "BOOLEAN":
            return f"COALESCE(TRY_CAST({quoted} AS BOOLEAN), FALSE)"
        if t == "DOUBLE":
            return f"COALESCE(TRY_CAST({quoted} AS DOUBLE), 0.0)"
        if t == "BIGINT":
            return f"COALESCE(TRY_CAST({quoted} AS BIGINT), 0)"
        if t == "TIMESTAMP":
            return f"COALESCE(TRY_CAST({quoted} AS TIMESTAMP), CURRENT_TIMESTAMP)"
        return f"CAST({quoted} AS {target_type})"

    def _default_expr(self, column: str, target_type: str) -> str:
        col = column.lower()
        t = self._normalize_type(target_type)
        if col == "tenant_id":
            return "'public'"
        if col.endswith("_id"):
            return "''"
        if t == "VARCHAR":
            return "''"
        if t == "BOOLEAN":
            return "FALSE"
        if t == "DOUBLE":
            return "0.0"
        if t == "BIGINT":
            return "0"
        if t == "TIMESTAMP":
            return "CURRENT_TIMESTAMP"
        return "NULL"

    def _rebuild_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str,
        expected_cols: list[tuple[str, str]],
    ) -> None:
        current = self._current_schema(conn, table)
        temp_table = f"{table}__schema_fix"
        column_defs = ",\n                        ".join(f"{name} {dtype}" for name, dtype in expected_cols)
        conn.execute(
            f"""
            CREATE TABLE {temp_table} (
                {column_defs}
            )
            """
        )

        select_exprs: list[str] = []
        for name, dtype in expected_cols:
            if name.lower() in current:
                select_exprs.append(f"{self._cast_expr(name, dtype)} AS \"{name}\"")
            else:
                select_exprs.append(f"{self._default_expr(name, dtype)} AS \"{name}\"")

        select_sql = ", ".join(select_exprs)
        conn.execute(
            f"""
            INSERT INTO {temp_table}
            SELECT {select_sql}
            FROM {table}
            """
        )
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {temp_table} RENAME TO {table}")

    def _ensure_schema_compatibility(self, conn: duckdb.DuckDBPyConnection) -> None:
        for table, expected_cols in self._EXPECTED_SCHEMAS.items():
            current = self._current_schema(conn, table)
            if not current:
                continue
            mismatch = False
            for col_name, expected_type in expected_cols:
                cur_type = current.get(col_name.lower())
                if not cur_type:
                    mismatch = True
                    break
                if cur_type != self._normalize_type(expected_type):
                    mismatch = True
                    break
            if mismatch:
                self._rebuild_table(conn, table, expected_cols)

    def store_turn(
        self,
        *,
        tenant_id: str | None,
        trace_id: str,
        goal: str,
        resolved_goal: str,
        runtime_mode: str,
        provider: str | None,
        success: bool,
        confidence_score: float,
        row_count: int | None,
        plan: dict[str, Any],
        sql: str | None,
        audit_warnings: list[str] | None,
        correction_applied: bool,
        correction_reason: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_agent_memory (
                        memory_id,
                        created_at,
                        tenant_id,
                        trace_id,
                        goal,
                        resolved_goal,
                        runtime_mode,
                        provider,
                        success,
                        confidence_score,
                        row_count,
                        table_name,
                        metric,
                        dimensions_json,
                        time_filter_json,
                        value_filters_json,
                        sql_text,
                        audit_warnings_json,
                        correction_applied,
                        correction_reason,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        memory_id,
                        datetime.utcnow(),
                        tenant,
                        trace_id,
                        goal,
                        resolved_goal,
                        runtime_mode,
                        provider or "",
                        bool(success),
                        float(confidence_score),
                        int(row_count or 0),
                        str(plan.get("table") or ""),
                        str(plan.get("metric") or ""),
                        json.dumps(plan.get("dimensions", []), default=str),
                        json.dumps(plan.get("time_filter"), default=str),
                        json.dumps(plan.get("value_filters", []), default=str),
                        sql or "",
                        json.dumps(audit_warnings or [], default=str),
                        bool(correction_applied),
                        correction_reason or "",
                        json.dumps(metadata or {}, default=str),
                    ],
                )
            finally:
                conn.close()
        return memory_id

    def recall(
        self,
        goal: str,
        *,
        tenant_id: str | None = None,
        limit: int = 3,
        scan_limit: int = 400,
    ) -> list[dict[str, Any]]:
        """Return similar successful historical runs."""

        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT goal, resolved_goal, table_name, metric, dimensions_json, time_filter_json,
                           value_filters_json, correction_applied, correction_reason, confidence_score, row_count,
                           runtime_mode, created_at
                    FROM datada_agent_memory
                    WHERE success = TRUE
                      AND tenant_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [tenant, scan_limit],
                ).fetchall()
            finally:
                conn.close()

        scored: list[dict[str, Any]] = []
        goal_focus = _focus_tokens(goal)
        for row in rows:
            past_goal = str(row[0] or "")
            lexical_sim = _similarity(goal, past_goal)
            semantic_sim = _semantic_similarity(goal, past_goal)
            past_focus = _focus_tokens(past_goal)
            focus_overlap = (
                len(goal_focus & past_focus) / max(1, len(goal_focus | past_focus))
                if goal_focus and past_focus
                else 0.0
            )
            sim = max(lexical_sim, semantic_sim * 0.92, focus_overlap)
            if sim < 0.35:
                continue
            # Precision guard: noisy lexical overlap without semantic/focus agreement is dropped.
            if sim < 0.50 and semantic_sim < 0.45 and focus_overlap < 0.25:
                continue
            dimensions = []
            time_filter = None
            value_filters = []
            try:
                dimensions = json.loads(row[4] or "[]")
            except Exception:
                dimensions = []
            try:
                time_filter = json.loads(row[5] or "null")
            except Exception:
                time_filter = None
            try:
                value_filters = json.loads(row[6] or "[]")
            except Exception:
                value_filters = []

            entry = {
                    "similarity": round(sim, 4),
                    "lexical_similarity": round(lexical_sim, 4),
                    "semantic_similarity": round(semantic_sim, 4),
                    "focus_overlap": round(focus_overlap, 4),
                    "goal": past_goal,
                    "resolved_goal": str(row[1] or ""),
                    "table": str(row[2] or ""),
                    "metric": str(row[3] or ""),
                    "dimensions": dimensions if isinstance(dimensions, list) else [],
                    "time_filter": time_filter if isinstance(time_filter, dict) else None,
                    "value_filters": value_filters if isinstance(value_filters, list) else [],
                    "correction_applied": bool(row[7]),
                    "correction_reason": str(row[8] or ""),
                    "confidence_score": float(row[9] or 0.0),
                    "row_count": int(row[10] or 0),
                    "runtime_mode": str(row[11] or ""),
                    "created_at": str(row[12] or ""),
            }
            # GAP 41c: Flag entries that represent learned corrections
            if entry["correction_applied"] and entry["confidence_score"] >= 0.6:
                entry["was_learned_correction"] = True
            scored.append(entry)

        scored.sort(
            key=lambda x: (
                x["similarity"],
                x["semantic_similarity"],
                x["focus_overlap"],
                x["confidence_score"],
                1.0 if x["correction_applied"] else 0.0,
            ),
            reverse=True,
        )
        return scored[: max(1, min(10, int(limit)))]

    def store_feedback(
        self,
        *,
        tenant_id: str | None,
        trace_id: str | None,
        session_id: str | None,
        goal: str | None,
        issue: str,
        suggested_fix: str | None,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_agent_feedback (
                        feedback_id,
                        created_at,
                        tenant_id,
                        trace_id,
                        session_id,
                        goal,
                        issue,
                        suggested_fix,
                        severity,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        feedback_id,
                        datetime.utcnow(),
                        tenant,
                        trace_id or "",
                        session_id or "",
                        goal or "",
                        issue,
                        suggested_fix or "",
                        severity,
                        json.dumps(metadata or {}, default=str),
                    ],
                )
            finally:
                conn.close()
        return feedback_id

    def upsert_correction(
        self,
        *,
        tenant_id: str | None,
        keyword: str,
        target_table: str,
        target_metric: str,
        target_dimensions: list[str] | None = None,
        notes: str = "",
        source: str = "feedback",
        weight: float = 1.0,
    ) -> str:
        """Insert a correction rule if an equivalent enabled rule does not exist."""

        norm_keyword = keyword.strip().lower()
        if not norm_keyword:
            return ""
        tenant = _clean_tenant(tenant_id)

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT correction_id
                    FROM datada_agent_corrections
                    WHERE enabled = TRUE
                      AND tenant_id = ?
                      AND LOWER(keyword) = ?
                      AND target_table = ?
                      AND target_metric = ?
                    LIMIT 1
                    """,
                    [tenant, norm_keyword, target_table, target_metric],
                ).fetchone()
                if existing and existing[0]:
                    return str(existing[0])

                correction_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO datada_agent_corrections (
                        correction_id,
                        created_at,
                        tenant_id,
                        source,
                        keyword,
                        target_table,
                        target_metric,
                        target_dimensions_json,
                        notes,
                        weight,
                        enabled
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        correction_id,
                        datetime.utcnow(),
                        tenant,
                        source,
                        norm_keyword,
                        target_table,
                        target_metric,
                        json.dumps(target_dimensions or [], default=str),
                        notes,
                        float(max(0.05, min(5.0, weight))),
                        True,
                    ],
                )
                return correction_id
            finally:
                conn.close()

    def get_matching_corrections(
        self,
        goal: str,
        *,
        tenant_id: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT correction_id, source, keyword, target_table, target_metric, target_dimensions_json, notes, weight
                    FROM datada_agent_corrections
                    WHERE enabled = TRUE
                      AND tenant_id = ?
                    ORDER BY created_at DESC
                    LIMIT 250
                    """
                    ,
                    [tenant],
                ).fetchall()
            finally:
                conn.close()

        lower_goal = goal.lower()
        out: list[dict[str, Any]] = []
        for row in rows:
            keyword = str(row[2] or "")
            if not keyword:
                continue
            if _is_low_specificity_keyword(keyword):
                continue
            substring_match = keyword in lower_goal
            sim = 1.0 if substring_match else _semantic_similarity(goal, keyword)
            min_sim = 0.24
            keyword_tokens = [t for t in re.findall(r"[a-z0-9_]+", keyword.lower()) if t]
            if len(keyword_tokens) >= 3:
                min_sim = 0.18
            elif len(keyword_tokens) == 1:
                min_sim = 0.55
            if sim < min_sim:
                continue
            dims = []
            try:
                dims = json.loads(row[5] or "[]")
            except Exception:
                dims = []
            weight = float(row[7] or 1.0)
            out.append(
                {
                    "correction_id": str(row[0]),
                    "source": str(row[1] or ""),
                    "keyword": keyword,
                    "target_table": str(row[3] or ""),
                    "target_metric": str(row[4] or ""),
                    "target_dimensions": dims if isinstance(dims, list) else [],
                    "notes": str(row[6] or ""),
                    "weight": weight,
                    "match_score": round(sim * weight, 4),
                }
            )
        out.sort(key=lambda x: x["match_score"], reverse=True)
        return out[: max(1, min(10, int(limit)))]

    def list_corrections(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 250,
        include_disabled: bool = True,
    ) -> list[dict[str, Any]]:
        tenant = _clean_tenant(tenant_id)
        where_clause = "WHERE tenant_id = ?" if include_disabled else "WHERE tenant_id = ? AND enabled = TRUE"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT correction_id, created_at, source, keyword, target_table, target_metric,
                           target_dimensions_json, notes, weight, enabled
                    FROM datada_agent_corrections
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [tenant, max(1, min(1000, int(limit)))],
                ).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            dims = []
            try:
                dims = json.loads(row[6] or "[]")
            except Exception:
                dims = []
            out.append(
                {
                    "correction_id": str(row[0] or ""),
                    "created_at": str(row[1] or ""),
                    "source": str(row[2] or ""),
                    "keyword": str(row[3] or ""),
                    "target_table": str(row[4] or ""),
                    "target_metric": str(row[5] or ""),
                    "target_dimensions": dims if isinstance(dims, list) else [],
                    "notes": str(row[7] or ""),
                    "weight": float(row[8] or 0.0),
                    "enabled": bool(row[9]),
                }
            )
        return out

    def set_correction_enabled(
        self,
        correction_id: str,
        enabled: bool,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        cid = (correction_id or "").strip()
        if not cid:
            return False
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT correction_id, enabled FROM datada_agent_corrections
                    WHERE correction_id = ?
                      AND tenant_id = ?
                    LIMIT 1
                    """,
                    [cid, tenant],
                ).fetchone()
                if not existing:
                    return False
                before = bool(existing[1])
                after = bool(enabled)
                if before == after:
                    return True
                conn.execute(
                    """
                    UPDATE datada_agent_corrections
                    SET enabled = ?
                    WHERE correction_id = ?
                      AND tenant_id = ?
                    """,
                    [bool(enabled), cid, tenant],
                )
                conn.execute(
                    """
                    INSERT INTO datada_agent_correction_events (
                        event_id,
                        created_at,
                        tenant_id,
                        correction_id,
                        action,
                        enabled_before,
                        enabled_after,
                        note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        datetime.utcnow(),
                        tenant,
                        cid,
                        "toggle",
                        before,
                        after,
                        f"manual_toggle:{before}->{after}",
                    ],
                )
                return True
            finally:
                conn.close()

    def rollback_correction(self, correction_id: str, *, tenant_id: str | None = None) -> dict[str, Any]:
        cid = (correction_id or "").strip()
        if not cid:
            return {"success": False, "message": "Missing correction_id."}
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                current = conn.execute(
                    """
                    SELECT enabled FROM datada_agent_corrections
                    WHERE correction_id = ?
                      AND tenant_id = ?
                    LIMIT 1
                    """,
                    [cid, tenant],
                ).fetchone()
                if not current:
                    return {"success": False, "message": f"Unknown correction_id '{cid}'."}
                last_change = conn.execute(
                    """
                    SELECT enabled_before, enabled_after
                    FROM datada_agent_correction_events
                    WHERE correction_id = ?
                      AND tenant_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [cid, tenant],
                ).fetchone()
                if not last_change:
                    return {"success": False, "message": "No correction state changes to rollback."}
                previous_state = bool(last_change[0])
                current_state = bool(current[0])
                if previous_state == current_state:
                    return {
                        "success": False,
                        "message": "Correction already matches the last rollback target state.",
                    }
                conn.execute(
                    """
                    UPDATE datada_agent_corrections
                    SET enabled = ?
                    WHERE correction_id = ?
                      AND tenant_id = ?
                    """,
                    [previous_state, cid, tenant],
                )
                conn.execute(
                    """
                    INSERT INTO datada_agent_correction_events (
                        event_id,
                        created_at,
                        tenant_id,
                        correction_id,
                        action,
                        enabled_before,
                        enabled_after,
                        note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        datetime.utcnow(),
                        tenant,
                        cid,
                        "rollback",
                        current_state,
                        previous_state,
                        "rollback_last_toggle",
                    ],
                )
                return {
                    "success": True,
                    "message": f"Correction {cid} rolled back to {'enabled' if previous_state else 'disabled'}.",
                    "enabled": previous_state,
                }
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Business rules (admin-governed domain knowledge layer)
    # ------------------------------------------------------------------

    def create_business_rule(
        self,
        *,
        tenant_id: str | None,
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
        clean_domain = (domain or "general").strip().lower() or "general"
        clean_name = (name or "").strip()
        clean_type = (rule_type or "").strip().lower()
        clean_status = (status or "draft").strip().lower()
        if not clean_name:
            return ""
        if clean_type not in {"plan_override", "add_filter", "override_metric_expr", "route_override"}:
            clean_type = "plan_override"
        if clean_status not in {"draft", "active", "disabled", "archived"}:
            clean_status = "draft"
        clean_triggers = [
            str(t).strip().lower()
            for t in (triggers or [])
            if str(t).strip()
        ]
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                dup = conn.execute(
                    """
                    SELECT rule_id
                    FROM datada_business_rules
                    WHERE tenant_id = ?
                      AND LOWER(name) = LOWER(?)
                      AND rule_type = ?
                      AND status != 'archived'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    [tenant, clean_name, clean_type],
                ).fetchone()
                if dup and dup[0]:
                    return str(dup[0])

                rule_id = str(uuid.uuid4())
                now = datetime.utcnow()
                row_after = {
                    "rule_id": rule_id,
                    "tenant_id": tenant,
                    "domain": clean_domain,
                    "name": clean_name,
                    "rule_type": clean_type,
                    "triggers": clean_triggers,
                    "action_payload": action_payload or {},
                    "notes": notes,
                    "priority": float(priority),
                    "status": clean_status,
                    "source": source,
                    "created_by": created_by,
                    "approved_by": approved_by,
                    "version": 1,
                }
                conn.execute(
                    """
                    INSERT INTO datada_business_rules (
                        rule_id, created_at, updated_at, tenant_id, domain, name, rule_type,
                        triggers_json, action_payload_json, notes, priority, status, source,
                        created_by, approved_by, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        rule_id,
                        now,
                        now,
                        tenant,
                        clean_domain,
                        clean_name,
                        clean_type,
                        json.dumps(clean_triggers, default=str),
                        json.dumps(action_payload or {}, default=str),
                        notes,
                        float(priority),
                        clean_status,
                        source,
                        created_by,
                        approved_by,
                        1,
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO datada_business_rule_events (
                        event_id, created_at, tenant_id, rule_id, action, before_json, after_json, note, actor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        now,
                        tenant,
                        rule_id,
                        "create",
                        "{}",
                        json.dumps(row_after, default=str),
                        "rule_created",
                        created_by or "admin",
                    ],
                )
                return rule_id
            finally:
                conn.close()

    def list_business_rules(
        self,
        *,
        tenant_id: str | None = None,
        status: str | None = None,
        domain: str | None = None,
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        tenant = _clean_tenant(tenant_id)
        where: list[str] = ["tenant_id = ?"]
        params: list[Any] = [tenant]
        clean_status = (status or "").strip().lower()
        clean_domain = (domain or "").strip().lower()
        if clean_status:
            where.append("LOWER(status) = ?")
            params.append(clean_status)
        if clean_domain:
            where.append("LOWER(domain) = ?")
            params.append(clean_domain)
        params.append(max(1, min(1000, int(limit))))
        where_sql = " AND ".join(where)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        rule_id, created_at, updated_at, tenant_id, domain, name, rule_type,
                        triggers_json, action_payload_json, notes, priority, status, source,
                        created_by, approved_by, version
                    FROM datada_business_rules
                    WHERE {where_sql}
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                triggers = json.loads(row[7] or "[]")
            except Exception:
                triggers = []
            try:
                action_payload = json.loads(row[8] or "{}")
            except Exception:
                action_payload = {}
            out.append(
                {
                    "rule_id": str(row[0] or ""),
                    "created_at": str(row[1] or ""),
                    "updated_at": str(row[2] or ""),
                    "tenant_id": str(row[3] or ""),
                    "domain": str(row[4] or ""),
                    "name": str(row[5] or ""),
                    "rule_type": str(row[6] or ""),
                    "triggers": triggers if isinstance(triggers, list) else [],
                    "action_payload": action_payload if isinstance(action_payload, dict) else {},
                    "notes": str(row[9] or ""),
                    "priority": float(row[10] or 0.0),
                    "status": str(row[11] or ""),
                    "source": str(row[12] or ""),
                    "created_by": str(row[13] or ""),
                    "approved_by": str(row[14] or ""),
                    "version": int(row[15] or 1),
                }
            )
        return out

    def upsert_org_knowledge_view(
        self,
        *,
        tenant_id: str | None = None,
        view_name: str,
        payload: dict[str, Any],
        goal_signature: str = "",
        source_agent: str = "OrganizationalKnowledgeAgent",
    ) -> dict[str, Any]:
        tenant = _clean_tenant(tenant_id)
        clean_view = (view_name or "").strip().lower()[:120]
        if not clean_view:
            return {"success": False, "message": "Missing view_name."}
        payload_json = json.dumps(payload or {}, default=str)
        now = datetime.utcnow()

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT view_id, version
                    FROM datada_org_knowledge_views
                    WHERE tenant_id = ? AND view_name = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    [tenant, clean_view],
                ).fetchone()
                if existing:
                    view_id = str(existing[0] or "")
                    next_version = int(existing[1] or 1) + 1
                    conn.execute(
                        """
                        UPDATE datada_org_knowledge_views
                        SET
                            updated_at = ?,
                            goal_signature = ?,
                            payload_json = ?,
                            source_agent = ?,
                            version = ?
                        WHERE view_id = ? AND tenant_id = ?
                        """,
                        [
                            now,
                            (goal_signature or "")[:160],
                            payload_json,
                            (source_agent or "OrganizationalKnowledgeAgent")[:120],
                            next_version,
                            view_id,
                            tenant,
                        ],
                    )
                    return {
                        "success": True,
                        "view_id": view_id,
                        "view_name": clean_view,
                        "version": next_version,
                    }

                view_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO datada_org_knowledge_views (
                        view_id, created_at, updated_at, tenant_id, view_name,
                        goal_signature, payload_json, source_agent, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        view_id,
                        now,
                        now,
                        tenant,
                        clean_view,
                        (goal_signature or "")[:160],
                        payload_json,
                        (source_agent or "OrganizationalKnowledgeAgent")[:120],
                        1,
                    ],
                )
                return {
                    "success": True,
                    "view_id": view_id,
                    "view_name": clean_view,
                    "version": 1,
                }
            finally:
                conn.close()

    def list_org_knowledge_views(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        view_id, created_at, updated_at, tenant_id, view_name,
                        goal_signature, payload_json, source_agent, version
                    FROM datada_org_knowledge_views
                    WHERE tenant_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    [tenant, max(1, min(500, int(limit)))],
                ).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row[6] or "{}")
            except Exception:
                payload = {}
            out.append(
                {
                    "view_id": str(row[0] or ""),
                    "created_at": str(row[1] or ""),
                    "updated_at": str(row[2] or ""),
                    "tenant_id": str(row[3] or ""),
                    "view_name": str(row[4] or ""),
                    "goal_signature": str(row[5] or ""),
                    "payload": payload if isinstance(payload, dict) else {},
                    "source_agent": str(row[7] or ""),
                    "version": int(row[8] or 1),
                }
            )
        return out

    def set_business_rule_status(
        self,
        rule_id: str,
        *,
        tenant_id: str | None = None,
        status: str,
        actor: str = "admin",
        note: str = "",
    ) -> dict[str, Any]:
        rid = (rule_id or "").strip()
        clean_status = (status or "").strip().lower()
        if not rid:
            return {"success": False, "message": "Missing rule_id."}
        if clean_status not in {"draft", "active", "disabled", "archived"}:
            return {"success": False, "message": f"Unsupported status '{status}'."}
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT
                        rule_id, created_at, updated_at, tenant_id, domain, name, rule_type,
                        triggers_json, action_payload_json, notes, priority, status, source,
                        created_by, approved_by, version
                    FROM datada_business_rules
                    WHERE rule_id = ? AND tenant_id = ?
                    LIMIT 1
                    """,
                    [rid, tenant],
                ).fetchone()
                if not row:
                    return {"success": False, "message": f"Unknown rule_id '{rid}'."}

                before = {
                    "rule_id": str(row[0] or ""),
                    "status": str(row[11] or ""),
                    "version": int(row[15] or 1),
                    "approved_by": str(row[14] or ""),
                }
                if before["status"] == clean_status:
                    return {"success": True, "message": "No status change required.", "status": clean_status}

                next_version = before["version"] + 1
                approved_by = actor if clean_status == "active" else str(row[14] or "")
                now = datetime.utcnow()
                conn.execute(
                    """
                    UPDATE datada_business_rules
                    SET updated_at = ?, status = ?, approved_by = ?, version = ?
                    WHERE rule_id = ? AND tenant_id = ?
                    """,
                    [now, clean_status, approved_by, next_version, rid, tenant],
                )
                after = {
                    "rule_id": rid,
                    "status": clean_status,
                    "version": next_version,
                    "approved_by": approved_by,
                }
                conn.execute(
                    """
                    INSERT INTO datada_business_rule_events (
                        event_id, created_at, tenant_id, rule_id, action, before_json, after_json, note, actor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        now,
                        tenant,
                        rid,
                        "status_change",
                        json.dumps(before, default=str),
                        json.dumps(after, default=str),
                        (note or f"status:{before['status']}->{clean_status}")[:300],
                        actor[:120],
                    ],
                )
                return {
                    "success": True,
                    "message": f"Rule {rid} status set to {clean_status}.",
                    "status": clean_status,
                }
            finally:
                conn.close()

    def update_business_rule(
        self,
        rule_id: str,
        *,
        tenant_id: str | None = None,
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
        rid = (rule_id or "").strip()
        if not rid:
            return {"success": False, "message": "Missing rule_id."}

        tenant = _clean_tenant(tenant_id)
        normalized_status = (status or "").strip().lower() if status is not None else None
        if normalized_status is not None and normalized_status not in {"draft", "active", "disabled", "archived"}:
            return {"success": False, "message": f"Unsupported status '{status}'."}
        if action_payload is not None and not isinstance(action_payload, dict):
            return {"success": False, "message": "action_payload must be a JSON object."}

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT
                        rule_id, created_at, updated_at, tenant_id, domain, name, rule_type,
                        triggers_json, action_payload_json, notes, priority, status, source,
                        created_by, approved_by, version
                    FROM datada_business_rules
                    WHERE rule_id = ? AND tenant_id = ?
                    LIMIT 1
                    """,
                    [rid, tenant],
                ).fetchone()
                if not row:
                    return {"success": False, "message": f"Unknown rule_id '{rid}'."}

                current_domain = str(row[4] or "")
                current_name = str(row[5] or "")
                current_rule_type = str(row[6] or "")
                try:
                    current_triggers = json.loads(row[7] or "[]")
                except Exception:
                    current_triggers = []
                if not isinstance(current_triggers, list):
                    current_triggers = []
                try:
                    current_payload = json.loads(row[8] or "{}")
                except Exception:
                    current_payload = {}
                if not isinstance(current_payload, dict):
                    current_payload = {}
                current_notes = str(row[9] or "")
                current_priority = float(row[10] or 1.0)
                current_status = str(row[11] or "").lower()
                current_approved_by = str(row[14] or "")
                current_version = int(row[15] or 1)

                next_domain = _clean_domain(domain) if domain is not None else current_domain
                next_name = (name or "").strip()[:180] if name is not None else current_name
                next_rule_type = (rule_type or "").strip()[:64] if rule_type is not None else current_rule_type
                if triggers is None:
                    next_triggers = list(current_triggers)
                else:
                    next_triggers = [str(t).strip().lower()[:120] for t in triggers if str(t).strip()]
                    next_triggers = list(dict.fromkeys(next_triggers))
                next_payload = dict(action_payload) if action_payload is not None else dict(current_payload)
                next_notes = (notes or "").strip()[:500] if notes is not None else current_notes
                next_priority = (
                    float(max(0.0, min(10.0, float(priority)))) if priority is not None else current_priority
                )
                next_status = normalized_status if normalized_status is not None else current_status
                next_approved_by = actor if next_status == "active" else current_approved_by

                if len(next_name) < 2:
                    return {"success": False, "message": "Rule name must have at least 2 characters."}
                if not next_triggers:
                    return {"success": False, "message": "At least one trigger is required."}

                before = {
                    "domain": current_domain,
                    "name": current_name,
                    "rule_type": current_rule_type,
                    "triggers": current_triggers,
                    "action_payload": current_payload,
                    "notes": current_notes,
                    "priority": current_priority,
                    "status": current_status,
                    "approved_by": current_approved_by,
                    "version": current_version,
                }
                after = {
                    "domain": next_domain,
                    "name": next_name,
                    "rule_type": next_rule_type,
                    "triggers": next_triggers,
                    "action_payload": next_payload,
                    "notes": next_notes,
                    "priority": next_priority,
                    "status": next_status,
                    "approved_by": next_approved_by,
                    "version": current_version + 1,
                }
                if before == after:
                    return {
                        "success": True,
                        "message": "No changes detected.",
                        "status": current_status,
                    }

                now = datetime.utcnow()
                conn.execute(
                    """
                    UPDATE datada_business_rules
                    SET
                        updated_at = ?,
                        domain = ?,
                        name = ?,
                        rule_type = ?,
                        triggers_json = ?,
                        action_payload_json = ?,
                        notes = ?,
                        priority = ?,
                        status = ?,
                        approved_by = ?,
                        version = ?
                    WHERE rule_id = ? AND tenant_id = ?
                    """,
                    [
                        now,
                        next_domain,
                        next_name,
                        next_rule_type,
                        json.dumps(next_triggers, default=str),
                        json.dumps(next_payload, default=str),
                        next_notes,
                        next_priority,
                        next_status,
                        next_approved_by,
                        current_version + 1,
                        rid,
                        tenant,
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO datada_business_rule_events (
                        event_id, created_at, tenant_id, rule_id, action, before_json, after_json, note, actor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        now,
                        tenant,
                        rid,
                        "update",
                        json.dumps(before, default=str),
                        json.dumps(after, default=str),
                        (note or "rule_updated")[:300],
                        actor[:120],
                    ],
                )
                return {
                    "success": True,
                    "message": f"Rule {rid} updated.",
                    "status": next_status,
                }
            finally:
                conn.close()

    def rollback_business_rule(
        self,
        rule_id: str,
        *,
        tenant_id: str | None = None,
        actor: str = "admin",
    ) -> dict[str, Any]:
        rid = (rule_id or "").strip()
        if not rid:
            return {"success": False, "message": "Missing rule_id."}
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                current = conn.execute(
                    """
                    SELECT status, version
                    FROM datada_business_rules
                    WHERE rule_id = ? AND tenant_id = ?
                    LIMIT 1
                    """,
                    [rid, tenant],
                ).fetchone()
                if not current:
                    return {"success": False, "message": f"Unknown rule_id '{rid}'."}
                last_change = conn.execute(
                    """
                    SELECT before_json
                    FROM datada_business_rule_events
                    WHERE rule_id = ? AND tenant_id = ? AND action = 'status_change'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [rid, tenant],
                ).fetchone()
                if not last_change:
                    return {"success": False, "message": "No prior status change found for rollback."}
                try:
                    before = json.loads(last_change[0] or "{}")
                except Exception:
                    before = {}
                target_status = str(before.get("status") or "").strip().lower()
                if target_status not in {"draft", "active", "disabled", "archived"}:
                    return {"success": False, "message": "Rollback target status unavailable."}

                now = datetime.utcnow()
                next_version = int(current[1] or 1) + 1
                conn.execute(
                    """
                    UPDATE datada_business_rules
                    SET updated_at = ?, status = ?, version = ?
                    WHERE rule_id = ? AND tenant_id = ?
                    """,
                    [now, target_status, next_version, rid, tenant],
                )
                conn.execute(
                    """
                    INSERT INTO datada_business_rule_events (
                        event_id, created_at, tenant_id, rule_id, action, before_json, after_json, note, actor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        now,
                        tenant,
                        rid,
                        "rollback",
                        json.dumps({"status": str(current[0] or ""), "version": int(current[1] or 1)}),
                        json.dumps({"status": target_status, "version": next_version}),
                        "rollback_last_status_change",
                        actor[:120],
                    ],
                )
                return {
                    "success": True,
                    "message": f"Rule {rid} rolled back to status '{target_status}'.",
                    "status": target_status,
                }
            finally:
                conn.close()

    def get_matching_business_rules(
        self,
        goal: str,
        *,
        tenant_id: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        lower_goal = (goal or "").lower()
        if not lower_goal.strip():
            return []
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        rule_id, domain, name, rule_type, triggers_json, action_payload_json,
                        notes, priority, source
                    FROM datada_business_rules
                    WHERE tenant_id = ?
                      AND status = 'active'
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT 500
                    """,
                    [tenant],
                ).fetchall()
            finally:
                conn.close()

        matched: list[dict[str, Any]] = []
        for row in rows:
            try:
                triggers_raw = json.loads(row[4] or "[]")
            except Exception:
                triggers_raw = []
            triggers = [str(t).strip().lower() for t in (triggers_raw or []) if str(t).strip()]
            if not triggers:
                continue
            trigger_hits = [t for t in triggers if t in lower_goal]
            semantic_hits: list[str] = []
            semantic_scores: list[float] = []
            for trig in triggers:
                if trig in trigger_hits:
                    semantic_hits.append(trig)
                    semantic_scores.append(1.0)
                    continue
                sim = _semantic_similarity(lower_goal, trig)
                if sim >= 0.38:
                    semantic_hits.append(trig)
                    semantic_scores.append(sim)
            if not semantic_hits:
                continue
            try:
                payload = json.loads(row[5] or "{}")
            except Exception:
                payload = {}
            normalized_hits = (len(semantic_hits) / max(1, len(triggers)))
            semantic_quality = (sum(semantic_scores) / max(1, len(semantic_scores)))
            match_score = normalized_hits * semantic_quality * float(row[7] or 1.0)
            matched.append(
                {
                    "rule_id": str(row[0] or ""),
                    "domain": str(row[1] or ""),
                    "name": str(row[2] or ""),
                    "rule_type": str(row[3] or ""),
                    "triggers": triggers,
                    "trigger_hits": semantic_hits,
                    "action_payload": payload if isinstance(payload, dict) else {},
                    "notes": str(row[6] or ""),
                    "priority": float(row[7] or 1.0),
                    "source": str(row[8] or ""),
                    "match_score": round(float(match_score), 4),
                    "semantic_quality": round(float(semantic_quality), 4),
                }
            )
        matched.sort(key=lambda x: (x["match_score"], x["priority"]), reverse=True)
        return matched[: max(1, min(40, int(limit)))]

    def register_tool_candidate(
        self,
        *,
        tenant_id: str | None,
        title: str,
        sql_text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        clean_sql = (sql_text or "").strip()
        clean_title = (title or "candidate_tool").strip()[:180]
        if not clean_sql:
            return ""
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT tool_id
                    FROM datada_agent_toolsmith
                    WHERE sql_text = ? AND status IN ('candidate', 'staged', 'promoted')
                      AND tenant_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [clean_sql, tenant],
                ).fetchone()
                if existing and existing[0]:
                    return str(existing[0])

                tool_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO datada_agent_toolsmith (
                        tool_id,
                        created_at,
                        updated_at,
                        tenant_id,
                        status,
                        source,
                        title,
                        sql_text,
                        test_sql_text,
                        test_success,
                        test_message,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tool_id,
                        datetime.utcnow(),
                        datetime.utcnow(),
                        tenant,
                        "candidate",
                        source,
                        clean_title,
                        clean_sql,
                        "",
                        False,
                        "candidate created",
                        json.dumps(metadata or {}, default=str),
                    ],
                )
                return tool_id
            finally:
                conn.close()

    def _is_safe_select_sql(self, sql_text: str) -> bool:
        text = (sql_text or "").strip().lower()
        if not text:
            return False
        blocked = ["drop ", "delete ", "truncate ", "update ", "insert ", "alter ", "create "]
        if any(token in text for token in blocked):
            return False
        return text.startswith("select") or text.startswith("with") or text.startswith("explain")

    def stage_tool_candidate(
        self,
        tool_id: str,
        *,
        tenant_id: str | None,
        db_path: Path | str,
    ) -> dict[str, Any]:
        tid = (tool_id or "").strip()
        if not tid:
            return {"success": False, "message": "Missing tool_id."}
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT tool_id, sql_text, status
                    FROM datada_agent_toolsmith
                    WHERE tool_id = ?
                      AND tenant_id = ?
                    LIMIT 1
                    """,
                    [tid, tenant],
                ).fetchone()
                if not row:
                    return {"success": False, "message": f"Unknown tool_id '{tid}'."}
                sql_text = str(row[1] or "").strip()
                if not self._is_safe_select_sql(sql_text):
                    conn.execute(
                        """
                        UPDATE datada_agent_toolsmith
                        SET updated_at = ?, test_success = ?, test_message = ?
                        WHERE tool_id = ?
                          AND tenant_id = ?
                        """,
                        [datetime.utcnow(), False, "Blocked: SQL is not read-only SELECT/WITH.", tid, tenant],
                    )
                    return {"success": False, "message": "Candidate SQL failed safety checks."}

                probe_sql = sql_text.rstrip().rstrip(";")
                if " limit " not in probe_sql.lower():
                    probe_sql = f"{probe_sql} LIMIT 5"
                try:
                    probe = duckdb.connect(str(Path(db_path).expanduser()), read_only=True)
                    probe.execute(probe_sql).fetchmany(1)
                    probe.close()
                except Exception as exc:
                    conn.execute(
                        """
                        UPDATE datada_agent_toolsmith
                        SET updated_at = ?, test_success = ?, test_message = ?, test_sql_text = ?
                        WHERE tool_id = ?
                          AND tenant_id = ?
                        """,
                        [
                            datetime.utcnow(),
                            False,
                            f"Stage test failed: {exc}",
                            probe_sql,
                            tid,
                            tenant,
                        ],
                    )
                    return {"success": False, "message": f"Stage test failed: {exc}"}

                conn.execute(
                    """
                    UPDATE datada_agent_toolsmith
                    SET updated_at = ?, status = ?, test_success = ?, test_message = ?, test_sql_text = ?
                    WHERE tool_id = ?
                      AND tenant_id = ?
                    """,
                    [datetime.utcnow(), "staged", True, "Stage test passed.", probe_sql, tid, tenant],
                )
                return {"success": True, "message": f"Tool {tid} staged successfully."}
            finally:
                conn.close()

    def promote_tool_candidate(self, tool_id: str, *, tenant_id: str | None = None) -> dict[str, Any]:
        tid = (tool_id or "").strip()
        if not tid:
            return {"success": False, "message": "Missing tool_id."}
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT status, test_success
                    FROM datada_agent_toolsmith
                    WHERE tool_id = ?
                      AND tenant_id = ?
                    LIMIT 1
                    """,
                    [tid, tenant],
                ).fetchone()
                if not row:
                    return {"success": False, "message": f"Unknown tool_id '{tid}'."}
                status = str(row[0] or "")
                tested = bool(row[1])
                if status not in {"staged", "promoted"} or not tested:
                    return {
                        "success": False,
                        "message": "Tool must be staged with passing tests before promotion.",
                    }
                conn.execute(
                    """
                    UPDATE datada_agent_toolsmith
                    SET updated_at = ?, status = ?, test_message = ?
                    WHERE tool_id = ?
                      AND tenant_id = ?
                    """,
                    [datetime.utcnow(), "promoted", "Promoted for autonomous reuse.", tid, tenant],
                )
                return {"success": True, "message": f"Tool {tid} promoted."}
            finally:
                conn.close()

    def rollback_tool_candidate(
        self,
        tool_id: str,
        *,
        tenant_id: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        tid = (tool_id or "").strip()
        if not tid:
            return {"success": False, "message": "Missing tool_id."}
        note = (reason or "manual rollback").strip()[:300]
        tenant = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT tool_id FROM datada_agent_toolsmith
                    WHERE tool_id = ?
                      AND tenant_id = ?
                    LIMIT 1
                    """,
                    [tid, tenant],
                ).fetchone()
                if not existing:
                    return {"success": False, "message": f"Unknown tool_id '{tid}'."}
                conn.execute(
                    """
                    UPDATE datada_agent_toolsmith
                    SET updated_at = ?, status = ?, test_message = ?
                    WHERE tool_id = ?
                      AND tenant_id = ?
                    """,
                    [datetime.utcnow(), "rolled_back", note, tid, tenant],
                )
                return {"success": True, "message": f"Tool {tid} rolled back."}
            finally:
                conn.close()

    def list_tool_candidates(
        self,
        *,
        tenant_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clean_status = (status or "").strip().lower()
        tenant = _clean_tenant(tenant_id)
        where = "WHERE tenant_id = ?"
        params: list[Any] = [tenant, max(1, min(500, int(limit)))]
        if clean_status:
            where = "WHERE tenant_id = ? AND LOWER(status) = ?"
            params = [tenant, clean_status, params[1]]
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT tool_id, created_at, updated_at, status, source, title, sql_text,
                           test_sql_text, test_success, test_message, metadata_json
                    FROM datada_agent_toolsmith
                    {where}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            metadata = {}
            try:
                parsed = json.loads(row[10] or "{}")
                if isinstance(parsed, dict):
                    metadata = parsed
            except Exception:
                metadata = {}
            out.append(
                {
                    "tool_id": str(row[0] or ""),
                    "created_at": str(row[1] or ""),
                    "updated_at": str(row[2] or ""),
                    "status": str(row[3] or ""),
                    "source": str(row[4] or ""),
                    "title": str(row[5] or ""),
                    "sql_text": str(row[6] or ""),
                    "test_sql_text": str(row[7] or ""),
                    "test_success": bool(row[8]),
                    "test_message": str(row[9] or ""),
                    "metadata": metadata,
                }
            )
        return out

    def learn_from_success(
        self,
        *,
        tenant_id: str | None,
        goal: str,
        plan: dict[str, Any],
        score: float,
        source: str = "autonomous_learning",
    ) -> str:
        """Promote successful corrections as reusable routing hints."""

        if score < 0.78:
            return ""

        goal_lower = goal.lower()
        keyword = ""
        for candidate in [
            "exchange rate",
            "fx charges",
            "forex charges",
            "forex markup",
            "forex",
            "markup",
            "mt103",
            "refund",
            "platform wise",
            "month wise",
            "deal type",
            "customer country",
        ]:
            if candidate in goal_lower:
                keyword = candidate
                break
        if not keyword:
            terms = [
                t
                for t in sorted(_tokens(goal_lower))
                if len(t) >= 4 and t not in _GENERIC_CORRECTION_TOKENS
            ]
            keyword = " ".join(terms[:2]) if terms else ""
        if not keyword or _is_low_specificity_keyword(keyword):
            return ""

        return self.upsert_correction(
            tenant_id=tenant_id,
            keyword=keyword,
            target_table=str(plan.get("table") or ""),
            target_metric=str(plan.get("metric") or ""),
            target_dimensions=list(plan.get("dimensions") or []),
            notes=f"Auto-learned from successful run (score={score:.2f})",
            source=source,
            weight=min(2.5, max(1.0, score)),
        )

    # ------------------------------------------------------------------
    # Domain Glossary
    # ------------------------------------------------------------------

    def upsert_glossary_term(
        self,
        *,
        term: str,
        definition: str,
        sql_expression: str = "",
        target_table: str = "",
        target_column: str = "",
        examples: list[str] | None = None,
        contributed_by: str = "system",
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Add or update a business term in the domain glossary."""
        tid = _clean_tenant(tenant_id)
        now = datetime.utcnow().isoformat()
        examples_json = json.dumps(examples or [])

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT term_id, version FROM datada_glossary WHERE tenant_id=? AND LOWER(term)=LOWER(?)",
                    [tid, term.strip()],
                ).fetchone()

                if existing:
                    term_id = existing[0]
                    version = int(existing[1] or 0) + 1
                    conn.execute(
                        """UPDATE datada_glossary SET definition=?, sql_expression=?,
                           target_table=?, target_column=?, examples_json=?,
                           contributed_by=?, version=?, updated_at=?, deprecated=FALSE
                           WHERE term_id=?""",
                        [definition, sql_expression, target_table, target_column,
                         examples_json, contributed_by, version, now, term_id],
                    )
                else:
                    term_id = str(uuid.uuid4())
                    conn.execute(
                        """INSERT INTO datada_glossary
                           (term_id, created_at, updated_at, tenant_id, term, definition,
                            sql_expression, target_table, target_column, examples_json,
                            contributed_by, version, deprecated, effectiveness_score, use_count)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,1,FALSE,0.5,0)""",
                        [term_id, now, now, tid, term.strip(), definition,
                         sql_expression, target_table, target_column, examples_json,
                         contributed_by],
                    )
                return {"term_id": term_id, "term": term.strip(), "status": "ok"}
            finally:
                conn.close()

    def list_glossary(self, *, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """List all active glossary terms for a tenant."""
        tid = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT term_id, term, definition, sql_expression,
                              target_table, target_column, examples_json,
                              contributed_by, version, deprecated,
                              effectiveness_score, use_count, updated_at
                       FROM datada_glossary WHERE tenant_id=?
                       ORDER BY term""",
                    [tid],
                ).fetchall()
                return [
                    {
                        "term_id": r[0], "term": r[1], "definition": r[2],
                        "sql_expression": r[3], "target_table": r[4],
                        "target_column": r[5],
                        "examples": json.loads(r[6]) if r[6] else [],
                        "contributed_by": r[7], "version": r[8],
                        "deprecated": bool(r[9]),
                        "effectiveness_score": float(r[10] or 0.5),
                        "use_count": int(r[11] or 0),
                        "updated_at": r[12],
                    }
                    for r in rows
                ]
            finally:
                conn.close()

    def resolve_glossary(self, text: str, *, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """Find glossary terms that match tokens in the given text."""
        tid = _clean_tenant(tenant_id)
        tokens = _expand_tokens(text)
        if not tokens:
            return []

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT term_id, term, definition, sql_expression,
                              target_table, target_column
                       FROM datada_glossary
                       WHERE tenant_id=? AND deprecated=FALSE""",
                    [tid],
                ).fetchall()
                matches = []
                for r in rows:
                    term = str(r[1] or "")
                    definition = str(r[2] or "")
                    term_tokens = _expand_tokens(term)
                    definition_tokens = _expand_tokens(definition)
                    union_tokens = term_tokens | definition_tokens
                    overlap = tokens & union_tokens
                    overlap_score = len(overlap) / max(1, len(term_tokens))
                    semantic_score = _semantic_similarity(text, f"{term} {definition}")
                    final_score = max(overlap_score, semantic_score)
                    if final_score < 0.30:
                        continue
                    matches.append(
                        {
                            "term_id": r[0],
                            "term": term,
                            "definition": definition,
                            "sql_expression": r[3],
                            "target_table": r[4],
                            "target_column": r[5],
                            "match_score": round(float(final_score), 4),
                            "matched_tokens": sorted(overlap)[:8],
                        }
                    )

                matches.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
                top = matches[:20]
                # Safety guard: avoid unsafe semantic remaps on near-tie cross-table terms.
                if len(top) >= 2:
                    first = top[0]
                    second = top[1]
                    first_score = float(first.get("match_score", 0.0) or 0.0)
                    second_score = float(second.get("match_score", 0.0) or 0.0)
                    first_table = str(first.get("target_table") or "")
                    second_table = str(second.get("target_table") or "")
                    if (
                        first_table
                        and second_table
                        and first_table != second_table
                        and first_score >= 0.55
                        and second_score >= 0.55
                        and abs(first_score - second_score) <= 0.04
                    ):
                        return []
                for item in top:
                    conn.execute(
                        "UPDATE datada_glossary SET use_count = use_count + 1 WHERE term_id = ?",
                        [item["term_id"]],
                    )
                return top
            finally:
                conn.close()

    def bootstrap_glossary_from_business_dictionary(
        self,
        *,
        business_dictionary: dict[str, Any] | None,
        tenant_id: str | None = None,
        contributed_by: str = "system_bootstrap",
    ) -> dict[str, int]:
        """Seed glossary rows from business dictionary payload.

        This keeps term-to-definition coverage high for business-facing schema
        dictionary output, and supports semantic retrieval of domain terms.
        """
        if not isinstance(business_dictionary, dict):
            return {"inserted": 0, "updated": 0, "total": 0}

        tid = _clean_tenant(tenant_id)
        existing_terms = {
            str(row.get("term") or "").strip().lower()
            for row in self.list_glossary(tenant_id=tid)
            if str(row.get("term") or "").strip()
        }
        inserted = 0
        updated = 0

        table_map = business_dictionary.get("tables")
        if not isinstance(table_map, dict):
            table_map = {}

        for table, table_payload in table_map.items():
            if not isinstance(table_payload, dict):
                continue
            col_defs = table_payload.get("column_definitions")
            if isinstance(col_defs, dict):
                for col, definition in col_defs.items():
                    term = str(col or "").strip()
                    desc = str(definition or "").strip()
                    if not term or not desc:
                        continue
                    before = term.lower() in existing_terms
                    self.upsert_glossary_term(
                        tenant_id=tid,
                        term=term,
                        definition=desc,
                        target_table=str(table or ""),
                        target_column=term,
                        contributed_by=contributed_by,
                    )
                    if before:
                        updated += 1
                    else:
                        inserted += 1
                        existing_terms.add(term.lower())

            metric_defs = table_payload.get("metric_definitions")
            if isinstance(metric_defs, dict):
                for metric, definition in metric_defs.items():
                    term = str(metric or "").strip()
                    desc = str(definition or "").strip()
                    if not term or not desc:
                        continue
                    before = term.lower() in existing_terms
                    self.upsert_glossary_term(
                        tenant_id=tid,
                        term=term,
                        definition=desc,
                        target_table=str(table or ""),
                        target_column="",
                        contributed_by=contributed_by,
                    )
                    if before:
                        updated += 1
                    else:
                        inserted += 1
                        existing_terms.add(term.lower())

        global_defs = business_dictionary.get("global_column_definitions")
        if isinstance(global_defs, dict):
            for term, definition in global_defs.items():
                clean_term = str(term or "").strip()
                clean_def = str(definition or "").strip()
                if not clean_term or not clean_def:
                    continue
                before = clean_term.lower() in existing_terms
                self.upsert_glossary_term(
                    tenant_id=tid,
                    term=clean_term,
                    definition=clean_def,
                    target_table="",
                    target_column=clean_term,
                    contributed_by=contributed_by,
                )
                if before:
                    updated += 1
                else:
                    inserted += 1
                    existing_terms.add(clean_term.lower())

        return {"inserted": inserted, "updated": updated, "total": inserted + updated}

    # ------------------------------------------------------------------
    # Expert Teachings
    # ------------------------------------------------------------------

    def add_teaching(
        self,
        *,
        teaching_text: str,
        expert_name: str = "anonymous",
        keyword: str = "",
        target_table: str = "",
        target_metric: str = "",
        target_dimensions: list[str] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an expert teaching as institutional knowledge.

        Teachings are natural language rules from subject matter experts.
        They get automatically converted to correction rules when possible.
        """
        tid = _clean_tenant(tenant_id)
        now = datetime.utcnow().isoformat()
        teaching_id = str(uuid.uuid4())
        dims_json = json.dumps(target_dimensions or [])

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO datada_teachings
                       (teaching_id, created_at, tenant_id, expert_name,
                        teaching_text, parsed_keyword, parsed_table,
                        parsed_metric, parsed_dimensions_json,
                        confidence, times_applied, times_correct, active)
                       VALUES (?,?,?,?,?,?,?,?,?,0.5,0,0,TRUE)""",
                    [teaching_id, now, tid, expert_name, teaching_text,
                     keyword, target_table, target_metric, dims_json],
                )

                # Auto-create a correction rule if keyword and table are provided
                correction_id = ""
                if keyword and target_table:
                    correction_id = self.upsert_correction(
                        tenant_id=tid,
                        keyword=keyword,
                        target_table=target_table,
                        target_metric=target_metric,
                        target_dimensions=target_dimensions or [],
                        notes=f"From teaching by {expert_name}: {teaching_text[:120]}",
                        source="teaching",
                    )

                return {
                    "teaching_id": teaching_id,
                    "correction_id": correction_id,
                    "status": "ok",
                    "auto_correction_created": bool(correction_id),
                }
            finally:
                conn.close()

    def list_teachings(self, *, tenant_id: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
        """List all teachings for a tenant."""
        tid = _clean_tenant(tenant_id)
        clause = "AND active = TRUE" if active_only else ""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""SELECT teaching_id, created_at, expert_name, teaching_text,
                               parsed_keyword, parsed_table, parsed_metric,
                               parsed_dimensions_json, confidence,
                               times_applied, times_correct, active
                        FROM datada_teachings WHERE tenant_id=? {clause}
                        ORDER BY created_at DESC""",
                    [tid],
                ).fetchall()
                return [
                    {
                        "teaching_id": r[0], "created_at": r[1],
                        "expert_name": r[2], "teaching_text": r[3],
                        "keyword": r[4], "target_table": r[5],
                        "target_metric": r[6],
                        "target_dimensions": json.loads(r[7]) if r[7] else [],
                        "confidence": float(r[8] or 0.5),
                        "times_applied": int(r[9] or 0),
                        "times_correct": int(r[10] or 0),
                        "active": bool(r[11]),
                    }
                    for r in rows
                ]
            finally:
                conn.close()

    def record_teaching_outcome(
        self, teaching_id: str, *, correct: bool, tenant_id: str | None = None
    ) -> None:
        """Record whether a teaching led to a correct answer."""
        tid = _clean_tenant(tenant_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE datada_teachings
                       SET times_applied = times_applied + 1,
                           times_correct = times_correct + CASE WHEN ? THEN 1 ELSE 0 END,
                           confidence = CASE
                             WHEN times_applied > 0
                             THEN (times_correct + CASE WHEN ? THEN 1.0 ELSE 0.0 END)
                                  / (times_applied + 1.0)
                             ELSE 0.5 END
                       WHERE teaching_id = ? AND tenant_id = ?""",
                    [correct, correct, teaching_id, tid],
                )
            finally:
                conn.close()
