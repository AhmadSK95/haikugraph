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

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._ensure_tables()

    def _connect(self) -> duckdb.DuckDBPyConnection:
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
            finally:
                conn.close()

    def store_turn(
        self,
        *,
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
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_agent_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        memory_id,
                        datetime.utcnow(),
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

    def recall(self, goal: str, *, limit: int = 3, scan_limit: int = 400) -> list[dict[str, Any]]:
        """Return similar successful historical runs."""

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
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [scan_limit],
                ).fetchall()
            finally:
                conn.close()

        scored: list[dict[str, Any]] = []
        for row in rows:
            past_goal = str(row[0] or "")
            sim = _similarity(goal, past_goal)
            if sim < 0.20:
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

            scored.append(
                {
                    "similarity": round(sim, 4),
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
            )

        scored.sort(
            key=lambda x: (
                x["similarity"],
                x["confidence_score"],
                1.0 if x["correction_applied"] else 0.0,
            ),
            reverse=True,
        )
        return scored[: max(1, min(10, int(limit)))]

    def store_feedback(
        self,
        *,
        trace_id: str | None,
        session_id: str | None,
        goal: str | None,
        issue: str,
        suggested_fix: str | None,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_agent_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        feedback_id,
                        datetime.utcnow(),
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

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT correction_id
                    FROM datada_agent_corrections
                    WHERE enabled = TRUE
                      AND LOWER(keyword) = ?
                      AND target_table = ?
                      AND target_metric = ?
                    LIMIT 1
                    """,
                    [norm_keyword, target_table, target_metric],
                ).fetchone()
                if existing and existing[0]:
                    return str(existing[0])

                correction_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO datada_agent_corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        correction_id,
                        datetime.utcnow(),
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

    def get_matching_corrections(self, goal: str, *, limit: int = 4) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT correction_id, source, keyword, target_table, target_metric, target_dimensions_json, notes, weight
                    FROM datada_agent_corrections
                    WHERE enabled = TRUE
                    ORDER BY created_at DESC
                    LIMIT 250
                    """
                ).fetchall()
            finally:
                conn.close()

        lower_goal = goal.lower()
        out: list[dict[str, Any]] = []
        for row in rows:
            keyword = str(row[2] or "")
            if not keyword:
                continue
            substring_match = keyword in lower_goal
            sim = 1.0 if substring_match else _similarity(goal, keyword)
            if sim < 0.18:
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

    def learn_from_success(
        self,
        *,
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
            "forex markup",
            "forex",
            "markup",
            "mt103",
            "refund",
            "platform wise",
            "month wise",
            "quotes",
            "customers",
            "bookings",
        ]:
            if candidate in goal_lower:
                keyword = candidate
                break
        if not keyword:
            terms = [t for t in sorted(_tokens(goal_lower)) if len(t) >= 5]
            keyword = " ".join(terms[:2]) if terms else ""
        if not keyword:
            return ""

        return self.upsert_correction(
            keyword=keyword,
            target_table=str(plan.get("table") or ""),
            target_metric=str(plan.get("metric") or ""),
            target_dimensions=list(plan.get("dimensions") or []),
            notes=f"Auto-learned from successful run (score={score:.2f})",
            source=source,
            weight=min(2.5, max(1.0, score)),
        )
