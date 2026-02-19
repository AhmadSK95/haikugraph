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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_agent_correction_events (
                        event_id VARCHAR,
                        created_at TIMESTAMP,
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

    def list_corrections(self, *, limit: int = 250, include_disabled: bool = True) -> list[dict[str, Any]]:
        where_clause = "" if include_disabled else "WHERE enabled = TRUE"
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
                    [max(1, min(1000, int(limit)))],
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

    def set_correction_enabled(self, correction_id: str, enabled: bool) -> bool:
        cid = (correction_id or "").strip()
        if not cid:
            return False
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT correction_id, enabled FROM datada_agent_corrections
                    WHERE correction_id = ?
                    LIMIT 1
                    """,
                    [cid],
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
                    """,
                    [bool(enabled), cid],
                )
                conn.execute(
                    """
                    INSERT INTO datada_agent_correction_events VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        datetime.utcnow(),
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

    def rollback_correction(self, correction_id: str) -> dict[str, Any]:
        cid = (correction_id or "").strip()
        if not cid:
            return {"success": False, "message": "Missing correction_id."}
        with self._lock:
            conn = self._connect()
            try:
                current = conn.execute(
                    """
                    SELECT enabled FROM datada_agent_corrections
                    WHERE correction_id = ?
                    LIMIT 1
                    """,
                    [cid],
                ).fetchone()
                if not current:
                    return {"success": False, "message": f"Unknown correction_id '{cid}'."}
                last_change = conn.execute(
                    """
                    SELECT enabled_before, enabled_after
                    FROM datada_agent_correction_events
                    WHERE correction_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [cid],
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
                    """,
                    [previous_state, cid],
                )
                conn.execute(
                    """
                    INSERT INTO datada_agent_correction_events VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        datetime.utcnow(),
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

    def register_tool_candidate(
        self,
        *,
        title: str,
        sql_text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        clean_sql = (sql_text or "").strip()
        clean_title = (title or "candidate_tool").strip()[:180]
        if not clean_sql:
            return ""
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT tool_id
                    FROM datada_agent_toolsmith
                    WHERE sql_text = ? AND status IN ('candidate', 'staged', 'promoted')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [clean_sql],
                ).fetchone()
                if existing and existing[0]:
                    return str(existing[0])

                tool_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO datada_agent_toolsmith VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tool_id,
                        datetime.utcnow(),
                        datetime.utcnow(),
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

    def stage_tool_candidate(self, tool_id: str, *, db_path: Path | str) -> dict[str, Any]:
        tid = (tool_id or "").strip()
        if not tid:
            return {"success": False, "message": "Missing tool_id."}
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT tool_id, sql_text, status
                    FROM datada_agent_toolsmith
                    WHERE tool_id = ?
                    LIMIT 1
                    """,
                    [tid],
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
                        """,
                        [datetime.utcnow(), False, "Blocked: SQL is not read-only SELECT/WITH.", tid],
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
                        """,
                        [
                            datetime.utcnow(),
                            False,
                            f"Stage test failed: {exc}",
                            probe_sql,
                            tid,
                        ],
                    )
                    return {"success": False, "message": f"Stage test failed: {exc}"}

                conn.execute(
                    """
                    UPDATE datada_agent_toolsmith
                    SET updated_at = ?, status = ?, test_success = ?, test_message = ?, test_sql_text = ?
                    WHERE tool_id = ?
                    """,
                    [datetime.utcnow(), "staged", True, "Stage test passed.", probe_sql, tid],
                )
                return {"success": True, "message": f"Tool {tid} staged successfully."}
            finally:
                conn.close()

    def promote_tool_candidate(self, tool_id: str) -> dict[str, Any]:
        tid = (tool_id or "").strip()
        if not tid:
            return {"success": False, "message": "Missing tool_id."}
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT status, test_success
                    FROM datada_agent_toolsmith
                    WHERE tool_id = ?
                    LIMIT 1
                    """,
                    [tid],
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
                    """,
                    [datetime.utcnow(), "promoted", "Promoted for autonomous reuse.", tid],
                )
                return {"success": True, "message": f"Tool {tid} promoted."}
            finally:
                conn.close()

    def rollback_tool_candidate(self, tool_id: str, *, reason: str = "") -> dict[str, Any]:
        tid = (tool_id or "").strip()
        if not tid:
            return {"success": False, "message": "Missing tool_id."}
        note = (reason or "manual rollback").strip()[:300]
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT tool_id FROM datada_agent_toolsmith
                    WHERE tool_id = ?
                    LIMIT 1
                    """,
                    [tid],
                ).fetchone()
                if not existing:
                    return {"success": False, "message": f"Unknown tool_id '{tid}'."}
                conn.execute(
                    """
                    UPDATE datada_agent_toolsmith
                    SET updated_at = ?, status = ?, test_message = ?
                    WHERE tool_id = ?
                    """,
                    [datetime.utcnow(), "rolled_back", note, tid],
                )
                return {"success": True, "message": f"Tool {tid} rolled back."}
            finally:
                conn.close()

    def list_tool_candidates(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clean_status = (status or "").strip().lower()
        where = ""
        params: list[Any] = [max(1, min(500, int(limit)))]
        if clean_status:
            where = "WHERE LOWER(status) = ?"
            params = [clean_status, params[0]]
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
