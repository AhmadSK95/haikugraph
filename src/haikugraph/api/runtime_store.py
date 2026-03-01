"""Durable runtime state for sessions, budgets, async jobs, and trust metrics."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb


def _utc_now() -> datetime:
    return datetime.utcnow()


def _utc_iso() -> str:
    return _utc_now().isoformat() + "Z"


def _percentile(values: list[float], pct: float) -> float:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return float(clean[0])
    rank = (float(pct) / 100.0) * (len(clean) - 1)
    idx = int(round(rank))
    idx = max(0, min(idx, len(clean) - 1))
    return float(clean[idx])


class RuntimeStore:
    """Thread-safe runtime persistence for API orchestration concerns."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).expanduser()
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
                    CREATE TABLE IF NOT EXISTS datada_session_turns (
                        turn_id VARCHAR,
                        created_at TIMESTAMP,
                        session_scope VARCHAR,
                        connection_id VARCHAR,
                        session_id VARCHAR,
                        goal VARCHAR,
                        answer_markdown VARCHAR,
                        success BOOLEAN,
                        sql_text VARCHAR,
                        confidence_score DOUBLE,
                        metadata_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_budget_usage (
                        tenant_id VARCHAR,
                        bucket_hour VARCHAR,
                        query_count BIGINT,
                        updated_at TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_async_jobs (
                        job_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        status VARCHAR,
                        tenant_id VARCHAR,
                        connection_id VARCHAR,
                        session_id VARCHAR,
                        request_json VARCHAR,
                        response_json VARCHAR,
                        error_text VARCHAR,
                        runtime_ms DOUBLE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_run_metrics (
                        metric_id VARCHAR,
                        created_at TIMESTAMP,
                        tenant_id VARCHAR,
                        connection_id VARCHAR,
                        session_scope VARCHAR,
                        success BOOLEAN,
                        confidence_score DOUBLE,
                        execution_ms DOUBLE,
                        llm_mode VARCHAR,
                        provider VARCHAR,
                        row_count BIGINT,
                        warning_count BIGINT,
                        metadata_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_incident_events (
                        incident_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR,
                        severity VARCHAR,
                        status VARCHAR,
                        source VARCHAR,
                        title VARCHAR,
                        summary VARCHAR,
                        fingerprint VARCHAR,
                        metadata_json VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_scenario_sets (
                        scenario_set_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR,
                        connection_id VARCHAR,
                        name VARCHAR,
                        assumptions_json VARCHAR,
                        status VARCHAR,
                        version BIGINT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datada_schema_signatures (
                        record_id VARCHAR,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        tenant_id VARCHAR,
                        connection_id VARCHAR,
                        dataset_signature VARCHAR,
                        schema_signature VARCHAR,
                        metadata_json VARCHAR
                    )
                    """
                )
            finally:
                conn.close()

    def load_session_turns(self, session_scope: str, *, limit: int = 20) -> list[dict[str, Any]]:
        scope = (session_scope or "").strip()
        if not scope:
            return []
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT goal, answer_markdown, success, sql_text, confidence_score, metadata_json
                    FROM (
                        SELECT *
                        FROM datada_session_turns
                        WHERE session_scope = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) t
                    ORDER BY created_at ASC
                    """,
                    [scope, max(1, min(200, int(limit)))],
                ).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            try:
                metadata_raw = json.loads(row[5] or "{}")
                if isinstance(metadata_raw, dict):
                    metadata = metadata_raw
            except Exception:
                metadata = {}
            out.append(
                {
                    "goal": str(row[0] or ""),
                    "answer_markdown": str(row[1] or ""),
                    "success": bool(row[2]),
                    "sql": str(row[3] or ""),
                    "confidence_score": float(row[4] or 0.0),
                    **metadata,
                }
            )
        return out

    def append_session_turn(
        self,
        *,
        session_scope: str,
        connection_id: str,
        session_id: str,
        goal: str,
        answer_markdown: str,
        success: bool,
        sql: str | None,
        confidence_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        turn_id = str(uuid.uuid4())
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_session_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        turn_id,
                        _utc_now(),
                        session_scope,
                        connection_id,
                        session_id,
                        goal,
                        answer_markdown,
                        bool(success),
                        sql or "",
                        float(confidence_score),
                        json.dumps(metadata or {}, default=str),
                    ],
                )
            finally:
                conn.close()
        return turn_id

    def clear_session(self, session_scope: str) -> int:
        scope = (session_scope or "").strip()
        if not scope:
            return 0
        with self._lock:
            conn = self._connect()
            try:
                deleted = conn.execute(
                    """
                    DELETE FROM datada_session_turns
                    WHERE session_scope = ?
                    RETURNING turn_id
                    """,
                    [scope],
                ).fetchall()
            finally:
                conn.close()
        return len(deleted)

    def consume_budget(self, *, tenant_id: str, limit_per_hour: int) -> dict[str, Any]:
        clean_tenant = (tenant_id or "public").strip() or "public"
        cap = max(1, int(limit_per_hour))
        now = _utc_now()
        hour_bucket = now.strftime("%Y%m%d%H")
        cutoff = (now - timedelta(hours=72)).strftime("%Y%m%d%H")

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    DELETE FROM datada_budget_usage
                    WHERE bucket_hour < ?
                    """,
                    [cutoff],
                )
                row = conn.execute(
                    """
                    SELECT query_count
                    FROM datada_budget_usage
                    WHERE tenant_id = ? AND bucket_hour = ?
                    LIMIT 1
                    """,
                    [clean_tenant, hour_bucket],
                ).fetchone()
                current = int(row[0] or 0) if row else 0
                if current >= cap:
                    return {
                        "allowed": False,
                        "remaining": 0,
                        "limit_per_hour": cap,
                        "message": f"Hourly query budget exhausted for tenant '{clean_tenant}'.",
                    }
                next_count = current + 1
                if row:
                    conn.execute(
                        """
                        UPDATE datada_budget_usage
                        SET query_count = ?, updated_at = ?
                        WHERE tenant_id = ? AND bucket_hour = ?
                        """,
                        [next_count, now, clean_tenant, hour_bucket],
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO datada_budget_usage VALUES (?, ?, ?, ?)
                        """,
                        [clean_tenant, hour_bucket, next_count, now],
                    )
            finally:
                conn.close()

        return {
            "allowed": True,
            "remaining": max(0, cap - next_count),
            "limit_per_hour": cap,
            "message": "Budget check passed.",
        }

    def create_async_job(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        session_id: str,
        request_payload: dict[str, Any],
    ) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_async_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        job_id,
                        _utc_now(),
                        _utc_now(),
                        "queued",
                        tenant_id,
                        connection_id,
                        session_id,
                        json.dumps(request_payload, default=str),
                        "",
                        "",
                        0.0,
                    ],
                )
            finally:
                conn.close()
        return job_id

    def update_async_job(
        self,
        *,
        job_id: str,
        status: str,
        response_payload: dict[str, Any] | None = None,
        error_text: str | None = None,
        runtime_ms: float | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE datada_async_jobs
                    SET updated_at = ?, status = ?, response_json = ?, error_text = ?, runtime_ms = ?
                    WHERE job_id = ?
                    """,
                    [
                        _utc_now(),
                        status,
                        json.dumps(response_payload or {}, default=str) if response_payload else "",
                        error_text or "",
                        float(runtime_ms or 0.0),
                        job_id,
                    ],
                )
            finally:
                conn.close()

    def get_async_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT job_id, created_at, updated_at, status, tenant_id, connection_id,
                           session_id, request_json, response_json, error_text, runtime_ms
                    FROM datada_async_jobs
                    WHERE job_id = ?
                    LIMIT 1
                    """,
                    [job_id],
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return None

        request_payload: dict[str, Any] = {}
        response_payload: dict[str, Any] = {}
        try:
            request_raw = json.loads(row[7] or "{}")
            if isinstance(request_raw, dict):
                request_payload = request_raw
        except Exception:
            request_payload = {}
        try:
            response_raw = json.loads(row[8] or "{}")
            if isinstance(response_raw, dict):
                response_payload = response_raw
        except Exception:
            response_payload = {}

        return {
            "job_id": str(row[0]),
            "created_at": str(row[1] or ""),
            "updated_at": str(row[2] or ""),
            "status": str(row[3] or "unknown"),
            "tenant_id": str(row[4] or ""),
            "connection_id": str(row[5] or ""),
            "session_id": str(row[6] or ""),
            "request": request_payload,
            "response": response_payload,
            "error": str(row[9] or ""),
            "runtime_ms": float(row[10] or 0.0),
        }

    def count_async_jobs(
        self,
        *,
        tenant_id: str | None = None,
        statuses: tuple[str, ...] = ("queued", "running"),
    ) -> dict[str, int]:
        """Return async job counts for load-shedding decisions."""
        clean_statuses = [str(s).strip().lower() for s in statuses if str(s).strip()]
        if not clean_statuses:
            clean_statuses = ["queued", "running"]

        placeholders = ", ".join(["?"] * len(clean_statuses))
        params: list[Any] = list(clean_statuses)
        sql = (
            "SELECT status, COUNT(*) AS cnt "
            "FROM datada_async_jobs "
            f"WHERE LOWER(status) IN ({placeholders})"
        )
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append((tenant_id or "public").strip() or "public")
        sql += " GROUP BY status"

        counts: dict[str, int] = {"total": 0}
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()

        for row in rows:
            status = str(row[0] or "").strip().lower()
            count = int(row[1] or 0)
            counts[status] = count
            counts["total"] += count

        for status in clean_statuses:
            counts.setdefault(status, 0)
        return counts

    def upsert_scenario_set(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        name: str,
        assumptions: list[str],
        scenario_set_id: str | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        clean_tenant = (tenant_id or "public").strip() or "public"
        clean_connection = (connection_id or "default").strip() or "default"
        clean_name = str(name or "").strip()[:180] or "Scenario set"
        clean_assumptions = [str(item).strip() for item in (assumptions or []) if str(item).strip()]
        clean_assumptions = clean_assumptions[:50]
        clean_status = str(status or "active").strip().lower() or "active"
        now = _utc_now()

        with self._lock:
            conn = self._connect()
            try:
                sid = str(scenario_set_id or "").strip()
                if sid:
                    existing = conn.execute(
                        """
                        SELECT version
                        FROM datada_scenario_sets
                        WHERE scenario_set_id = ? AND tenant_id = ?
                        LIMIT 1
                        """,
                        [sid, clean_tenant],
                    ).fetchone()
                    if existing:
                        next_version = int(existing[0] or 0) + 1
                        conn.execute(
                            """
                            UPDATE datada_scenario_sets
                            SET updated_at = ?, connection_id = ?, name = ?, assumptions_json = ?, status = ?, version = ?
                            WHERE scenario_set_id = ? AND tenant_id = ?
                            """,
                            [
                                now,
                                clean_connection,
                                clean_name,
                                json.dumps(clean_assumptions, default=str),
                                clean_status,
                                next_version,
                                sid,
                                clean_tenant,
                            ],
                        )
                        return {
                            "scenario_set_id": sid,
                            "tenant_id": clean_tenant,
                            "connection_id": clean_connection,
                            "name": clean_name,
                            "assumptions": clean_assumptions,
                            "status": clean_status,
                            "version": next_version,
                            "updated": True,
                        }

                sid = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO datada_scenario_sets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        sid,
                        now,
                        now,
                        clean_tenant,
                        clean_connection,
                        clean_name,
                        json.dumps(clean_assumptions, default=str),
                        clean_status,
                        1,
                    ],
                )
            finally:
                conn.close()

        return {
            "scenario_set_id": sid,
            "tenant_id": clean_tenant,
            "connection_id": clean_connection,
            "name": clean_name,
            "assumptions": clean_assumptions,
            "status": clean_status,
            "version": 1,
            "updated": False,
        }

    def get_scenario_set(self, *, scenario_set_id: str, tenant_id: str) -> dict[str, Any] | None:
        sid = str(scenario_set_id or "").strip()
        clean_tenant = (tenant_id or "public").strip() or "public"
        if not sid:
            return None
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT scenario_set_id, created_at, updated_at, tenant_id, connection_id, name,
                           assumptions_json, status, version
                    FROM datada_scenario_sets
                    WHERE scenario_set_id = ? AND tenant_id = ?
                    LIMIT 1
                    """,
                    [sid, clean_tenant],
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return None
        assumptions: list[str] = []
        try:
            parsed = json.loads(row[6] or "[]")
            if isinstance(parsed, list):
                assumptions = [str(item) for item in parsed if str(item).strip()]
        except Exception:
            assumptions = []
        return {
            "scenario_set_id": str(row[0] or ""),
            "created_at": str(row[1] or ""),
            "updated_at": str(row[2] or ""),
            "tenant_id": str(row[3] or ""),
            "connection_id": str(row[4] or ""),
            "name": str(row[5] or ""),
            "assumptions": assumptions,
            "status": str(row[7] or "active"),
            "version": int(row[8] or 1),
        }

    def list_scenario_sets(
        self,
        *,
        tenant_id: str,
        connection_id: str | None = None,
        status: str | None = "active",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clean_tenant = (tenant_id or "public").strip() or "public"
        clean_connection = (connection_id or "").strip()
        clean_status = (status or "").strip().lower()
        where = ["tenant_id = ?"]
        params: list[Any] = [clean_tenant]
        if clean_connection:
            where.append("connection_id = ?")
            params.append(clean_connection)
        if clean_status:
            where.append("LOWER(status) = ?")
            params.append(clean_status)
        where_sql = " AND ".join(where)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT scenario_set_id, created_at, updated_at, tenant_id, connection_id, name,
                           assumptions_json, status, version
                    FROM datada_scenario_sets
                    WHERE {where_sql}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    [*params, max(1, min(500, int(limit)))],
                ).fetchall()
            finally:
                conn.close()

        out: list[dict[str, Any]] = []
        for row in rows:
            assumptions: list[str] = []
            try:
                parsed = json.loads(row[6] or "[]")
                if isinstance(parsed, list):
                    assumptions = [str(item) for item in parsed if str(item).strip()]
            except Exception:
                assumptions = []
            out.append(
                {
                    "scenario_set_id": str(row[0] or ""),
                    "created_at": str(row[1] or ""),
                    "updated_at": str(row[2] or ""),
                    "tenant_id": str(row[3] or ""),
                    "connection_id": str(row[4] or ""),
                    "name": str(row[5] or ""),
                    "assumptions": assumptions,
                    "status": str(row[7] or "active"),
                    "version": int(row[8] or 1),
                }
            )
        return out

    def record_schema_signature(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        dataset_signature: str,
        schema_signature: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_tenant = (tenant_id or "public").strip() or "public"
        clean_connection = (connection_id or "default").strip() or "default"
        clean_dataset = str(dataset_signature or "").strip()
        clean_schema = str(schema_signature or "").strip()

        previous_schema = ""
        previous_dataset = ""
        previous_updated_at = ""
        now = _utc_now()
        record_id = str(uuid.uuid4())

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT dataset_signature, schema_signature, updated_at
                    FROM datada_schema_signatures
                    WHERE tenant_id = ? AND connection_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    [clean_tenant, clean_connection],
                ).fetchone()
                if row:
                    previous_dataset = str(row[0] or "")
                    previous_schema = str(row[1] or "")
                    previous_updated_at = str(row[2] or "")
                conn.execute(
                    """
                    DELETE FROM datada_schema_signatures
                    WHERE tenant_id = ? AND connection_id = ?
                    """,
                    [clean_tenant, clean_connection],
                )
                conn.execute(
                    """
                    INSERT INTO datada_schema_signatures VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record_id,
                        now,
                        now,
                        clean_tenant,
                        clean_connection,
                        clean_dataset,
                        clean_schema,
                        json.dumps(metadata or {}, default=str),
                    ],
                )
            finally:
                conn.close()

        drift_detected = bool(previous_schema) and bool(clean_schema) and previous_schema != clean_schema
        return {
            "record_id": record_id,
            "tenant_id": clean_tenant,
            "connection_id": clean_connection,
            "dataset_signature": clean_dataset,
            "schema_signature": clean_schema,
            "previous_dataset_signature": previous_dataset,
            "previous_schema_signature": previous_schema,
            "previous_updated_at": previous_updated_at,
            "drift_detected": drift_detected,
        }

    def record_run_metric(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        session_scope: str,
        success: bool,
        confidence_score: float,
        execution_ms: float,
        llm_mode: str,
        provider: str,
        row_count: int,
        warning_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metric_id = str(uuid.uuid4())
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO datada_run_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        metric_id,
                        _utc_now(),
                        tenant_id,
                        connection_id,
                        session_scope,
                        bool(success),
                        float(confidence_score),
                        float(execution_ms),
                        llm_mode,
                        provider,
                        int(row_count),
                        int(warning_count),
                        json.dumps(metadata or {}, default=str),
                    ],
                )
            finally:
                conn.close()
        return metric_id

    def trust_dashboard(self, *, tenant_id: str | None = None, hours: int = 168) -> dict[str, Any]:
        window_hours = max(1, min(24 * 90, int(hours)))
        since = _utc_now() - timedelta(hours=window_hours)
        tenant_filter = (tenant_id or "").strip()

        where = "WHERE created_at >= ?"
        params: list[Any] = [since]
        if tenant_filter:
            where += " AND tenant_id = ?"
            params.append(tenant_filter)

        with self._lock:
            conn = self._connect()
            try:
                totals = conn.execute(
                    f"""
                    SELECT
                      COUNT(*) AS runs,
                      SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_runs,
                      AVG(confidence_score) AS avg_confidence,
                      AVG(execution_ms) AS avg_execution_ms,
                      QUANTILE_CONT(execution_ms, 0.95) AS p95_execution_ms,
                      SUM(warning_count) AS total_warnings
                    FROM datada_run_metrics
                    {where}
                    """,
                    params,
                ).fetchone()
                by_mode = conn.execute(
                    f"""
                    SELECT llm_mode,
                           COUNT(*) AS runs,
                           SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_runs,
                           AVG(confidence_score) AS avg_confidence,
                           AVG(execution_ms) AS avg_execution_ms
                    FROM datada_run_metrics
                    {where}
                    GROUP BY 1
                    ORDER BY runs DESC, llm_mode ASC
                    """,
                    params,
                ).fetchall()

                recent_failures = conn.execute(
                    f"""
                    SELECT created_at, connection_id, llm_mode, metadata_json
                    FROM datada_run_metrics
                    {where} AND success = FALSE
                    ORDER BY created_at DESC
                    LIMIT 8
                    """,
                    params,
                ).fetchall()
                recent_contract_rows = conn.execute(
                    f"""
                    SELECT llm_mode, metadata_json
                    FROM datada_run_metrics
                    {where}
                    ORDER BY created_at DESC
                    LIMIT 320
                    """,
                    params,
                ).fetchall()
            finally:
                conn.close()

        run_count = int(totals[0] or 0) if totals else 0
        success_runs = int(totals[1] or 0) if totals else 0
        success_rate = (success_runs / run_count) if run_count > 0 else 0.0

        modes: list[dict[str, Any]] = []
        for row in by_mode:
            mode_runs = int(row[1] or 0)
            mode_success = int(row[2] or 0)
            modes.append(
                {
                    "mode": str(row[0] or "unknown"),
                    "runs": mode_runs,
                    "success_rate": round((mode_success / mode_runs) if mode_runs else 0.0, 4),
                    "avg_confidence": round(float(row[3] or 0.0), 4),
                    "avg_execution_ms": round(float(row[4] or 0.0), 2),
                }
            )

        failures: list[dict[str, Any]] = []
        for row in recent_failures:
            metadata = {}
            try:
                parsed = json.loads(row[3] or "{}")
                if isinstance(parsed, dict):
                    metadata = parsed
            except Exception:
                metadata = {}
            failures.append(
                {
                    "created_at": str(row[0] or ""),
                    "connection_id": str(row[1] or ""),
                    "llm_mode": str(row[2] or ""),
                    "goal": str(metadata.get("goal") or ""),
                    "warning_terms": metadata.get("warning_terms") or [],
                }
            )

        parity_summary: dict[str, Any] = {
            "available": False,
            "mode_deltas": {},
            "contract_drift_cases": 0,
            "contract_drift_rate": 0.0,
            "alerts": [],
        }
        by_mode_map = {str(item.get("mode") or ""): item for item in modes}
        deterministic = by_mode_map.get("deterministic")
        if deterministic:
            base_success = float(deterministic.get("success_rate") or 0.0)
            deltas: dict[str, dict[str, Any]] = {}
            alerts: list[dict[str, Any]] = []
            for mode_name, mode_stats in by_mode_map.items():
                if mode_name == "deterministic":
                    continue
                mode_success = float(mode_stats.get("success_rate") or 0.0)
                delta = round(base_success - mode_success, 4)
                deltas[mode_name] = {
                    "success_rate": round(mode_success, 4),
                    "success_delta_vs_deterministic": delta,
                }
                if delta > 0.05:
                    alerts.append(
                        {
                            "type": "success_drift",
                            "mode": mode_name,
                            "baseline": round(base_success, 4),
                            "actual": round(mode_success, 4),
                            "delta": delta,
                        }
                    )

            goal_contracts: dict[str, dict[str, set[str]]] = {}
            for row in recent_contract_rows:
                metadata: dict[str, Any] = {}
                try:
                    parsed = json.loads(row[1] or "{}")
                    if isinstance(parsed, dict):
                        metadata = parsed
                except Exception:
                    metadata = {}
                goal = str(metadata.get("goal") or "").strip().lower()
                contract_sig = str(metadata.get("contract_signature") or "").strip()
                mode_name = str(row[0] or "").strip().lower() or "unknown"
                if not goal or not contract_sig:
                    continue
                goal_contracts.setdefault(goal, {}).setdefault(mode_name, set()).add(contract_sig)
            drift_cases = 0
            compared = 0
            for mode_map in goal_contracts.values():
                if len(mode_map) < 2:
                    continue
                compared += 1
                union_count = len(set().union(*mode_map.values()))
                if union_count > 1:
                    drift_cases += 1
            drift_rate = (drift_cases / max(1, compared)) if compared else 0.0
            if drift_rate > 0.10:
                alerts.append(
                    {
                        "type": "contract_drift",
                        "drift_rate": round(drift_rate, 4),
                        "threshold": 0.10,
                        "compared_cases": compared,
                    }
                )
            parity_summary = {
                "available": True,
                "mode_deltas": deltas,
                "contract_drift_cases": drift_cases,
                "contract_drift_rate": round(drift_rate, 4),
                "alerts": alerts,
            }

        return {
            "generated_at": _utc_iso(),
            "tenant_id": tenant_filter or "all",
            "window_hours": window_hours,
            "runs": run_count,
            "success_runs": success_runs,
            "success_rate": round(success_rate, 4),
            "avg_confidence": round(float(totals[2] or 0.0), 4) if totals else 0.0,
            "avg_execution_ms": round(float(totals[3] or 0.0), 2) if totals else 0.0,
            "p95_execution_ms": round(float(totals[4] or 0.0), 2) if totals else 0.0,
            "total_warnings": int(totals[5] or 0) if totals else 0,
            "by_mode": modes,
            "parity_summary": parity_summary,
            "recent_failures": failures,
        }

    def evaluate_slo(
        self,
        *,
        tenant_id: str | None = None,
        hours: int = 24,
        success_rate_target: float = 0.95,
        p95_execution_ms_target: float = 3500.0,
        warning_rate_target: float = 0.15,
        min_runs: int = 20,
    ) -> dict[str, Any]:
        trust = self.trust_dashboard(tenant_id=tenant_id, hours=hours)
        runs = int(trust.get("runs") or 0)
        success_rate = float(trust.get("success_rate") or 0.0)
        p95_execution_ms = float(trust.get("p95_execution_ms") or 0.0)
        total_warnings = int(trust.get("total_warnings") or 0)
        warning_rate = (total_warnings / runs) if runs > 0 else 0.0

        breaches: list[dict[str, Any]] = []
        if runs >= max(1, int(min_runs)):
            if success_rate < float(success_rate_target):
                breaches.append(
                    {
                        "metric": "success_rate",
                        "actual": round(success_rate, 4),
                        "target": round(float(success_rate_target), 4),
                        "direction": "min",
                        "delta": round(float(success_rate_target) - success_rate, 4),
                    }
                )
            if p95_execution_ms > float(p95_execution_ms_target):
                breaches.append(
                    {
                        "metric": "p95_execution_ms",
                        "actual": round(p95_execution_ms, 2),
                        "target": round(float(p95_execution_ms_target), 2),
                        "direction": "max",
                        "delta": round(p95_execution_ms - float(p95_execution_ms_target), 2),
                    }
                )
            if warning_rate > float(warning_rate_target):
                breaches.append(
                    {
                        "metric": "warning_rate",
                        "actual": round(warning_rate, 4),
                        "target": round(float(warning_rate_target), 4),
                        "direction": "max",
                        "delta": round(warning_rate - float(warning_rate_target), 4),
                    }
                )

        status = "insufficient_data"
        if runs >= max(1, int(min_runs)):
            status = "healthy" if not breaches else "breach"

        burn_rate = 0.0
        if runs >= max(1, int(min_runs)):
            ratios: list[float] = []
            if float(success_rate_target) > 0:
                ratios.append(max(0.0, (float(success_rate_target) - success_rate) / float(success_rate_target)))
            if float(p95_execution_ms_target) > 0:
                ratios.append(max(0.0, (p95_execution_ms - float(p95_execution_ms_target)) / float(p95_execution_ms_target)))
            if float(warning_rate_target) > 0:
                ratios.append(max(0.0, (warning_rate - float(warning_rate_target)) / float(warning_rate_target)))
            burn_rate = max(ratios) if ratios else 0.0

        return {
            "generated_at": _utc_iso(),
            "tenant_id": str(trust.get("tenant_id") or (tenant_id or "all")),
            "window_hours": int(hours),
            "runs": runs,
            "status": status,
            "success_rate": round(success_rate, 4),
            "p95_execution_ms": round(p95_execution_ms, 2),
            "warning_rate": round(warning_rate, 4),
            "targets": {
                "success_rate_min": round(float(success_rate_target), 4),
                "p95_execution_ms_max": round(float(p95_execution_ms_target), 2),
                "warning_rate_max": round(float(warning_rate_target), 4),
                "min_runs": max(1, int(min_runs)),
            },
            "breaches": breaches,
            "burn_rate": round(float(burn_rate), 4),
            "trust": trust,
        }

    def stage_slo_snapshot(
        self,
        *,
        tenant_id: str | None = None,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Aggregate recent per-stage timings from run metadata for stage-level SLO views."""
        window_hours = max(1, min(24 * 30, int(hours)))
        since = _utc_now() - timedelta(hours=window_hours)
        tenant_filter = (tenant_id or "").strip()

        where = "WHERE created_at >= ?"
        params: list[Any] = [since]
        if tenant_filter:
            where += " AND tenant_id = ?"
            params.append(tenant_filter)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT metadata_json
                    FROM datada_run_metrics
                    {where}
                    ORDER BY created_at DESC
                    LIMIT 2000
                    """,
                    params,
                ).fetchall()
            finally:
                conn.close()

        stage_samples: dict[str, list[float]] = {}
        for row in rows:
            metadata: dict[str, Any] = {}
            try:
                parsed = json.loads(row[0] or "{}")
                if isinstance(parsed, dict):
                    metadata = parsed
            except Exception:
                metadata = {}
            stage_timings = metadata.get("stage_timings_ms") or {}
            if not isinstance(stage_timings, dict):
                continue
            for stage_name, raw_value in stage_timings.items():
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                if value < 0:
                    continue
                stage_samples.setdefault(str(stage_name), []).append(value)

        observed_p95 = {
            stage: round(_percentile(values, 95.0), 2)
            for stage, values in sorted(stage_samples.items())
            if values
        }
        return {
            "generated_at": _utc_iso(),
            "tenant_id": tenant_filter or "all",
            "window_hours": window_hours,
            "runs_considered": len(rows),
            "observed_p95_ms": observed_p95,
        }

    def record_incident(
        self,
        *,
        tenant_id: str,
        severity: str,
        title: str,
        summary: str,
        source: str = "runtime",
        fingerprint: str = "",
        metadata: dict[str, Any] | None = None,
        dedupe_window_minutes: int = 60,
    ) -> dict[str, Any]:
        clean_tenant = (tenant_id or "public").strip() or "public"
        sev = (severity or "medium").strip().lower()
        if sev not in {"info", "low", "medium", "high", "critical"}:
            sev = "medium"
        fp = (fingerprint or "").strip()
        dedupe_minutes = max(1, int(dedupe_window_minutes))
        cutoff = _utc_now() - timedelta(minutes=dedupe_minutes)

        with self._lock:
            conn = self._connect()
            try:
                if fp:
                    existing = conn.execute(
                        """
                        SELECT incident_id, created_at
                        FROM datada_incident_events
                        WHERE tenant_id = ?
                          AND fingerprint = ?
                          AND status IN ('open', 'acknowledged')
                          AND created_at >= ?
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        [clean_tenant, fp, cutoff],
                    ).fetchone()
                    if existing:
                        return {
                            "incident_id": str(existing[0]),
                            "created_at": str(existing[1] or ""),
                            "created": False,
                            "deduped": True,
                        }

                incident_id = str(uuid.uuid4())
                now = _utc_now()
                conn.execute(
                    """
                    INSERT INTO datada_incident_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        incident_id,
                        now,
                        now,
                        clean_tenant,
                        sev,
                        "open",
                        source,
                        (title or "").strip()[:240],
                        (summary or "").strip()[:2000],
                        fp,
                        json.dumps(metadata or {}, default=str),
                    ],
                )
            finally:
                conn.close()

        return {
            "incident_id": incident_id,
            "created_at": _utc_iso(),
            "created": True,
            "deduped": False,
        }

    def list_incidents(
        self,
        *,
        tenant_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        clean_tenant = (tenant_id or "").strip()
        clean_status = (status or "").strip().lower()
        if clean_tenant:
            where_parts.append("tenant_id = ?")
            params.append(clean_tenant)
        if clean_status:
            where_parts.append("LOWER(status) = ?")
            params.append(clean_status)

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        params.append(max(1, min(500, int(limit))))

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT incident_id, created_at, updated_at, tenant_id, severity, status, source,
                           title, summary, fingerprint, metadata_json
                    FROM datada_incident_events
                    {where_clause}
                    ORDER BY created_at DESC
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
                    "incident_id": str(row[0] or ""),
                    "created_at": str(row[1] or ""),
                    "updated_at": str(row[2] or ""),
                    "tenant_id": str(row[3] or ""),
                    "severity": str(row[4] or ""),
                    "status": str(row[5] or ""),
                    "source": str(row[6] or ""),
                    "title": str(row[7] or ""),
                    "summary": str(row[8] or ""),
                    "fingerprint": str(row[9] or ""),
                    "metadata": metadata,
                }
            )
        return out

    def update_incident_status(
        self,
        *,
        incident_id: str,
        status: str,
        note: str = "",
    ) -> dict[str, Any]:
        clean_id = (incident_id or "").strip()
        if not clean_id:
            return {"success": False, "message": "Missing incident_id."}
        clean_status = (status or "").strip().lower()
        if clean_status not in {"open", "acknowledged", "resolved", "dismissed"}:
            return {"success": False, "message": "Invalid incident status."}

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT incident_id, metadata_json
                    FROM datada_incident_events
                    WHERE incident_id = ?
                    LIMIT 1
                    """,
                    [clean_id],
                ).fetchone()
                if not row:
                    return {"success": False, "message": f"Unknown incident_id '{clean_id}'."}

                metadata = {}
                try:
                    parsed = json.loads(row[1] or "{}")
                    if isinstance(parsed, dict):
                        metadata = parsed
                except Exception:
                    metadata = {}
                if note.strip():
                    metadata["status_note"] = note.strip()[:400]
                metadata["status_updated_at"] = _utc_iso()

                conn.execute(
                    """
                    UPDATE datada_incident_events
                    SET updated_at = ?, status = ?, metadata_json = ?
                    WHERE incident_id = ?
                    """,
                    [_utc_now(), clean_status, json.dumps(metadata, default=str), clean_id],
                )
            finally:
                conn.close()

        return {"success": True, "message": f"Incident {clean_id} set to {clean_status}."}
