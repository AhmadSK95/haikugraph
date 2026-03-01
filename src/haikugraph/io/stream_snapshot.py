"""Governed bounded stream snapshot ingestion into DuckDB mirrors."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb


SUPPORTED_SCHEMES = {"kafka", "kinesis"}


def _utc_now() -> datetime:
    return datetime.utcnow()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _stream_config(stream_uri: str) -> dict[str, Any]:
    parsed = urlparse(stream_uri)
    scheme = (parsed.scheme or "").lower()
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError("Stream URI must use kafka:// or kinesis:// scheme.")

    query = parse_qs(parsed.query)
    file_path = str((query.get("file") or [""])[0]).strip()
    fmt = str((query.get("format") or [""])[0]).strip().lower()
    table_name = str((query.get("table") or [""])[0]).strip() or "datada_stream_events"
    freshness_minutes = max(1, _as_int((query.get("freshness_minutes") or [30])[0], 30))
    max_rows = max(1, _as_int((query.get("max_rows") or [20000])[0], 20000))

    if not fmt:
        if file_path.endswith(".json") or file_path.endswith(".jsonl"):
            fmt = "json"
        elif file_path.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "json"

    if fmt not in {"json", "csv", "parquet"}:
        raise ValueError("Stream format must be one of: json, csv, parquet.")

    return {
        "scheme": scheme,
        "topic": (parsed.netloc or parsed.path or "").strip("/"),
        "file_path": file_path,
        "format": fmt,
        "table_name": table_name,
        "freshness_minutes": freshness_minutes,
        "max_rows": max_rows,
    }


def _is_fresh(manifest_row: tuple[Any, ...] | None) -> bool:
    if not manifest_row or not manifest_row[0]:
        return False
    try:
        expires_at = manifest_row[0]
        if isinstance(expires_at, datetime):
            return expires_at >= _utc_now()
        return datetime.fromisoformat(str(expires_at).replace("Z", "")) >= _utc_now()
    except Exception:
        return False


def ingest_stream_snapshot_to_duckdb(
    *,
    stream_uri: str,
    db_path: Path | str,
    connection_id: str,
    force: bool = False,
) -> dict[str, Any]:
    cfg = _stream_config(stream_uri)
    db = Path(db_path).expanduser()
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db), read_only=False)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datada_stream_snapshot_manifest (
                snapshot_id VARCHAR,
                connection_id VARCHAR,
                stream_uri VARCHAR,
                source_scheme VARCHAR,
                source_topic VARCHAR,
                table_name VARCHAR,
                source_file VARCHAR,
                source_format VARCHAR,
                row_count BIGINT,
                schema_json VARCHAR,
                captured_at TIMESTAMP,
                freshness_expires_at TIMESTAMP,
                status VARCHAR,
                notes VARCHAR
            )
            """
        )

        latest = conn.execute(
            """
            SELECT freshness_expires_at, snapshot_id
            FROM datada_stream_snapshot_manifest
            WHERE connection_id = ?
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            [connection_id],
        ).fetchone()
        if not force and _is_fresh(latest):
            current = conn.execute(
                """
                SELECT table_name, row_count, source_format, source_file, snapshot_id, freshness_expires_at
                FROM datada_stream_snapshot_manifest
                WHERE connection_id = ?
                ORDER BY captured_at DESC
                LIMIT 1
                """,
                [connection_id],
            ).fetchone()
            return {
                "success": True,
                "reused": True,
                "snapshot_id": str((current or ["", 0, "", "", "", ""])[4] or ""),
                "table_name": str((current or [cfg["table_name"]])[0] or cfg["table_name"]),
                "row_count": int((current or [0, 0])[1] or 0),
                "source_format": str((current or ["", "", ""])[2] or cfg["format"]),
                "source_file": str((current or ["", "", "", ""])[3] or cfg["file_path"]),
                "freshness_expires_at": str((current or ["", "", "", "", "", ""])[5] or ""),
                "db_path": str(db),
            }

        table = cfg["table_name"]
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')

        file_path = Path(cfg["file_path"]).expanduser() if cfg["file_path"] else None
        row_count = 0
        schema_json = "[]"
        notes = ""
        status = "ok"

        if file_path and file_path.exists() and file_path.is_file():
            if cfg["format"] == "json":
                conn.execute(
                    f'CREATE TABLE "{table}" AS SELECT * FROM read_json_auto(?)',
                    [str(file_path)],
                )
            elif cfg["format"] == "csv":
                conn.execute(
                    f'CREATE TABLE "{table}" AS SELECT * FROM read_csv_auto(?)',
                    [str(file_path)],
                )
            else:
                conn.execute(
                    f'CREATE TABLE "{table}" AS SELECT * FROM read_parquet(?)',
                    [str(file_path)],
                )
            row_count_before = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] or 0)
            if row_count_before > int(cfg["max_rows"]):
                temp_table = f"{table}__bounded"
                conn.execute(
                    f'CREATE TABLE "{temp_table}" AS SELECT * FROM "{table}" LIMIT ?',
                    [int(cfg["max_rows"])],
                )
                conn.execute(f'DROP TABLE "{table}"')
                conn.execute(f'ALTER TABLE "{temp_table}" RENAME TO "{table}"')
            row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] or 0)
            schema_rows = conn.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='main' AND table_name=?
                ORDER BY ordinal_position
                """,
                [table],
            ).fetchall()
            schema_json = json.dumps(
                [{"name": str(col[0]), "type": str(col[1])} for col in schema_rows],
                default=str,
            )
        else:
            conn.execute(
                f'CREATE TABLE "{table}" (event_ts TIMESTAMP, event_type VARCHAR, payload_json VARCHAR)'
            )
            notes = "No local file configured for stream URI; created empty governed snapshot table."
            status = "empty"

        snapshot_id = str(uuid.uuid4())
        captured_at = _utc_now()
        freshness_expires_at = captured_at + timedelta(minutes=int(cfg["freshness_minutes"]))
        conn.execute(
            """
            INSERT INTO datada_stream_snapshot_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_id,
                connection_id,
                stream_uri,
                cfg["scheme"],
                cfg["topic"],
                table,
                str(file_path) if file_path else "",
                cfg["format"],
                int(row_count),
                schema_json,
                captured_at,
                freshness_expires_at,
                status,
                notes,
            ],
        )
    finally:
        conn.close()

    return {
        "success": True,
        "reused": False,
        "snapshot_id": snapshot_id,
        "table_name": cfg["table_name"],
        "row_count": int(row_count),
        "source_format": cfg["format"],
        "source_file": str(file_path) if file_path else "",
        "freshness_expires_at": freshness_expires_at.isoformat() + "Z",
        "db_path": str(db),
        "status": status,
        "notes": notes,
    }
