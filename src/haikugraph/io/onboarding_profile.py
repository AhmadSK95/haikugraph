"""Runtime onboarding profile loader and versioning helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


def _table_profile(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, Any]:
    row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] or 0)
    columns = conn.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='main' AND table_name=?
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()
    column_rows = [{"name": str(col[0]), "type": str(col[1])} for col in columns]
    id_columns = [
        c["name"] for c in column_rows if c["name"].lower().endswith("_id") or "key" in c["name"].lower()
    ]
    time_columns = [
        c["name"]
        for c in column_rows
        if any(token in c["name"].lower() for token in ("_ts", "_at", "date", "time"))
    ]
    metric_columns = [
        c["name"]
        for c in column_rows
        if any(tok in c["type"].upper() for tok in ("INT", "DECIMAL", "DOUBLE", "REAL", "NUMERIC"))
    ]
    return {
        "table": table_name,
        "row_count": row_count,
        "column_count": len(column_rows),
        "id_columns": id_columns,
        "time_columns": time_columns,
        "metric_columns": metric_columns,
        "columns": column_rows,
    }


def build_onboarding_profile(db_path: Path, *, include_datada_views: bool = False) -> dict[str, Any]:
    conn = duckdb.connect(str(db_path), read_only=False)
    try:
        rows = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='main'
            ORDER BY table_name
            """
        ).fetchall()
        table_names = [str(r[0]) for r in rows]
        if not include_datada_views:
            table_names = [name for name in table_names if not name.startswith("datada_")]
        profiles = [_table_profile(conn, name) for name in table_names]
    finally:
        conn.close()

    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "db_path": str(db_path),
        "table_count": len(profiles),
        "tables": profiles,
        "notes": [
            "Runtime-managed onboarding profile.",
            "Keep glossary mappings and business semantics in sync with this profile version.",
        ],
    }


def profile_version(profile: dict[str, Any]) -> str:
    tables = profile.get("tables") if isinstance(profile, dict) else []
    payload = json.dumps(tables if isinstance(tables, list) else [], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_or_create_onboarding_profile(
    *,
    db_path: Path,
    profile_path: Path,
    include_datada_views: bool = False,
) -> dict[str, Any]:
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    profile: dict[str, Any]
    if profile_path.exists():
        try:
            loaded = json.loads(profile_path.read_text(encoding="utf-8"))
            profile = loaded if isinstance(loaded, dict) else {}
        except Exception:
            profile = {}
    else:
        profile = {}

    if not profile:
        profile = build_onboarding_profile(db_path, include_datada_views=include_datada_views)
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    version = profile_version(profile)
    return {
        "path": str(profile_path),
        "version": version,
        "table_count": int(profile.get("table_count") or 0),
        "generated_at": str(profile.get("generated_at") or ""),
        "profile": profile,
    }
