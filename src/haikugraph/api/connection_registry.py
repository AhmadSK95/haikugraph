"""Connection registry for routing analytics requests across data sources.

Current supported connection kind:
- duckdb

The registry is persisted as JSON to keep configuration explicit and portable.
"""

from __future__ import annotations

import json
import importlib.util
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")
_SUPPORTED_KINDS = {"duckdb", "postgres", "snowflake", "bigquery", "stream", "documents"}


class ConnectionRegistry:
    """Thread-safe persisted registry for logical data connections."""

    def __init__(self, path: Path | str, default_db_path: Path | str):
        self.path = Path(path)
        self.default_db_path = Path(default_db_path).expanduser()
        self._lock = threading.RLock()
        self._state = self._load_or_initialize()

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _validate_id(self, connection_id: str) -> str:
        clean = connection_id.strip()
        if not _ID_RE.match(clean):
            raise ValueError(
                "connection_id must match [a-zA-Z0-9._-] and be at most 64 chars"
            )
        return clean

    def _normalize_path(self, value: str) -> str:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return str(p)

    def _normalize_entry(
        self,
        *,
        connection_id: str,
        kind: str,
        path: str,
        description: str = "",
        enabled: bool = True,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        cid = self._validate_id(connection_id)
        clean_kind = kind.strip().lower() or "duckdb"
        if clean_kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported connection kind '{clean_kind}'. "
                f"Supported kinds: {', '.join(sorted(_SUPPORTED_KINDS))}"
            )
        normalized_path = (
            self._normalize_path(path)
            if clean_kind in {"duckdb", "documents"}
            else str(path or "").strip()
        )
        return {
            "id": cid,
            "kind": clean_kind,
            "path": normalized_path,
            "description": (description or "").strip(),
            "enabled": bool(enabled),
            "created_at": created_at or self._now(),
            "updated_at": updated_at or self._now(),
        }

    def _default_state(self) -> dict[str, Any]:
        default = self._normalize_entry(
            connection_id="default",
            kind="duckdb",
            path=str(self.default_db_path),
            description="Primary local DuckDB connection",
            enabled=True,
        )
        return {
            "default_connection_id": "default",
            "connections": {"default": default},
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                state = self._default_state()
                self._save_unlocked(state)
                return state

            try:
                raw = json.loads(self.path.read_text())
            except Exception:
                state = self._default_state()
                self._save_unlocked(state)
                return state

            # Support both old list-style and map-style formats.
            raw_default = str(raw.get("default_connection_id") or "default")
            raw_connections = raw.get("connections") or {}
            entries: dict[str, dict[str, Any]] = {}
            if isinstance(raw_connections, list):
                for item in raw_connections:
                    if not isinstance(item, dict):
                        continue
                    cid = str(item.get("id") or "").strip()
                    if not cid:
                        continue
                    try:
                        entries[cid] = self._normalize_entry(
                            connection_id=cid,
                            kind=str(item.get("kind") or "duckdb"),
                            path=str(item.get("path") or ""),
                            description=str(item.get("description") or ""),
                            enabled=bool(item.get("enabled", True)),
                            created_at=str(item.get("created_at") or self._now()),
                            updated_at=str(item.get("updated_at") or self._now()),
                        )
                    except Exception:
                        continue
            elif isinstance(raw_connections, dict):
                for cid, item in raw_connections.items():
                    if not isinstance(item, dict):
                        continue
                    try:
                        entries[str(cid)] = self._normalize_entry(
                            connection_id=str(cid),
                            kind=str(item.get("kind") or "duckdb"),
                            path=str(item.get("path") or ""),
                            description=str(item.get("description") or ""),
                            enabled=bool(item.get("enabled", True)),
                            created_at=str(item.get("created_at") or self._now()),
                            updated_at=str(item.get("updated_at") or self._now()),
                        )
                    except Exception:
                        continue

            if not entries:
                state = self._default_state()
                self._save_unlocked(state)
                return state

            if raw_default not in entries:
                raw_default = next(iter(entries.keys()))

            state = {
                "default_connection_id": raw_default,
                "connections": entries,
            }
            self._save_unlocked(state)
            return state

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "default_connection_id": state["default_connection_id"],
            "connections": list(state["connections"].values()),
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=False))

    def default_connection_id(self) -> str:
        with self._lock:
            return str(self._state["default_connection_id"])

    def list_connections(self) -> dict[str, Any]:
        with self._lock:
            default_id = str(self._state["default_connection_id"])
            rows = []
            for cid, entry in self._state["connections"].items():
                path = Path(str(entry["path"]))
                exists = path.exists()
                size = path.stat().st_size if exists and path.is_file() else 0
                rows.append(
                    {
                        **entry,
                        "is_default": cid == default_id,
                        "exists": exists,
                        "db_size_bytes": size,
                    }
                )
            rows.sort(key=lambda x: (0 if x["is_default"] else 1, x["id"]))
            return {
                "default_connection_id": default_id,
                "connections": rows,
            }

    def get(self, connection_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._state["connections"].get(connection_id)
            return dict(item) if item else None

    def resolve(self, requested: str | None) -> dict[str, Any] | None:
        with self._lock:
            if not requested or requested == "default":
                cid = str(self._state["default_connection_id"])
                item = self._state["connections"].get(cid)
                return dict(item) if item else None
            item = self._state["connections"].get(requested)
            return dict(item) if item else None

    def upsert(
        self,
        *,
        connection_id: str,
        kind: str,
        path: str,
        description: str = "",
        enabled: bool = True,
        set_default: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            existing = self._state["connections"].get(connection_id)
            entry = self._normalize_entry(
                connection_id=connection_id,
                kind=kind,
                path=path,
                description=description or (existing or {}).get("description", ""),
                enabled=enabled,
                created_at=(existing or {}).get("created_at"),
                updated_at=self._now(),
            )
            self._state["connections"][connection_id] = entry
            if set_default:
                self._state["default_connection_id"] = connection_id
            self._save_unlocked(self._state)
            return dict(entry)

    def set_default(self, connection_id: str) -> dict[str, Any]:
        with self._lock:
            cid = self._validate_id(connection_id)
            if cid not in self._state["connections"]:
                raise KeyError(f"Unknown connection_id '{cid}'")
            self._state["default_connection_id"] = cid
            self._state["connections"][cid]["updated_at"] = self._now()
            self._save_unlocked(self._state)
            return dict(self._state["connections"][cid])

    def test(self, *, kind: str, path: str) -> tuple[bool, str]:
        clean_kind = kind.strip().lower() or "duckdb"
        if clean_kind not in _SUPPORTED_KINDS:
            return False, f"Unsupported connection kind '{clean_kind}'."

        if clean_kind == "duckdb":
            db_path = Path(self._normalize_path(path))
            if not db_path.exists():
                return False, f"Database file not found at {db_path}"
            try:
                conn = duckdb.connect(str(db_path), read_only=True)
                conn.execute("SELECT 1").fetchone()
                conn.close()
                return True, f"Connected successfully to {db_path}"
            except Exception as exc:
                return False, f"Connection failed for {db_path}: {exc}"

        if clean_kind == "postgres":
            raw = (path or "").strip()
            if not raw.startswith(("postgres://", "postgresql://")):
                return (
                    False,
                    "Postgres connection expects a DSN like postgresql://user:pass@host:5432/db",
                )
            if importlib.util.find_spec("psycopg") is None:
                return False, "Install psycopg to test Postgres connections."
            try:
                import psycopg  # type: ignore

                with psycopg.connect(raw, connect_timeout=3) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                return True, "Connected successfully to Postgres."
            except Exception as exc:
                return False, f"Postgres connection failed: {exc}"

        if clean_kind == "snowflake":
            if importlib.util.find_spec("snowflake.connector") is None:
                return (
                    False,
                    "Install snowflake-connector-python to test Snowflake connections.",
                )
            return (
                True,
                "Snowflake connector package found. Use connector sync workflow for mirrored ingestion.",
            )

        if clean_kind == "bigquery":
            if importlib.util.find_spec("google.cloud.bigquery") is None:
                return (
                    False,
                    "Install google-cloud-bigquery to test BigQuery connections.",
                )
            return (
                True,
                "BigQuery client package found. Use connector sync workflow for mirrored ingestion.",
            )

        if clean_kind == "stream":
            raw = (path or "").strip().lower()
            if raw.startswith(("kafka://", "kinesis://")):
                return True, "Stream URI accepted. Runtime uses bounded stream snapshot ingestion."
            return False, "Stream path must start with kafka:// or kinesis://."

        if clean_kind == "documents":
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if not p.exists():
                return False, f"Document path not found at {p}"
            return True, f"Document source path ready at {p}"

        return False, f"Connection kind '{clean_kind}' is not supported yet."
