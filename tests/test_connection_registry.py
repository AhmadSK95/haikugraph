from pathlib import Path

from haikugraph.api.connection_registry import ConnectionRegistry


def test_registry_initializes_default_connection(tmp_path: Path) -> None:
    default_db = tmp_path / "default.duckdb"
    default_db.write_text("placeholder")
    registry_path = tmp_path / "connections.json"

    registry = ConnectionRegistry(registry_path, default_db)
    payload = registry.list_connections()

    assert payload["default_connection_id"] == "default"
    assert len(payload["connections"]) == 1
    assert payload["connections"][0]["id"] == "default"


def test_upsert_and_set_default_connection(tmp_path: Path) -> None:
    default_db = tmp_path / "default.duckdb"
    default_db.write_text("placeholder")
    other_db = tmp_path / "other.duckdb"
    other_db.write_text("placeholder")

    registry = ConnectionRegistry(tmp_path / "connections.json", default_db)
    registry.upsert(
        connection_id="analytics_secondary",
        kind="duckdb",
        path=str(other_db),
        description="secondary",
        enabled=True,
        set_default=False,
    )
    entry = registry.set_default("analytics_secondary")

    assert entry["id"] == "analytics_secondary"
    assert registry.default_connection_id() == "analytics_secondary"
    resolved = registry.resolve("default")
    assert resolved is not None
    assert resolved["id"] == "analytics_secondary"


def test_connection_test_fails_for_missing_file(tmp_path: Path) -> None:
    default_db = tmp_path / "default.duckdb"
    default_db.write_text("placeholder")
    registry = ConnectionRegistry(tmp_path / "connections.json", default_db)

    ok, reason = registry.test(kind="duckdb", path=str(tmp_path / "missing.duckdb"))
    assert ok is False
    assert "not found" in reason.lower()
