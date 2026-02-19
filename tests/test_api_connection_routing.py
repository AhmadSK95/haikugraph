import json
from pathlib import Path
import shutil

import duckdb
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app
from haikugraph.io.document_ingest import ingest_documents_to_duckdb


def _close_app_teams(app) -> None:
    for item in app.state.teams.values():
        try:
            item["team"].close()
        except Exception:
            pass


def test_query_routes_to_named_connection(tmp_path: Path) -> None:
    db1 = tmp_path / "default.db"
    db2 = tmp_path / "secondary.db"
    shutil.copyfile("data/haikugraph.db", db1)
    shutil.copyfile("data/haikugraph.db", db2)

    app = create_app(db_path=db1)
    client = TestClient(app)

    upsert = client.post(
        "/api/assistant/connections/upsert",
        headers={"x-datada-role": "admin", "x-datada-tenant-id": "public"},
        json={
            "connection_id": "secondary",
            "kind": "duckdb",
            "path": str(db2),
            "description": "secondary connection for routing test",
            "enabled": True,
            "validate_connection": True,
        },
    )
    assert upsert.status_code == 200

    response = client.post(
        "/api/assistant/query",
        json={
            "goal": "What kind of data do I have?",
            "db_connection_id": "secondary",
            "llm_mode": "deterministic",
            "session_id": "routing-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["runtime"]["db_connection_id"] == "secondary"
    assert payload["runtime"]["db_kind"] == "duckdb"

    _close_app_teams(app)


def test_feedback_uses_connection_scope(tmp_path: Path) -> None:
    db1 = tmp_path / "default.db"
    shutil.copyfile("data/haikugraph.db", db1)

    app = create_app(db_path=db1)
    client = TestClient(app)

    response = client.post(
        "/api/assistant/feedback",
        json={
            "db_connection_id": "default",
            "issue": "routing smoke feedback",
            "keyword": "forex",
            "target_table": "datada_mart_quotes",
            "target_metric": "forex_markup_revenue",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["feedback_id"]

    _close_app_teams(app)


def test_create_app_self_heals_pytest_default_path(tmp_path: Path, monkeypatch) -> None:
    stable_db = tmp_path / "stable.db"
    shutil.copyfile("data/haikugraph.db", stable_db)
    stale_default = tmp_path / "pytest-of-user" / "pytest-11" / "dead-default.db"
    registry_path = tmp_path / "connections.json"
    registry_path.write_text(
        json.dumps(
            {
                "default_connection_id": "default",
                "connections": [
                    {
                        "id": "default",
                        "kind": "duckdb",
                        "path": str(stale_default),
                        "description": "stale pytest default",
                        "enabled": True,
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("HG_CONNECTION_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("HG_DB_PATH", str(stable_db))

    app = create_app()
    try:
        resolved = app.state.connection_registry.resolve("default")
        assert resolved is not None
        assert Path(str(resolved["path"])).resolve() == stable_db.resolve()
        assert app.state.db_path.resolve() == stable_db.resolve()
    finally:
        _close_app_teams(app)


def test_create_app_self_heals_temp_default_without_semantic_layer(tmp_path: Path, monkeypatch) -> None:
    stable_db = tmp_path / "stable.db"
    shutil.copyfile("data/haikugraph.db", stable_db)

    stale_default = tmp_path / "var" / "folders" / "1y" / "tmp862divrc.duckdb"
    stale_default.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(stale_default))
    try:
        conn.execute("CREATE TABLE only_noise(id INTEGER)")
        conn.execute("INSERT INTO only_noise VALUES (1)")
    finally:
        conn.close()

    registry_path = tmp_path / "connections.json"
    registry_path.write_text(
        json.dumps(
            {
                "default_connection_id": "default",
                "connections": [
                    {
                        "id": "default",
                        "kind": "duckdb",
                        "path": str(stale_default),
                        "description": "stale temp default",
                        "enabled": True,
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("HG_CONNECTION_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("HG_DB_PATH", str(stable_db))

    app = create_app()
    try:
        resolved = app.state.connection_registry.resolve("default")
        assert resolved is not None
        assert Path(str(resolved["path"])).resolve() == stable_db.resolve()
        assert app.state.db_path.resolve() == stable_db.resolve()
    finally:
        _close_app_teams(app)


def test_document_connection_routes_via_mirror(tmp_path: Path) -> None:
    default_db = tmp_path / "default.db"
    shutil.copyfile("data/haikugraph.db", default_db)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "guide.txt").write_text(
        "Compliance guide: forex markup disclosure is mandatory for all quote flows.",
        encoding="utf-8",
    )

    app = create_app(db_path=default_db)
    client = TestClient(app)

    upsert = client.post(
        "/api/assistant/connections/upsert",
        headers={"x-datada-role": "admin", "x-datada-tenant-id": "public"},
        json={
            "connection_id": "docs",
            "kind": "documents",
            "path": str(docs_dir),
            "description": "documents mirror source",
            "enabled": True,
            "validate_connection": True,
        },
    )
    assert upsert.status_code == 200

    response = client.post(
        "/api/assistant/query",
        json={
            "goal": "What does the guide say about forex markup disclosure?",
            "db_connection_id": "docs",
            "llm_mode": "deterministic",
            "session_id": "docs-routing",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["runtime"]["db_connection_id"] == "docs"
    assert payload["runtime"]["db_kind"] == "documents"
    assert payload["row_count"] >= 1

    mirror_path = Path(app.state.db_path.parent) / "document_mirrors" / "docs.duckdb"
    assert mirror_path.exists()
    ingest_check = ingest_documents_to_duckdb(docs_dir=docs_dir, db_path=mirror_path, force=False)
    assert ingest_check["success"] is True

    _close_app_teams(app)
