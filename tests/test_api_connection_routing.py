from pathlib import Path
import shutil

from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


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
