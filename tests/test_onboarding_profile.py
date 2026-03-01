from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


def test_generate_onboarding_profile_script_outputs_json_and_markdown() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_onboarding_profile.py"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "seed.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE sales_txn (transaction_id VARCHAR, customer_id VARCHAR, amount DOUBLE, created_at VARCHAR)")
            conn.execute("INSERT INTO sales_txn VALUES ('t1','c1',120.0,'2025-01-01')")
            conn.execute("CREATE TABLE customer_dim (customer_id VARCHAR, region VARCHAR, updated_ts VARCHAR)")
            conn.execute("INSERT INTO customer_dim VALUES ('c1','NA','2025-01-02')")
        finally:
            conn.close()

        out_json = tmp_path / "onboarding.json"
        out_md = tmp_path / "onboarding.md"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--db-path",
                str(db_path),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert out_json.exists()
        assert out_md.exists()

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        table_names = {t["table"] for t in payload.get("tables", [])}
        assert "sales_txn" in table_names
        assert "customer_dim" in table_names
        assert int(payload.get("table_count", 0)) >= 2


def test_runtime_health_and_query_expose_onboarding_profile_metadata(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "seed.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE sales_txn (transaction_id VARCHAR, amount DOUBLE, created_at VARCHAR)")
        conn.execute("INSERT INTO sales_txn VALUES ('t1', 10.0, '2026-01-01')")
    finally:
        conn.close()

    profile_dir = tmp_path / "profiles"
    monkeypatch.setenv("HG_ONBOARDING_PROFILE_DIR", str(profile_dir))
    app = create_app(db_path=db_path)
    client = TestClient(app)

    health = client.get("/api/assistant/health")
    assert health.status_code == 200
    hp = health.json()
    assert hp.get("onboarding_profile_version")
    assert str(hp.get("onboarding_profile_path") or "").endswith("default.json")
    assert Path(hp["onboarding_profile_path"]).exists()

    query = client.post(
        "/api/assistant/query",
        json={"goal": "How many transactions are there?", "llm_mode": "deterministic"},
    )
    assert query.status_code == 200
    runtime = query.json().get("runtime") or {}
    assert runtime.get("onboarding_profile_version")
    assert runtime.get("onboarding_profile_path")
