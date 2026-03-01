from __future__ import annotations

from pathlib import Path
import tempfile

import duckdb

import haikugraph.poc.agentic_team as team_module


def test_agent_memory_lock_conflict_falls_back_to_temp_path(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "seed.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS seed_table(id INTEGER)")
    conn.close()

    calls: list[Path] = []

    class _FakeMemoryStore:
        def __init__(self, path):
            p = Path(path)
            calls.append(p)
            if len(calls) == 1:
                raise RuntimeError(
                    "IO Error: Could not set lock on file "
                    f"\"{p}\": Conflicting lock is held"
                )

    monkeypatch.delenv("HG_MEMORY_DB_PATH", raising=False)
    monkeypatch.setattr(team_module, "AgentMemoryStore", _FakeMemoryStore)

    team = team_module.AgenticAnalyticsTeam(db_path)
    try:
        assert len(calls) == 2
        assert team.memory_db_path == calls[-1]
        assert team.memory_db_path != calls[0]
        assert str(team.memory_db_path).startswith(str(Path(tempfile.gettempdir())))
    finally:
        team.close()
