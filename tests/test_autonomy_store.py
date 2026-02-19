from pathlib import Path

from haikugraph.poc.autonomy import AgentMemoryStore


def test_agent_memory_store_creates_parent_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "agent" / "memory.duckdb"
    assert not db_path.parent.exists()
    store = AgentMemoryStore(db_path)
    assert db_path.parent.exists()
    assert db_path.exists()
    # Smoke query ensures schema bootstrap completed.
    assert store.recall("anything", limit=1) == []
