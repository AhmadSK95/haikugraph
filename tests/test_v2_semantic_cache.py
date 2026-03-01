from __future__ import annotations

import tempfile
import time
from pathlib import Path

import duckdb

from haikugraph.v2.semantic_cache import SemanticProfileCache
from haikugraph.v2.semantic_profiler import profile_dataset


def _seed_db(path: Path) -> None:
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE TABLE transactions (
            transaction_id VARCHAR,
            customer_id VARCHAR,
            amount DOUBLE,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO transactions VALUES
        ('t1', 'c1', 10.0, '2025-01-01'),
        ('t2', 'c2', 11.0, '2025-01-02')
        """
    )
    conn.close()


def test_semantic_cache_hits_and_invalidates_on_db_change() -> None:
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as fh:
        db_path = Path(fh.name)
    db_path.unlink(missing_ok=True)
    _seed_db(db_path)
    cache = SemanticProfileCache(max_entries=8, ttl_seconds=900)
    calls = {"count": 0}

    def _builder(path: str):
        calls["count"] += 1
        return profile_dataset(path)

    first_profile, first_meta = cache.get_or_build(db_path, _builder)
    second_profile, second_meta = cache.get_or_build(db_path, _builder)

    assert calls["count"] == 1
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True
    assert second_profile.dataset_signature == first_profile.dataset_signature

    # mutate db to force fingerprint change
    time.sleep(0.01)
    conn = duckdb.connect(str(db_path))
    conn.execute("INSERT INTO transactions VALUES ('t3', 'c3', 12.0, '2025-01-03')")
    conn.close()

    third_profile, third_meta = cache.get_or_build(db_path, _builder)
    assert calls["count"] == 2
    assert third_meta["cache_hit"] is False
    assert third_profile.dataset_signature != first_profile.dataset_signature

    db_path.unlink(missing_ok=True)
