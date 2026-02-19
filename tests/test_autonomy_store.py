from pathlib import Path

import duckdb

from haikugraph.poc.autonomy import AgentMemoryStore


def test_agent_memory_store_creates_parent_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "agent" / "memory.duckdb"
    assert not db_path.parent.exists()
    store = AgentMemoryStore(db_path)
    assert db_path.parent.exists()
    assert db_path.exists()
    # Smoke query ensures schema bootstrap completed.
    assert store.recall("anything", limit=1) == []


def test_agent_memory_store_handles_legacy_column_order(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy" / "memory.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE datada_agent_memory (
            memory_id VARCHAR,
            created_at TIMESTAMP,
            trace_id VARCHAR,
            goal VARCHAR,
            resolved_goal VARCHAR,
            runtime_mode VARCHAR,
            provider VARCHAR,
            success BOOLEAN,
            confidence_score DOUBLE,
            row_count BIGINT,
            table_name VARCHAR,
            metric VARCHAR,
            dimensions_json VARCHAR,
            time_filter_json VARCHAR,
            value_filters_json VARCHAR,
            sql_text VARCHAR,
            audit_warnings_json VARCHAR,
            correction_applied BOOLEAN,
            correction_reason VARCHAR,
            metadata_json VARCHAR,
            tenant_id VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE datada_agent_corrections (
            correction_id VARCHAR,
            created_at TIMESTAMP,
            source VARCHAR,
            keyword VARCHAR,
            target_table VARCHAR,
            target_metric VARCHAR,
            target_dimensions_json VARCHAR,
            notes VARCHAR,
            weight DOUBLE,
            enabled BOOLEAN,
            tenant_id VARCHAR
        )
        """
    )
    conn.close()

    store = AgentMemoryStore(db_path)
    memory_id = store.store_turn(
        tenant_id="public",
        trace_id="legacy-trace",
        goal="What is forex markup revenue?",
        resolved_goal="What is forex markup revenue?",
        runtime_mode="openai",
        provider="openai",
        success=True,
        confidence_score=0.9,
        row_count=3,
        plan={
            "table": "datada_mart_quotes",
            "metric": "forex_markup_revenue",
            "dimensions": ["__month__"],
            "time_filter": {"kind": "month_year", "month": 12, "year": 2025},
            "value_filters": [],
        },
        sql="SELECT 1",
        audit_warnings=[],
        correction_applied=False,
        correction_reason="",
        metadata={"runtime": "openai"},
    )
    assert memory_id

    correction_id = store.learn_from_success(
        tenant_id="public",
        goal="forex markup by month",
        plan={
            "table": "datada_mart_quotes",
            "metric": "forex_markup_revenue",
            "dimensions": ["__month__"],
        },
        score=0.91,
    )
    assert correction_id
