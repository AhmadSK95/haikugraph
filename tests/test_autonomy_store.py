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


def test_agent_memory_store_rebuilds_incompatible_legacy_types(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-types" / "memory.duckdb"
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
            runtime_mode BOOLEAN,
            provider BOOLEAN,
            success VARCHAR,
            confidence_score VARCHAR,
            row_count VARCHAR,
            table_name VARCHAR,
            metric VARCHAR,
            dimensions_json VARCHAR,
            time_filter_json VARCHAR,
            value_filters_json VARCHAR,
            sql_text VARCHAR,
            audit_warnings_json VARCHAR,
            correction_applied VARCHAR,
            correction_reason DOUBLE,
            metadata_json VARCHAR,
            tenant_id VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO datada_agent_memory VALUES (
            'old-id', NOW(), 'old-trace', 'old goal', 'old goal', TRUE, FALSE, 'true', '0.7', '4',
            't', 'm', '[]', 'null', '[]', 'SELECT 1', '[]', 'false', 0.0, '{}', 'public'
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
            notes DOUBLE,
            weight VARCHAR,
            enabled VARCHAR,
            tenant_id VARCHAR
        )
        """
    )
    conn.close()

    store = AgentMemoryStore(db_path)
    memory_id = store.store_turn(
        tenant_id="public",
        trace_id="type-fix",
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

    verify = duckdb.connect(str(db_path))
    try:
        row = verify.execute(
            """
            SELECT runtime_mode, provider, success
            FROM datada_agent_memory
            WHERE trace_id = 'type-fix'
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        assert row[0] == "openai"
        assert row[1] == "openai"
        assert bool(row[2]) is True

        correction_row = verify.execute(
            """
            SELECT notes, weight
            FROM datada_agent_corrections
            WHERE correction_id = ?
            LIMIT 1
            """,
            [correction_id],
        ).fetchone()
        assert correction_row is not None
        assert "Auto-learned from successful run" in str(correction_row[0] or "")
        assert float(correction_row[1]) >= 1.0
    finally:
        verify.close()


def test_memory_recall_precision_prefers_semantic_and_focus_overlap(tmp_path: Path) -> None:
    db_path = tmp_path / "precision" / "memory.duckdb"
    store = AgentMemoryStore(db_path)
    store.store_turn(
        tenant_id="public",
        trace_id="r1",
        goal="Show forex markup charges by month",
        resolved_goal="Show forex markup charges by month",
        runtime_mode="deterministic",
        provider="",
        success=True,
        confidence_score=0.9,
        row_count=4,
        plan={"table": "datada_mart_quotes", "metric": "forex_markup_revenue", "dimensions": ["__month__"]},
        sql="SELECT 1",
        audit_warnings=[],
        correction_applied=False,
        correction_reason="",
        metadata={},
    )
    store.store_turn(
        tenant_id="public",
        trace_id="r2",
        goal="List booking status for active customers",
        resolved_goal="List booking status for active customers",
        runtime_mode="deterministic",
        provider="",
        success=True,
        confidence_score=0.9,
        row_count=4,
        plan={"table": "datada_mart_bookings", "metric": "booking_count", "dimensions": ["status"]},
        sql="SELECT 1",
        audit_warnings=[],
        correction_applied=False,
        correction_reason="",
        metadata={},
    )

    recalled = store.recall("foreign exchange surcharge trend", tenant_id="public", limit=2)
    assert recalled
    assert str(recalled[0].get("metric") or "") == "forex_markup_revenue"
    assert float(recalled[0].get("semantic_similarity") or 0.0) >= 0.4


def test_memory_recall_respects_tenant_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "tenant" / "memory.duckdb"
    store = AgentMemoryStore(db_path)
    store.store_turn(
        tenant_id="tenant_a",
        trace_id="ta",
        goal="How many transactions are there?",
        resolved_goal="How many transactions are there?",
        runtime_mode="deterministic",
        provider="",
        success=True,
        confidence_score=0.8,
        row_count=2,
        plan={"table": "datada_mart_transactions", "metric": "transaction_count", "dimensions": []},
        sql="SELECT 1",
        audit_warnings=[],
        correction_applied=False,
        correction_reason="",
        metadata={},
    )
    store.store_turn(
        tenant_id="tenant_b",
        trace_id="tb",
        goal="How many quotes are there?",
        resolved_goal="How many quotes are there?",
        runtime_mode="deterministic",
        provider="",
        success=True,
        confidence_score=0.8,
        row_count=2,
        plan={"table": "datada_mart_quotes", "metric": "quote_count", "dimensions": []},
        sql="SELECT 1",
        audit_warnings=[],
        correction_applied=False,
        correction_reason="",
        metadata={},
    )

    a_rows = store.recall("transactions", tenant_id="tenant_a", limit=5)
    b_rows = store.recall("quotes", tenant_id="tenant_b", limit=5)
    assert all(str(item.get("table") or "").startswith("datada_mart_transactions") for item in a_rows)
    assert all(str(item.get("table") or "").startswith("datada_mart_quotes") for item in b_rows)
