from __future__ import annotations

from pathlib import Path

import duckdb

from haikugraph.io.stream_snapshot import ingest_stream_snapshot_to_duckdb


def test_ingest_stream_snapshot_from_json_file(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                '{"event_ts":"2026-03-01T10:00:00Z","event_type":"quote_created","amount":120.0}',
                '{"event_ts":"2026-03-01T10:01:00Z","event_type":"quote_created","amount":180.0}',
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "stream.duckdb"
    uri = f"kafka://quotes/live?file={events}&format=json&table=datada_stream_events&freshness_minutes=15"

    result = ingest_stream_snapshot_to_duckdb(
        stream_uri=uri,
        db_path=db_path,
        connection_id="stream_test",
    )
    assert result["success"] is True
    assert result["row_count"] == 2

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row_count = int(conn.execute("SELECT COUNT(*) FROM datada_stream_events").fetchone()[0] or 0)
        assert row_count == 2
        manifest_count = int(
            conn.execute("SELECT COUNT(*) FROM datada_stream_snapshot_manifest WHERE connection_id='stream_test'").fetchone()[0]
            or 0
        )
        assert manifest_count >= 1
    finally:
        conn.close()


def test_stream_snapshot_reuse_when_fresh(tmp_path: Path) -> None:
    events = tmp_path / "events.csv"
    events.write_text("event_type,amount\nq,10\nq,20\n", encoding="utf-8")
    db_path = tmp_path / "stream.duckdb"
    uri = f"kinesis://quotes/live?file={events}&format=csv&table=datada_stream_events&freshness_minutes=60"

    first = ingest_stream_snapshot_to_duckdb(
        stream_uri=uri,
        db_path=db_path,
        connection_id="stream_test",
    )
    second = ingest_stream_snapshot_to_duckdb(
        stream_uri=uri,
        db_path=db_path,
        connection_id="stream_test",
    )
    assert first["success"] is True
    assert second["success"] is True
    assert second["reused"] is True
