from __future__ import annotations

from pathlib import Path

from haikugraph.api.runtime_store import RuntimeStore


def test_runtime_store_scenario_set_crud(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.duckdb")

    created = store.upsert_scenario_set(
        tenant_id="public",
        connection_id="default",
        name="fx downside",
        assumptions=["volume down 10%", "fees up 2%"],
    )
    sid = str(created.get("scenario_set_id") or "")
    assert sid

    fetched = store.get_scenario_set(scenario_set_id=sid, tenant_id="public")
    assert fetched is not None
    assert fetched["name"] == "fx downside"
    assert len(fetched["assumptions"]) == 2

    updated = store.upsert_scenario_set(
        tenant_id="public",
        connection_id="default",
        name="fx downside v2",
        assumptions=["volume down 8%"],
        scenario_set_id=sid,
    )
    assert updated["updated"] is True
    assert int(updated["version"] or 0) >= 2

    rows = store.list_scenario_sets(tenant_id="public", connection_id="default")
    assert any(str(row.get("scenario_set_id") or "") == sid for row in rows)


def test_runtime_store_schema_signature_drift_detection(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.duckdb")

    first = store.record_schema_signature(
        tenant_id="public",
        connection_id="default",
        dataset_signature="d1",
        schema_signature="s1",
        metadata={"table_count": 2},
    )
    assert first["drift_detected"] is False
    assert first["schema_signature"] == "s1"

    second = store.record_schema_signature(
        tenant_id="public",
        connection_id="default",
        dataset_signature="d2",
        schema_signature="s2",
        metadata={"table_count": 3},
    )
    assert second["drift_detected"] is True
    assert second["previous_schema_signature"] == "s1"
    assert second["schema_signature"] == "s2"
