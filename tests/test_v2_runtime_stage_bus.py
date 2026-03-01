from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pytest

from haikugraph.agents.contracts import AssistantQueryResponse, ConfidenceLevel
from haikugraph.poc import RuntimeSelection
from haikugraph.v2 import STAGE_ORDER, StageEventBusV2, StageTransitionError, V2Orchestrator
from haikugraph.v2.compat_adapter import apply_v2_compat_fields


class _DummyTeam:
    def run(self, *_args, **_kwargs) -> AssistantQueryResponse:
        return AssistantQueryResponse(
            success=True,
            answer_markdown="ok",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.9,
            definition_used="metric",
            sql="SELECT 1 AS metric_value",
            row_count=1,
            columns=["metric_value"],
            sample_rows=[{"metric_value": 1}],
            execution_time_ms=1.0,
            trace_id="trace-v2-stage-bus",
            runtime={"mode": "deterministic", "provider": "deterministic"},
        )


@pytest.fixture
def _mini_db_path() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as fh:
        path = Path(fh.name)
    path.unlink(missing_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE TABLE transactions (
            transaction_id VARCHAR,
            customer_id VARCHAR,
            payment_amount DOUBLE,
            created_ts TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO transactions VALUES
        ('t1','c1',100.0,'2025-01-01'),
        ('t2','c2',120.0,'2025-01-02')
        """
    )
    conn.close()
    yield path
    path.unlink(missing_ok=True)


def test_stage_event_bus_enforces_order_and_records_events() -> None:
    bus = StageEventBusV2()

    for stage in STAGE_ORDER:
        bus.start_stage(stage)
        bus.complete_stage(stage, detail={"duration_ms": 1.23})

    events = bus.events()
    assert len(events) == len(STAGE_ORDER) * 2
    assert events[0].stage == STAGE_ORDER[0]
    assert events[0].status == "started"
    assert events[1].status == "completed"
    assert events[-1].stage == STAGE_ORDER[-1]
    assert events[-1].status == "completed"
    assert bus.state == "completed"


def test_stage_event_bus_rejects_invalid_transition() -> None:
    bus = StageEventBusV2()
    with pytest.raises(StageTransitionError):
        bus.start_stage("planner")


def test_v2_orchestrator_emits_stage_events_and_adapter_projects_them(_mini_db_path: Path) -> None:
    runtime = RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )
    orchestrator = V2Orchestrator(_DummyTeam())
    run = orchestrator.run(
        goal="How many transactions are there?",
        runtime=runtime,
        db_path=str(_mini_db_path),
        history=[],
        tenant_id="public",
        storyteller_mode=False,
        autonomy=None,
        scenario_context=None,
        session_id="v2-stage-session",
    )

    v2 = run.v2_payload
    assert v2.analysis_version == "v2"
    assert set(v2.stage_timings_ms.keys()) == set(STAGE_ORDER)
    assert len(v2.stage_events) == len(STAGE_ORDER) * 2

    # Stage events should be emitted in deterministic order: started/completed pairs.
    for idx, stage in enumerate(STAGE_ORDER):
        started = v2.stage_events[idx * 2]
        completed = v2.stage_events[idx * 2 + 1]
        assert started.stage == stage
        assert completed.stage == stage
        assert started.status == "started"
        assert completed.status == "completed"
        assert completed.elapsed_ms >= started.elapsed_ms

    adapted = apply_v2_compat_fields(run.response, v2, analysis_version="v2")
    assert adapted.analysis_version == "v2"
    assert adapted.slice_signature
    assert set(adapted.stage_timings_ms.keys()) == set(STAGE_ORDER)
    assert adapted.runtime.get("stage_state") == "completed"
    assert len(adapted.runtime.get("stage_events") or []) == len(STAGE_ORDER) * 2


def test_v2_orchestrator_carries_prior_slice_signature_in_followup_ops(_mini_db_path: Path) -> None:
    runtime = RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )
    orchestrator = V2Orchestrator(_DummyTeam())
    run = orchestrator.run(
        goal="Keep same slice and add total amount too.",
        runtime=runtime,
        db_path=str(_mini_db_path),
        history=[
            {
                "goal": "Show count by customer",
                "sql": "SELECT customer_id, COUNT(*) FROM transactions GROUP BY 1",
                "slice_signature": "abc123slice",
            }
        ],
        tenant_id="public",
        storyteller_mode=False,
        autonomy=None,
        scenario_context=None,
        session_id="v2-followup-session",
    )
    ops = set((run.v2_payload.intent.operations if run.v2_payload.intent else []) or [])
    assert "carry_scope" in ops
    assert "carry_slice_signature" in ops
    assert "add_secondary_metric" in ops
