from __future__ import annotations

from pathlib import Path

import haikugraph.llm.router as router
from haikugraph.poc.agentic_team import AgenticAnalyticsTeam, RuntimeSelection
from haikugraph.poc.decision_spine import (
    deterministic_runtime_snapshot,
    grouped_signal,
    validate_handoff_contract,
)


def _det_runtime() -> RuntimeSelection:
    return RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )


def test_slice2_ambiguity_requires_clarification(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run("split by month and platform", _det_runtime())
        assert resp.success is False
        assert str(resp.error or "") == "clarification_required"
        assert "clarification" in str(resp.answer_markdown or "").lower()
        assert not (resp.sql or "").strip()


def test_slice2_time_column_clash_requires_followup_for_amount_metric(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run("Total transaction amount in December 2025", _det_runtime())
        assert resp.success is False
        assert str(resp.error or "") == "clarification_required"
        answer = str(resp.answer_markdown or "").lower()
        assert "created_ts" in answer
        assert "event_ts" in answer


def test_slice3_decision_spine_runtime_snapshot_contract_guard():
    snap = deterministic_runtime_snapshot(
        requested_mode="openai",
        mode="openai",
        provider="openai",
    )
    assert snap.mode == "deterministic"
    assert snap.use_llm is False
    assert snap.provider is None
    assert "contract_guard_from_openai" in snap.reason
    assert grouped_signal("split transactions by month and platform") is True


def test_slice3_decision_spine_handoff_contract_validation():
    ok = validate_handoff_contract(
        "query_plan",
        {"sql": "SELECT 1", "table": "datada_mart_transactions"},
    )
    assert ok.valid is True
    bad = validate_handoff_contract("query_plan", {"sql": "SELECT 1"})
    assert bad.valid is False
    assert "handoff_missing_table" in bad.reason_codes
    schema_ok = validate_handoff_contract(
        "execution_plan",
        {"intent": "schema_exploration"},
    )
    assert schema_ok.valid is True


def test_slice4_ui_explain_contract_is_single_timeline_view():
    ui_path = Path(__file__).resolve().parents[1] / "src" / "haikugraph" / "api" / "ui.html"
    text = ui_path.read_text(encoding="utf-8")
    assert "Business view" in text
    assert "Technical drill-down" in text
    assert "Agent decision timeline" in text
    assert "Advanced diagnostics (JSON)" in text
    assert "Explain Yourself" in text


def test_slice4_mt103_policy_note_is_disclosed(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run("Show valid transaction spend by month", _det_runtime())
        assert resp.success is True
        answer = str(resp.answer_markdown or "")
        assert "Policy note applied before analysis" in answer
        assert "has_mt103=true" in answer


def test_slice4_multi_question_deterministic_split():
    rt = RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )
    parts = AgenticAnalyticsTeam._detect_multi_part(  # type: ignore[misc]
        AgenticAnalyticsTeam.__new__(AgenticAnalyticsTeam),  # bypass init; pure helper behavior
        "How many transactions in Dec-2025? Also split by platform?",
        rt,
    )
    assert isinstance(parts, list)
    assert len(parts) >= 2


def test_slice4_mixed_intent_without_sequence_requires_clarification(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run(
            "What kind of data do I have and total transaction amount by platform in December 2025",
            _det_runtime(),
        )
        assert resp.success is False
        assert str(resp.error or "") == "clarification_required"
        dq = resp.data_quality or {}
        clarification = dq.get("clarification") if isinstance(dq, dict) else {}
        reason = str((clarification or {}).get("reason") or "")
        assert "intent_collision_unresolved" in reason


def test_slice4_explicit_sequence_multi_part_handles_overview_and_schema(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run(
            "What kind of data do I have? Then generate me a glossary of schema each field and table and what it means.",
            _det_runtime(),
        )
        assert resp.success is True
        answer = str(resp.answer_markdown or "")
        assert "Data map" in answer
        assert "Schema Dictionary" in answer
        sql = str(resp.sql or "")
        assert "Sub-query" in sql


def test_slice_r13_mt103_amount_month_year_executes_without_time_clarification(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run(
            "For November 2025 and MT103-tagged transactions only, what was total transaction amount?",
            _det_runtime(),
        )
        assert resp.success is True
        sql = str(resp.sql or "").lower()
        assert "has_mt103" in sql
        assert "extract(year from created_ts) = 2025" in sql
        assert "extract(month from created_ts) = 11" in sql


def test_slice_r13_refund_count_uses_refund_flag_not_payment_status_filter(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run("How many refunded transactions happened in Q1 2026?", _det_runtime())
        assert resp.success is True
        sql = str(resp.sql or "").lower()
        assert "has_refund" in sql
        assert "payment_status" not in sql


def test_slice_r13_followup_switch_metric_to_average_forex_markup(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        first_goal = "Show top source currencies by quote count in Jan-2026"
        first = team.run(first_goal, _det_runtime())
        context = [{"goal": first_goal, "sql": first.sql or "", "answer_markdown": first.answer_markdown or ""}]
        second = team.run(
            "Keep that scope but switch metric to average forex markup and keep top currencies.",
            _det_runtime(),
            conversation_context=context,
        )
        assert second.success is True
        sql = str(second.sql or "").lower()
        assert "avg(forex_markup)" in sql
        assert "from_currency" in sql
        assert "extract(year from created_ts) = 2026" in sql
        assert "extract(month from created_ts) = 1" in sql


def test_slice_r13_followup_add_amount_keeps_dual_metric_same_scope(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        first_goal = "What is the transaction count of MT103 by state for November 2025?"
        first = team.run(first_goal, _det_runtime())
        context = [{"goal": first_goal, "sql": first.sql or "", "answer_markdown": first.answer_markdown or ""}]
        second = team.run(
            "Now keep that same slice and add total amount too.",
            _det_runtime(),
            conversation_context=context,
        )
        assert second.success is True
        sql = str(second.sql or "").lower()
        assert "secondary_metric_value" in sql
        assert "has_mt103" in sql
        assert "state" in sql
        assert "2025" in sql
        assert "11" in sql


def test_slice_r13_enforce_intake_locks_explicit_month_year_and_ranking_dimension(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        goal = "Which platform had the highest MT103 transaction amount in December 2025?"
        deterministic = {
            "goal": goal,
            "domain": "transactions",
            "intent": "grouped_metric",
            "metric": "total_amount",
            "dimensions": ["platform_name"],
            "time_filter": {"kind": "month_year", "month": 12, "year": 2025},
            "value_filters": [{"column": "has_mt103", "value": "true"}],
        }
        parsed = {
            "goal": goal,
            "domain": "transactions",
            "intent": "metric",
            "metric": "total_amount",
            "dimensions": [],
            "dimension": None,
            "time_filter": {"kind": "month_year", "month": 11, "year": 2025},
            "value_filters": [{"column": "has_mt103", "value": "true"}],
        }
        catalog = {
            "marts": {"datada_mart_transactions": {"columns": ["platform_name", "created_ts", "has_mt103"]}},
            "metrics_by_table": {"datada_mart_transactions": {"total_amount": "SUM(amount)"}},
        }
        team._pipeline_warnings = []
        out = team._enforce_intake_consistency(goal, deterministic, parsed, catalog, is_llm_mode=True)
        assert out.get("time_filter", {}).get("month") == 12
        assert out.get("dimensions") == ["platform_name"]
        assert out.get("intent") == "grouped_metric"


def test_slice_r13_local_mode_skips_llm_intake_refinement(monkeypatch, known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        calls = {"count": 0}

        def _fake_intake_with_llm(*args, **kwargs):
            calls["count"] += 1
            return {"metric": "quote_count"}

        monkeypatch.setattr(team, "_intake_with_llm", _fake_intake_with_llm)
        local_runtime = RuntimeSelection(
            requested_mode="local",
            mode="local",
            use_llm=True,
            provider="ollama",
            reason="test",
            intent_model="qwen2.5:7b-instruct",
            narrator_model="qwen2.5:7b-instruct",
        )
        resp = team.run("How many transactions are there?", local_runtime)
        assert resp.success is True
        assert calls["count"] == 0


def test_slice_r13_followup_same_grouped_output_skips_llm_context_rewrite(monkeypatch, known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        calls = {"count": 0}

        def _fake_call_llm(*args, **kwargs):
            calls["count"] += 1
            return '{"standalone_goal":"bad rewrite"}'

        monkeypatch.setattr("haikugraph.poc.agentic_team.call_llm", _fake_call_llm)
        runtime = RuntimeSelection(
            requested_mode="openai",
            mode="openai",
            use_llm=True,
            provider="openai",
            reason="test",
            intent_model="gpt-5.3",
            narrator_model="gpt-5.3",
        )
        goal = "Now only January and include booked amount in the same grouped output."
        resolved = team._resolve_contextual_goal(
            goal,
            [{"goal": "Show booking count by deal_type for Q1 2026", "sql": "SELECT 1"}],
            runtime,
        )
        assert "Follow-up:" in resolved
        assert goal in resolved
        assert calls["count"] == 0


def test_slice5_router_enforces_bounded_timeout(monkeypatch):
    captured: dict[str, int] = {}

    def _fake_openai(messages, model, temperature, max_tokens, timeout):
        del messages, model, temperature, max_tokens
        captured["timeout"] = int(timeout)
        return "ok", {"model": "gpt-4o-mini", "input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

    monkeypatch.setenv("HG_BOUNDED_LATENCY_ENABLED", "1")
    monkeypatch.setenv("HG_LLM_CACHE_ENABLED", "0")
    monkeypatch.setenv("HG_OPENAI_PLANNER_MAX_SECONDS", "7")
    monkeypatch.setattr(router, "_call_openai", _fake_openai)

    router.call_llm(
        [{"role": "user", "content": "Count transactions"}],
        role="planner",
        provider="openai",
        model="gpt-4o",
        timeout=60,
    )
    assert captured["timeout"] == 7


def test_slice5_router_model_fallback_after_failure(monkeypatch):
    attempts: list[str] = []

    def _fake_openai(messages, model, temperature, max_tokens, timeout):
        del messages, temperature, max_tokens, timeout
        attempts.append(model)
        if model == "gpt-4o":
            raise ValueError("primary model unavailable")
        return "ok-from-fallback", {"model": model, "input_tokens": 8, "output_tokens": 3, "total_tokens": 11}

    monkeypatch.setenv("HG_BOUNDED_LATENCY_ENABLED", "0")
    monkeypatch.setenv("HG_LLM_CACHE_ENABLED", "0")
    monkeypatch.setenv("HG_ENABLE_MODEL_FALLBACK", "1")
    monkeypatch.setattr(router, "_call_openai", _fake_openai)

    out = router.call_llm(
        [{"role": "user", "content": "Compare transaction and quote trends"}],
        role="planner",
        provider="openai",
        model="gpt-4o",
        timeout=5,
    )
    assert out == "ok-from-fallback"
    assert attempts[:2] == ["gpt-4o", "gpt-4o-mini"]
