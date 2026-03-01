from __future__ import annotations

from unittest.mock import patch

from haikugraph.poc.agentic_team import AgenticAnalyticsTeam, RuntimeSelection


def _det_runtime() -> RuntimeSelection:
    return RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )


def test_grouped_signal_preserved_for_split_typo(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run("What are the transaction split my month and platform", _det_runtime())
        assert resp.success
        assert resp.contract_spec.get("intent") == "grouped_metric"
        dims = list(resp.contract_spec.get("dimensions") or [])
        assert "__month__" in dims
        assert "platform_name" in dims
        assert "GROUP BY" in (resp.sql or "").upper()


def test_contract_guard_recompiles_invalid_group_sql(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        team._pipeline_warnings = []
        catalog = team.semantic.prepare()
        intake = team._intake_deterministic("Transaction count by platform", catalog)
        retrieval = team._semantic_retrieval_agent(intake, catalog)
        plan = team._planning_agent(intake, retrieval, catalog, _det_runtime())
        bad_query_plan = {
            "sql": f"SELECT {plan['metric_expr']} AS metric_value FROM {plan['table']} WHERE 1=1",
            "table": plan["table"],
        }

        guarded_qp, _, guarded_validation = team._enforce_contract_guard(
            plan=plan,
            query_plan=bad_query_plan,
            specialist_findings=[],
            runtime=_det_runtime(),
            catalog=catalog,
            stage="unit_test",
        )

        assert guarded_validation.get("valid") is True
        assert "GROUP BY" in str(guarded_qp.get("sql") or "").upper()


def test_currency_pair_query_keeps_virtual_dimension_under_llm_refinement(known_data_db):
    llm_runtime = RuntimeSelection(
        requested_mode="openai",
        mode="openai",
        use_llm=True,
        provider="openai",
        reason="test",
        intent_model="gpt-4.1-mini",
        narrator_model="gpt-4.1-mini",
    )
    with AgenticAnalyticsTeam(known_data_db) as team:
        team._pipeline_warnings = []
        catalog = team.semantic.prepare()
        intake = team._intake_deterministic("What is the most frequent currency pair in quotes?", catalog)
        retrieval = team._semantic_retrieval_agent(intake, catalog)
        with patch.object(
            team,
            "_planning_agent_with_llm",
            return_value={"metric": "quote_count", "dimensions": ["from_currency", "to_currency"], "reasoning": "mock"},
        ):
            plan = team._planning_agent(intake, retrieval, catalog, llm_runtime)
        assert plan.get("dimensions") == ["currency_pair"]


def test_execution_agent_includes_analysis_rows(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        team._pipeline_warnings = []
        catalog = team.semantic.prepare()
        intake = team._intake_deterministic("Transaction count by platform", catalog)
        retrieval = team._semantic_retrieval_agent(intake, catalog)
        plan = team._planning_agent(intake, retrieval, catalog, _det_runtime())
        query_plan = team._query_engine_agent(plan, [], _det_runtime(), catalog)
        execution = team._execution_agent(query_plan)
        assert execution["success"] is True
        assert isinstance(execution.get("analysis_rows"), list)
        assert len(execution["analysis_rows"]) >= len(execution["sample_rows"])


def test_contract_spec_and_trace_include_canonical_metric_registry_fields(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        resp = team.run("How many transactions are there?", _det_runtime())
        assert resp.success is True
        assert resp.contract_spec.get("canonical_metric_id") == "transactions.transaction_count"
        canonical = resp.contract_spec.get("canonical_metric") or {}
        assert canonical.get("metric_name") == "transaction_count"
        assert canonical.get("table") == "datada_mart_transactions"

        planning_packet = next(
            (pkt for pkt in (resp.evidence_packets or []) if pkt.get("agent") == "PlanningAgent"),
            {},
        )
        assert planning_packet.get("canonical_metric_id") == "transactions.transaction_count"
        assert (resp.data_quality or {}).get("canonical_metric_contract", {}).get("metric_id") == "transactions.transaction_count"


def test_canonical_registry_alias_binding_for_fx_charges(known_data_db):
    with AgenticAnalyticsTeam(known_data_db) as team:
        catalog = team.semantic.prepare()
        intake = team._intake_agent(
            "show fx charges by month",
            _det_runtime(),
            {},
            catalog,
            [],
        )
        assert intake.get("metric") == "forex_markup_revenue"
        assert intake.get("canonical_metric_id") == "quotes.forex_markup_revenue"

        retrieval = team._semantic_retrieval_agent(intake, catalog)
        plan = team._planning_agent(intake, retrieval, catalog, _det_runtime())
        assert plan.get("canonical_metric_id") == "quotes.forex_markup_revenue"

        contract = team._build_contract_spec(plan)
        assert contract.get("canonical_metric_id") == "quotes.forex_markup_revenue"
