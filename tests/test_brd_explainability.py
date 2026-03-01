"""BRD §7.1.4 — Explainability suite (>=15 cases).

Covers: decision-flow completeness, SQL/contract alignment,
        contradiction resolution visibility.
Acceptance: >=95% completeness.

BRD closure criterion #6: "Explain yourself" renders:
  - chosen contract
  - rejected alternatives and why
  - final SQL
  - audit checks
  - confidence decomposition
"""
from __future__ import annotations

import pytest
from haikugraph.poc.agentic_team import AgenticAnalyticsTeam


def _query(team, goal: str):
    from haikugraph.poc.agentic_team import RuntimeSelection

    rt = RuntimeSelection(
        requested_mode="deterministic",
        mode="deterministic",
        use_llm=False,
        provider=None,
        reason="test",
    )
    return team.run(goal, rt)


class TestDecisionFlowCompleteness:
    """E01-E08: Decision flow must be present and complete."""

    def test_e01_simple_query_has_decision_flow(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions are there?")
            assert resp.success
            flow = resp.decision_flow
            assert isinstance(flow, list)
            assert len(flow) >= 5, f"Decision flow has only {len(flow)} steps"

    def test_e02_flow_has_question_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "question_understanding" in step_names

    def test_e03_flow_has_contract_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total quote value in Dec-2025")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "contract_binding" in step_names

    def test_e04_flow_has_sql_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Customer count by country")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "sql_generation" in step_names

    def test_e05_flow_has_audit_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Average rate by deal type")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "audit_checks" in step_names

    def test_e06_flow_has_confidence_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many bookings?")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "confidence_decomposition" in step_names

    def test_e07_flow_has_execution_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Total transaction amount")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "execution" in step_names

    def test_e08_flow_has_validation_step(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Quote count by source currency")
            flow = resp.decision_flow
            step_names = [s.get("step") for s in flow]
            assert "contract_validation" in step_names


class TestContractSpec:
    """E09-E12: Contract spec is present and accurate."""

    def test_e09_contract_has_metric(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions?")
            cs = resp.contract_spec
            assert cs.get("metric"), "Contract spec missing metric"
            assert cs.get("table"), "Contract spec missing table"

    def test_e10_contract_has_time_scope(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count in Dec-2025")
            cs = resp.contract_spec
            assert "month_year" in cs.get("time_scope", ""), f"Expected month_year scope, got {cs.get('time_scope')}"

    def test_e11_contract_has_dimensions(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform_name")
            cs = resp.contract_spec
            assert len(cs.get("dimensions", [])) >= 1

    def test_e12_contract_validation_present(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count in Dec-2025 by platform")
            cv = resp.contract_validation
            assert "valid" in cv, "Contract validation missing 'valid' field"
            assert "checks" in cv, "Contract validation missing 'checks' field"

    def test_e12b_contract_has_canonical_metric_binding(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions?")
            cs = resp.contract_spec
            assert cs.get("canonical_metric_id") == "transactions.transaction_count"
            canonical = cs.get("canonical_metric") or {}
            assert canonical.get("metric_name") == "transaction_count"
            assert canonical.get("unit") in {"count", "number"}

    def test_e12f_data_quality_contains_kpi_decomposition_tree(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform_name")
            dq = resp.data_quality or {}
            kpi_tree = dq.get("kpi_decomposition") or {}
            root = kpi_tree.get("root") or {}
            assert root.get("kpi_id"), "KPI decomposition root is missing kpi_id"
            children = kpi_tree.get("children") or []
            assert isinstance(children, list)

    def test_e12g_kpi_decomposition_contains_owner_and_target(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions?")
            root = ((resp.data_quality or {}).get("kpi_decomposition") or {}).get("root") or {}
            assert root.get("owner")
            assert "target" in root
            assert root.get("target_source") in {"kpi_catalog", "domain_default"}


class TestExplainabilityLevels:
    """E12c-E12e: Explainability has business + technical levels."""

    def test_e12c_explainability_has_dual_levels(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count in Dec-2025 by platform_name")
            explainability = resp.explainability
            assert isinstance(explainability, dict)
            assert "business_view" in explainability
            assert "technical_view" in explainability

    def test_e12d_business_view_has_plain_steps(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions?")
            business = (resp.explainability or {}).get("business_view", {})
            steps = business.get("plain_steps") or []
            assert isinstance(steps, list)
            assert len(steps) >= 3
            assert any("interpreted" in str(step).lower() for step in steps)

    def test_e12e_technical_view_contains_contract_and_flow(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform")
            technical = (resp.explainability or {}).get("technical_view", {})
            assert isinstance(technical.get("contract_spec"), dict)
            assert isinstance(technical.get("contract_validation"), dict)
            flow = technical.get("decision_flow")
            assert isinstance(flow, list)
            assert len(flow) >= 5


class TestConfidenceDecomposition:
    """E13-E15: Confidence reasoning is detailed and accurate."""

    def test_e13_confidence_reasoning_present(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "How many transactions?")
            assert resp.confidence_reasoning, "confidence_reasoning is empty"
            assert "Score" in resp.confidence_reasoning

    def test_e14_confidence_factors_in_reasoning(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform")
            assert "Factors:" in resp.confidence_reasoning or "base_score" in resp.confidence_reasoning

    def test_e15_low_confidence_explains_why(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            # "joy score" is now an unsupported concept — policy gate blocks it.
            # A blocked response with confidence 0.0 and a refusal message also
            # satisfies the spirit of this test (low confidence is explained).
            resp = _query(team, "Customer joy score trend")
            # Blocked by policy gate → refusal answer explains why
            if resp.confidence_score == 0.0:
                assert "cannot" in resp.answer_markdown.lower() or "unsupported" in resp.answer_markdown.lower()
            else:
                assert resp.confidence_reasoning
                assert len(resp.confidence_reasoning) > 20


class TestSQLContractAlignment:
    """E16-E18: SQL aligns with contract."""

    def test_e16_month_in_contract_appears_in_sql(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transactions in Dec-2025")
            cs = resp.contract_spec
            if "month_year" in cs.get("time_scope", ""):
                assert "12" in (resp.sql or ""), "Month 12 not in SQL despite contract"

    def test_e17_group_by_in_contract_appears_in_sql(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Transaction count by platform_name")
            cs = resp.contract_spec
            if cs.get("dimensions"):
                assert "GROUP BY" in (resp.sql or "").upper()

    def test_e18_future_blocked_produces_false(self, known_data_db):
        with AgenticAnalyticsTeam(known_data_db) as team:
            resp = _query(team, "Platform split for 2032")
            cs = resp.contract_spec
            if "future_blocked" in cs.get("time_scope", ""):
                assert "FALSE" in (resp.sql or "").upper()
