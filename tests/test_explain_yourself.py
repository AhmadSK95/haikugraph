"""Comprehensive tests for the Explain Yourself decision-grade explainability panel.

Covers all 7 required sections, completeness scoring, serialisation formats,
and edge cases for Gate 7 evaluation (>=95% completeness target).
"""

from __future__ import annotations

import pytest

from haikugraph.explain.explain_yourself import (
    REQUIRED_SECTIONS,
    ExplainSection,
    ExplainYourselfPanel,
    build_explain_yourself,
    compute_panel_completeness,
)


# ---------------------------------------------------------------------------
# Fixtures: reusable pipeline artifacts for building panels
# ---------------------------------------------------------------------------

@pytest.fixture()
def full_intent() -> dict:
    return {
        "type": "grouped_metric",
        "confidence": 0.92,
        "rationale": "User wants transaction count broken down by platform",
        "requires_comparison": False,
    }


@pytest.fixture()
def full_contract() -> dict:
    return {
        "metric": "count_transactions",
        "domain": "test_1_1_merged",
        "grain": ["platform_name"],
        "time_scope": {"type": "month_year", "month": 12, "year": 2025},
        "filters": [{"column": "payment_status", "op": "=", "value": "completed"}],
        "exclusions": [],
    }


@pytest.fixture()
def full_contract_validation() -> dict:
    return {
        "valid": True,
        "violations": [],
        "checks": [
            {"name": "metric_in_select", "passed": True},
            {"name": "grain_in_group_by", "passed": True},
            {"name": "time_filter_present", "passed": True},
        ],
    }


@pytest.fixture()
def full_alternatives() -> list[dict]:
    return [
        {
            "description": "Use test_3_1 (quotes) table instead",
            "rejection_reason": "No transaction count metric available in quotes",
            "score": 0.3,
        },
        {
            "description": "Use test_5_1 (bookings) table instead",
            "rejection_reason": "Bookings table lacks platform_name dimension",
            "score": 0.25,
        },
    ]


@pytest.fixture()
def full_sql() -> str:
    return (
        "SELECT platform_name, COUNT(*) AS transaction_count "
        "FROM test_1_1_merged "
        "WHERE payment_status = 'completed' "
        "AND created_at >= '2025-12-01' AND created_at < '2026-01-01' "
        "GROUP BY platform_name "
        "ORDER BY transaction_count DESC "
        "LIMIT 100"
    )


@pytest.fixture()
def full_audit() -> dict:
    return {
        "overall_pass": True,
        "passed": 3,
        "warned": 1,
        "failed": 0,
        "checks": [
            {"check_name": "null_rate", "status": "pass", "message": "Null rate < 5%"},
            {"check_name": "row_count", "status": "pass", "message": "Row count > 0"},
            {"check_name": "join_validity", "status": "pass", "message": "No joins used"},
            {"check_name": "time_filter", "status": "warn", "message": "Time filter uses VARCHAR comparison"},
        ],
    }


@pytest.fixture()
def full_confidence() -> dict:
    return {
        "overall": 0.87,
        "level": "HIGH",
        "factors": [
            {"name": "schema_match", "score": 0.95, "weight": 0.3, "reason": "Exact column match"},
            {"name": "intent_clarity", "score": 0.92, "weight": 0.25, "reason": "Clear grouped metric intent"},
            {"name": "audit_pass", "score": 0.85, "weight": 0.25, "reason": "3/4 checks passed, 1 warning"},
            {"name": "data_freshness", "score": 0.75, "weight": 0.2, "reason": "Data from current month"},
        ],
        "penalties": ["-0.02 for VARCHAR time comparison"],
        "reasoning": "High confidence: exact schema match, clear intent, passing audits.",
    }


@pytest.fixture()
def full_panel_kwargs(
    full_intent,
    full_contract,
    full_contract_validation,
    full_alternatives,
    full_sql,
    full_audit,
    full_confidence,
):
    """All keyword arguments needed to build a 100 %-complete panel."""
    return {
        "goal": "Transaction count by platform in Dec-2025",
        "intent": full_intent,
        "contract": full_contract,
        "contract_validation": full_contract_validation,
        "rejected_alternatives": full_alternatives,
        "sql": full_sql,
        "audit": full_audit,
        "confidence": full_confidence,
        "narrative": "B2C-APP leads with 3 completed transactions in December 2025.",
        "answer_summary": "3 platforms have completed transactions in Dec-2025.",
    }


# =========================================================================
# 1. Full panel tests
# =========================================================================

class TestFullPanel:
    """Tests for a fully-populated Explain Yourself panel."""

    def test_full_panel_all_sections_present(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        present_ids = {s.section_id for s in panel.sections if s.present}
        for req in REQUIRED_SECTIONS:
            assert req in present_ids, f"Required section '{req}' is not present"

    def test_panel_completeness_100_percent(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        assert panel.completeness == pytest.approx(1.0)
        assert panel.missing_sections == []

    def test_panel_section_count_equals_seven(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        assert len(panel.sections) == 7

    def test_panel_to_dict_structure(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        d = panel.to_dict()
        assert "sections" in d
        assert "completeness" in d
        assert "missing_sections" in d
        assert isinstance(d["sections"], list)
        assert len(d["sections"]) == 7

    def test_to_decision_flow_format(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        flow = panel.to_decision_flow()
        assert isinstance(flow, list)
        # All 7 sections are present so all should appear in the flow
        assert len(flow) == 7
        for step in flow:
            assert "step" in step
            assert "title" in step


# =========================================================================
# 2. Completeness scoring
# =========================================================================

class TestCompleteness:
    """Tests for completeness computation and missing section reporting."""

    def test_panel_completeness_partial(self):
        """Panel with only goal (no contract, no SQL, no audit, etc.)."""
        panel = build_explain_yourself(goal="How many transactions?")
        # query_intent is always present; rejected_alternatives is always present
        # That gives 2 of 7 required sections
        assert panel.completeness == pytest.approx(2 / 7, abs=0.01)

    def test_panel_missing_sections_reported(self):
        panel = build_explain_yourself(goal="How many transactions?")
        assert "chosen_contract" in panel.missing_sections
        assert "sql_annotation" in panel.missing_sections
        assert "audit_summary" in panel.missing_sections
        assert "confidence_decomposition" in panel.missing_sections
        assert "narrative_summary" in panel.missing_sections

    def test_completeness_with_sql_only(self):
        panel = build_explain_yourself(
            goal="Count rows",
            sql="SELECT COUNT(*) FROM t",
        )
        # present: query_intent, rejected_alternatives, sql_annotation = 3/7
        assert panel.completeness == pytest.approx(3 / 7, abs=0.01)

    def test_completeness_with_narrative_only(self):
        panel = build_explain_yourself(
            goal="Count rows",
            narrative="There are 42 rows.",
        )
        # present: query_intent, rejected_alternatives, narrative_summary = 3/7
        assert panel.completeness == pytest.approx(3 / 7, abs=0.01)

    def test_compute_panel_completeness_helper(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        assert compute_panel_completeness(panel) == pytest.approx(1.0)


# =========================================================================
# 3. Section 1: Query Intent
# =========================================================================

class TestIntentSection:
    """Tests for the query_intent section."""

    def test_intent_section_with_classification(self, full_intent):
        panel = build_explain_yourself(
            goal="Transaction count by platform",
            intent=full_intent,
        )
        sec = _find_section(panel, "query_intent")
        assert sec.present is True
        assert sec.content["original_question"] == "Transaction count by platform"
        assert sec.content["detected_intent"] == "grouped_metric"
        assert sec.content["intent_confidence"] == pytest.approx(0.92)
        assert sec.content["rationale"] != ""
        assert sec.content["requires_comparison"] is False

    def test_intent_section_without_classification(self):
        panel = build_explain_yourself(goal="Some question")
        sec = _find_section(panel, "query_intent")
        assert sec.present is True
        assert sec.content["detected_intent"] == "not_classified"
        assert sec.content["intent_confidence"] == 0.0

    def test_intent_section_empty_goal(self):
        panel = build_explain_yourself(goal="")
        sec = _find_section(panel, "query_intent")
        assert sec.present is True
        assert sec.content["original_question"] == ""


# =========================================================================
# 4. Section 2: Chosen Contract
# =========================================================================

class TestContractSection:
    """Tests for the chosen_contract section."""

    def test_contract_section_with_validation(
        self, full_contract, full_contract_validation
    ):
        panel = build_explain_yourself(
            goal="Count txns",
            contract=full_contract,
            contract_validation=full_contract_validation,
        )
        sec = _find_section(panel, "chosen_contract")
        assert sec.present is True
        assert sec.content["metric"] == "count_transactions"
        assert sec.content["domain"] == "test_1_1_merged"
        assert sec.content["grain"] == ["platform_name"]
        assert "validation" in sec.content
        assert sec.content["validation"]["valid"] is True
        assert sec.content["validation"]["checks_count"] == 3

    def test_contract_section_missing(self):
        panel = build_explain_yourself(goal="Question without contract")
        sec = _find_section(panel, "chosen_contract")
        assert sec.present is False
        assert sec.content["status"] == "no_contract"

    def test_contract_section_without_validation(self, full_contract):
        panel = build_explain_yourself(goal="Q", contract=full_contract)
        sec = _find_section(panel, "chosen_contract")
        assert sec.present is True
        assert "validation" not in sec.content

    def test_contract_section_preserves_filters(self, full_contract):
        panel = build_explain_yourself(goal="Q", contract=full_contract)
        sec = _find_section(panel, "chosen_contract")
        assert len(sec.content["filters"]) == 1
        assert sec.content["filters"][0]["column"] == "payment_status"


# =========================================================================
# 5. Section 3: Rejected Alternatives
# =========================================================================

class TestAlternativesSection:
    """Tests for the rejected_alternatives section."""

    def test_alternatives_section_with_items(self, full_alternatives):
        panel = build_explain_yourself(
            goal="Q",
            rejected_alternatives=full_alternatives,
        )
        sec = _find_section(panel, "rejected_alternatives")
        assert sec.present is True
        assert sec.content["alternatives_considered"] == 2
        assert len(sec.content["items"]) == 2
        assert sec.content["items"][0]["option"] == "Use test_3_1 (quotes) table instead"
        assert sec.content["items"][1]["score"] == pytest.approx(0.25)

    def test_alternatives_section_empty(self):
        panel = build_explain_yourself(goal="Q")
        sec = _find_section(panel, "rejected_alternatives")
        assert sec.present is True
        assert sec.content["alternatives_considered"] == 0
        assert sec.content["items"] == []
        assert "note" in sec.content


# =========================================================================
# 6. Section 4: SQL Annotation
# =========================================================================

class TestSQLAnnotationSection:
    """Tests for the sql_annotation section."""

    def test_sql_annotation_all_clauses(self, full_sql, full_contract):
        panel = build_explain_yourself(
            goal="Q",
            sql=full_sql,
            contract=full_contract,
        )
        sec = _find_section(panel, "sql_annotation")
        assert sec.present is True
        assert sec.content["sql"] == full_sql

        clause_names = [a["clause"] for a in sec.content["annotations"]]
        assert "SELECT" in clause_names
        assert "FROM" in clause_names
        assert "WHERE" in clause_names
        assert "GROUP BY" in clause_names
        assert "ORDER BY" in clause_names
        assert "LIMIT" in clause_names
        assert "CONTRACT" in clause_names

    def test_sql_annotation_simple_query(self):
        simple_sql = "SELECT COUNT(*) FROM my_table"
        panel = build_explain_yourself(goal="Q", sql=simple_sql)
        sec = _find_section(panel, "sql_annotation")
        assert sec.present is True

        clause_names = [a["clause"] for a in sec.content["annotations"]]
        assert "SELECT" in clause_names
        assert "FROM" in clause_names
        # Simple query has no WHERE, GROUP BY, ORDER BY, LIMIT
        assert "WHERE" not in clause_names
        assert "GROUP BY" not in clause_names

    def test_sql_annotation_from_extracts_table_name(self):
        panel = build_explain_yourself(
            goal="Q",
            sql="SELECT x FROM fancy_table WHERE y > 1",
        )
        sec = _find_section(panel, "sql_annotation")
        from_ann = [a for a in sec.content["annotations"] if a["clause"] == "FROM"]
        assert len(from_ann) == 1
        assert "fancy_table" in from_ann[0]["explanation"]

    def test_sql_annotation_no_sql(self):
        panel = build_explain_yourself(goal="Q")
        sec = _find_section(panel, "sql_annotation")
        assert sec.present is False

    def test_sql_annotation_contract_binding_message(self, full_contract):
        panel = build_explain_yourself(
            goal="Q",
            sql="SELECT 1 FROM t",
            contract=full_contract,
        )
        sec = _find_section(panel, "sql_annotation")
        contract_ann = [a for a in sec.content["annotations"] if a["clause"] == "CONTRACT"]
        assert len(contract_ann) == 1
        assert "count_transactions" in contract_ann[0]["explanation"]
        assert "test_1_1_merged" in contract_ann[0]["explanation"]


# =========================================================================
# 7. Section 5: Audit Summary
# =========================================================================

class TestAuditSection:
    """Tests for the audit_summary section."""

    def test_audit_section_with_checks(self, full_audit):
        panel = build_explain_yourself(goal="Q", audit=full_audit)
        sec = _find_section(panel, "audit_summary")
        assert sec.present is True
        assert sec.content["overall_pass"] is True
        assert sec.content["passed"] == 3
        assert sec.content["warned"] == 1
        assert sec.content["failed"] == 0
        assert len(sec.content["checks"]) == 4

    def test_audit_section_check_details(self, full_audit):
        panel = build_explain_yourself(goal="Q", audit=full_audit)
        sec = _find_section(panel, "audit_summary")
        first = sec.content["checks"][0]
        assert first["name"] == "null_rate"
        assert first["status"] == "pass"
        assert first["message"] == "Null rate < 5%"

    def test_audit_section_absent(self):
        panel = build_explain_yourself(goal="Q")
        sec = _find_section(panel, "audit_summary")
        assert sec.present is False
        assert sec.content["status"] == "no_audit"


# =========================================================================
# 8. Section 6: Confidence Decomposition
# =========================================================================

class TestConfidenceSection:
    """Tests for the confidence_decomposition section."""

    def test_confidence_section_with_factors(self, full_confidence):
        panel = build_explain_yourself(goal="Q", confidence=full_confidence)
        sec = _find_section(panel, "confidence_decomposition")
        assert sec.present is True
        assert sec.content["overall"] == pytest.approx(0.87)
        assert sec.content["level"] == "HIGH"
        assert len(sec.content["factors"]) == 4
        assert sec.content["penalties"] == ["-0.02 for VARCHAR time comparison"]
        assert "High confidence" in sec.content["reasoning"]

    def test_confidence_factor_structure(self, full_confidence):
        panel = build_explain_yourself(goal="Q", confidence=full_confidence)
        sec = _find_section(panel, "confidence_decomposition")
        f0 = sec.content["factors"][0]
        assert f0["name"] == "schema_match"
        assert f0["score"] == pytest.approx(0.95)
        assert f0["weight"] == pytest.approx(0.3)
        assert f0["reason"] == "Exact column match"

    def test_confidence_section_absent(self):
        panel = build_explain_yourself(goal="Q")
        sec = _find_section(panel, "confidence_decomposition")
        assert sec.present is False
        assert sec.content["status"] == "no_confidence_data"


# =========================================================================
# 9. Section 7: Narrative Summary
# =========================================================================

class TestNarrativeSection:
    """Tests for the narrative_summary section."""

    def test_narrative_section(self):
        panel = build_explain_yourself(
            goal="Q",
            narrative="Full narrative paragraph here.",
            answer_summary="Short summary.",
        )
        sec = _find_section(panel, "narrative_summary")
        assert sec.present is True
        assert sec.content["answer_summary"] == "Short summary."
        assert sec.content["full_narrative"] == "Full narrative paragraph here."

    def test_narrative_section_summary_from_narrative(self):
        """When answer_summary is empty, it should fall back to narrative."""
        panel = build_explain_yourself(
            goal="Q",
            narrative="A moderately long narrative text that describes the result.",
        )
        sec = _find_section(panel, "narrative_summary")
        assert sec.present is True
        assert sec.content["answer_summary"] == "A moderately long narrative text that describes the result."

    def test_narrative_section_truncation(self):
        """Long narratives should be truncated in the summary when no explicit summary given."""
        long_narrative = "A" * 300
        panel = build_explain_yourself(goal="Q", narrative=long_narrative)
        sec = _find_section(panel, "narrative_summary")
        assert sec.content["answer_summary"].endswith("...")
        assert len(sec.content["answer_summary"]) == 203  # 200 chars + "..."

    def test_narrative_section_absent(self):
        panel = build_explain_yourself(goal="Q")
        sec = _find_section(panel, "narrative_summary")
        assert sec.present is False


# =========================================================================
# 10. Decision flow serialisation
# =========================================================================

class TestDecisionFlowFormat:
    """Tests for to_decision_flow() output format."""

    def test_decision_flow_excludes_absent_sections(self):
        panel = build_explain_yourself(goal="Q")
        flow = panel.to_decision_flow()
        step_ids = [s["step"] for s in flow]
        # Only query_intent and rejected_alternatives should be present
        assert "query_intent" in step_ids
        assert "rejected_alternatives" in step_ids
        assert "chosen_contract" not in step_ids
        assert "sql_annotation" not in step_ids

    def test_decision_flow_step_has_content_keys(self, full_panel_kwargs):
        panel = build_explain_yourself(**full_panel_kwargs)
        flow = panel.to_decision_flow()
        intent_step = [s for s in flow if s["step"] == "query_intent"][0]
        assert "original_question" in intent_step
        assert "detected_intent" in intent_step

    def test_decision_flow_compatible_with_response_contract(self, full_panel_kwargs):
        """Verify flow entries can be placed in AssistantQueryResponse.decision_flow."""
        panel = build_explain_yourself(**full_panel_kwargs)
        flow = panel.to_decision_flow()
        # decision_flow is list[dict[str, Any]] - ensure all entries are dicts
        for entry in flow:
            assert isinstance(entry, dict)
            assert isinstance(entry.get("step"), str)
            assert isinstance(entry.get("title"), str)


# =========================================================================
# 11. ExplainSection dataclass
# =========================================================================

class TestExplainSection:
    """Tests for the ExplainSection dataclass itself."""

    def test_to_dict(self):
        sec = ExplainSection(
            section_id="test",
            title="Test section",
            content={"key": "value"},
            present=True,
        )
        d = sec.to_dict()
        assert d["section_id"] == "test"
        assert d["title"] == "Test section"
        assert d["content"] == {"key": "value"}
        assert d["present"] is True

    def test_to_dict_absent_section(self):
        sec = ExplainSection(
            section_id="empty",
            title="Empty",
            content={},
            present=False,
        )
        d = sec.to_dict()
        assert d["present"] is False


# =========================================================================
# 12. Edge cases and regressions
# =========================================================================

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_all_none_inputs(self):
        """Calling with absolutely no data should not raise."""
        panel = build_explain_yourself()
        assert panel.completeness > 0  # query_intent + rejected_alternatives
        assert len(panel.sections) == 7

    def test_empty_contract_dict_is_falsy(self):
        """An empty dict {} for contract should behave like no contract."""
        panel = build_explain_yourself(goal="Q", contract={})
        sec = _find_section(panel, "chosen_contract")
        # Empty dict is falsy in Python, so contract section should be absent
        assert sec.present is False

    def test_answer_summary_only_makes_narrative_present(self):
        panel = build_explain_yourself(goal="Q", answer_summary="Just a summary.")
        sec = _find_section(panel, "narrative_summary")
        assert sec.present is True
        assert sec.content["answer_summary"] == "Just a summary."
        assert sec.content["full_narrative"] == ""

    def test_sql_with_no_contract_skips_contract_annotation(self):
        panel = build_explain_yourself(
            goal="Q",
            sql="SELECT 1 FROM t",
        )
        sec = _find_section(panel, "sql_annotation")
        clause_names = [a["clause"] for a in sec.content["annotations"]]
        assert "CONTRACT" not in clause_names

    def test_audit_with_empty_checks_list(self):
        panel = build_explain_yourself(
            goal="Q",
            audit={"overall_pass": True, "passed": 0, "warned": 0, "failed": 0, "checks": []},
        )
        sec = _find_section(panel, "audit_summary")
        assert sec.present is True
        assert sec.content["checks"] == []

    def test_confidence_with_empty_factors(self):
        panel = build_explain_yourself(
            goal="Q",
            confidence={"overall": 0.5, "level": "MEDIUM", "factors": []},
        )
        sec = _find_section(panel, "confidence_decomposition")
        assert sec.present is True
        assert sec.content["factors"] == []
        assert sec.content["overall"] == pytest.approx(0.5)

    def test_required_sections_constant_has_seven_entries(self):
        assert len(REQUIRED_SECTIONS) == 7

    def test_alternatives_with_missing_fields(self):
        """Alternatives with missing optional fields use defaults."""
        panel = build_explain_yourself(
            goal="Q",
            rejected_alternatives=[{"description": "Alt A"}],
        )
        sec = _find_section(panel, "rejected_alternatives")
        item = sec.content["items"][0]
        assert item["option"] == "Alt A"
        assert item["reason_rejected"] == "Lower confidence or fitness"
        assert item["score"] == 0.0


# =========================================================================
# Helpers
# =========================================================================

def _find_section(panel: ExplainYourselfPanel, section_id: str) -> ExplainSection:
    """Find a section by ID in the panel or fail."""
    for sec in panel.sections:
        if sec.section_id == section_id:
            return sec
    raise AssertionError(f"Section '{section_id}' not found in panel")
