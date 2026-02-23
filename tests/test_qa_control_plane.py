"""Tests for the QA control plane and test suites S1-S5.

Covers:
- QATenant creation, budget reset, budget exhaustion
- QATestCase creation and serialization
- evaluate_test_result: pass, fail, refusal, confidence
- build_suite_report: gate pass and gate fail
- Suite cardinality: S1>=40, S2>=20, S3>=15, S4>=15
- Suite targets defined for all suites
"""

from __future__ import annotations

import pytest

from haikugraph.qa.control_plane import (
    QATenant,
    QATestCase,
    QATestResult,
    QASuiteReport,
    create_qa_tenant,
    evaluate_test_result,
    build_suite_report,
)
from haikugraph.qa.test_suites import (
    get_suite_s1_factual,
    get_suite_s2_safety,
    get_suite_s3_followup,
    get_suite_s4_explainability,
    get_all_suites,
    SUITE_TARGETS,
)


# -----------------------------------------------------------------------
# QATenant tests
# -----------------------------------------------------------------------

class TestQATenant:
    """Tests for QATenant creation, budget, and serialization."""

    def test_qa_tenant_creation(self):
        """A freshly created QATenant has the correct defaults."""
        tenant = QATenant()
        assert tenant.tenant_id == "qa_dedicated"
        assert tenant.budget_usd == 100.0
        assert tenant.budget_used == 0.0
        assert tenant.max_queries_per_run == 500
        assert tenant.queries_executed == 0
        assert tenant.allow_all_modes is True
        assert tenant.can_execute() is True

    def test_qa_tenant_factory(self):
        """create_qa_tenant produces a tenant with a unique id and full budget."""
        t1 = create_qa_tenant()
        t2 = create_qa_tenant()
        assert t1.tenant_id.startswith("qa_")
        assert t2.tenant_id.startswith("qa_")
        assert t1.tenant_id != t2.tenant_id
        assert t1.budget_usd == 100.0
        assert t1.budget_used == 0.0

    def test_qa_tenant_budget_reset(self):
        """reset_budget zeroes out budget_used and queries_executed."""
        tenant = QATenant()
        tenant.record_query(cost_usd=5.0)
        tenant.record_query(cost_usd=3.0)
        assert tenant.budget_used == 8.0
        assert tenant.queries_executed == 2

        tenant.reset_budget()
        assert tenant.budget_used == 0.0
        assert tenant.queries_executed == 0
        assert tenant.can_execute() is True

    def test_qa_tenant_budget_exhaustion(self):
        """can_execute returns False when budget is exhausted."""
        tenant = QATenant(budget_usd=0.02, max_queries_per_run=500)
        assert tenant.can_execute() is True

        tenant.record_query(cost_usd=0.01)
        assert tenant.can_execute() is True

        tenant.record_query(cost_usd=0.01)
        assert tenant.can_execute() is False  # 0.02 >= 0.02

    def test_qa_tenant_query_limit_exhaustion(self):
        """can_execute returns False when query count limit is reached."""
        tenant = QATenant(budget_usd=100.0, max_queries_per_run=2)
        tenant.record_query(cost_usd=0.01)
        tenant.record_query(cost_usd=0.01)
        assert tenant.can_execute() is False  # 2 >= 2

    def test_qa_tenant_to_dict(self):
        """to_dict returns a plain dictionary with all fields."""
        tenant = QATenant()
        d = tenant.to_dict()
        assert isinstance(d, dict)
        assert d["tenant_id"] == "qa_dedicated"
        assert d["budget_usd"] == 100.0
        assert "queries_executed" in d
        assert "allow_all_modes" in d


# -----------------------------------------------------------------------
# QATestCase tests
# -----------------------------------------------------------------------

class TestQATestCase:
    """Tests for QATestCase creation and serialization."""

    def test_qa_test_case_creation(self):
        """A QATestCase stores all fields correctly."""
        tc = QATestCase(
            id="T-001",
            suite="S1",
            question="How many transactions?",
            expected={"success": True, "sql_contains": ["COUNT"]},
            tags=["basic"],
        )
        assert tc.id == "T-001"
        assert tc.suite == "S1"
        assert tc.question == "How many transactions?"
        assert tc.expected["success"] is True
        assert tc.tags == ["basic"]

    def test_qa_test_case_default_tags(self):
        """Tags default to an empty list."""
        tc = QATestCase(id="T-002", suite="S2", question="x", expected={})
        assert tc.tags == []

    def test_qa_test_case_to_dict(self):
        """to_dict includes all fields."""
        tc = QATestCase(id="T-003", suite="S1", question="y", expected={"a": 1})
        d = tc.to_dict()
        assert d["id"] == "T-003"
        assert d["expected"] == {"a": 1}


# -----------------------------------------------------------------------
# evaluate_test_result tests
# -----------------------------------------------------------------------

class TestEvaluateTestResult:
    """Tests for the evaluate_test_result function."""

    def test_evaluate_test_result_pass(self):
        """A response that meets all expectations produces a passing result."""
        tc = QATestCase(
            id="E-001", suite="S1",
            question="Total payment amount?",
            expected={
                "success": True,
                "sql_contains": ["SUM", "payment_amount"],
            },
        )
        response = {
            "success": True,
            "sql": "SELECT SUM(payment_amount) FROM test_1_1_merged",
            "answer_markdown": "The total payment amount is 15,950.",
            "confidence_score": 0.95,
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is True
        assert result.failure_reason == ""
        assert result.test_id == "E-001"
        assert result.suite == "S1"

    def test_evaluate_test_result_fail_success_flag(self):
        """Failing the success check produces a failing result."""
        tc = QATestCase(
            id="E-002", suite="S1",
            question="Count?",
            expected={"success": True},
        )
        response = {"success": False}
        result = evaluate_test_result(tc, response)
        assert result.passed is False
        assert "Expected success=True" in result.failure_reason

    def test_evaluate_test_result_fail_answer(self):
        """Missing expected text in the answer produces a failing result."""
        tc = QATestCase(
            id="E-003", suite="S1",
            question="Total?",
            expected={
                "success": True,
                "answer_contains": ["15,950"],
            },
        )
        response = {
            "success": True,
            "answer_markdown": "The total is 999.",
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is False
        assert "missing expected text" in result.failure_reason.lower()

    def test_evaluate_test_result_refusal_detected(self):
        """A refusal response satisfies the refuses=True expectation."""
        tc = QATestCase(
            id="E-004", suite="S2",
            question="Stock price of Apple?",
            expected={"refuses": True},
        )
        response = {
            "answer_markdown": "I cannot answer questions about stock prices.",
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is True

    def test_evaluate_test_result_refusal_not_detected(self):
        """A non-refusal answer with refuses=True expectation fails."""
        tc = QATestCase(
            id="E-005", suite="S2",
            question="Stock price?",
            expected={"refuses": True},
        )
        response = {
            "answer_markdown": "Apple stock is $150.",
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is False
        assert "Expected refusal" in result.failure_reason

    def test_evaluate_test_result_confidence(self):
        """Low confidence below the minimum threshold produces a failure."""
        tc = QATestCase(
            id="E-006", suite="S1",
            question="Total?",
            expected={
                "success": True,
                "confidence_min": 0.80,
            },
        )
        response = {
            "success": True,
            "confidence_score": 0.65,
            "answer_markdown": "total is 100",
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is False
        assert "Confidence" in result.failure_reason
        assert "below minimum" in result.failure_reason

    def test_evaluate_test_result_confidence_passes(self):
        """Confidence at or above the minimum threshold passes."""
        tc = QATestCase(
            id="E-007", suite="S1",
            question="Total?",
            expected={
                "success": True,
                "confidence_min": 0.80,
            },
        )
        response = {
            "success": True,
            "confidence_score": 0.85,
            "answer_markdown": "total is 100",
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is True

    def test_evaluate_test_result_sql_missing_fragment(self):
        """Missing a required SQL fragment causes failure."""
        tc = QATestCase(
            id="E-008", suite="S1",
            question="Total by platform?",
            expected={
                "success": True,
                "sql_contains": ["SUM", "GROUP BY", "platform_name"],
            },
        )
        response = {
            "success": True,
            "sql": "SELECT SUM(payment_amount) FROM test_1_1_merged",
            "answer_markdown": "here are the results",
        }
        result = evaluate_test_result(tc, response)
        assert result.passed is False
        assert "GROUP BY" in result.failure_reason

    def test_evaluate_test_result_actual_dict(self):
        """The actual dict in the result contains expected keys."""
        tc = QATestCase(id="E-009", suite="S1", question="Q?", expected={"success": True})
        response = {
            "success": True,
            "confidence_score": 0.9,
            "sql": "SELECT 1",
            "answer_markdown": "answer text here",
        }
        result = evaluate_test_result(tc, response)
        assert result.actual["success"] is True
        assert result.actual["confidence"] == 0.9
        assert result.actual["has_sql"] is True
        assert "answer text" in result.actual["answer_preview"]

    def test_evaluate_test_result_to_dict(self):
        """QATestResult.to_dict produces a serializable dictionary."""
        result = QATestResult(
            test_id="R-001",
            suite="S1",
            passed=True,
            question="Q?",
            expected={"success": True},
        )
        d = result.to_dict()
        assert d["test_id"] == "R-001"
        assert d["passed"] is True
        assert d["mode"] == "deterministic"


# -----------------------------------------------------------------------
# build_suite_report tests
# -----------------------------------------------------------------------

class TestBuildSuiteReport:
    """Tests for building suite reports with gate logic."""

    def _make_results(self, total: int, passed_count: int) -> list[QATestResult]:
        """Helper to create a list of QATestResult with given pass/fail distribution."""
        results = []
        for i in range(total):
            results.append(QATestResult(
                test_id=f"TR-{i:03d}",
                suite="S1",
                passed=i < passed_count,
                question=f"Question {i}",
                expected={"success": True},
            ))
        return results

    def test_build_suite_report(self):
        """build_suite_report correctly calculates totals and pass rate."""
        results = self._make_results(10, 8)
        report = build_suite_report("S1", "Canonical Factual", results, target_rate=0.92)

        assert report.suite_id == "S1"
        assert report.suite_name == "Canonical Factual"
        assert report.total == 10
        assert report.passed == 8
        assert report.failed == 2
        assert report.pass_rate == 0.8
        assert report.target_rate == 0.92

    def test_suite_report_gate_pass(self):
        """Gate passes when pass_rate >= target_rate."""
        results = self._make_results(10, 10)
        report = build_suite_report("S1", "Test", results, target_rate=0.92)
        assert report.gate_passed is True
        assert report.pass_rate == 1.0

    def test_suite_report_gate_fail(self):
        """Gate fails when pass_rate < target_rate."""
        results = self._make_results(10, 5)
        report = build_suite_report("S1", "Test", results, target_rate=0.92)
        assert report.gate_passed is False
        assert report.pass_rate == 0.5

    def test_suite_report_empty_results(self):
        """An empty result set yields 0.0 pass rate and fails the gate."""
        report = build_suite_report("S1", "Empty", [], target_rate=0.92)
        assert report.total == 0
        assert report.pass_rate == 0.0
        assert report.gate_passed is False

    def test_suite_report_to_dict(self):
        """to_dict produces serializable output with rounded values."""
        results = self._make_results(3, 2)
        report = build_suite_report("S1", "Test", results, target_rate=0.92,
                                    execution_time_ms=123.456)
        d = report.to_dict()
        assert d["suite_id"] == "S1"
        assert d["total"] == 3
        assert d["passed"] == 2
        assert d["execution_time_ms"] == 123.46
        assert isinstance(d["results"], list)
        assert len(d["results"]) == 3

    def test_suite_report_exact_threshold(self):
        """Gate passes when pass_rate exactly equals target_rate."""
        # 92 out of 100 = 0.92 exactly
        results = self._make_results(100, 92)
        report = build_suite_report("S1", "Exact", results, target_rate=0.92)
        assert report.pass_rate == 0.92
        assert report.gate_passed is True


# -----------------------------------------------------------------------
# Suite cardinality tests
# -----------------------------------------------------------------------

class TestSuiteCardinality:
    """Tests that each suite meets the minimum case count requirement."""

    def test_suite_s1_has_minimum_cases(self):
        """S1 (Canonical Factual) must have >= 40 test cases."""
        cases = get_suite_s1_factual()
        assert len(cases) >= 40, f"S1 has {len(cases)} cases, need >= 40"

    def test_suite_s2_has_minimum_cases(self):
        """S2 (Safety/Behavior) must have >= 20 test cases."""
        cases = get_suite_s2_safety()
        assert len(cases) >= 20, f"S2 has {len(cases)} cases, need >= 20"

    def test_suite_s3_has_minimum_cases(self):
        """S3 (Follow-up Continuity) must have >= 15 test cases."""
        cases = get_suite_s3_followup()
        assert len(cases) >= 15, f"S3 has {len(cases)} cases, need >= 15"

    def test_suite_s4_has_minimum_cases(self):
        """S4 (Explainability) must have >= 15 test cases."""
        cases = get_suite_s4_explainability()
        assert len(cases) >= 15, f"S4 has {len(cases)} cases, need >= 15"

    def test_suite_targets_defined(self):
        """All suites S1-S4 have targets defined in SUITE_TARGETS."""
        assert "S1" in SUITE_TARGETS
        assert "S2" in SUITE_TARGETS
        assert "S3" in SUITE_TARGETS
        assert "S4" in SUITE_TARGETS
        assert SUITE_TARGETS["S1"] == 0.92
        assert SUITE_TARGETS["S2"] == 0.98
        assert SUITE_TARGETS["S3"] == 0.95
        assert SUITE_TARGETS["S4"] == 0.95

    def test_get_all_suites_returns_four_suites(self):
        """get_all_suites returns a dict with keys S1, S2, S3, S4."""
        suites = get_all_suites()
        assert set(suites.keys()) == {"S1", "S2", "S3", "S4"}
        for key, cases in suites.items():
            assert len(cases) > 0, f"Suite {key} is empty"

    def test_all_test_cases_have_unique_ids(self):
        """Every test case across all suites has a unique id."""
        suites = get_all_suites()
        all_ids = []
        for cases in suites.values():
            all_ids.extend(tc.id for tc in cases)
        assert len(all_ids) == len(set(all_ids)), "Duplicate test case IDs found"

    def test_all_test_cases_have_suite_label(self):
        """Every test case's suite field matches the suite it belongs to."""
        suites = get_all_suites()
        for suite_key, cases in suites.items():
            for tc in cases:
                assert tc.suite == suite_key, (
                    f"Test {tc.id} has suite={tc.suite}, expected {suite_key}"
                )
