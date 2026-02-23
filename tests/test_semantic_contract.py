"""Tests for the semantic analysis contract system (Goal G1).

Covers all Gate 1 closure criteria:
- Contract construction from plan/intake artifacts
- Domain table validation
- Grain (GROUP BY) enforcement
- Time scope enforcement
- Metric presence validation
- Full-pass scenarios
- Multiple-violation detection
"""

from __future__ import annotations

import pytest

from haikugraph.contracts.analysis_contract import (
    AnalysisContract,
    ContractValidationResult,
    build_contract_from_pipeline,
    validate_sql_against_contract,
)


# ---------------------------------------------------------------------------
# Helpers: reusable plan fixtures
# ---------------------------------------------------------------------------

def _make_plan(
    *,
    tables: list[str] | None = None,
    aggregations: list[dict] | None = None,
    group_by: list | None = None,
    constraints: list[dict] | None = None,
) -> dict:
    """Build a minimal valid plan dict for testing."""
    sq: dict = {
        "id": "SQ1",
        "tables": tables or ["test_1_1_merged"],
    }
    if aggregations is not None:
        sq["aggregations"] = aggregations
    if group_by is not None:
        sq["group_by"] = group_by

    plan: dict = {
        "original_question": "test question",
        "subquestions": [sq],
    }
    if constraints is not None:
        plan["constraints"] = constraints
    return plan


# ===================================================================
# 1. Contract construction from plan (basic)
# ===================================================================

class TestContractFromPlanBasic:
    """Test basic contract construction from plan dictionaries."""

    def test_contract_from_plan_basic(self):
        """A plan with a single table and aggregation produces the correct contract."""
        plan = _make_plan(
            tables=["test_1_1_merged"],
            aggregations=[{"agg": "sum", "col": "payment_amount"}],
        )
        contract = AnalysisContract.from_plan(plan)

        assert contract.metric == "sum(payment_amount)"
        assert contract.domain == "transactions"
        assert contract.grain == []
        assert contract.time_scope == {}
        assert contract.filters == []
        assert contract.exclusions == []

    def test_contract_from_plan_unknown_table(self):
        """A plan referencing an unknown table maps domain to 'unknown'."""
        plan = _make_plan(tables=["some_random_table"])
        contract = AnalysisContract.from_plan(plan)
        assert contract.domain == "unknown"

    def test_contract_from_plan_no_aggregations(self):
        """A plan with no aggregations results in metric='unknown'."""
        plan = _make_plan(tables=["test_3_1"])
        contract = AnalysisContract.from_plan(plan)
        assert contract.metric == "unknown"
        assert contract.domain == "quotes"

    def test_contract_from_plan_with_intake_exclusions(self):
        """Intake exclusions are propagated into the contract."""
        plan = _make_plan(tables=["test_1_1_merged"])
        intake = {"exclusions": ["refunds", "pending"]}
        contract = AnalysisContract.from_plan(plan, intake=intake)
        assert contract.exclusions == ["refunds", "pending"]

    def test_contract_to_dict_roundtrip(self):
        """to_dict returns a plain dict matching constructor args."""
        plan = _make_plan(
            tables=["test_1_1_merged"],
            aggregations=[{"agg": "count", "col": "transaction_id"}],
        )
        contract = AnalysisContract.from_plan(plan)
        d = contract.to_dict()
        assert d["metric"] == "count(transaction_id)"
        assert d["domain"] == "transactions"
        assert isinstance(d["grain"], list)
        assert isinstance(d["time_scope"], dict)


# ===================================================================
# 2. Contract construction with time constraint
# ===================================================================

class TestContractFromPlanWithTimeConstraint:
    """Test time scope extraction from plan constraints."""

    def test_contract_from_plan_with_time_constraint(self):
        """A time constraint is parsed into time_scope dict."""
        plan = _make_plan(
            tables=["test_1_1_merged"],
            aggregations=[{"agg": "sum", "col": "payment_amount"}],
            constraints=[
                {"type": "time", "expression": "month = 12, year = 2025"},
            ],
        )
        contract = AnalysisContract.from_plan(plan)
        assert contract.time_scope.get("month") == 12
        assert contract.time_scope.get("year") == 2025

    def test_contract_from_plan_with_date_ref(self):
        """A time constraint with a date literal populates date_ref."""
        plan = _make_plan(
            constraints=[
                {"type": "time", "expression": "created_at >= '2025-11-01'"},
            ],
        )
        contract = AnalysisContract.from_plan(plan)
        assert contract.time_scope.get("date_ref") == "2025-11-01"

    def test_contract_from_plan_filter_constraint(self):
        """A filter constraint ends up in the filters list."""
        plan = _make_plan(
            constraints=[
                {"type": "filter", "expression": "payment_status = 'completed'"},
            ],
        )
        contract = AnalysisContract.from_plan(plan)
        assert len(contract.filters) == 1
        assert contract.filters[0]["expression"] == "payment_status = 'completed'"


# ===================================================================
# 3. Contract construction with grain (GROUP BY)
# ===================================================================

class TestContractFromPlanWithGrain:
    """Test grain extraction from group_by in the plan."""

    def test_contract_from_plan_with_grain(self):
        """String group_by entries are added to grain."""
        plan = _make_plan(
            tables=["test_1_1_merged"],
            aggregations=[{"agg": "sum", "col": "payment_amount"}],
            group_by=["platform_name", "state"],
        )
        contract = AnalysisContract.from_plan(plan)
        assert contract.grain == ["platform_name", "state"]

    def test_contract_from_plan_with_time_bucket_grain(self):
        """Dict group_by entries with type=time_bucket extract grain."""
        plan = _make_plan(
            tables=["test_1_1_merged"],
            aggregations=[{"agg": "sum", "col": "payment_amount"}],
            group_by=[
                {"type": "time_bucket", "grain": "month", "column": "created_at"},
                "platform_name",
            ],
        )
        contract = AnalysisContract.from_plan(plan)
        assert "month" in contract.grain
        assert "platform_name" in contract.grain


# ===================================================================
# 4. SQL validation: domain table check
# ===================================================================

class TestValidateSqlDomainCheck:
    """Validate that SQL references a table from the contract's domain."""

    def test_validate_sql_domain_check_pass(self):
        """SQL containing the correct domain table passes."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(payment_amount) FROM test_1_1_merged WHERE 1=1"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True
        domain_check = next(c for c in result.checks if c["check"] == "domain_table")
        assert domain_check["passed"] is True

    def test_validate_sql_domain_check_fail(self):
        """SQL missing the domain table triggers a violation."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(booked_amount) FROM test_5_1"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is False
        assert any("domain" in v.lower() for v in result.violations)

    def test_validate_sql_domain_check_alternate_table(self):
        """SQL using the mart alias for the domain still passes."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(payment_amount) FROM datada_mart_transactions"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True


# ===================================================================
# 5. SQL validation: grain enforcement
# ===================================================================

class TestValidateSqlGrainEnforcement:
    """Validate that SQL enforces the contract's grain dimensions."""

    def test_validate_sql_grain_enforcement_pass(self):
        """SQL with matching GROUP BY passes grain check."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=["platform_name"],
            time_scope={},
            filters=[],
        )
        sql = (
            "SELECT platform_name, SUM(payment_amount) "
            "FROM test_1_1_merged "
            "GROUP BY platform_name"
        )
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True
        grain_check = next(c for c in result.checks if c["check"] == "grain_enforcement")
        assert grain_check["passed"] is True

    def test_validate_sql_grain_enforcement_no_group_by(self):
        """SQL without GROUP BY fails when contract requires grain."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=["platform_name"],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(payment_amount) FROM test_1_1_merged"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is False
        assert any("GROUP BY" in v for v in result.violations)

    def test_validate_sql_grain_enforcement_missing_dimension(self):
        """SQL GROUP BY missing a required dimension fails."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=["platform_name", "state"],
            time_scope={},
            filters=[],
        )
        sql = (
            "SELECT platform_name, SUM(payment_amount) "
            "FROM test_1_1_merged "
            "GROUP BY platform_name"
        )
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is False
        assert any("state" in v for v in result.violations)

    def test_validate_sql_grain_enforcement_time_dimension(self):
        """SQL with EXTRACT(MONTH ...) satisfies a 'month' grain dimension."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=["month"],
            time_scope={},
            filters=[],
        )
        sql = (
            "SELECT EXTRACT(MONTH FROM created_at) AS month, SUM(payment_amount) "
            "FROM test_1_1_merged "
            "GROUP BY EXTRACT(MONTH FROM created_at)"
        )
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True


# ===================================================================
# 6. SQL validation: time scope enforcement
# ===================================================================

class TestValidateSqlTimeScope:
    """Validate that SQL enforces the contract's time scope."""

    def test_validate_sql_time_scope_pass_month(self):
        """SQL filtering by month satisfies month time scope."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={"month": 12, "year": 2025},
            filters=[],
        )
        sql = (
            "SELECT SUM(payment_amount) FROM test_1_1_merged "
            "WHERE EXTRACT(MONTH FROM created_at) = 12 "
            "AND EXTRACT(YEAR FROM created_at) = 2025"
        )
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True
        time_check = next(c for c in result.checks if c["check"] == "time_scope")
        assert time_check["passed"] is True

    def test_validate_sql_time_scope_pass_date_literal(self):
        """SQL with date literal containing the year passes."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={"year": 2025},
            filters=[],
        )
        sql = (
            "SELECT SUM(payment_amount) FROM test_1_1_merged "
            "WHERE created_at >= '2025-01-01'"
        )
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True

    def test_validate_sql_time_scope_fail(self):
        """SQL missing time filter for a contract with time scope fails."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={"month": 12, "year": 2025},
            filters=[],
        )
        sql = "SELECT SUM(payment_amount) FROM test_1_1_merged"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is False
        assert any("time scope" in v.lower() for v in result.violations)


# ===================================================================
# 7. SQL validation: metric presence
# ===================================================================

class TestValidateSqlMetricPresence:
    """Validate that SQL contains the contract's required metric."""

    def test_validate_sql_metric_presence_pass(self):
        """SQL containing SUM(payment_amount) passes metric check."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(payment_amount) FROM test_1_1_merged"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True
        metric_check = next(c for c in result.checks if c["check"] == "metric_presence")
        assert metric_check["passed"] is True

    def test_validate_sql_metric_presence_fail(self):
        """SQL missing the required metric function/column fails."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT COUNT(*) FROM test_1_1_merged"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is False
        assert any("metric" in v.lower() for v in result.violations)

    def test_validate_sql_metric_unknown_skipped(self):
        """When metric is 'unknown', the metric check is skipped."""
        contract = AnalysisContract(
            metric="unknown",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT COUNT(*) FROM test_1_1_merged"
        result = validate_sql_against_contract(sql, contract)
        # No metric check should be emitted
        metric_checks = [c for c in result.checks if c["check"] == "metric_presence"]
        assert len(metric_checks) == 0
        # Should still be valid (only domain check, which passes)
        assert result.valid is True


# ===================================================================
# 8. SQL validation: all checks pass together
# ===================================================================

class TestValidateSqlAllPass:
    """End-to-end: a well-formed SQL passes all contract checks."""

    def test_validate_sql_all_pass(self):
        """SQL satisfying domain, grain, time scope, and metric passes."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=["platform_name"],
            time_scope={"month": 12, "year": 2025},
            filters=[],
        )
        sql = (
            "SELECT platform_name, SUM(payment_amount) "
            "FROM test_1_1_merged "
            "WHERE EXTRACT(MONTH FROM created_at) = 12 "
            "AND EXTRACT(YEAR FROM created_at) = 2025 "
            "GROUP BY platform_name"
        )
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is True
        assert result.violations == []
        assert len(result.checks) == 4  # domain, grain, time_scope, metric

    def test_validate_sql_all_pass_to_dict(self):
        """ContractValidationResult.to_dict produces serializable output."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(payment_amount) FROM test_1_1_merged"
        result = validate_sql_against_contract(sql, contract)
        d = result.to_dict()
        assert d["valid"] is True
        assert isinstance(d["violations"], list)
        assert isinstance(d["checks"], list)


# ===================================================================
# 9. SQL validation: multiple violations
# ===================================================================

class TestValidateSqlMultipleViolations:
    """Detect multiple violations in a single validation pass."""

    def test_validate_sql_multiple_violations(self):
        """SQL violating domain, grain, time scope, and metric all at once."""
        contract = AnalysisContract(
            metric="sum(payment_amount)",
            domain="transactions",
            grain=["platform_name"],
            time_scope={"month": 12, "year": 2025},
            filters=[],
        )
        # Wrong table, no GROUP BY, no time filter, wrong metric
        sql = "SELECT AVG(booked_amount) FROM test_5_1"
        result = validate_sql_against_contract(sql, contract)

        assert result.valid is False
        # Expect at least 4 violations: domain, grain, time_scope, metric
        assert len(result.violations) >= 4

        violation_text = " ".join(result.violations).lower()
        assert "domain" in violation_text
        assert "group by" in violation_text
        assert "time scope" in violation_text
        assert "metric" in violation_text

    def test_validate_sql_two_violations_domain_and_metric(self):
        """SQL referencing wrong domain table and wrong metric."""
        contract = AnalysisContract(
            metric="count(transaction_id)",
            domain="transactions",
            grain=[],
            time_scope={},
            filters=[],
        )
        sql = "SELECT SUM(booked_amount) FROM test_5_1"
        result = validate_sql_against_contract(sql, contract)
        assert result.valid is False
        assert len(result.violations) == 2


# ===================================================================
# 10. build_contract_from_pipeline integration
# ===================================================================

class TestBuildContractFromPipeline:
    """Test the main pipeline entry point."""

    def test_build_contract_from_pipeline_basic(self):
        """Pipeline builder delegates to from_plan correctly."""
        plan = _make_plan(
            tables=["test_3_1"],
            aggregations=[{"agg": "avg", "col": "exchange_rate"}],
        )
        contract = build_contract_from_pipeline(plan)
        assert contract.metric == "avg(exchange_rate)"
        assert contract.domain == "quotes"

    def test_build_contract_from_pipeline_schema_context_enrichment(self):
        """Schema context enriches unknown domain."""
        plan = _make_plan(tables=["unknown_table"])
        schema_ctx = {"primary_domain": "customers"}
        contract = build_contract_from_pipeline(plan, schema_context=schema_ctx)
        assert contract.domain == "customers"

    def test_build_contract_from_pipeline_schema_context_no_override(self):
        """Schema context does NOT override a known domain."""
        plan = _make_plan(tables=["test_1_1_merged"])
        schema_ctx = {"primary_domain": "quotes"}
        contract = build_contract_from_pipeline(plan, schema_context=schema_ctx)
        # Domain should remain 'transactions' because it was already known
        assert contract.domain == "transactions"

    def test_build_contract_from_pipeline_with_intake(self):
        """Intake result flows through to exclusions."""
        plan = _make_plan(tables=["test_1_1_merged"])
        intake = {"exclusions": ["refunds"]}
        contract = build_contract_from_pipeline(plan, intake=intake)
        assert contract.exclusions == ["refunds"]
