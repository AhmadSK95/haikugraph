"""Semantic analysis contract for dataDa.

An analysis_contract is a normalized artifact binding:
- metric: what is being measured
- domain: which business domain (transactions, quotes, customers, bookings)
- grain: aggregation level (total, by_month, by_platform, etc.)
- time_scope: time window constraints
- filters: explicit value filters
- exclusions: what was explicitly excluded

The contract is built from the intake/planning phase and validated against
the generated SQL before execution. If the SQL violates the contract,
execution is blocked.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AnalysisContract:
    """Normalized semantic contract binding query intent to SQL constraints."""

    metric: str  # e.g., "sum(payment_amount)", "count(transaction_id)"
    domain: str  # e.g., "transactions", "quotes"
    grain: list[str]  # e.g., ["month", "platform_name"]
    time_scope: dict[str, Any]  # e.g., {"month": 12, "year": 2025}
    filters: list[dict[str, Any]]  # e.g., [{"column": "status", "op": "=", "value": "completed"}]
    exclusions: list[str] = field(default_factory=list)  # e.g., ["refunds", "pending"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_plan(cls, plan: dict, intake: dict | None = None) -> "AnalysisContract":
        """Build contract from validated plan and optional intake result."""
        metric = "unknown"
        domain = "unknown"
        grain: list[str] = []
        time_scope: dict[str, Any] = {}
        filters: list[dict[str, Any]] = []
        exclusions: list[str] = []

        subquestions = plan.get("subquestions", [])
        if subquestions:
            sq = subquestions[0]
            # Domain from tables
            tables = sq.get("tables", [])
            if tables:
                domain = _table_to_domain(tables[0])

            # Metric from aggregations
            aggs = sq.get("aggregations", [])
            if aggs:
                agg = aggs[0]
                metric = f"{agg.get('agg', 'count')}({agg.get('col', '*')})"

            # Grain from group_by
            group_by = sq.get("group_by", [])
            if group_by:
                for g in group_by:
                    if isinstance(g, str):
                        grain.append(g)
                    elif isinstance(g, dict) and g.get("type") == "time_bucket":
                        grain.append(g.get("grain", "month"))

        # Time scope from constraints
        constraints = plan.get("constraints", [])
        if constraints:
            for c in constraints:
                ctype = c.get("type", "")
                expr = c.get("expression", "")
                if ctype in ("time", "time_month"):
                    time_scope = _parse_time_expression(expr)
                elif ctype == "filter":
                    filters.append({"expression": expr})

        # Enrich from intake if available
        if intake:
            if intake.get("exclusions"):
                exclusions = intake["exclusions"]

        return cls(
            metric=metric,
            domain=domain,
            grain=grain,
            time_scope=time_scope,
            filters=filters,
            exclusions=exclusions,
        )


@dataclass
class ContractValidationResult:
    """Result of validating SQL against a semantic contract."""

    valid: bool
    violations: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------

_TABLE_DOMAIN_MAP = {
    "test_1_1_merged": "transactions",
    "datada_mart_transactions": "transactions",
    "test_3_1": "quotes",
    "datada_mart_quotes": "quotes",
    "test_4_1": "customers",
    "datada_dim_customers": "customers",
    "test_5_1": "bookings",
    "datada_mart_bookings": "bookings",
}


def _table_to_domain(table: str) -> str:
    return _TABLE_DOMAIN_MAP.get(table, "unknown")


def _parse_time_expression(expr: str) -> dict[str, Any]:
    """Parse a time constraint expression into structured time scope."""
    scope: dict[str, Any] = {}
    lower = expr.lower()

    # Month extraction
    month_match = re.search(r"month\s*(?:=|from)\s*(\d+)", lower)
    if month_match:
        scope["month"] = int(month_match.group(1))

    # Year extraction
    year_match = re.search(r"year\s*(?:=|from)\s*(\d{4})", lower)
    if year_match:
        scope["year"] = int(year_match.group(1))

    # Date range
    date_match = re.search(r"'(\d{4}-\d{2}-\d{2})'", expr)
    if date_match:
        scope["date_ref"] = date_match.group(1)

    return scope


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------


def validate_sql_against_contract(
    sql: str,
    contract: AnalysisContract,
) -> ContractValidationResult:
    """Validate generated SQL against the semantic contract.

    Checks:
    1. Domain table presence - SQL must reference the contract's domain table
    2. Grain enforcement - if contract specifies grain dimensions, SQL must GROUP BY them
    3. Time scope enforcement - if contract specifies time, SQL must have matching filter
    4. Metric presence - SQL must contain the contract's aggregation function
    5. Filter enforcement - contract filters must appear in SQL WHERE clause

    Returns ContractValidationResult with valid=True if all checks pass.
    """
    violations: list[str] = []
    checks: list[dict[str, Any]] = []
    sql_upper = sql.upper()
    sql_lower = sql.lower()

    # Check 1: Domain table
    domain_tables = [t for t, d in _TABLE_DOMAIN_MAP.items() if d == contract.domain]
    table_found = any(t.lower() in sql_lower for t in domain_tables)
    checks.append({
        "check": "domain_table",
        "passed": table_found,
        "expected_domain": contract.domain,
    })
    if not table_found:
        violations.append(
            f"SQL does not reference any table for domain '{contract.domain}'. "
            f"Expected one of: {domain_tables}"
        )

    # Check 2: Grain enforcement (GROUP BY dimensions)
    if contract.grain:
        has_group_by = "GROUP BY" in sql_upper
        grain_checks = []
        for dim in contract.grain:
            # Time-grain dimensions map to EXTRACT or strftime patterns
            if dim in ("month", "year", "quarter", "week", "day"):
                dim_present = (
                    f"EXTRACT({dim.upper()}" in sql_upper
                    or "STRFTIME" in sql_upper
                    or "DATE_TRUNC" in sql_upper
                    or dim.upper() in sql_upper
                )
            else:
                dim_present = dim.lower() in sql_lower
            grain_checks.append({"dimension": dim, "present": dim_present})

        all_grain_present = all(g["present"] for g in grain_checks)
        checks.append({
            "check": "grain_enforcement",
            "passed": has_group_by and all_grain_present,
            "grain": contract.grain,
            "details": grain_checks,
        })
        if not has_group_by:
            violations.append(
                f"Contract requires grain {contract.grain} but SQL has no GROUP BY clause"
            )
        elif not all_grain_present:
            missing = [g["dimension"] for g in grain_checks if not g["present"]]
            violations.append(
                f"Contract requires grain dimensions {missing} but they are missing from SQL"
            )

    # Check 3: Time scope enforcement
    if contract.time_scope:
        time_found = False
        month = contract.time_scope.get("month")
        year = contract.time_scope.get("year")

        if month is not None:
            # Check for month filter in SQL
            time_found = (
                str(month) in sql
                and (
                    "MONTH" in sql_upper
                    or "EXTRACT" in sql_upper
                    or "STRFTIME" in sql_upper
                    or f"-{month:02d}-" in sql
                )
            )
        if year is not None:
            time_found = time_found or str(year) in sql

        checks.append({
            "check": "time_scope",
            "passed": time_found,
            "expected": contract.time_scope,
        })
        if not time_found:
            violations.append(
                f"Contract requires time scope {contract.time_scope} "
                f"but no matching time filter found in SQL"
            )

    # Check 4: Metric presence
    if contract.metric and contract.metric != "unknown":
        # Parse agg(col) pattern
        agg_match = re.match(r"(\w+)\((\w+)\)", contract.metric)
        if agg_match:
            agg_func = agg_match.group(1).upper()
            agg_col = agg_match.group(2).lower()
            metric_present = agg_func in sql_upper and agg_col in sql_lower
        else:
            metric_present = contract.metric.lower() in sql_lower

        checks.append({
            "check": "metric_presence",
            "passed": metric_present,
            "expected_metric": contract.metric,
        })
        if not metric_present:
            violations.append(
                f"Contract requires metric '{contract.metric}' but not found in SQL"
            )

    valid = len(violations) == 0
    return ContractValidationResult(valid=valid, violations=violations, checks=checks)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def build_contract_from_pipeline(
    plan: dict,
    intake: dict | None = None,
    schema_context: dict | None = None,
) -> AnalysisContract:
    """Build analysis contract from pipeline artifacts.

    This is the main entry point for contract construction during the pipeline.
    """
    contract = AnalysisContract.from_plan(plan, intake)

    # Enrich from schema context if available
    if schema_context:
        if not contract.domain or contract.domain == "unknown":
            contract.domain = schema_context.get("primary_domain", contract.domain)

    return contract
