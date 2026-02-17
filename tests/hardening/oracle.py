"""
Oracle/invariant checker for HaikuGraph hardening.

Validates that SQL and results conform to expected patterns based on question intent.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from tests.hardening.cli_runner import CLITestResult
from tests.hardening.question_generator import QuestionSpec, IntentType, TimeWindow


class ViolationType(Enum):
    """Types of oracle violations"""
    # GROUP BY violations
    SCALAR_HAS_GROUP_BY = "scalar_has_group_by"  # Scalar query has GROUP BY
    COMPARISON_HAS_GROUP_BY = "comparison_has_group_by"  # Comparison returns series instead of 2 scalars
    MISSING_GROUP_BY = "missing_group_by"  # Breakdown query missing GROUP BY
    
    # DISTINCT violations
    UNIQUE_MISSING_DISTINCT = "unique_missing_distinct"  # "unique customers" without COUNT(DISTINCT)
    
    # Time filter violations
    WRONG_TIME_FILTER = "wrong_time_filter"  # Time filter doesn't match intent
    MISSING_TIME_FILTER = "missing_time_filter"  # Time window specified but no filter
    
    # Shape violations
    WRONG_ROW_COUNT = "wrong_row_count"  # Scalar should return 1 row
    WRONG_COLUMN_COUNT = "wrong_column_count"  # Unexpected number of columns
    
    # Execution violations
    SQL_ERROR = "sql_error"  # SQL failed to execute
    PLAN_ERROR = "plan_error"  # Planning failed
    TIMEOUT = "timeout"  # Query timed out
    
    # Comparison-specific violations
    COMPARISON_WRONG_SHAPE = "comparison_wrong_shape"  # Comparison doesn't return 2 values
    
    # Grain violations
    WRONG_TIME_GRAIN = "wrong_time_grain"  # date_trunc grain doesn't match breakdown


@dataclass
class OracleViolation:
    """A detected violation of expected behavior"""
    violation_type: ViolationType
    description: str
    expected: str
    actual: str
    severity: str = "error"  # "error", "warning"
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.violation_type.value}: {self.description}\\n  Expected: {self.expected}\\n  Actual: {self.actual}"


def check_oracle_invariants(
    spec: QuestionSpec,
    result: CLITestResult,
) -> List[OracleViolation]:
    """
    Check if CLI result violates expected invariants based on question spec.
    
    Args:
        spec: Question specification with expected behavior
        result: CLI test result with actual behavior
    
    Returns:
        List of violations (empty if all checks pass)
    """
    violations = []
    
    # ========================================================================
    # 1. EXECUTION CHECKS
    # ========================================================================
    if result.error:
        # Ambiguity errors are EXPECTED for ambiguous queries - don't treat as violation
        if "ambiguities" in result.error.lower() or "ambiguous" in result.error.lower():
            # This is correct behavior - system detected ambiguity
            return []  # No violations - pass!
        
        if "timed out" in result.error.lower():
            violations.append(OracleViolation(
                violation_type=ViolationType.TIMEOUT,
                description="Query timed out",
                expected="Query completes within timeout",
                actual=result.error,
            ))
        elif result.plan is None:
            violations.append(OracleViolation(
                violation_type=ViolationType.PLAN_ERROR,
                description="Planning failed",
                expected="Valid plan generated",
                actual=result.error,
            ))
        else:
            violations.append(OracleViolation(
                violation_type=ViolationType.SQL_ERROR,
                description="SQL execution failed",
                expected="SQL executes successfully",
                actual=result.error,
            ))
        
        # If execution failed, skip other checks
        return violations
    
    # ========================================================================
    # 2. GROUP BY CHECKS
    # ========================================================================
    if spec.expected_group_by is False:
        # Scalar queries should NOT have GROUP BY
        if result.has_group_by:
            violations.append(OracleViolation(
                violation_type=ViolationType.SCALAR_HAS_GROUP_BY,
                description=f"Scalar {spec.intent.value} query has GROUP BY",
                expected="No GROUP BY for scalar result",
                actual="GROUP BY found in SQL",
                severity="error",
            ))
    
    elif spec.expected_group_by is True:
        # Breakdown/trend queries SHOULD have GROUP BY
        if not result.has_group_by:
            violations.append(OracleViolation(
                violation_type=ViolationType.MISSING_GROUP_BY,
                description=f"{spec.intent.value} query missing GROUP BY",
                expected="GROUP BY for grouped result",
                actual="No GROUP BY in SQL",
                severity="error",
            ))
    
    # ========================================================================
    # 3. DISTINCT CHECKS
    # ========================================================================
    if spec.expected_distinct:
        # "unique X" should use COUNT(DISTINCT)
        if not result.has_distinct:
            violations.append(OracleViolation(
                violation_type=ViolationType.UNIQUE_MISSING_DISTINCT,
                description=f"Unique {spec.metric} query missing DISTINCT",
                expected="COUNT(DISTINCT ...)",
                actual="COUNT without DISTINCT",
                severity="error",
            ))
    
    # ========================================================================
    # 4. SHAPE CHECKS (Row/Column count)
    # ========================================================================
    if spec.expected_shape == "scalar":
        # Scalar queries should return exactly 1 row
        if result.row_count is not None and result.row_count != 1:
            violations.append(OracleViolation(
                violation_type=ViolationType.WRONG_ROW_COUNT,
                description="Scalar query returned multiple rows",
                expected="1 row",
                actual=f"{result.row_count} rows",
                severity="error",
            ))
        
        # Scalar queries should return 1 column (the aggregation)
        if result.column_count is not None and result.column_count > 1:
            # Allow 2 columns if one is a group key (will trigger GROUP BY check)
            if result.column_count > 2:
                violations.append(OracleViolation(
                    violation_type=ViolationType.WRONG_COLUMN_COUNT,
                    description="Scalar query returned too many columns",
                    expected="1-2 columns",
                    actual=f"{result.column_count} columns",
                    severity="warning",
                ))
    
    elif spec.expected_shape == "comparison":
        # Comparison queries should return 1 row with 2-4 columns (period1, period2, delta, pct_change)
        # OR 2 rows with 1 column each
        if result.row_count is not None:
            if result.row_count not in [1, 2]:
                violations.append(OracleViolation(
                    violation_type=ViolationType.COMPARISON_WRONG_SHAPE,
                    description="Comparison query returned wrong shape",
                    expected="1-2 rows (2 scalar values or delta row)",
                    actual=f"{result.row_count} rows",
                    severity="error",
                ))
    
    elif spec.expected_shape in ["grouped", "series"]:
        # Grouped/series queries should return multiple rows
        if result.row_count is not None and result.row_count <= 1:
            violations.append(OracleViolation(
                violation_type=ViolationType.WRONG_ROW_COUNT,
                description=f"{spec.expected_shape} query returned too few rows",
                expected=">1 rows",
                actual=f"{result.row_count} rows",
                severity="warning",
            ))
    
    # ========================================================================
    # 5. TIME FILTER CHECKS
    # ========================================================================
    if spec.time_window and spec.time_window != TimeWindow.ALL_TIME:
        # Check if time filter is present in SQL
        if result.sql and not has_time_filter_in_sql(result.sql):
            violations.append(OracleViolation(
                violation_type=ViolationType.MISSING_TIME_FILTER,
                description=f"Time window '{spec.time_window.value}' specified but no filter in SQL",
                expected=f"WHERE clause with {spec.time_window.value} filter",
                actual="No time filter found",
                severity="error",
            ))
        
        # Validate time filter correctness (basic check)
        if result.sql:
            filter_type = infer_time_filter_type(result.sql)
            expected_filter = get_expected_time_filter_type(spec.time_window)
            
            if filter_type and expected_filter and filter_type != expected_filter:
                violations.append(OracleViolation(
                    violation_type=ViolationType.WRONG_TIME_FILTER,
                    description=f"Time filter type mismatch for '{spec.time_window.value}'",
                    expected=expected_filter,
                    actual=filter_type,
                    severity="warning",
                ))
    
    # ========================================================================
    # 6. GRAIN CHECKS (for trends)
    # ========================================================================
    if spec.expected_shape == "series" and spec.breakdown:
        # Check if date_trunc grain matches breakdown
        if result.sql:
            grain = extract_date_trunc_grain(result.sql)
            expected_grain = get_expected_grain(spec.breakdown)
            
            if grain and expected_grain and grain != expected_grain:
                violations.append(OracleViolation(
                    violation_type=ViolationType.WRONG_TIME_GRAIN,
                    description=f"Time grain mismatch for {spec.breakdown.value}",
                    expected=f"date_trunc('{expected_grain}', ...)",
                    actual=f"date_trunc('{grain}', ...)",
                    severity="error",
                ))
    
    return violations


def has_time_filter_in_sql(sql: str) -> bool:
    """Check if SQL contains any time-related filter"""
    time_keywords = [
        "EXTRACT(MONTH",
        "EXTRACT(YEAR",
        "EXTRACT(DAY",
        "date_trunc",
        "CURRENT_DATE",
        "INTERVAL",
        ">= DATE",
        "<= DATE",
    ]
    return any(keyword in sql.upper() for keyword in time_keywords)


def infer_time_filter_type(sql: str) -> Optional[str]:
    """
    Infer what type of time filter is used in SQL.
    
    Returns: "calendar_month", "calendar_year", "rolling_days", "specific_date", etc
    """
    sql_upper = sql.upper()
    
    # Check for rolling days FIRST (last N days)
    # Pattern: col >= CURRENT_DATE - INTERVAL 'N days'
    # Must check before specific_date because ">= CURRENT_DATE" contains "= CURRENT_DATE"
    if "CURRENT_DATE - INTERVAL" in sql_upper and "DAY" in sql_upper:
        return "rolling_days"
    elif "INTERVAL" in sql_upper and "DAY" in sql_upper and "CURRENT_DATE" in sql_upper:
        return "rolling_days"
    
    # Check for DATE_TRUNC-based filters (this week, this month, this year, last month, etc.)
    if "DATE_TRUNC('WEEK'" in sql_upper and "CURRENT_DATE" in sql_upper:
        return "calendar_week"
    elif "DATE_TRUNC('MONTH'" in sql_upper and "CURRENT_DATE" in sql_upper:
        return "calendar_month"
    elif "DATE_TRUNC('YEAR'" in sql_upper and "CURRENT_DATE" in sql_upper:
        return "calendar_year"
    
    # Check for specific date filters (today, yesterday)
    # Pattern: CAST(col AS DATE) = CURRENT_DATE
    if "AS DATE) = CURRENT_DATE" in sql_upper:
        return "specific_date"
    
    # Check for EXTRACT-based filters (legacy)
    if "EXTRACT(MONTH" in sql_upper and "EXTRACT(YEAR" in sql_upper:
        return "calendar_month"
    elif "EXTRACT(YEAR" in sql_upper and "EXTRACT(MONTH" not in sql_upper:
        return "calendar_year"
    elif "EXTRACT(MONTH" in sql_upper and "EXTRACT(YEAR" not in sql_upper:
        return "month_only"  # Potentially wrong - should specify year too
    
    return "unknown"


def get_expected_time_filter_type(time_window: TimeWindow) -> Optional[str]:
    """Get expected time filter type for a given time window"""
    if time_window in [TimeWindow.THIS_MONTH, TimeWindow.LAST_MONTH]:
        return "calendar_month"
    # Bare month names like "December" match month_only (any year)
    elif time_window in [TimeWindow.DECEMBER, TimeWindow.JANUARY]:
        return "month_only"
    elif time_window in [TimeWindow.THIS_YEAR, TimeWindow.LAST_YEAR]:
        return "calendar_year"
    elif time_window in [TimeWindow.LAST_7_DAYS, TimeWindow.LAST_30_DAYS, TimeWindow.LAST_90_DAYS]:
        return "rolling_days"
    elif time_window in [TimeWindow.THIS_WEEK, TimeWindow.LAST_WEEK]:
        return "calendar_week"
    elif time_window in [TimeWindow.TODAY, TimeWindow.YESTERDAY]:
        return "specific_date"
    
    return None


def extract_date_trunc_grain(sql: str) -> Optional[str]:
    """Extract grain from date_trunc call in SQL"""
    import re
    match = re.search(r"date_trunc\\('([^']+)',", sql, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_expected_grain(breakdown: "BreakdownType") -> Optional[str]:
    """Get expected date_trunc grain for a breakdown type"""
    from tests.hardening.question_generator import BreakdownType
    
    if breakdown == BreakdownType.BY_DAY:
        return "day"
    elif breakdown == BreakdownType.BY_WEEK:
        return "week"
    elif breakdown == BreakdownType.BY_MONTH:
        return "month"
    
    return None


def classify_failure(violations: List[OracleViolation]) -> str:
    """
    Classify failure into buckets for reporting.
    
    Returns bucket name like "planner_error", "group_by_error", "time_filter_error", etc.
    """
    if not violations:
        return "success"
    
    # Priority order for classification
    violation_types = [v.violation_type for v in violations]
    
    if ViolationType.PLAN_ERROR in violation_types:
        return "planner_error"
    elif ViolationType.SQL_ERROR in violation_types:
        return "sql_error"
    elif ViolationType.TIMEOUT in violation_types:
        return "timeout"
    elif ViolationType.SCALAR_HAS_GROUP_BY in violation_types or ViolationType.COMPARISON_HAS_GROUP_BY in violation_types:
        return "group_by_over_aggregation"
    elif ViolationType.MISSING_GROUP_BY in violation_types:
        return "group_by_missing"
    elif ViolationType.UNIQUE_MISSING_DISTINCT in violation_types:
        return "distinct_missing"
    elif ViolationType.MISSING_TIME_FILTER in violation_types or ViolationType.WRONG_TIME_FILTER in violation_types:
        return "time_filter_error"
    elif ViolationType.WRONG_TIME_GRAIN in violation_types:
        return "time_grain_error"
    elif ViolationType.COMPARISON_WRONG_SHAPE in violation_types:
        return "comparison_shape_error"
    elif ViolationType.WRONG_ROW_COUNT in violation_types:
        return "shape_error"
    else:
        return "other_error"


if __name__ == "__main__":
    # Test the oracle checker
    from tests.hardening.question_generator import QuestionSpec, IntentType, TimeWindow, BreakdownType
    from tests.hardening.cli_runner import CLITestResult
    
    # Create a test spec
    spec = QuestionSpec(
        intent=IntentType.SCALAR_METRIC,
        metric="revenue",
        time_window=TimeWindow.THIS_MONTH,
        expected_shape="scalar",
        expected_group_by=False,
    )
    
    # Create a mock result with GROUP BY (violation)
    result = CLITestResult(
        question="What is the total revenue this month?",
        exit_code=0,
        stdout="",
        stderr="",
        sql="SELECT SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
        has_group_by=True,
        row_count=4,
    )
    
    # Check invariants
    violations = check_oracle_invariants(spec, result)
    
    print(f"Found {len(violations)} violations:\\n")
    for v in violations:
        print(v)
        print()
    
    print(f"Failure bucket: {classify_failure(violations)}")
