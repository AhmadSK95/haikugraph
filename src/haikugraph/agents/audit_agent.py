"""AuditAgent for validating query results.

This agent performs sanity checks on query results:
1. Join validity checks (no Cartesian products)
2. Time filter validation
3. Null rate checks
4. Duplicate detection
5. Outlier detection
6. Cardinality checks
"""

from typing import Any

from haikugraph.agents.base import BaseAgent
from haikugraph.agents.contracts import (
    AgentStatus,
    AuditCheck,
    AuditCheckStatus,
    AuditResult,
    QueryPlanResult,
    SchemaResult,
)


class AuditAgent(BaseAgent[AuditResult]):
    """Agent for auditing query results.
    
    This agent validates that query results are reasonable and
    identifies potential issues that may require refinement.
    """
    
    name = "audit_agent"
    
    # Thresholds for checks
    MAX_EXPECTED_ROWS = 100000
    HIGH_NULL_RATE_THRESHOLD = 0.5
    OUTLIER_MULTIPLIER = 3.0  # Values > mean + 3*std considered outliers
    
    @property
    def output_schema(self) -> type[AuditResult]:
        return AuditResult
    
    def run(
        self,
        query_result: QueryPlanResult,
        schema_result: SchemaResult,
    ) -> AuditResult:
        """Run audit checks on query results.
        
        Args:
            query_result: Result from QueryAgent
            schema_result: Result from SchemaAgent
        
        Returns:
            AuditResult with check results and recommendations
        """
        self._start_timer()
        
        checks: list[AuditCheck] = []
        
        # Skip if query failed
        if query_result.status == AgentStatus.FAILED or not query_result.final_result:
            elapsed = self._stop_timer()
            return AuditResult(
                status=AgentStatus.FAILED,
                checks=[AuditCheck(
                    check_name="query_status",
                    check_type="query_validity",
                    status=AuditCheckStatus.FAIL,
                    message="Query execution failed",
                    severity="error",
                )],
                passed=0,
                warned=0,
                failed=1,
                skipped=0,
                overall_pass=False,
                requires_refinement=True,
                refinement_suggestions=["Fix query execution errors"],
                confidence=0.0,
                reasoning="Query failed, cannot audit results",
                processing_time_ms=elapsed,
            )
        
        final_result = query_result.final_result
        
        # Check 1: Row count sanity
        checks.append(self._check_row_count(final_result.row_count))
        
        # Check 2: Empty results
        checks.append(self._check_empty_results(final_result.row_count))
        
        # Check 3: Null rate in results
        checks.append(self._check_null_rate(final_result.sample_rows, final_result.columns))
        
        # Check 4: Duplicate detection
        checks.append(self._check_duplicates(final_result.sample_rows, final_result.columns))
        
        # Check 5: Join validity (based on query plan)
        checks.append(self._check_join_validity(query_result))
        
        # Check 6: Time filter applied (if expected)
        checks.append(self._check_time_filter(query_result))
        
        # Check 7: Outlier detection
        checks.append(self._check_outliers(final_result.sample_rows, final_result.columns))
        
        # Compute summary
        passed = sum(1 for c in checks if c.status == AuditCheckStatus.PASS)
        warned = sum(1 for c in checks if c.status == AuditCheckStatus.WARN)
        failed = sum(1 for c in checks if c.status == AuditCheckStatus.FAIL)
        skipped = sum(1 for c in checks if c.status == AuditCheckStatus.SKIPPED)
        
        overall_pass = failed == 0
        requires_refinement = failed > 0 or warned > 1
        
        # Build refinement suggestions
        suggestions = []
        for check in checks:
            if check.status in [AuditCheckStatus.FAIL, AuditCheckStatus.WARN] and check.remediation:
                suggestions.append(check.remediation)
        
        elapsed = self._stop_timer()
        
        return AuditResult(
            status=AgentStatus.SUCCESS,
            checks=checks,
            passed=passed,
            warned=warned,
            failed=failed,
            skipped=skipped,
            overall_pass=overall_pass,
            requires_refinement=requires_refinement,
            refinement_suggestions=suggestions,
            confidence=0.9 if overall_pass else 0.5,
            reasoning=f"Audit complete: {passed} passed, {warned} warnings, {failed} failed",
            processing_time_ms=elapsed,
        )
    
    def _check_row_count(self, row_count: int) -> AuditCheck:
        """Check if row count is reasonable."""
        if row_count > self.MAX_EXPECTED_ROWS:
            return AuditCheck(
                check_name="row_count_sanity",
                check_type="cardinality",
                status=AuditCheckStatus.WARN,
                message=f"High row count ({row_count:,}), may indicate missing filters",
                details={"row_count": row_count, "threshold": self.MAX_EXPECTED_ROWS},
                severity="warning",
                remediation="Add more specific filters to reduce result set",
            )
        return AuditCheck(
            check_name="row_count_sanity",
            check_type="cardinality",
            status=AuditCheckStatus.PASS,
            message=f"Row count ({row_count:,}) within expected range",
            details={"row_count": row_count},
            severity="info",
        )
    
    def _check_empty_results(self, row_count: int) -> AuditCheck:
        """Check for empty results."""
        if row_count == 0:
            return AuditCheck(
                check_name="empty_results",
                check_type="cardinality",
                status=AuditCheckStatus.WARN,
                message="Query returned no results",
                details={"row_count": 0},
                severity="warning",
                remediation="Verify filters are not too restrictive",
            )
        return AuditCheck(
            check_name="empty_results",
            check_type="cardinality",
            status=AuditCheckStatus.PASS,
            message=f"Query returned {row_count:,} rows",
            details={"row_count": row_count},
            severity="info",
        )
    
    def _check_null_rate(
        self,
        sample_rows: list[dict[str, Any]],
        columns: list[str],
    ) -> AuditCheck:
        """Check null rate in result columns."""
        if not sample_rows or not columns:
            return AuditCheck(
                check_name="null_rate",
                check_type="null_rate",
                status=AuditCheckStatus.SKIPPED,
                message="No data to check",
                severity="info",
            )
        
        high_null_cols = []
        for col in columns:
            values = [row.get(col) for row in sample_rows]
            null_count = sum(1 for v in values if v is None)
            null_rate = null_count / len(values)
            
            if null_rate > self.HIGH_NULL_RATE_THRESHOLD:
                high_null_cols.append((col, null_rate))
        
        if high_null_cols:
            return AuditCheck(
                check_name="null_rate",
                check_type="null_rate",
                status=AuditCheckStatus.WARN,
                message=f"High null rate in columns: {', '.join(c[0] for c in high_null_cols)}",
                details={"columns": {c[0]: f"{c[1]:.1%}" for c in high_null_cols}},
                severity="warning",
                remediation="Check if null values are expected or indicate data issues",
            )
        
        return AuditCheck(
            check_name="null_rate",
            check_type="null_rate",
            status=AuditCheckStatus.PASS,
            message="Null rates within acceptable range",
            severity="info",
        )
    
    def _check_duplicates(
        self,
        sample_rows: list[dict[str, Any]],
        columns: list[str],
    ) -> AuditCheck:
        """Check for duplicate rows in sample."""
        if not sample_rows:
            return AuditCheck(
                check_name="duplicates",
                check_type="duplicates",
                status=AuditCheckStatus.SKIPPED,
                message="No data to check",
                severity="info",
            )
        
        # Convert rows to tuples for comparison
        seen = set()
        duplicates = 0
        for row in sample_rows:
            row_tuple = tuple(sorted(row.items()))
            if row_tuple in seen:
                duplicates += 1
            seen.add(row_tuple)
        
        if duplicates > 0:
            dup_rate = duplicates / len(sample_rows)
            if dup_rate > 0.1:  # More than 10% duplicates
                return AuditCheck(
                    check_name="duplicates",
                    check_type="duplicates",
                    status=AuditCheckStatus.WARN,
                    message=f"Found {duplicates} duplicate rows ({dup_rate:.1%} of sample)",
                    details={"duplicate_count": duplicates, "rate": dup_rate},
                    severity="warning",
                    remediation="Add DISTINCT or review join conditions",
                )
        
        return AuditCheck(
            check_name="duplicates",
            check_type="duplicates",
            status=AuditCheckStatus.PASS,
            message="No significant duplicates detected",
            severity="info",
        )
    
    def _check_join_validity(self, query_result: QueryPlanResult) -> AuditCheck:
        """Check join validity based on query plan."""
        tables_used = query_result.tables_used
        joins_used = query_result.joins_used
        
        if len(tables_used) > 1 and not joins_used:
            return AuditCheck(
                check_name="join_validity",
                check_type="join_validity",
                status=AuditCheckStatus.WARN,
                message=f"Multiple tables ({len(tables_used)}) used without explicit joins",
                details={"tables": tables_used},
                severity="warning",
                remediation="Add explicit JOIN conditions to avoid Cartesian product",
            )
        
        return AuditCheck(
            check_name="join_validity",
            check_type="join_validity",
            status=AuditCheckStatus.PASS,
            message="Join conditions appear valid",
            details={"tables": tables_used, "joins": joins_used},
            severity="info",
        )
    
    def _check_time_filter(self, query_result: QueryPlanResult) -> AuditCheck:
        """Check if time filter was applied when expected."""
        filters = query_result.filters_applied
        
        # Check if any filter looks like a time filter
        time_keywords = ["date", "time", "created", "updated", "year", "month", "day"]
        has_time_filter = any(
            any(kw in f.lower() for kw in time_keywords)
            for f in filters
        )
        
        # This is just informational - we can't know if time filter was needed
        if has_time_filter:
            return AuditCheck(
                check_name="time_filter",
                check_type="time_filter",
                status=AuditCheckStatus.PASS,
                message="Time filter detected in query",
                details={"filters": filters},
                severity="info",
            )
        
        return AuditCheck(
            check_name="time_filter",
            check_type="time_filter",
            status=AuditCheckStatus.PASS,
            message="No time filter detected (may or may not be needed)",
            details={"filters": filters},
            severity="info",
        )
    
    def _check_outliers(
        self,
        sample_rows: list[dict[str, Any]],
        columns: list[str],
    ) -> AuditCheck:
        """Check for outliers in numeric columns."""
        if not sample_rows or len(sample_rows) < 3:
            return AuditCheck(
                check_name="outliers",
                check_type="outliers",
                status=AuditCheckStatus.SKIPPED,
                message="Not enough data for outlier detection",
                severity="info",
            )
        
        outlier_cols = []
        for col in columns:
            values = [row.get(col) for row in sample_rows if row.get(col) is not None]
            
            # Only check numeric columns
            if not values or not all(isinstance(v, (int, float)) for v in values):
                continue
            
            # Simple outlier detection using IQR
            if len(values) < 4:
                continue
            
            sorted_vals = sorted(values)
            q1_idx = len(sorted_vals) // 4
            q3_idx = 3 * len(sorted_vals) // 4
            q1 = sorted_vals[q1_idx]
            q3 = sorted_vals[q3_idx]
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_count = sum(1 for v in values if v < lower_bound or v > upper_bound)
            if outlier_count > 0:
                outlier_cols.append((col, outlier_count))
        
        if outlier_cols:
            return AuditCheck(
                check_name="outliers",
                check_type="outliers",
                status=AuditCheckStatus.PASS,  # Outliers are not necessarily bad
                message=f"Potential outliers in: {', '.join(c[0] for c in outlier_cols)}",
                details={"columns": {c[0]: c[1] for c in outlier_cols}},
                severity="info",
            )
        
        return AuditCheck(
            check_name="outliers",
            check_type="outliers",
            status=AuditCheckStatus.PASS,
            message="No significant outliers detected",
            severity="info",
        )
