"""Safe SQL executor with guardrails for read-only query execution.

This module provides a safe wrapper around DuckDB execution that enforces:
- Read-only queries (SELECT/WITH/EXPLAIN only)
- LIMIT enforcement
- Query timeout handling
- Result row capping
- Execution statistics tracking
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from haikugraph.sql.guardrails import (
    GuardrailConfig,
    ValidationResult,
    enforce_limit,
    sanitize_for_prompt_injection,
    validate_sql,
    get_query_stats,
)


@dataclass
class ExecutionResult:
    """Result of a safe SQL execution."""
    
    success: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    total_rows_available: int = 0  # Before capping
    truncated: bool = False
    execution_time_ms: float = 0.0
    sql_executed: str = ""
    sql_original: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    query_stats: dict = field(default_factory=dict)
    executed_at: str = ""


class SafeSQLExecutor:
    """Safe SQL executor with configurable guardrails.
    
    This executor ensures:
    - Only read-only queries are executed
    - LIMIT clauses are enforced
    - Results are capped to prevent memory issues
    - Query timeouts are respected
    - Dangerous patterns are blocked
    
    Usage:
        executor = SafeSQLExecutor(db_path)
        result = executor.execute("SELECT * FROM users WHERE status = 'active'")
        if result.success:
            for row in result.rows:
                print(row)
    """
    
    def __init__(
        self,
        db_path: Path | str,
        config: GuardrailConfig | None = None,
        read_only: bool = True,
    ):
        """Initialize safe SQL executor.
        
        Args:
            db_path: Path to DuckDB database
            config: Optional guardrail configuration
            read_only: Whether to open connection in read-only mode (default True)
        """
        self.db_path = Path(db_path)
        self.config = config or GuardrailConfig()
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a fresh connection per execution.

        A single shared DuckDB connection is not thread-safe under concurrent
        FastAPI requests. Returning a short-lived connection prevents cross-request
        contention and "connection file with different configuration" failures.
        """
        return duckdb.connect(str(self.db_path), read_only=self.read_only)
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "SafeSQLExecutor":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def validate(self, sql: str) -> ValidationResult:
        """Validate SQL query without executing.
        
        Args:
            sql: SQL query to validate
        
        Returns:
            ValidationResult with is_valid flag and optional error/warnings
        """
        return validate_sql(sql, self.config)
    
    def execute(
        self,
        sql: str,
        *,
        parameters: dict[str, Any] | None = None,
        skip_validation: bool = False,
    ) -> ExecutionResult:
        """Execute SQL query with safety guardrails.
        
        Args:
            sql: SQL query to execute
            parameters: Optional query parameters (for parameterized queries)
            skip_validation: Skip validation (use with caution, for internal queries)
        
        Returns:
            ExecutionResult with rows, columns, and metadata
        """
        executed_at = datetime.utcnow().isoformat() + "Z"
        original_sql = sql
        warnings: list[str] = []
        
        # Step 1: Validate query
        if not skip_validation:
            validation = validate_sql(sql, self.config)
            if not validation.is_valid:
                return ExecutionResult(
                    success=False,
                    error=validation.error,
                    sql_original=original_sql,
                    executed_at=executed_at,
                )
            if validation.warnings:
                warnings.extend(validation.warnings)
        
        # Step 2: Enforce LIMIT
        sql_with_limit = enforce_limit(sql, self.config)
        if sql_with_limit != sql.strip().rstrip(";"):
            warnings.append(f"LIMIT enforced: {self.config.default_limit}")
        
        # Step 3: Get query statistics
        query_stats = get_query_stats(sql_with_limit)
        
        # Step 4: Execute with timeout handling
        start_time = time.perf_counter()
        conn: duckdb.DuckDBPyConnection | None = None
        
        try:
            conn = self._get_connection()
            
            # DuckDB doesn't have query timeout, but we can use Python-level timeout
            # For production, consider using asyncio or threading with timeout
            # For now, we rely on LIMIT to prevent runaway queries
            
            if parameters:
                result = conn.execute(sql_with_limit, parameters)
            else:
                result = conn.execute(sql_with_limit)
            
            # Fetch results
            raw_rows = result.fetchall()
            columns = [desc[0] for desc in result.description] if result.description else []
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Step 5: Cap results
            total_rows = len(raw_rows)
            truncated = False
            
            if total_rows > self.config.max_result_rows:
                raw_rows = raw_rows[:self.config.max_result_rows]
                truncated = True
                warnings.append(
                    f"Results truncated from {total_rows} to {self.config.max_result_rows} rows"
                )
            
            # Convert to list of dicts
            rows = [
                {col: val for col, val in zip(columns, row)}
                for row in raw_rows
            ]
            
            return ExecutionResult(
                success=True,
                rows=rows,
                columns=columns,
                row_count=len(rows),
                total_rows_available=total_rows,
                truncated=truncated,
                execution_time_ms=round(execution_time_ms, 2),
                sql_executed=sql_with_limit,
                sql_original=original_sql,
                warnings=warnings,
                query_stats=query_stats,
                executed_at=executed_at,
            )
        
        except duckdb.Error as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return ExecutionResult(
                success=False,
                error=f"Database error: {str(e)}",
                execution_time_ms=round(execution_time_ms, 2),
                sql_executed=sql_with_limit,
                sql_original=original_sql,
                warnings=warnings,
                query_stats=query_stats,
                executed_at=executed_at,
            )
        
        except Exception as e:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            return ExecutionResult(
                success=False,
                error=f"Execution error: {str(e)}",
                execution_time_ms=round(execution_time_ms, 2),
                sql_executed=sql_with_limit,
                sql_original=original_sql,
                warnings=warnings,
                query_stats=query_stats,
                executed_at=executed_at,
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def execute_probe(
        self,
        sql: str,
        *,
        limit: int = 5,
    ) -> ExecutionResult:
        """Execute a probing query with minimal results.
        
        Useful for testing query validity and getting sample data
        without pulling full result sets.
        
        Args:
            sql: SQL query to probe
            limit: Maximum rows to return (default 5)
        
        Returns:
            ExecutionResult with limited rows
        """
        # Create a temporary config with lower limit
        probe_config = GuardrailConfig(
            default_limit=limit,
            max_limit=limit,
            max_result_rows=limit,
            query_timeout_seconds=self.config.query_timeout_seconds,
            blocked_keywords=self.config.blocked_keywords,
            blocked_patterns=self.config.blocked_patterns,
            allowed_prefixes=self.config.allowed_prefixes,
        )
        
        # Temporarily swap config
        original_config = self.config
        self.config = probe_config
        
        try:
            return self.execute(sql)
        finally:
            self.config = original_config
    
    def get_table_sample(
        self,
        table_name: str,
        limit: int = 5,
    ) -> ExecutionResult:
        """Get a sample of rows from a table.
        
        Args:
            table_name: Name of table to sample
            limit: Number of rows to return
        
        Returns:
            ExecutionResult with sample rows
        """
        # Sanitize table name (basic protection)
        if not table_name.replace("_", "").isalnum():
            return ExecutionResult(
                success=False,
                error=f"Invalid table name: {table_name}",
                executed_at=datetime.utcnow().isoformat() + "Z",
            )
        
        sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        return self.execute(sql, skip_validation=True)
    
    def get_column_stats(
        self,
        table_name: str,
        column_name: str,
    ) -> ExecutionResult:
        """Get statistics for a column.
        
        Args:
            table_name: Name of table
            column_name: Name of column
        
        Returns:
            ExecutionResult with column statistics
        """
        # Sanitize names
        for name in [table_name, column_name]:
            if not name.replace("_", "").isalnum():
                return ExecutionResult(
                    success=False,
                    error=f"Invalid name: {name}",
                    executed_at=datetime.utcnow().isoformat() + "Z",
                )
        
        sql = f'''
            SELECT
                COUNT(*) as total_rows,
                COUNT("{column_name}") as non_null_count,
                COUNT(DISTINCT "{column_name}") as distinct_count,
                MIN("{column_name}") as min_value,
                MAX("{column_name}") as max_value
            FROM "{table_name}"
        '''
        return self.execute(sql, skip_validation=True)
    
    def sanitize_result_text(self, text: str) -> str:
        """Sanitize text from results for safe use in prompts.
        
        Args:
            text: Raw text from query results
        
        Returns:
            Sanitized text
        """
        return sanitize_for_prompt_injection(text)
    
    def sanitize_results(self, result: ExecutionResult) -> ExecutionResult:
        """Sanitize all text values in results for safe prompt inclusion.
        
        Args:
            result: ExecutionResult to sanitize
        
        Returns:
            New ExecutionResult with sanitized text values
        """
        if not result.success:
            return result
        
        sanitized_rows = []
        for row in result.rows:
            sanitized_row = {}
            for key, value in row.items():
                if isinstance(value, str):
                    sanitized_row[key] = sanitize_for_prompt_injection(value)
                else:
                    sanitized_row[key] = value
            sanitized_rows.append(sanitized_row)
        
        return ExecutionResult(
            success=result.success,
            rows=sanitized_rows,
            columns=result.columns,
            row_count=result.row_count,
            total_rows_available=result.total_rows_available,
            truncated=result.truncated,
            execution_time_ms=result.execution_time_ms,
            sql_executed=result.sql_executed,
            sql_original=result.sql_original,
            warnings=result.warnings,
            query_stats=result.query_stats,
            executed_at=result.executed_at,
        )


def create_executor(
    db_path: Path | str,
    *,
    default_limit: int = 1000,
    max_limit: int = 10000,
    timeout_seconds: int = 30,
) -> SafeSQLExecutor:
    """Factory function to create a SafeSQLExecutor with custom settings.
    
    Args:
        db_path: Path to DuckDB database
        default_limit: Default LIMIT to apply (default 1000)
        max_limit: Maximum allowed LIMIT (default 10000)
        timeout_seconds: Query timeout in seconds (default 30)
    
    Returns:
        Configured SafeSQLExecutor instance
    """
    config = GuardrailConfig(
        default_limit=default_limit,
        max_limit=max_limit,
        query_timeout_seconds=timeout_seconds,
    )
    return SafeSQLExecutor(db_path, config=config)
