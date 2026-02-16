"""QueryAgent for SQL plan generation and execution.

This agent:
1. Generates SQL query plan based on intake and schema
2. Executes probing queries to validate approach
3. Executes final query with SafeSQLExecutor
4. Learns from errors and retries with fixes
5. Uses schema cards and memory for better queries
"""

import json
import re
from pathlib import Path
from typing import Any

from haikugraph.agents.base import LLMAgent, AgentError
from haikugraph.agents.contracts import (
    AgentStatus,
    IntakeResult,
    QueryExecution,
    QueryPlanResult,
    QueryStep,
    SchemaResult,
)
from haikugraph.sql.safe_executor import SafeSQLExecutor

# Try to import memory and cards (optional)
try:
    from haikugraph.memory import MemoryStore, QueryMemory
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

try:
    from haikugraph.cards.store import load_index, load_card
    from haikugraph.cards.schemas import TableCard, ColumnCard
    HAS_CARDS = True
except ImportError:
    HAS_CARDS = False


QUERY_SYSTEM_PROMPT = """You are an expert SQL query generator for DuckDB. Generate safe, efficient SQL queries based on the user's goal and available schema.

CRITICAL RULES:
1. Generate SELECT queries only (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Always include LIMIT clause (max 1000 rows) - NEVER use LIMIT 0
3. Use double quotes for column names: "column_name"
4. Include appropriate aggregations (SUM, COUNT, AVG) when requested
5. Add GROUP BY when using aggregations with other columns
6. Add ORDER BY for deterministic results
7. For counting entities (transactions, customers), use COUNT(DISTINCT id_column) to avoid duplicates

DATE/TIME HANDLING (VERY IMPORTANT):
- For filtering by month: WHERE EXTRACT(MONTH FROM column::TIMESTAMP) = 12
- For filtering by year: WHERE EXTRACT(YEAR FROM column::TIMESTAMP) = 2025
- For December 2025: WHERE EXTRACT(YEAR FROM column::TIMESTAMP) = 2025 AND EXTRACT(MONTH FROM column::TIMESTAMP) = 12
- If a date/time column is stored as VARCHAR, cast it: column::TIMESTAMP or CAST(column AS TIMESTAMP)
- DuckDB date functions require DATE, TIMESTAMP, or INTERVAL types - NOT VARCHAR.
- NEVER use MySQL functions like STR_TO_DATE, DATE_FORMAT. Use DuckDB's strptime() and strftime() instead.

BUSINESS RULES:
- MT103 transactions are valid when mt103_created_at IS NOT NULL
- Always use the PRIMARY TIMESTAMP column specified in schema for date filters
- Always use the PRIMARY AMOUNT column specified in schema for SUM/AVG

Output ONLY valid JSON. No prose, no markdown."""


QUERY_USER_PROMPT_TEMPLATE = """Generate a SQL query for this analysis request.

User's Goal: {goal}
Intent Type: {intent_type}

{semantic_context}

Available Schema:
{schema_summary}

Relevant Tables: {relevant_tables}
Suggested Metrics: {suggested_metrics}
Suggested Dimensions: {suggested_dimensions}
Time Column (USE THIS FOR DATE FILTERS): {time_column}

{time_filter_instructions}

IMPORTANT REMINDERS:
1. NEVER use LIMIT 0 - use LIMIT 1000
2. For transaction counts: SELECT COUNT(DISTINCT transaction_id) FROM table
3. For December 2025: WHERE EXTRACT(YEAR FROM time_column::TIMESTAMP) = 2025 AND EXTRACT(MONTH FROM time_column::TIMESTAMP) = 12
4. Use the time column specified above, not random date columns

Output JSON schema:
{{
  "plan_steps": [
    {{
      "step_id": "step_1",
      "description": "What this step does",
      "sql": "SELECT ...",
      "depends_on": [],
      "is_probe": false
    }}
  ],
  "final_sql": "SELECT ... LIMIT 1000",
  "tables_used": ["table1", "table2"],
  "joins_used": ["table1.col = table2.col"],
  "filters_applied": ["date filter", "value filter"],
  "reasoning": "Brief explanation of query strategy"
}}

Generate the SQL query plan now. Output ONLY the JSON object."""


class QueryAgent(LLMAgent[QueryPlanResult]):
    """Agent for SQL plan generation and execution.
    
    This agent generates SQL based on the intake and schema results,
    then executes the query using SafeSQLExecutor.
    """
    
    name = "query_agent"
    llm_role = "planner"
    system_prompt = QUERY_SYSTEM_PROMPT
    user_prompt_template = QUERY_USER_PROMPT_TEMPLATE
    
    def __init__(
        self,
        db_path: Path | str,
        *,
        max_retries: int = 2,
    ):
        """Initialize query agent.
        
        Args:
            db_path: Path to DuckDB database
            max_retries: Max LLM retries
        """
        super().__init__(max_retries=max_retries)
        self.db_path = Path(db_path)
        self.executor = SafeSQLExecutor(db_path)
    
    @property
    def output_schema(self) -> type[QueryPlanResult]:
        return QueryPlanResult
    
    def run(
        self,
        intake_result: IntakeResult,
        schema_result: SchemaResult,
        *,
        execute: bool = True,
    ) -> QueryPlanResult:
        """Generate and optionally execute SQL query.
        
        Args:
            intake_result: Result from IntakeAgent
            schema_result: Result from SchemaAgent
            execute: Whether to execute the query
        
        Returns:
            QueryPlanResult with plan and execution results
        """
        self._start_timer()
        
        try:
            # Build prompt
            user_prompt = self._build_prompt(intake_result, schema_result)
            
            # Call LLM to generate plan
            data = self.call_llm_with_retry(user_prompt)
            
            # Extract plan steps
            plan_steps = []
            for step_data in data.get("plan_steps", []):
                plan_steps.append(QueryStep(
                    step_id=step_data.get("step_id", "step_1"),
                    description=step_data.get("description", ""),
                    sql=step_data.get("sql", ""),
                    depends_on=step_data.get("depends_on", []),
                    is_probe=step_data.get("is_probe", False),
                ))
            
            final_sql = data.get("final_sql", "")
            if not final_sql and plan_steps:
                final_sql = plan_steps[-1].sql
            
            # Fix LIMIT 0 issue - never allow LIMIT 0
            if 'LIMIT 0' in final_sql.upper():
                final_sql = final_sql.replace('LIMIT 0', 'LIMIT 1000').replace('limit 0', 'LIMIT 1000')
            
            # Pre-emptively fix VARCHAR date column issues
            final_sql = self._fix_varchar_date_operations(final_sql, schema_result)
            
            # Pre-emptively fix MySQL functions (LLM sometimes generates MySQL syntax)
            final_sql = self._fix_mysql_date_functions(final_sql)
            
            # Execute if requested
            executions = []
            final_result = None
            result_summary = {}
            
            if execute and final_sql:
                exec_result = self.executor.execute(final_sql)
                
                # If we got a type error, try to fix and retry
                if not exec_result.success and exec_result.error:
                    fixed_sql = self._try_fix_sql_error(final_sql, exec_result.error, schema_result)
                    if fixed_sql and fixed_sql != final_sql:
                        # Retry with fixed SQL
                        final_sql = fixed_sql
                        exec_result = self.executor.execute(final_sql)
                
                final_result = QueryExecution(
                    step_id="final",
                    success=exec_result.success,
                    sql_executed=exec_result.sql_executed,
                    row_count=exec_result.row_count,
                    columns=exec_result.columns,
                    sample_rows=exec_result.rows[:10],  # Limit to 10 samples
                    execution_time_ms=exec_result.execution_time_ms,
                    error=exec_result.error,
                    warnings=exec_result.warnings,
                )
                executions.append(final_result)
                
                # Build result summary
                if exec_result.success:
                    result_summary = self._build_result_summary(exec_result.rows, exec_result.columns)
            
            elapsed = self._stop_timer()
            
            return QueryPlanResult(
                status=AgentStatus.SUCCESS if (not execute or (final_result and final_result.success)) else AgentStatus.FAILED,
                plan_steps=plan_steps,
                final_sql=final_sql,
                executions=executions,
                final_result=final_result,
                result_summary=result_summary,
                tables_used=data.get("tables_used", []),
                joins_used=data.get("joins_used", []),
                filters_applied=data.get("filters_applied", []),
                confidence=0.8 if final_result and final_result.success else 0.4,
                reasoning=data.get("reasoning", "Generated SQL query"),
                processing_time_ms=elapsed,
            )
        
        except AgentError:
            raise
        except Exception as e:
            elapsed = self._stop_timer()
            return QueryPlanResult(
                status=AgentStatus.FAILED,
                plan_steps=[],
                final_sql="",
                executions=[],
                final_result=None,
                result_summary={},
                tables_used=[],
                joins_used=[],
                filters_applied=[],
                confidence=0.0,
                reasoning=f"Query generation failed: {str(e)}",
                processing_time_ms=elapsed,
            )
    
    def _try_fix_sql_error(self, sql: str, error: str, schema: SchemaResult) -> str | None:
        """Try to fix SQL based on error message."""
        # Check for STR_TO_DATE (MySQL function) - convert to DuckDB strptime
        if "str_to_date" in error.lower():
            fixed_sql = self._fix_mysql_date_functions(sql)
            if fixed_sql != sql:
                return fixed_sql
        
        # Check for DATE_TRUNC on VARCHAR error
        if "date_trunc" in error.lower() and "VARCHAR" in error:
            # Extract column name from error or SQL
            col_match = re.search(r"DATE_TRUNC\s*\(\s*'[^']+'\s*,\s*(\w+)", sql, re.IGNORECASE)
            if col_match:
                col_name = col_match.group(1)
                # Add CAST if not already present
                if f"CAST({col_name}" not in sql.upper():
                    return self._fix_varchar_date_operations(sql, schema)
        
        # Check for EXTRACT on VARCHAR error  
        if "extract" in error.lower() and "VARCHAR" in error:
            col_match = re.search(r"EXTRACT\s*\(\s*\w+\s+FROM\s+(\w+)", sql, re.IGNORECASE)
            if col_match:
                col_name = col_match.group(1)
                if f"CAST({col_name}" not in sql.upper():
                    return self._fix_varchar_date_operations(sql, schema)
        
        return None
    
    def _fix_mysql_date_functions(self, sql: str) -> str:
        """Convert MySQL date functions to DuckDB equivalents."""
        fixed_sql = sql
        
        # Convert STR_TO_DATE(col, format) to strptime(col, format)
        # MySQL format: %Y-%m-%d -> DuckDB format: %Y-%m-%d (same)
        fixed_sql = re.sub(
            r"STR_TO_DATE\s*\(",
            "strptime(",
            fixed_sql,
            flags=re.IGNORECASE
        )
        
        # Convert DATE_FORMAT(col, format) to strftime(col, format)
        fixed_sql = re.sub(
            r"DATE_FORMAT\s*\(",
            "strftime(",
            fixed_sql,
            flags=re.IGNORECASE
        )
        
        return fixed_sql
    
    def _build_prompt(
        self,
        intake: IntakeResult,
        schema: SchemaResult,
    ) -> str:
        """Build the user prompt for query generation."""
        # Get semantic context if available
        semantic_context = ""
        if hasattr(schema, '_schema_agent') and hasattr(schema._schema_agent, 'get_semantic_context'):
            ctx = schema._schema_agent.get_semantic_context()
            if ctx:
                semantic_context = f"\n=== DATABASE KNOWLEDGE ===\n{ctx}\n"
        
        # Build detailed schema summary with types
        schema_lines = []
        varchar_date_warnings = []
        
        for table in schema.tables[:5]:  # Limit to 5 tables for prompt size
            col_details = []
            for c in table.columns[:15]:
                col_type = c.data_type.upper()
                col_name = c.name
                col_details.append(f"{col_name} ({col_type})")
                
                # Detect VARCHAR columns that look like dates
                is_varchar = col_type == "VARCHAR"
                looks_like_date = any(p in col_name.lower() for p in [
                    "_at", "_date", "created", "updated", "expires", "time", "timestamp"
                ])
                if is_varchar and looks_like_date:
                    varchar_date_warnings.append(
                        f"  ⚠️ {table.name}.{col_name} is VARCHAR but looks like a date - USE CAST({col_name} AS DATE) for date functions!"
                    )
            
            schema_lines.append(f"  {table.name}: {', '.join(col_details)}")
        
        schema_summary = "\n".join(schema_lines) if schema_lines else "No schema available"
        
        # Add VARCHAR date warnings
        if varchar_date_warnings:
            schema_summary += "\n\n⚠️ VARCHAR DATE COLUMNS (MUST CAST FOR DATE FUNCTIONS):\n"
            schema_summary += "\n".join(varchar_date_warnings)
        
        # Build time filter instructions
        time_filter_instructions = ""
        time_col = schema.suggested_time_column
        
        if intake.time_window and intake.time_window.has_time_filter:
            tw = intake.time_window
            
            # Check if time column is VARCHAR
            time_col_type = self._get_column_type(schema, time_col) if time_col else None
            cast_warning = ""
            if time_col_type and time_col_type.upper() == "VARCHAR":
                cast_warning = f"\n⚠️ IMPORTANT: {time_col} is VARCHAR! You MUST use CAST({time_col} AS DATE) for all date operations!"
            
            if tw.relative_period:
                time_filter_instructions = f"""
Time Filter Required:
- Period: {tw.relative_period}
- Time column: {time_col or 'auto-detect'}{cast_warning}
- For last month: WHERE DATE_TRUNC('month', CAST({time_col or 'date_column'} AS DATE)) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
- For current month: WHERE EXTRACT(MONTH FROM CAST({time_col or 'date_column'} AS DATE)) = EXTRACT(MONTH FROM CURRENT_DATE)
"""
            elif tw.start_date or tw.end_date:
                time_filter_instructions = f"""
Time Filter Required:
- Start: {tw.start_date or 'not specified'}
- End: {tw.end_date or 'not specified'}
- Time column: {time_col or 'auto-detect'}{cast_warning}
- Use: WHERE CAST({time_col or 'date_column'} AS DATE) BETWEEN '{tw.start_date}' AND '{tw.end_date}'
"""
        
        return self.user_prompt_template.format(
            goal=intake.clarified_goal,
            intent_type=intake.intent_type,
            semantic_context=semantic_context,
            schema_summary=schema_summary,
            relevant_tables=", ".join(schema.relevant_tables) or "all tables",
            suggested_metrics=", ".join(schema.suggested_metrics) or "none",
            suggested_dimensions=", ".join(schema.suggested_dimensions) or "none",
            time_column=time_col or "auto-detect",
            time_filter_instructions=time_filter_instructions,
        )
    
    def _get_column_type(self, schema: SchemaResult, column_name: str | None) -> str | None:
        """Get the type of a column from schema."""
        if not column_name:
            return None
        # Strip table prefix if present (e.g., "table.column" -> "column")
        col_simple = column_name.split(".")[-1] if "." in column_name else column_name
        for table in schema.tables:
            for col in table.columns:
                if col.name == col_simple or col.name == column_name:
                    return col.data_type
        return None
    
    def _fix_varchar_date_operations(self, sql: str, schema: SchemaResult) -> str:
        """Fix DATE_TRUNC/EXTRACT on VARCHAR columns by adding CAST."""
        # Find all VARCHAR columns that look like dates
        varchar_date_cols = []
        for table in schema.tables:
            for col in table.columns:
                if col.data_type.upper() == "VARCHAR":
                    if any(p in col.name.lower() for p in ["_at", "_date", "created", "updated", "time"]):
                        # Add both simple name and qualified name
                        varchar_date_cols.append(col.name)
                        varchar_date_cols.append(f"{table.name}.{col.name}")
        
        if not varchar_date_cols:
            return sql
        
        fixed_sql = sql
        for col_name in varchar_date_cols:
            # Escape special regex characters in column name (e.g., dots)
            col_escaped = re.escape(col_name)
            
            # Fix DATE_TRUNC('part', col) -> DATE_TRUNC('part', CAST(col AS DATE))
            # Make sure we don't match already-cast columns
            pattern = rf"(DATE_TRUNC\s*\(\s*'[^']+'\s*,\s*)(?!CAST\()({col_escaped})(\s*\))"
            if re.search(pattern, fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(pattern, rf"\1CAST(\2 AS DATE)\3", fixed_sql, flags=re.IGNORECASE)
            
            # Fix EXTRACT(part FROM col) -> EXTRACT(part FROM CAST(col AS DATE))
            pattern = rf"(EXTRACT\s*\(\s*\w+\s+FROM\s+)(?!CAST\()({col_escaped})(\s*\))"
            if re.search(pattern, fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(pattern, rf"\1CAST(\2 AS DATE)\3", fixed_sql, flags=re.IGNORECASE)
        
        return fixed_sql
    
    def _build_result_summary(
        self,
        rows: list[dict[str, Any]],
        columns: list[str],
    ) -> dict[str, Any]:
        """Build a summary of query results."""
        summary: dict[str, Any] = {
            "row_count": len(rows),
            "columns": columns,
        }
        
        if not rows:
            summary["data"] = []
            return summary
        
        # For single-row results (aggregations), include the values
        if len(rows) == 1:
            summary["single_result"] = rows[0]
        else:
            # For multi-row results, include first few
            summary["sample_data"] = rows[:5]
            
            # Try to compute summary stats for numeric columns
            for col in columns:
                values = [r.get(col) for r in rows if r.get(col) is not None]
                if values and all(isinstance(v, (int, float)) for v in values):
                    summary[f"{col}_min"] = min(values)
                    summary[f"{col}_max"] = max(values)
                    summary[f"{col}_sum"] = sum(values)
        
        return summary
    
    def run_simple(
        self,
        intake_result: IntakeResult,
        schema_result: SchemaResult,
        *,
        execute: bool = True,
    ) -> QueryPlanResult:
        """Generate and execute SQL using rule-based approach (no LLM).
        
        Use this when LLM is unavailable or for simple queries.
        
        Args:
            intake_result: Result from IntakeAgent
            schema_result: Result from SchemaAgent
            execute: Whether to execute the query
        
        Returns:
            QueryPlanResult with plan and execution results
        """
        self._start_timer()
        
        try:
            # Generate SQL using rule-based patterns
            final_sql = self._generate_sql_simple(intake_result, schema_result)
            
            # Fix VARCHAR date operations
            final_sql = self._fix_varchar_date_operations(final_sql, schema_result)
            
            # Execute if requested
            executions = []
            final_result = None
            result_summary = {}
            
            if execute and final_sql:
                exec_result = self.executor.execute(final_sql)
                
                # If we got a type error, try to fix and retry
                if not exec_result.success and exec_result.error:
                    fixed_sql = self._try_fix_sql_error(final_sql, exec_result.error, schema_result)
                    if fixed_sql and fixed_sql != final_sql:
                        final_sql = fixed_sql
                        exec_result = self.executor.execute(final_sql)
                
                final_result = QueryExecution(
                    step_id="final",
                    success=exec_result.success,
                    sql_executed=exec_result.sql_executed,
                    row_count=exec_result.row_count,
                    columns=exec_result.columns,
                    sample_rows=exec_result.rows[:10],
                    execution_time_ms=exec_result.execution_time_ms,
                    error=exec_result.error,
                    warnings=exec_result.warnings,
                )
                executions.append(final_result)
                
                if exec_result.success:
                    result_summary = self._build_result_summary(exec_result.rows, exec_result.columns)
            
            elapsed = self._stop_timer()
            
            return QueryPlanResult(
                status=AgentStatus.SUCCESS if (not execute or (final_result and final_result.success)) else AgentStatus.FAILED,
                plan_steps=[QueryStep(
                    step_id="step_1",
                    description="Rule-based query generation",
                    sql=final_sql,
                    depends_on=[],
                    is_probe=False,
                )],
                final_sql=final_sql,
                executions=executions,
                final_result=final_result,
                result_summary=result_summary,
                tables_used=schema_result.relevant_tables[:1] if schema_result.relevant_tables else [],
                joins_used=[],
                filters_applied=[],
                confidence=0.6 if final_result and final_result.success else 0.3,
                reasoning="Simple rule-based SQL generation",
                processing_time_ms=elapsed,
            )
        
        except Exception as e:
            elapsed = self._stop_timer()
            return QueryPlanResult(
                status=AgentStatus.FAILED,
                plan_steps=[],
                final_sql="",
                executions=[],
                final_result=None,
                result_summary={},
                tables_used=[],
                joins_used=[],
                filters_applied=[],
                confidence=0.0,
                reasoning=f"Query generation failed: {str(e)}",
                processing_time_ms=elapsed,
            )
    
    def _generate_sql_simple(
        self,
        intake: IntakeResult,
        schema: SchemaResult,
    ) -> str:
        """Generate SQL using rule-based patterns."""
        goal_lower = intake.clarified_goal.lower()
        
        # Get primary table
        table = schema.relevant_tables[0] if schema.relevant_tables else "unknown_table"
        
        # Get time column for date filtering
        time_col = schema.suggested_time_column
        time_col_type = self._get_column_type(schema, time_col) if time_col else None
        needs_cast = time_col_type and time_col_type.upper() == "VARCHAR"
        time_expr = f"CAST({time_col} AS DATE)" if needs_cast and time_col else time_col
        
        # Detect query pattern and generate SQL
        sql = ""
        
        # Pattern: count/total queries
        if any(kw in goal_lower for kw in ["count", "how many", "total", "number of"]):
            # Check for time filter
            where_clause = ""
            if "december" in goal_lower and time_col:
                where_clause = f" WHERE EXTRACT(MONTH FROM {time_expr}) = 12"
            elif "january" in goal_lower and time_col:
                where_clause = f" WHERE EXTRACT(MONTH FROM {time_expr}) = 1"
            elif "this month" in goal_lower and time_col:
                where_clause = f" WHERE DATE_TRUNC('month', {time_expr}) = DATE_TRUNC('month', CURRENT_DATE)"
            elif "last month" in goal_lower and time_col:
                where_clause = f" WHERE DATE_TRUNC('month', {time_expr}) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
            
            sql = f"SELECT COUNT(*) AS transaction_count FROM {table}{where_clause} LIMIT 1000"
        
        # Pattern: grouped by month/time
        elif "per month" in goal_lower or "by month" in goal_lower or "each month" in goal_lower:
            if time_col:
                sql = f"SELECT DATE_TRUNC('month', {time_expr}) AS month, COUNT(*) AS count FROM {table} GROUP BY 1 ORDER BY 1 LIMIT 1000"
            else:
                sql = f"SELECT COUNT(*) AS count FROM {table} LIMIT 1000"
        
        # Pattern: unique/distinct values
        elif "unique" in goal_lower or "distinct" in goal_lower:
            # Try to find the column being asked about
            col_to_check = None
            for col_word in ["status", "type", "category", "state"]:
                if col_word in goal_lower:
                    # Find matching column in schema
                    for t in schema.tables:
                        for c in t.columns:
                            if col_word in c.name.lower():
                                col_to_check = c.name
                                break
            if col_to_check:
                sql = f"SELECT DISTINCT {col_to_check} FROM {table} LIMIT 100"
            else:
                # Get first non-id column
                first_col = self._get_first_interesting_column(schema)
                sql = f"SELECT DISTINCT {first_col} FROM {table} LIMIT 100"
        
        # Pattern: sum/average of numeric columns
        elif "sum" in goal_lower or "total amount" in goal_lower:
            numeric_col = self._find_numeric_column(schema, goal_lower)
            if numeric_col:
                sql = f"SELECT SUM({numeric_col}) AS total FROM {table} LIMIT 1000"
            else:
                sql = f"SELECT COUNT(*) AS count FROM {table} LIMIT 1000"
        
        elif "average" in goal_lower or "avg" in goal_lower:
            numeric_col = self._find_numeric_column(schema, goal_lower)
            if numeric_col:
                sql = f"SELECT AVG({numeric_col}) AS average FROM {table} LIMIT 1000"
            else:
                sql = f"SELECT COUNT(*) AS count FROM {table} LIMIT 1000"
        
        # Pattern: show/list/display
        elif any(kw in goal_lower for kw in ["show", "list", "display", "get", "what are"]):
            # Check for specific column requests
            cols = self._extract_columns_from_goal(goal_lower, schema)
            if cols:
                sql = f"SELECT {', '.join(cols)} FROM {table} LIMIT 100"
            else:
                sql = f"SELECT * FROM {table} LIMIT 100"
        
        # Fallback: simple select
        else:
            sql = f"SELECT * FROM {table} LIMIT 100"
        
        return sql
    
    def _get_first_interesting_column(self, schema: SchemaResult) -> str:
        """Get first non-id column from schema."""
        for t in schema.tables:
            for c in t.columns:
                if not c.name.lower().endswith("_id") and c.name.lower() != "id":
                    return c.name
        return "*"
    
    def _find_numeric_column(self, schema: SchemaResult, goal: str) -> str | None:
        """Find a numeric column that matches the goal."""
        keywords = ["amount", "value", "total", "price", "cost", "revenue", "count"]
        for t in schema.tables:
            for c in t.columns:
                if c.data_type.upper() in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]:
                    if any(kw in c.name.lower() for kw in keywords):
                        return c.name
        # Return first numeric column if no match
        for t in schema.tables:
            for c in t.columns:
                if c.data_type.upper() in ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]:
                    return c.name
        return None
    
    def _extract_columns_from_goal(self, goal: str, schema: SchemaResult) -> list[str]:
        """Extract column names mentioned in the goal."""
        columns = []
        for t in schema.tables:
            for c in t.columns:
                # Check if column name (without underscores) appears in goal
                readable_name = c.name.replace("_", " ").lower()
                if readable_name in goal or c.name.lower() in goal:
                    columns.append(c.name)
        return columns[:5]  # Limit to 5 columns
    
    def close(self) -> None:
        """Close the executor."""
        self.executor.close()
    
    def __enter__(self) -> "QueryAgent":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
