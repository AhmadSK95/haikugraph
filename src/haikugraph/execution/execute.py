"""Execute plans by generating and running SQL queries against DuckDB."""

import copy
import re
from datetime import datetime
from pathlib import Path
import json
from haikugraph.execution.type_detector import (
    get_column_info,
    get_sql_cast_expression,
    detect_column_type
)

import duckdb

from haikugraph.planning.ambiguity import validate_no_unresolved_ambiguities
from haikugraph.planning.schema import validate_plan_or_raise
from haikugraph.execution.comparison import extract_comparison_from_results, ComparisonResult
from haikugraph.rules import load_rules, apply_entity_rules


def execute_plan(plan: dict, db_path: Path) -> dict:
    """
    Execute a plan by running SQL queries for each subquestion.

    Args:
        plan: Plan dict from planner
        db_path: Path to DuckDB database

    Returns:
        Execution result dict with subquestion results

    Raises:
        ValueError: If plan does not conform to schema
    """
    # Validate plan schema before execution
    validate_plan_or_raise(plan)

    # Validate no unresolved ambiguities remain
    validate_no_unresolved_ambiguities(plan)

    # Apply ambiguity resolutions to plan
    resolved_plan = apply_resolutions(plan)

    conn = duckdb.connect(str(db_path), read_only=True)

    # Process ambiguity resolutions for result output
    resolutions = []
    for ambiguity in plan.get("ambiguities", []):
        resolutions.append(
            {
                "issue": ambiguity["issue"],
                "chosen": ambiguity["recommended"],
                "alternatives": ambiguity["options"],
            }
        )

    # Execute each subquestion
    subquestion_results = []
    for sq in resolved_plan.get("subquestions", []):
        result = execute_subquestion(sq, resolved_plan, conn)
        subquestion_results.append(result)

    conn.close()

    # A11: Extract and normalize comparison if this is a comparison query
    # This MUST happen before narration to enforce structural correctness
    comparison_result = None
    try:
        comparison_result = extract_comparison_from_results(resolved_plan, subquestion_results)
    except ValueError as e:
        # Comparison extraction failed - this is a fatal error for comparison queries
        # Include the error in final_summary so narrator can report it
        final_summary = f"Comparison extraction failed: {e}"
    else:
        # Generate final summary (only if comparison extraction succeeded or not a comparison)
        final_summary = generate_summary(plan, subquestion_results)

    # Generate human-readable applied resolutions summary
    applied_resolutions_summary = summarize_applied_resolutions(
        resolved_plan.get("resolutions_applied", [])
    )

    result = {
        "original_question": plan.get("original_question", ""),
        "executed_at": datetime.utcnow().isoformat() + "Z",
        "resolutions": resolutions,
        "applied_resolutions_summary": applied_resolutions_summary,
        "subquestion_results": subquestion_results,
        "final_summary": final_summary,
    }
    
    # A11: Include normalized comparison result if present
    # This provides a first-class comparison structure for the narrator
    if comparison_result is not None:
        result["comparison"] = comparison_result.model_dump()
    
    return result


def execute_subquestion(sq: dict, plan: dict, conn) -> dict:
    """
    Execute a single subquestion.

    Args:
        sq: Subquestion dict
        plan: Full plan for context (constraints, join_paths, etc.)
        conn: DuckDB connection

    Returns:
        Result dict with SQL, row count, preview rows
    """
    sql = None
    metadata = {
        "joins_used": [],
        "constraints_applied": [],
        "resolutions": [],
        "resolutions_applied": plan.get("resolutions_applied", []),
    }

    try:
        # Build SQL for this subquestion
        sql, build_metadata = build_sql(sq, plan, conn)
        # Merge build_metadata into metadata
        metadata.update(build_metadata)
        
        # DEBUG: Log generated SQL
        print(f"\n{'='*80}")
        print(f"EXECUTING SQL for subquestion {sq.get('id', 'unknown')}:")
        print(f"{'='*80}")
        print(f"Subquestion: {sq}")
        print(f"\nGenerated SQL:\n{sql}")
        print(f"{'='*80}\n")

        # Execute query
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []

        # Format preview rows (limit to 20)
        preview_rows = []
        for row in result[:20]:
            preview_rows.append({col: val for col, val in zip(columns, row)})

        return {
            "id": sq.get("id", "unknown"),
            "status": "success",
            "description": sq.get("description", ""),
            "sql": sql,
            "row_count": len(result),
            "columns": columns,
            "preview_rows": preview_rows,
            "metadata": metadata,
        }
    except Exception as e:
        # Return failure with SQL preserved
        return {
            "id": sq.get("id", "unknown"),
            "status": "failed",
            "description": sq.get("description", ""),
            "sql": sql,
            "error": str(e),
            "row_count": 0,
            "columns": [],
            "preview_rows": [],
            "metadata": metadata,
        }


def _strip_sql_literals(sql: str) -> str:
    """Cheap scrub of literals/identifiers to reduce false FROM matches.
    - Replaces single-quoted strings (including escaped '' inside) with ''
    - Replaces double-quoted identifiers (including escaped "") with ""
    Note: Not a full SQL parser; intended for simple guard only.
    """
    # Single-quoted string literals: '...'
    sql = re.sub(r"'([^']|'')*'", "''", sql)
    # Double-quoted identifiers: "..."
    sql = re.sub(r'"([^"]|"")*"', '""', sql)
    return sql


def get_timestamp_expression(table: str, col: str, conn) -> str:
    """
    Get a timestamp expression that works with DuckDB date_trunc.
    
    DuckDB's date_trunc requires DATE/TIMESTAMP types, but columns may be VARCHAR.
    This utility introspects the column type and returns an appropriate expression:
    - If TIMESTAMP/DATE: use column as-is
    - If VARCHAR: wrap with TRY_CAST(col AS TIMESTAMP)
    
    Args:
        table: Table name
        col: Column name
        conn: DuckDB connection
    
    Returns:
        SQL expression that evaluates to a timestamp
    """
    try:
        # Query DuckDB catalog to get column type
        type_query = f"""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{col}'
        """
        result = conn.execute(type_query).fetchone()
        
        if result:
            data_type = result[0].upper()
            
            # If already a timestamp/date type, use as-is
            if any(t in data_type for t in ['TIMESTAMP', 'DATE', 'TIME']):
                return f'"{table}"."{col}"'
            
            # If VARCHAR/TEXT, wrap with TRY_CAST
            if any(t in data_type for t in ['VARCHAR', 'TEXT', 'CHAR']):
                return f'TRY_CAST("{table}"."{col}" AS TIMESTAMP)'
        
        # Default: assume it's a timestamp column (for compatibility)
        return f'"{table}"."{col}"'
        
    except Exception:
        # If schema introspection fails, use TRY_CAST as safe default
        return f'TRY_CAST("{table}"."{col}" AS TIMESTAMP)'


def build_sql(sq: dict, plan: dict, conn=None) -> tuple[str, dict]:
    """
    Build SQL query for a subquestion.

    Args:
        sq: Subquestion dict
        plan: Full plan for context
        conn: Optional DuckDB connection for schema introspection

    Returns:
        Tuple of (sql_string, metadata_dict)
    """
    metadata = {"joins_used": [], "constraints_applied": [], "resolutions": []}

    tables = sq.get("tables", [])
    columns = sq.get("columns", [])
    group_by = sq.get("group_by", [])
    aggregations = sq.get("aggregations", [])

    if not tables:
        raise ValueError("No tables specified in subquestion")

    # Determine primary table
    primary_table = tables[0]

    # Build SELECT clause
    if aggregations:
        # Aggregation query (with or without GROUP BY)
        select_parts = []
        time_bucket_exprs = []  # Track time bucket expressions for GROUP BY

        # Add group by columns if present
        if group_by:
            for col_spec in group_by:
                # Support both simple strings and dict specs for time bucketing
                if isinstance(col_spec, dict):
                    # Time bucket: {"type": "time_bucket", "grain": "month", "col": "date_col"}
                    if col_spec.get("type") == "time_bucket":
                        grain = col_spec.get("grain", "month")
                        col_name = col_spec["col"]
                        bucket_alias = f"{grain}"
                        
                        # Get timestamp expression (handles VARCHAR columns)
                        if conn:
                            ts_expr = get_timestamp_expression(primary_table, col_name, conn)
                        else:
                            ts_expr = f'"{primary_table}"."{col_name}"'
                        
                # DuckDB date_trunc function with COALESCE for NULL handling
                        # Use COALESCE to show NULL timestamps as 'Unknown' or keep as NULL for counting
                        bucket_expr = f"date_trunc('{grain}', {ts_expr})"
                        select_parts.append(f'{bucket_expr} AS "{bucket_alias}"')
                        time_bucket_exprs.append((bucket_alias, bucket_expr))
                    else:
                        # Unknown dict type, skip
                        pass
                elif isinstance(col_spec, str):
                    # Simple column name
                    select_parts.append(f'"{primary_table}"."{col_spec}"')
                else:
                    # Unknown type, skip
                    pass

        # Add aggregations with safe type casting
        for agg_spec in aggregations:
            agg_func = agg_spec["agg"].upper()
            agg_col = agg_spec["col"]
            is_distinct = agg_spec.get("distinct", False)
            
            # Handle count_distinct as alias for count with distinct=true
            if agg_func == "COUNT_DISTINCT":
                agg_func = "COUNT"
                is_distinct = True
            
            # Safety net: auto-apply DISTINCT for COUNT on _id columns
            # These columns typically have duplicates in merged/denormalized tables,
            # and counting without DISTINCT inflates the result.
            if (
                agg_func == "COUNT"
                and not is_distinct
                and agg_col.lower().endswith("_id")
            ):
                is_distinct = True
            
            # Build alias
            if is_distinct:
                alias = f"{agg_spec['agg']}_distinct_{agg_col}"
            else:
                alias = f"{agg_spec['agg']}_{agg_col}"
            
            # Build column expression with improved type detection
            if agg_func in ("SUM", "AVG", "MIN", "MAX"):
                if conn:
                    # Use improved type detection
                    declared_type, semantic_type = get_column_info(conn, primary_table, agg_col)
                    col_expr = get_sql_cast_expression(
                        primary_table, 
                        agg_col, 
                        semantic_type,
                        declared_type
                    )
                else:
                    # Fallback: detect from name only
                    semantic_type = detect_column_type(agg_col, "VARCHAR")
                    if semantic_type == "timestamp":
                        col_expr = f'TRY_CAST("{primary_table}"."{agg_col}" AS TIMESTAMP)'
                    else:
                        col_expr = f'TRY_CAST("{primary_table}"."{agg_col}" AS DOUBLE)'
            else:
                # COUNT - just column reference
                col_expr = f'"{primary_table}"."{agg_col}"'
            
            # Apply DISTINCT if requested (typically for COUNT)
            if is_distinct:
                agg_expr = f'{agg_func}(DISTINCT {col_expr})'
            else:
                agg_expr = f'{agg_func}({col_expr})'
            
            select_parts.append(f'{agg_expr} AS "{alias}"')

        select_clause = "SELECT " + ", ".join(select_parts)
    else:
        # Regular select with smart casting for display
        if columns:
            select_parts = []
            for col in sorted(columns)[:50]:  # Limit columns
                if conn:
                    # Apply smart casting for known types
                    declared_type, semantic_type = get_column_info(conn, primary_table, col)
                    
                    # For regular SELECTs, we want to preserve original data
                    # but cast numeric/timestamp columns for better display
                    if semantic_type in ("timestamp", "numeric_amount", "numeric_rate", "numeric_decimal"):
                        cast_expr = get_sql_cast_expression(
                            primary_table, col, semantic_type, declared_type
                        )
                        select_parts.append(f'{cast_expr} AS "{col}"')
                    else:
                        # Keep as-is (identifiers, text, booleans)
                        select_parts.append(f'"{primary_table}"."{col}"')
                else:
                    select_parts.append(f'"{primary_table}"."{col}"')
            
            select_clause = "SELECT " + ", ".join(select_parts)
        else:
            # SELECT * without FROM in clause (FROM added separately below)
            select_clause = "SELECT *"

    # Build FROM clause with joins if needed
    from_clause = f'FROM "{primary_table}"'

    # Apply joins if multiple tables needed
    if len(tables) > 1:
        # Find join paths for these tables
        join_paths = find_relevant_joins(tables, plan.get("join_paths", []))
        for jp in join_paths:
            join_from = jp["from"]
            join_to = jp["to"]
            join_cols = jp["via"]

            # Build JOIN clause
            if join_cols:
                join_col = join_cols[0]  # Use first column for now
                from_clause += (
                    f' INNER JOIN "{join_to}" '
                    f'ON "{join_from}"."{join_col}" = "{join_to}"."{join_col}"'
                )
                metadata["joins_used"].append(
                    {
                        "from": join_from,
                        "to": join_to,
                        "on": join_col,
                    }
                )

    # Build WHERE clause from constraints
    where_clauses = []
    
    # Handle time_filter from comparison subquestions
    time_filter = sq.get("time_filter")
    if time_filter and time_filter.get("column"):
        time_col = time_filter["column"]
        period = time_filter["period"]
        
        # Get timestamp expression with proper casting
        if conn:
            ts_expr = get_timestamp_expression(primary_table, time_col, conn)
        else:
            ts_expr = f'TRY_CAST("{primary_table}"."{time_col}" AS TIMESTAMP)'
        
        # Parse period and build WHERE clause
        # Format: "september_2025", "october_2025", "year_2024"
        if "_" in period:
            parts = period.split("_")
            if len(parts) == 2:
                period_type, period_value = parts
                
                # Month comparison
                if period_type in ["january", "february", "march", "april", "may", "june",
                                  "july", "august", "september", "october", "november", "december"]:
                    month_num = {
                        "january": 1, "february": 2, "march": 3, "april": 4,
                        "may": 5, "june": 6, "july": 7, "august": 8,
                        "september": 9, "october": 10, "november": 11, "december": 12
                    }[period_type]
                    
                    where_clauses.append(
                        f"EXTRACT(YEAR FROM {ts_expr}) = {period_value} AND "
                        f"EXTRACT(MONTH FROM {ts_expr}) = {month_num}"
                    )
                    metadata["constraints_applied"].append({
                        "type": "time_filter",
                        "period": period,
                        "column": time_col
                    })
                
                # Year comparison
                elif period_type == "year":
                    where_clauses.append(f"EXTRACT(YEAR FROM {ts_expr}) = {period_value}")
                    metadata["constraints_applied"].append({
                        "type": "time_filter",
                        "period": period,
                        "column": time_col
                    })
    
    # Apply plan-level constraints AND subquestion-level constraints
    plan_constraints = plan.get("constraints", [])
    sq_constraints = sq.get("constraints", [])
    constraints = plan_constraints + sq_constraints

    for constraint in constraints:
        expr = constraint.get("expression", "")
        constraint_type = constraint.get("type")

        # Check if constraint is scoped to a specific subquestion (A5 comparison support)
        applies_to = constraint.get("applies_to")
        if applies_to and applies_to != sq.get("id"):
            continue

        # Handle time_month constraints
        if constraint_type == "time_month":
            month_num = constraint.get("month")
            year_num = constraint.get("year")
            time_col = constraint.get("column")
            constraint_table = constraint.get("table") or primary_table  # Default to primary table if missing
            
            if constraint_table in tables:
                # Get timestamp expression with proper casting
                if conn:
                    ts_expr = get_timestamp_expression(constraint_table, time_col, conn)
                else:
                    ts_expr = f'TRY_CAST("{constraint_table}"."{time_col}" AS TIMESTAMP)'
                
                sql_expr = f"EXTRACT(MONTH FROM {ts_expr}) = {month_num}"
                if year_num:
                    sql_expr += f" AND EXTRACT(YEAR FROM {ts_expr}) = {year_num}"
                where_clauses.append(sql_expr)
                metadata["constraints_applied"].append({
                    "type": "time_month",
                    "month": month_num,
                    "year": year_num,
                    "column": time_col,
                    "expression": sql_expr
                })
            continue
        
        # Handle time_relative constraints (today, this week, last month, etc.)
        if constraint_type == "time_relative":
            period = constraint.get("period")
            time_col = constraint.get("column")
            constraint_table = constraint.get("table")
            days = constraint.get("days")  # For "last N days" patterns
            
            if constraint_table in tables:
                # Get timestamp expression with proper casting
                if conn:
                    ts_expr = get_timestamp_expression(constraint_table, time_col, conn)
                else:
                    ts_expr = f'TRY_CAST("{constraint_table}"."{time_col}" AS TIMESTAMP)'
                
                # Translate period to SQL
                if period == "today":
                    sql_expr = f"CAST({ts_expr} AS DATE) = CURRENT_DATE"
                elif period == "yesterday":
                    sql_expr = f"CAST({ts_expr} AS DATE) = CURRENT_DATE - INTERVAL '1 day'"
                elif period == "this_week":
                    sql_expr = f"{ts_expr} >= DATE_TRUNC('week', CURRENT_DATE) AND {ts_expr} < DATE_TRUNC('week', CURRENT_DATE) + INTERVAL '1 week'"
                elif period == "last_week":
                    sql_expr = f"{ts_expr} >= DATE_TRUNC('week', CURRENT_DATE) - INTERVAL '1 week' AND {ts_expr} < DATE_TRUNC('week', CURRENT_DATE)"
                elif period == "this_month":
                    sql_expr = f"{ts_expr} >= DATE_TRUNC('month', CURRENT_DATE) AND {ts_expr} < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'"
                elif period == "last_month":
                    sql_expr = f"{ts_expr} >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' AND {ts_expr} < DATE_TRUNC('month', CURRENT_DATE)"
                elif period == "this_year":
                    sql_expr = f"{ts_expr} >= DATE_TRUNC('year', CURRENT_DATE) AND {ts_expr} < DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '1 year'"
                elif period == "last_year":
                    sql_expr = f"{ts_expr} >= DATE_TRUNC('year', CURRENT_DATE) - INTERVAL '1 year' AND {ts_expr} < DATE_TRUNC('year', CURRENT_DATE)"
                elif period == "last_N_days" and days:
                    sql_expr = f"{ts_expr} >= CURRENT_DATE - INTERVAL '{days} days'"
                else:
                    # Fallback for unknown periods
                    sql_expr = f"{ts_expr} >= CURRENT_DATE - INTERVAL '30 days'"
                
                where_clauses.append(sql_expr)
                metadata["constraints_applied"].append({
                    "type": "time_relative",
                    "period": period,
                    "column": time_col,
                    "expression": sql_expr
                })
            continue
        
        # Handle value_filter constraints
        if constraint_type == "value_filter":
            constraint_table = constraint.get("table") or primary_table
            col = constraint.get("column")
            operator = constraint.get("operator", "eq")
            value = constraint.get("value")
            
            if constraint_table in tables and col:
                # Build SQL expression based on operator
                col_ref = f'"{constraint_table}"."{col}"'
                if operator == "is_not_null":
                    sql_expr = f"{col_ref} IS NOT NULL"
                elif operator == "is_null":
                    sql_expr = f"{col_ref} IS NULL"
                elif operator == "eq" and value is not None:
                    # Escape single quotes in value
                    safe_value = str(value).replace("'", "''")
                    sql_expr = f"{col_ref} = '{safe_value}'"
                elif operator == "neq" and value is not None:
                    safe_value = str(value).replace("'", "''")
                    sql_expr = f"{col_ref} != '{safe_value}'"
                elif operator == "gt" and value is not None:
                    sql_expr = f"{col_ref} > {value}"
                elif operator == "lt" and value is not None:
                    sql_expr = f"{col_ref} < {value}"
                elif operator == "gte" and value is not None:
                    sql_expr = f"{col_ref} >= {value}"
                elif operator == "lte" and value is not None:
                    sql_expr = f"{col_ref} <= {value}"
                elif expr:
                    # Fallback: use expression as-is
                    sql_expr = expr
                else:
                    continue
                
                where_clauses.append(sql_expr)
                metadata["constraints_applied"].append({
                    "type": "value_filter",
                    "column": col,
                    "operator": operator,
                    "value": value,
                    "expression": sql_expr
                })
            continue

        # Check if constraint's table is in our tables (for other constraint types)
        constraint_table = extract_table_from_constraint(expr)
        if constraint_table and constraint_table in tables:
            # Translate constraint
            if constraint_type == "time":
                sql_expr = translate_time_constraint(expr, conn)
            else:
                sql_expr = expr

            where_clauses.append(sql_expr)
            metadata["constraints_applied"].append(
                {"type": constraint_type, "expression": sql_expr}
            )

    # ==========================================================================
    # APPLY DATA RULES FROM rules.yaml
    # ==========================================================================
    # Load and apply configured data rules (validity rules, default filters, etc.)
    # These rules are applied AFTER plan constraints but BEFORE assembling WHERE clause.
    try:
        original_question = plan.get("original_question", "")
        where_clauses, applied_rules = apply_entity_rules(
            table_name=primary_table,
            existing_where_clauses=where_clauses,
            question=original_question,
            include_defaults=True,
            include_global=True,
        )
        
        # Track applied rules in metadata
        if applied_rules:
            metadata["data_rules_applied"] = applied_rules
            for rule in applied_rules:
                metadata["constraints_applied"].append({
                    "type": f"data_rule:{rule['type']}",
                    "expression": rule["condition"],
                    "source": "rules.yaml",
                })
    except Exception as e:
        # Rules loading failed - log but don't break query execution
        # This allows queries to work even if rules.yaml has issues
        print(f"Warning: Failed to apply data rules: {e}")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    # Build GROUP BY clause
    group_by_clause = ""
    if group_by:
        group_by_parts = []
        for col_spec in group_by:
            if isinstance(col_spec, dict) and col_spec.get("type") == "time_bucket":
                # Use the bucket expression directly (with VARCHAR handling)
                grain = col_spec.get("grain", "month")
                col_name = col_spec["col"]
                
                # Get timestamp expression (handles VARCHAR columns)
                if conn:
                    ts_expr = get_timestamp_expression(primary_table, col_name, conn)
                else:
                    ts_expr = f'"{primary_table}"."{col_name}"'
                
                bucket_expr = f"date_trunc('{grain}', {ts_expr})"
                group_by_parts.append(bucket_expr)
            elif isinstance(col_spec, str):
                group_by_parts.append(f'"{primary_table}"."{col_spec}"')
        
        if group_by_parts:
            group_by_clause = "GROUP BY " + ", ".join(group_by_parts)

    # Build ORDER BY clause for determinism
    order_by_clause = ""
    if group_by:
        # Order by group_by columns (preserving order for time series)
        order_by_parts = []
        for col_spec in group_by:
            if isinstance(col_spec, dict) and col_spec.get("type") == "time_bucket":
                # Order by time bucket alias with NULLS LAST
                grain = col_spec.get("grain", "month")
                order_by_parts.append(f'"{grain}" NULLS LAST')
            elif isinstance(col_spec, str):
                order_by_parts.append(f'"{primary_table}"."{col_spec}"')
        
        if order_by_parts:
            order_by_clause = "ORDER BY " + ", ".join(order_by_parts)
    elif aggregations and not group_by:
        # Pure aggregation: no ordering
        order_by_clause = ""
    elif columns:
        # Order by first column
        order_by_clause = f'ORDER BY "{primary_table}"."{columns[0]}"'

    # Build LIMIT clause for non-grouped, non-aggregated queries
    limit_clause = ""
    if not group_by and not aggregations:
        # Only add LIMIT for regular SELECT queries (not aggregations)
        # Check for plan-level row_limit (from A5 follow-ups)
        row_limit = plan.get("row_limit")
        if row_limit and isinstance(row_limit, int) and row_limit > 0:
            limit_clause = f"LIMIT {row_limit}"
        else:
            limit_clause = "LIMIT 200"

    # Assemble final SQL
    sql_parts = [select_clause, from_clause]
    if where_clause:
        sql_parts.append(where_clause)
    if group_by_clause:
        sql_parts.append(group_by_clause)
    if order_by_clause:
        sql_parts.append(order_by_clause)
    if limit_clause:
        sql_parts.append(limit_clause)

    sql = " ".join(sql_parts)

    # Defensive guard: ensure exactly one FROM clause in simple queries.
    # Scrub literals/identifiers first to avoid false matches in skip logic and FROM counting.
    # Only enforce guard when we don't detect CTEs or subqueries.
    scrubbed = _strip_sql_literals(sql)
    scrubbed_upper = scrubbed.upper()
    
    # Skip guard if CTEs (WITH) or subqueries ((SELECT with optional whitespace) detected
    has_cte = bool(re.search(r"\bWITH\b", scrubbed_upper))
    has_subquery = bool(re.search(r"\(\s*SELECT\b", scrubbed_upper))
    
    # Also skip guard if EXTRACT(...FROM...) is present (temporal functions)
    has_extract = bool(re.search(r"\bEXTRACT\s*\(", scrubbed_upper))
    
    if not has_cte and not has_subquery and not has_extract:
        # Count FROM clauses in scrubbed SQL (already uppercase, no case flag needed)
        from_matches = re.findall(r"\bFROM\b", scrubbed_upper)
        if len(from_matches) != 1:
            raise ValueError(
                f"Invalid SQL generated: expected exactly 1 FROM clause, got {len(from_matches)}.\n"
                f"SQL: {sql}"
            )

    return sql, metadata


def find_relevant_joins(tables: list[str], join_paths: list[dict]) -> list[dict]:
    """
    Find join paths connecting the given tables.

    Args:
        tables: List of table names to connect
        join_paths: Available join paths from plan

    Returns:
        List of relevant join path dicts
    """
    relevant = []
    for jp in join_paths:
        if jp["from"] in tables and jp["to"] in tables:
            relevant.append(jp)
    return relevant


def extract_table_from_constraint(expr: str) -> str | None:
    """Extract table name from constraint expression like 'test_1_1.status = ...'"""
    match = re.match(r"^([a-z0-9_]+)\.", expr)
    return match.group(1) if match else None


def translate_time_constraint(expr: str, conn=None) -> str:
    """
    Translate symbolic time constraint to SQL.

    Args:
        expr: Symbolic expression like "test_2_1.created_at in last_30_days"
              or "table.col in yesterday" or "table.col in this_month"
        conn: Optional DuckDB connection for VARCHAR handling

    Returns:
        SQL expression like "test_2_1.created_at >= now() - interval '30 days'"
    """
    # Parse: table.column in <period>
    pattern = r"^([a-z0-9_]+)\.([a-z0-9_]+)\s+in\s+(.+)$"
    match = re.match(pattern, expr)

    if not match:
        return expr  # Can't parse, return original

    table_name = match.group(1)
    col_name = match.group(2)
    period = match.group(3).strip()

    # Get timestamp expression (handles VARCHAR columns)
    if conn:
        qualified_col = get_timestamp_expression(table_name, col_name, conn)
    else:
        qualified_col = f'"{table_name}"."{col_name}"'

    # Handle different period formats
    # Pattern: last_N_days, last_N_weeks, last_N_months, last_N_years
    last_n_match = re.match(r"^last_(\d+)_(day|week|month|year)s?$", period)
    if last_n_match:
        count = last_n_match.group(1)
        unit = last_n_match.group(2)
        return f"{qualified_col} >= now() - interval '{count} {unit}s'"

    # Simple periods
    simple_periods = {
        "yesterday": (
            f"{qualified_col} >= current_date - interval '1 day' AND {qualified_col} < current_date"
        ),
        "today": f"{qualified_col} >= current_date",
        "this_week": f"{qualified_col} >= date_trunc('week', current_date)",
        "this_month": f"{qualified_col} >= date_trunc('month', current_date)",
        "this_year": f"{qualified_col} >= date_trunc('year', current_date)",
        "previous_month": (
            f"{qualified_col} >= date_trunc('month', current_date) - interval '1 month' "
            f"AND {qualified_col} < date_trunc('month', current_date)"
        ),
        "previous_year": (
            f"{qualified_col} >= date_trunc('year', current_date) - interval '1 year' "
            f"AND {qualified_col} < date_trunc('year', current_date)"
        ),
        "previous_week": (
            f"{qualified_col} >= date_trunc('week', current_date) - interval '1 week' "
            f"AND {qualified_col} < date_trunc('week', current_date)"
        ),
        "previous_period": f"{qualified_col} >= now() - interval '30 days'",
    }

    if period in simple_periods:
        return simple_periods[period]

    # Fallback: return original
    return expr


def generate_summary(plan: dict, results: list[dict]) -> str:
    """Generate human-readable summary of execution results."""
    question = plan.get("original_question", "Unknown question")
    total_rows = sum(r.get("row_count", 0) for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    summary_parts = [f'Answered: "{question}"']
    summary_parts.append(f"Executed {len(results)} subquestion(s)")

    if failed > 0:
        summary_parts.append(f"{failed} failed")

    summary_parts.append(f"Total rows retrieved: {total_rows}")

    return " | ".join(summary_parts)


def summarize_applied_resolutions(resolutions_applied: list[dict]) -> list[str]:
    """
    Generate human-readable summary of applied resolutions.

    Args:
        resolutions_applied: List of resolution dicts from apply_resolutions

    Returns:
        List of readable summary strings
    """
    summary = []

    for resolution in resolutions_applied:
        kind = resolution.get("kind")
        context = resolution.get("context")

        if kind == "entity_table":
            # Entity resolution
            entity = resolution.get("entity", "unknown")
            table = resolution.get("chosen_table", "unknown")
            summary.append(f"Entity '{entity}' resolved to table {table}")

        elif kind == "column_table":
            column = resolution.get("column", "unknown")
            table = resolution.get("chosen_table", "unknown")

            if context == "metric_rewrite":
                # Metric column rewrite
                summary.append(f"Column '{column}' rewritten to {table} (metric)")

            elif context == "subquestion_switch":
                # Subquestion table switch
                sq_id = resolution.get("subquestion_id", "unknown")
                orig_table = resolution.get("original_table", "unknown")
                summary.append(
                    f"Subquestion {sq_id} switched '{column}' from {orig_table} → {table}"
                )

    return summary


def apply_resolutions(plan: dict) -> dict:
    """
    Apply ambiguity resolutions to plan (entity and column/table ambiguities).

    Args:
        plan: Original plan dict

    Returns:
        Deep-copied plan with resolutions applied and resolutions_applied key added
    """
    resolved = copy.deepcopy(plan)
    resolutions_applied = []

    # Extract entity-to-table mappings from ambiguities
    entity_table_choices = {}
    # Extract column ambiguity rules (not global dict - structured list)
    column_ambiguities = []

    for ambiguity in plan.get("ambiguities", []):
        issue = ambiguity.get("issue", "")
        recommended = ambiguity.get("recommended")
        options = ambiguity.get("options", [])

        # Match pattern: "Entity 'X' found in multiple tables"
        entity_match = re.match(r"Entity '([^']+)' found in multiple tables", issue)
        if entity_match and recommended:
            entity_name = entity_match.group(1)
            entity_table_choices[entity_name] = recommended
            resolutions_applied.append(
                {"kind": "entity_table", "entity": entity_name, "chosen_table": recommended}
            )

        # Match patterns for column ambiguities:
        # "Multiple tables contain test_2_1.payment_amount"
        # "Multiple tables contain payment_amount"
        column_match = re.match(r"^Multiple tables contain (?:([a-z0-9_]+)\.)?([a-z0-9_]+)$", issue)
        if column_match and recommended:
            original_table = column_match.group(1)  # May be None
            column_name = column_match.group(2)
            column_ambiguities.append(
                {
                    "column": column_name,
                    "recommended": recommended,
                    "options": options,
                    "original": f"{original_table}.{column_name}"
                    if original_table
                    else column_name,
                }
            )

    # Apply to entities_detected: filter mapped_to to only chosen table columns
    for entity in resolved.get("entities_detected", []):
        entity_name = entity.get("name")
        if entity_name in entity_table_choices:
            chosen_table = entity_table_choices[entity_name]
            original_mapped = entity.get("mapped_to", [])
            # Keep only columns from chosen table
            filtered = [col for col in original_mapped if col.startswith(f"{chosen_table}.")]
            if filtered:
                entity["mapped_to"] = sorted(filtered)

    # Apply to metrics_requested: rewrite column references with overlap logic
    applied_metric_resolutions = set()  # Track (column, recommended) to avoid duplicates
    for metric in resolved.get("metrics_requested", []):
        mapped_cols = metric.get("mapped_columns", [])
        new_mapped_cols = []
        for col_ref in mapped_cols:
            # Parse table.column
            if "." in col_ref:
                parts = col_ref.split(".", 1)
                if len(parts) == 2:
                    old_table, col_name = parts
                    # Find matching column ambiguity rule
                    matching_rule = next(
                        (rule for rule in column_ambiguities if rule["column"] == col_name),
                        None,
                    )
                    if (
                        matching_rule
                        and old_table in matching_rule["options"]
                        and matching_rule["recommended"] in matching_rule["options"]
                    ):
                        # Rewrite to recommended table (overlap condition satisfied)
                        new_mapped_cols.append(f"{matching_rule['recommended']}.{col_name}")
                        # Record metric rewrite (once per column/recommended pair)
                        resolution_key = (col_name, matching_rule["recommended"])
                        if resolution_key not in applied_metric_resolutions:
                            applied_metric_resolutions.add(resolution_key)
                            resolutions_applied.append(
                                {
                                    "kind": "column_table",
                                    "column": col_name,
                                    "chosen_table": matching_rule["recommended"],
                                    "original": matching_rule["original"],
                                    "context": "metric_rewrite",
                                }
                            )
                    else:
                        new_mapped_cols.append(col_ref)
                else:
                    new_mapped_cols.append(col_ref)
            else:
                new_mapped_cols.append(col_ref)
        if new_mapped_cols:
            metric["mapped_columns"] = sorted(new_mapped_cols)

    # Apply to subquestions with strict overlap logic
    applied_sq_resolutions = set()  # Dedupe subquestion switches
    for sq in resolved.get("subquestions", []):
        sq_tables = sq.get("tables", [])
        sq_columns = sq.get("columns", [])
        aggregations = sq.get("aggregations", [])
        sq_id = sq.get("id", "unknown")

        # Collect all columns used in this subquestion
        columns_used = sq_columns + [agg.get("col") for agg in aggregations if agg.get("col")]

        # Apply column ambiguity rules with overlap check
        for rule in column_ambiguities:
            if rule["column"] in columns_used:
                # Compute overlap between sq_tables and rule options
                overlap = [t for t in sq_tables if t in rule["options"]]

                # Case A: Multiple overlapping tables - narrow to recommended
                if len(overlap) > 1 and rule["recommended"] in overlap:
                    # Determine original table (first in overlap that's not recommended)
                    original_table = next(
                        (t for t in overlap if t != rule["recommended"]),
                        overlap[0] if overlap else "unknown",
                    )
                    sq["tables"] = [rule["recommended"]]
                    sq_tables = sq["tables"]
                    # Record this application (deduplicated)
                    resolution_key = (
                        sq_id,
                        rule["column"],
                        rule["recommended"],
                        "subquestion_switch",
                    )
                    if resolution_key not in applied_sq_resolutions:
                        applied_sq_resolutions.add(resolution_key)
                        resolutions_applied.append(
                            {
                                "kind": "column_table",
                                "column": rule["column"],
                                "chosen_table": rule["recommended"],
                                "original": rule["original"],
                                "context": "subquestion_switch",
                                "subquestion_id": sq_id,
                                "original_table": original_table,
                            }
                        )
                    break  # Only narrow once per subquestion

                # Case B: Collapsed subquestion - switch if current != recommended
                elif (
                    len(overlap) == 1
                    and overlap[0] != rule["recommended"]
                    and rule["recommended"] in rule["options"]
                    and overlap[0] in rule["options"]
                ):
                    # Subquestion pinned to one ambiguous option, switch
                    original_table = overlap[0]
                    sq["tables"] = [rule["recommended"]]
                    sq_tables = sq["tables"]
                    # Record this application (deduplicated)
                    resolution_key = (
                        sq_id,
                        rule["column"],
                        rule["recommended"],
                        "subquestion_switch",
                    )
                    if resolution_key not in applied_sq_resolutions:
                        applied_sq_resolutions.add(resolution_key)
                        resolutions_applied.append(
                            {
                                "kind": "column_table",
                                "column": rule["column"],
                                "chosen_table": rule["recommended"],
                                "original": rule["original"],
                                "context": "subquestion_switch",
                                "subquestion_id": sq_id,
                                "original_table": original_table,
                            }
                        )
                    break  # Only narrow once per subquestion

        # For each entity ambiguity, check if subquestion tables overlap with options
        for ambiguity in plan.get("ambiguities", []):
            issue = ambiguity.get("issue", "")
            if issue.startswith("Entity '") and "found in multiple tables" in issue:
                options = ambiguity.get("options", [])
                recommended = ambiguity.get("recommended")
                # Compute overlap between subquestion tables and ambiguity options
                overlap = [t for t in sq_tables if t in options]
                # If multiple overlapping tables and recommended is in overlap, narrow to it
                if len(overlap) > 1 and recommended in overlap:
                    sq["tables"] = [recommended]
                    sq_tables = sq["tables"]
                    break

    resolved["resolutions_applied"] = resolutions_applied
    return resolved


def _demo_apply_resolutions() -> None:
    """Demo/test function to verify apply_resolutions logic with scoped ambiguity."""
    # Construct a minimal test plan
    test_plan = {
        "original_question": "Test question",
        "ambiguities": [
            {
                "issue": "Entity 'customer' found in multiple tables",
                "recommended": "test_1_1",
                "options": ["test_1_1", "test_2_1", "test_4_1"],
            },
            {
                "issue": "Multiple tables contain test_2_1.payment_amount",
                "recommended": "test_1_1",  # Changed to test_1_1 for collapsed test
                "options": ["test_1_1", "test_2_1"],
            },
            {
                "issue": "Multiple tables contain transaction_id",
                "recommended": "test_3_1",
                "options": ["test_1_1", "test_3_1"],
            },
        ],
        "entities_detected": [
            {
                "name": "customer",
                "mapped_to": [
                    "test_1_1.customer_id",
                    "test_2_1.customer_id",
                    "test_4_1.customer_id",
                ],
            }
        ],
        "metrics_requested": [
            {
                "name": "sum_payment_amount",
                "mapped_columns": ["test_1_1.payment_amount", "test_2_1.payment_amount"],
            }
        ],
        "subquestions": [
            {
                "id": "SQ1",
                "tables": ["test_1_1", "test_2_1"],
                "columns": ["customer_id", "payment_amount"],
                "aggregations": [{"agg": "sum", "col": "payment_amount"}],
            },
            {
                "id": "SQ2",
                "tables": ["test_3_1"],
                "columns": ["transaction_id"],
            },
            {
                "id": "SQ3",
                "tables": ["test_2_1"],
                "columns": ["transaction_id"],
            },
            {
                "id": "SQ4",
                "tables": ["test_2_1"],  # Collapsed to single table
                "columns": ["payment_amount"],
                "aggregations": [{"agg": "sum", "col": "payment_amount"}],
            },
        ],
    }

    resolved = apply_resolutions(test_plan)

    # Assertions for verification
    # Check entity resolution applied
    assert any(
        r["kind"] == "entity_table"
        and r["entity"] == "customer"
        and r["chosen_table"] == "test_1_1"
        for r in resolved["resolutions_applied"]
    ), "Entity resolution not applied"

    # Check column resolution applied (now test_1_1 instead of test_2_1)
    assert any(
        r["kind"] == "column_table"
        and r["column"] == "payment_amount"
        and r["chosen_table"] == "test_1_1"
        for r in resolved["resolutions_applied"]
    ), "Column resolution not applied"

    # Check entities_detected filtered
    customer_entity = next(
        (e for e in resolved["entities_detected"] if e["name"] == "customer"), None
    )
    assert customer_entity is not None
    assert all(col.startswith("test_1_1.") for col in customer_entity["mapped_to"]), (
        "Entity mapped_to not filtered"
    )

    # Check metrics_requested rewritten to test_1_1
    metric = resolved["metrics_requested"][0]
    assert "test_1_1.payment_amount" in metric["mapped_columns"], (
        "Metric column not rewritten to chosen table"
    )

    # Check subquestion tables narrowed correctly (now test_1_1)
    sq1 = resolved["subquestions"][0]
    assert sq1["tables"] == ["test_1_1"], f"SQ1 tables not narrowed correctly: {sq1['tables']}"

    # Check SQ2 remains test_3_1 (already single table, no narrowing needed)
    sq2 = resolved["subquestions"][1]
    assert sq2["tables"] == ["test_3_1"], f"SQ2 tables should remain test_3_1: {sq2['tables']}"

    # Check SQ3 remains test_2_1 (no overlap with transaction_id options)
    sq3 = resolved["subquestions"][2]
    assert sq3["tables"] == ["test_2_1"], (
        f"SQ3 tables should remain test_2_1 (no overlap): {sq3['tables']}"
    )

    # Check SQ4 switched from test_2_1 to test_1_1 (collapsed handling)
    sq4 = resolved["subquestions"][3]
    assert sq4["tables"] == ["test_1_1"], f"SQ4 tables should switch to test_1_1: {sq4['tables']}"

    # Verify column_table resolutions are properly deduped and contextualized
    column_resolutions = [r for r in resolved["resolutions_applied"] if r["kind"] == "column_table"]

    # Check exactly 1 metric_rewrite for payment_amount -> test_1_1
    metric_rewrites = [
        r
        for r in column_resolutions
        if r.get("context") == "metric_rewrite"
        and r["column"] == "payment_amount"
        and r["chosen_table"] == "test_1_1"
    ]
    assert len(metric_rewrites) == 1, (
        f"Expected exactly 1 metric_rewrite, got {len(metric_rewrites)}"
    )

    # Check exactly 1 subquestion_switch for SQ1
    sq1_switches = [
        r
        for r in column_resolutions
        if r.get("context") == "subquestion_switch"
        and r.get("subquestion_id") == "SQ1"
        and r["column"] == "payment_amount"
        and r["chosen_table"] == "test_1_1"
    ]
    assert len(sq1_switches) == 1, f"Expected exactly 1 SQ1 switch, got {len(sq1_switches)}"
    # Verify original_table is captured
    assert "original_table" in sq1_switches[0], "Missing original_table in SQ1"

    # Check exactly 1 subquestion_switch for SQ4
    sq4_switches = [
        r
        for r in column_resolutions
        if r.get("context") == "subquestion_switch"
        and r.get("subquestion_id") == "SQ4"
        and r["column"] == "payment_amount"
        and r["chosen_table"] == "test_1_1"
    ]
    assert len(sq4_switches) == 1, f"Expected exactly 1 SQ4 switch, got {len(sq4_switches)}"
    # Verify original_table is test_2_1 (collapsed case)
    assert sq4_switches[0]["original_table"] == "test_2_1", (
        f"SQ4 original_table should be test_2_1, got {sq4_switches[0]['original_table']}"
    )

    print("✅ _demo_apply_resolutions: All assertions passed (deduped logging)")
    res_count = len(resolved["resolutions_applied"])
    print(f"Resolutions applied ({res_count}): {resolved['resolutions_applied']}")


def _demo_summarize_applied_resolutions() -> None:
    """Inline demo showing summarize_applied_resolutions output."""
    sample_resolutions = [
        {"kind": "entity_table", "entity": "customer", "chosen_table": "test_1_1"},
        {
            "kind": "column_table",
            "column": "payment_amount",
            "chosen_table": "test_1_1",
            "original": "test_2_1.payment_amount",
            "context": "metric_rewrite",
        },
        {
            "kind": "column_table",
            "column": "payment_amount",
            "chosen_table": "test_1_1",
            "original": "test_2_1.payment_amount",
            "context": "subquestion_switch",
            "subquestion_id": "SQ1",
            "original_table": "test_2_1",
        },
    ]

    summary = summarize_applied_resolutions(sample_resolutions)
    print("\n📋 Applied Resolutions Summary Demo:")
    for i, line in enumerate(summary, 1):
        print(f"  {i}. {line}")


def save_result(result: dict, output_path: Path) -> None:
    """Save execution result to JSON file."""
    import json

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
