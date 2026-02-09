"""LLM-powered plan generator for HaikuGraph.

This module generates Plan JSON objects from natural language questions
by introspecting database schema and using an LLM to produce structured plans.
"""

import json
from pathlib import Path
from typing import Any

import duckdb

from haikugraph.llm.client import call_openai, parse_json_response
from haikugraph.planning.schema import validate_plan


def generate_plan(
    question: str,
    db_path: Path,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Generate a Plan JSON from a natural language question.

    This function:
    1. Introspects the database schema
    2. Calls LLM to generate a Plan JSON
    3. Validates the plan against the schema
    4. If invalid, attempts repair with LLM (max 2 retries)
    5. Returns the validated plan dict

    Args:
        question: Natural language question about the data
        db_path: Path to DuckDB database
        model: LLM model name (optional, defaults to gpt-4o-mini)

    Returns:
        Validated Plan dict conforming to schema.py

    Raises:
        ValueError: If plan validation fails after all repair attempts
        FileNotFoundError: If database file not found
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Step 1: Introspect schema
    schema_text = introspect_schema(db_path)

    # Step 2: Generate initial plan with LLM
    initial_prompt = create_initial_plan_prompt(schema_text, question)
    response = call_openai(
        messages=[{"role": "user", "content": initial_prompt}],
        model=model,
        temperature=0.0,
    )

    # Step 3: Parse and validate
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            plan_dict = parse_json_response(response)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                # Try to repair JSON parse error
                repair_prompt = create_repair_prompt(
                    schema_text,
                    question,
                    response,
                    [f"JSON parse error: {e}"],
                )
                response = call_openai(
                    messages=[{"role": "user", "content": repair_prompt}],
                    model=model,
                    temperature=0.0,
                )
                continue
            else:
                raise ValueError(
                    f"Failed to parse valid JSON after {max_retries} retries. Last error: {e}"
                ) from e

        # Validate plan
        is_valid, errors = validate_plan(plan_dict)
        if is_valid:
            return plan_dict

        # Plan invalid, attempt repair if retries remain
        if attempt < max_retries:
            repair_prompt = create_repair_prompt(
                schema_text,
                question,
                json.dumps(plan_dict, indent=2),
                errors,
            )
            response = call_openai(
                messages=[{"role": "user", "content": repair_prompt}],
                model=model,
                temperature=0.0,
            )
        else:
            # Final attempt failed
            raise ValueError(
                f"Plan validation failed after {max_retries} retries.\n"
                f"Errors:\n" + "\n".join(f"  - {err}" for err in errors)
            )

    # Should not reach here
    raise ValueError("Unexpected error in plan generation")


def introspect_schema(db_path: Path) -> str:
    """
    Introspect database schema and return formatted text.

    Args:
        db_path: Path to DuckDB database

    Returns:
        Formatted schema text with tables and columns
    """
    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        # Get list of tables
        tables_result = conn.execute("PRAGMA show_tables").fetchall()
        tables = [row[0] for row in tables_result]

        schema_parts = [f"Database contains {len(tables)} tables:\n"]

        for table in tables:
            # Get column info
            columns_result = conn.execute(f'DESCRIBE "{table}"').fetchall()

            schema_parts.append(f"\nTable: {table}")
            schema_parts.append("Columns:")
            for col_info in columns_result:
                col_name = col_info[0]
                col_type = col_info[1]
                schema_parts.append(f"  - {col_name} ({col_type})")

            # Sample a few rows for type hints (optional, keep minimal)
            try:
                sample = conn.execute(f'SELECT * FROM "{table}" LIMIT 2').fetchall()
                if sample:
                    schema_parts.append(f"  Sample row count: {len(sample)}")
            except Exception:
                pass  # Skip if sampling fails

    finally:
        conn.close()

    return "\n".join(schema_parts)


def create_initial_plan_prompt(schema_text: str, question: str) -> str:
    """
    Create the initial prompt for plan generation.

    Args:
        schema_text: Introspected database schema
        question: User's natural language question

    Returns:
        Formatted prompt string
    """
    return f"""You are a data assistant that generates structured execution plans for
database queries.

Given a natural language question and database schema, generate a Plan JSON object
that conforms to this structure:

{{
  "original_question": "<the question>",
  "subquestions": [
    {{
      "id": "SQ1",
      "description": "Description of what this subquestion answers",
      "tables": ["table_name"],
      "columns": ["column1", "column2"],
      "group_by": ["column"] OR [{{"type": "time_bucket", "grain": "month|year|day", "col": "date_column"}}] (optional),
      "aggregations": [{{"agg": "sum|avg|count|min|max|count_distinct", "col": "column", "distinct": true (optional for count)}}] (optional)
    }}
  ],
  "intent": {{
    "type": "metric|lookup|comparison|diagnostic|other",
    "confidence": 0.0-1.0
  }},
  "entities_detected": [
    {{
      "name": "entity_name",
      "mapped_to": ["table.column"],
      "confidence": 0.0-1.0
    }}
  ] (optional),
  "metrics_requested": [
    {{
      "name": "metric_name",
      "mapped_columns": ["table.column"],
      "aggregation": "sum|avg|count|min|max",
      "confidence": 0.0-1.0
    }}
  ] (optional),
  "join_paths": [
    {{
      "from": "table1",
      "to": "table2",
      "via": ["join_column"],
      "confidence": 0.0-1.0,
      "cardinality": "one-to-many|many-to-one|one-to-one"
    }}
  ] (optional, only if multiple tables needed),
  "constraints": [
    {{
      "type": "time|filter",
      "expression": "table.column in last_30_days" or "table.col = 'value'"
    }}
  ] (optional),
  "ambiguities": [
    {{
      "issue": "Description of ambiguity",
      "recommended": "Recommended option",
      "options": ["option1", "option2"]
    }}
  ] (optional),
  "plan_confidence": 0.0-1.0
}}

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown, no comments, no explanations.
2. Every subquestion MUST have at least one table in the "tables" list.
3. If group_by is present and aggregations is present, aggregations must be non-empty.
4. Use deterministic subquestion IDs: SQ1, SQ2, etc.
5. For time constraints, use expressions like "table.column in last_30_days" or "last_7_days".
6. If multiple tables are used, include join_paths.
7. For ambiguities, use issue strings like:
   - "Entity 'X' found in multiple tables"
   - "Multiple tables contain column"
8. DISTINCT COUNTS:
   - For "unique customers", "distinct values": use {{"agg": "count", "col": "column", "distinct": true}}
   - OR use {{"agg": "count_distinct", "col": "column"}}
   - NEVER put "DISTINCT" in the column name: WRONG: {{"col": "DISTINCT customer_id"}}
   - Column names must be simple identifiers without spaces or SQL keywords
9. TIME BUCKETING (for "monthly", "by month", "month-wise", "trend"):
   - Use group_by with time_bucket: [{{"type": "time_bucket", "grain": "month", "col": "date_column"}}]
   - Supported grains: "month", "year", "day", "week", "quarter"
   - This produces monthly aggregation with ORDER BY month
   - Example: "monthly revenue" -> group_by: [{{"type": "time_bucket", "grain": "month", "col": "created_at"}}]

DATABASE SCHEMA:
{schema_text}

USER QUESTION:
{question}

Generate the Plan JSON now:"""


def create_repair_prompt(
    schema_text: str,
    question: str,
    invalid_plan: str,
    validation_errors: list[str],
) -> str:
    """
    Create a repair prompt when validation fails.

    Args:
        schema_text: Database schema
        question: Original question
        invalid_plan: The invalid plan JSON or text
        validation_errors: List of validation error messages

    Returns:
        Formatted repair prompt
    """
    errors_text = "\n".join(f"  - {err}" for err in validation_errors)

    return f"""The previous Plan JSON has validation errors. Please fix them and
output a corrected Plan JSON.

VALIDATION ERRORS:
{errors_text}

ORIGINAL QUESTION:
{question}

DATABASE SCHEMA:
{schema_text}

PREVIOUS ATTEMPT:
{invalid_plan}

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown, no comments.
2. Every subquestion MUST have at least one table in "tables" list.
3. If group_by exists and aggregations exists, aggregations must be non-empty.
4. All required fields must be present: original_question, subquestions.
5. Fix ALL the validation errors listed above.

Generate the CORRECTED Plan JSON now:"""
