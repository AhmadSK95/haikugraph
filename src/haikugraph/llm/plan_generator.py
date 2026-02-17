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

    Includes uniqueness annotations for _id columns so the LLM planner
    knows when to use COUNT(DISTINCT ...) instead of COUNT(...).

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

            # Get row count and distinct counts for _id columns
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

            # ── Auto-detect table business domains from column signatures ──
            col_names = {ci[0].lower() for ci in columns_result}

            def _detect_domains(cn: set[str]) -> tuple[list[str], str]:
                """Return (domain_list, use_for_hint)."""
                domains = []
                use_for = []
                if "transaction_id" in cn:
                    domains.append("transactions")
                    use_for.append("transaction counts")
                    use_for.append("transaction amounts")
                if "customer_id" in cn and len(cn) > 15:
                    domains.append("customers")
                    use_for.append("customer queries")
                if {"payment_amount", "payment_status"} & cn:
                    domains.append("payments")
                if {"mt103_document_id", "mt103_created_at"} & cn:
                    domains.append("mt103")
                if {"invoice_document_document_id"} & cn:
                    domains.append("invoices")
                if {"refund_refund_id"} & cn:
                    domains.append("refunds")
                if "platform_name" in cn:
                    domains.append("platforms")
                    use_for.append("platform breakdowns")
                if {"total_amount_to_be_paid", "amount_at_source"} & cn:
                    domains.append("settlements")
                    use_for.append("revenue/settlement totals")
                if {"exchange_rate", "from_currency", "to_currency"} & cn:
                    domains.append("forex")
                    use_for.append("forex/exchange rate queries")
                if {"total_gst", "swift_charges", "platform_charges"} & cn:
                    domains.append("charges")
                    use_for.append("fee/charge breakdowns")
                if "payee_id" in cn and {"address_city", "address_country"} & cn:
                    domains.append("payees/beneficiaries")
                    use_for.append("payee/beneficiary lookups")
                if {"booked_amount", "booked_at"} & cn:
                    domains.append("bookings")
                    use_for.append("booking/deal queries")
                if {"deal_id", "deal_type"} & cn:
                    domains.append("deals")
                return domains, use_for

            domains, use_for = _detect_domains(col_names)
            if domains:
                domain_tag = f" [DOMAINS: {', '.join(domains)}]"
                if use_for:
                    domain_tag += f" [USE THIS TABLE for: {', '.join(use_for)}]"
            else:
                domain_tag = ""
            schema_parts.append(f"\nTable: {table} ({row_count:,} rows){domain_tag}")

            # Detect timestamp columns and pick the best one using
            # semantic name understanding:
            #   "created_at"   = when the record was created (event time) — best default
            #   "booked_at"    = domain event time — good
            #   "*_created_at" = sub-entity creation — okay but narrower
            #   "expires_at"   = expiration — NOT event time
            #   "updated_at"   = last edit — NOT event time
            #   "value_date"   = financial settlement — secondary
            def _ts_name_priority(name: str) -> int:
                """Higher = better candidate for the primary time filter."""
                nl = name.lower()
                if nl == "created_at":
                    return 100          # canonical event timestamp
                if nl in ("booked_at", "ordered_at", "transacted_at"):
                    return 90           # clear domain-event columns
                if nl.endswith("_date") and "value" not in nl and "expir" not in nl:
                    return 70           # generic date columns
                if nl.endswith("_created_at"):
                    return 50           # sub-entity timestamps (payment_created_at)
                if "expir" in nl or "updated" in nl:
                    return 10           # not event time
                return 30               # anything else

            ts_candidates = []  # (col_name, non_null_count, name_priority)
            for col_info in columns_result:
                cn = col_info[0]
                ct = col_info[1]
                is_ts_name = (
                    cn.lower().endswith("_at")
                    or cn.lower().endswith("_date")
                    or cn.lower() == "created_at"
                )
                is_ts_type = any(t in ct.upper() for t in ["TIMESTAMP", "DATE"])
                if is_ts_name or is_ts_type:
                    try:
                        non_null = conn.execute(
                            f'SELECT COUNT(*) FROM "{table}" '
                            f'WHERE TRY_CAST("{cn}" AS TIMESTAMP) IS NOT NULL'
                        ).fetchone()[0]
                        ts_candidates.append((cn, non_null, _ts_name_priority(cn)))
                    except Exception:
                        pass

            # Sort: name priority first, then non-null coverage as tiebreaker
            best_ts = None
            if ts_candidates:
                ts_candidates.sort(
                    key=lambda x: (x[2], x[1]),
                    reverse=True,
                )
                best_ts = ts_candidates[0][0]

            # ── Detect money / amount columns and pick PRIMARY AMOUNT ──
            # Semantic name priority (higher = better candidate for
            # "transaction amount" or "revenue" queries):
            #   "payment_amount"       → canonical payment value
            #   "amount" (standalone)  → generic but direct
            #   "total_amount*"        → totals (may include fees)
            #   "amount_collected"     → downstream collection amount
            #   "amount_at_source"     → source-side amount
            #   "*_charges", "*_fee"   → fee/surcharge sub-amounts
            #   "refund_*"             → refund amounts
            def _money_name_priority(name: str) -> int:
                """Higher = better candidate for the primary amount column."""
                nl = name.lower()
                if nl == "payment_amount":
                    return 100          # canonical transaction payment amount
                if nl == "amount":
                    return 95           # generic standalone amount
                if nl == "transaction_amount":
                    return 98           # explicit transaction amount
                if nl.startswith("total_amount"):
                    return 80           # total including fees
                if nl == "amount_collected":
                    return 60           # collected (downstream)
                if nl == "amount_at_source":
                    return 55           # source-side
                if "deal" in nl and "amount" in nl:
                    return 30           # deal-level (different entity)
                if any(w in nl for w in ("charge", "fee", "surcharge", "tax", "commission")):
                    return 20           # fee/charge sub-amounts
                if nl.startswith("refund"):
                    return 15           # refund amounts
                if "amount" in nl or "price" in nl or "revenue" in nl:
                    return 50           # other amount-like columns
                return 0                # not a money column

            money_candidates = []  # (col_name, non_null_pct, avg_val, name_priority)
            for col_info in columns_result:
                cn = col_info[0]
                prio = _money_name_priority(cn)
                if prio <= 0:
                    continue
                try:
                    stats = conn.execute(
                        f'SELECT '
                        f'COUNT(*) FILTER (WHERE TRY_CAST("{cn}" AS DOUBLE) IS NOT NULL), '
                        f'AVG(TRY_CAST("{cn}" AS DOUBLE)) '
                        f'FROM "{table}"'
                    ).fetchone()
                    non_null = stats[0] or 0
                    avg_val = stats[1]
                    non_null_pct = round(non_null / row_count * 100, 1) if row_count > 0 else 0
                    # Skip columns with very low coverage (<5%) or NULL avg
                    if non_null_pct < 5 or avg_val is None:
                        continue
                    money_candidates.append((cn, non_null_pct, avg_val, prio))
                except Exception:
                    pass

            # Pick PRIMARY AMOUNT: name priority first, then coverage
            best_money = None
            if money_candidates:
                money_candidates.sort(
                    key=lambda x: (x[3], x[1]),
                    reverse=True,
                )
                best_money = money_candidates[0][0]

            schema_parts.append("Columns:")
            for col_info in columns_result:
                col_name = col_info[0]
                col_type = col_info[1]

                # For _id columns, check uniqueness and annotate
                annotation = ""
                if col_name.lower().endswith("_id") and row_count > 0:
                    try:
                        distinct = conn.execute(
                            f'SELECT approx_count_distinct("{col_name}") FROM "{table}"'
                        ).fetchone()[0]
                        if distinct < 0.98 * row_count:
                            dup_ratio = round(row_count / distinct, 1)
                            annotation = f" [NOT UNIQUE — {distinct:,} distinct / {row_count:,} rows, {dup_ratio}x duplication. Use COUNT(DISTINCT) when counting.]"
                        else:
                            annotation = " [UNIQUE]"
                    except Exception:
                        pass

                # For timestamp columns, annotate coverage and primary flag
                ts_entry = next((c for c in ts_candidates if c[0] == col_name), None)
                if ts_entry:
                    non_null = ts_entry[1]
                    pct = round(non_null / row_count * 100, 1) if row_count > 0 else 0
                    if col_name == best_ts:
                        annotation += f" [PRIMARY TIMESTAMP — {pct}% populated. USE THIS for time filters]"
                    else:
                        annotation += f" [timestamp — {pct}% populated]"

                # For money / amount columns, annotate coverage and primary flag
                money_entry = next((c for c in money_candidates if c[0] == col_name), None)
                if money_entry:
                    m_pct = money_entry[1]
                    m_avg = money_entry[2]
                    if col_name == best_money:
                        annotation += f" [PRIMARY AMOUNT — {m_pct}% populated, avg={m_avg:,.0f}. USE THIS for transaction amount / revenue queries]"
                    else:
                        annotation += f" [amount — {m_pct}% populated, avg={m_avg:,.0f}]"

                schema_parts.append(f"  - {col_name} ({col_type}){annotation}")

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
