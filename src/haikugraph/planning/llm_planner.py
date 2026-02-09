"""LLM-based planner with strict JSON validation and auto-repair.

This module uses a local LLM (planner role) to generate and repair Plan JSON
that strictly conforms to the schema defined in schema.py.
"""

import json
import os
from typing import Any

from haikugraph.llm.router import call_llm
from haikugraph.planning.intent import classify_intent, Intent
from haikugraph.planning.schema import validate_plan_or_raise


# System prompt for planner LLM - embedded as constant
PLANNER_SYSTEM_PROMPT = """You are a strict planning engine. Output ONLY valid JSON. No markdown. No commentary."""

# User prompt template for initial plan generation
PLANNER_USER_PROMPT_TEMPLATE = """Generate a Plan JSON for the following question:

Question: {question}

Database schema:
{schema}

{context}

Rules for Plan JSON:
1. Output a single JSON object representing a Plan
2. Required fields:
   - original_question (string): the user's question
   - subquestions (non-empty list): at least one subquestion
3. Each subquestion MUST have:
   - id (string): unique identifier like "SQ1", "SQ2"
   - tables (non-empty list): at least one table name
   - Optional: description, columns, group_by, aggregations
4. Subquestion IDs MUST be unique
5. Aggregations MUST be objects with "agg" and "col" fields:
   - CORRECT: {{"agg": "sum", "col": "payment_amount"}}
   - WRONG: "SUM(payment_amount)"
   - WRONG: {{"type": "sum", "column": "payment_amount"}}
   - Valid agg values: "sum", "avg", "count", "min", "max", "count_distinct"
   - For DISTINCT counts: {{"agg": "count", "col": "customer_id", "distinct": true}}
   - NEVER put SQL keywords in column names: WRONG {{"col": "DISTINCT customer_id"}}
6. For comparison queries, create exactly TWO subquestions:
   - SQ1_current (current period/cohort)
   - SQ2_comparison (comparison period/cohort)
   - DEFAULT: NO GROUP BY (produces two scalar totals for comparison)
   - ONLY add group_by when explicitly asked for time-series ("monthly trend", "by month", "over time")
   - Examples:
     * "revenue this month vs last month" → TWO scalars (no group_by)
     * "this year vs last year revenue" → TWO scalars (no group_by)
     * "monthly revenue this year vs last year" → monthly group_by (time-series comparison)
7. Constraints (optional list at TOP LEVEL of Plan, NOT inside subquestions):
   - type: "time" or "filter"
   - expression: constraint expression
   - applies_to: subquestion ID (required ONLY for comparison queries with multiple subquestions)
   - For single-subquestion plans, constraints go at top level WITHOUT applies_to
8. CRITICAL - Comparison time scoping rule:
   - For comparison queries (SQ1_current + SQ2_comparison), EVERY time constraint MUST have applies_to
   - BOTH subquestions MUST have their own scoped time constraint
   - NEVER leave time constraints unscoped in comparison queries
   - CORRECT: {{"type": "time", "expression": "this_year", "applies_to": "SQ1_current"}}
   - CORRECT: {{"type": "time", "expression": "previous_year", "applies_to": "SQ2_comparison"}}
   - WRONG: {{"type": "time", "expression": "this_year"}} (missing applies_to)
9. Any constraint with applies_to MUST match an existing subquestion id
10. Do not invent tables/columns outside schema unless explicitly present
11. TIME BUCKETING for "monthly", "by month", "month-wise" queries:
    - Use group_by with time_bucket dict: [{{"type": "time_bucket", "grain": "month", "col": "date_col"}}]
    - Supported grains: "month", "year", "day", "week", "quarter"
    - Example: "monthly unique customers" -> group_by: [{{"type": "time_bucket", "grain": "month", "col": "created_at"}}]
12. MONTH FILTERS for "in December", "during January", "[month name]" queries:
    - Add a constraint to filter by specific month
    - Constraint format: {{"type": "time_month", "expression": "table.date_col month=N", "month": N, "column": "date_col", "table": "table_name"}}
    - Month numbers: January=1, February=2, March=3, April=4, May=5, June=6, July=7, August=8, September=9, October=10, November=11, December=12
    - Examples:
      * "transactions in December" -> constraint: {{"type": "time_month", "expression": "orders.created_at month=12", "month": 12, "column": "created_at", "table": "orders"}}
      * "January payments" -> constraint: {{"type": "time_month", "expression": "payments.payment_date month=1", "month": 1, "column": "payment_date", "table": "payments"}}
13. SPLIT BY / BREAKDOWN queries for "split by X", "breakdown by X", "by each X", "per X":
    - These are GROUPED_METRIC queries, NOT comparisons
    - Use ONE subquestion (SQ1) with group_by containing the dimension column NAME (simple string)
    - Add aggregations for the metrics being split
    - CRITICAL: group_by should be simple column names like ["platform_name"], NOT time_bucket objects
    - Time filters go in constraints, NOT in group_by
    - Example: "revenue split by platform in December" ->
      {{
        "original_question": "revenue split by platform in December",
        "subquestions": [
          {{
            "id": "SQ1",
            "tables": ["orders"],
            "group_by": ["platform_name"],
            "aggregations": [{{"agg": "sum", "col": "revenue"}}]
          }}
        ],
        "constraints": [
          {{"type": "time_month", "expression": "orders.created_at month=12", "month": 12, "column": "created_at", "table": "orders"}}
        ]
      }}
    - WRONG: {{"group_by": [{{"type": "time_bucket", "grain": "month", "col": "created_at"}}]}} when question asks for "split by platform"
14. CRITICAL - Only answer what was asked:
    - Create ONLY the subquestions needed to answer the user's question
    - DO NOT add extra subquestions for related/tangential data
    - DO NOT fetch data from unrelated tables
    - Example: "revenue by platform" → ONLY query platform+revenue, DO NOT add customer addresses
    - Example: "count transactions" → ONLY count, DO NOT add revenue or customer details
    - Keep it minimal - one subquestion unless comparison or multi-step calculation required
15. Column names must be simple identifiers - NO spaces, NO SQL keywords like DISTINCT

EXAMPLE for "What is total revenue?":
{{
  "original_question": "What is total revenue?",
  "subquestions": [
    {{
      "id": "SQ1",
      "tables": ["orders"],
      "columns": ["revenue"],
      "aggregations": [{{"agg": "sum", "col": "revenue"}}]
    }}
  ]
}}

Output only the JSON object. Do not wrap in markdown code blocks."""

# Repair prompt template for fixing JSON/validation errors
REPAIR_PROMPT_TEMPLATE = """The previous JSON output had errors. Fix the JSON to satisfy the errors and rules.

Previous output:
{previous_output}

Errors:
{errors}

Rules reminder:
- Output ONLY valid JSON (no markdown)
- Required: original_question (string), subquestions (non-empty list)
- Subquestion IDs MUST be unique
- Aggregations MUST be objects: {{"agg": "sum", "col": "column_name"}}
  NEVER use strings like "SUM(column)" or objects with wrong field names
- Constraints go at TOP LEVEL, never inside aggregations array
- Constraints with applies_to MUST reference existing subquestion IDs
- For time-scoped constraints: {{"type": "time", "expression": "previous_month", "applies_to": "<valid_sq_id>"}}

Return ONLY the corrected JSON."""


def generate_or_patch_plan(
    question: str,
    schema: str,
    prev_plan: dict | None = None,
    prev_question: str | None = None,
    classification: dict | None = None,
    intent: Intent | None = None,
) -> dict:
    """Generate or patch a plan using LLM with strict validation.

    This function:
    1. Classifies intent (if not provided)
    2. Calls planner LLM with strict JSON-only prompt including intent context
    3. Attempts to parse JSON
    4. Validates against schema using validate_plan_or_raise
    5. On failure, enters repair loop (max HG_MAX_RETRIES attempts)
    6. Returns validated plan dict

    Args:
        question: User's natural language question
        schema: Database schema text
        prev_plan: Previous plan dict (for follow-ups)
        prev_question: Previous question (for follow-ups)
        classification: Follow-up classification (for context)
        intent: Pre-classified intent (optional, will classify if not provided)

    Returns:
        Validated plan dict conforming to schema.py

    Raises:
        ValueError: If plan validation fails after all repair attempts
    """
    max_retries = int(os.environ.get("HG_MAX_RETRIES", "2"))
    
    # Classify intent if not provided (skip for follow-ups to avoid redundant classification)
    if intent is None and not prev_plan:
        try:
            intent = classify_intent(question)
        except ValueError as e:
            # Intent classification failed - log but continue without intent context
            # This ensures planner remains functional even if intent classifier fails
            import sys
            print(f"Warning: Intent classification failed: {e}", file=sys.stderr)
            intent = None
    
    # Build context for follow-ups and intent
    context = ""
    
    # Add intent context if available (A8 integration)
    if intent:
        intent_hints = {
            "metric": "This is a METRIC query - single aggregated value without grouping.",
            "grouped_metric": "This is a GROUPED_METRIC query - aggregation WITH group_by dimension.",
            "comparison": "This is a COMPARISON query - generate SQ1_current and SQ2_comparison with scoped time constraints.",
            "lookup": "This is a LOOKUP query - retrieve raw rows without aggregation.",
            "diagnostic": "This is a DIAGNOSTIC query - health/gaps/anomalies, may need time-series.",
            "unknown": "Intent unclear - use best judgment based on question.",
        }
        intent_hint = intent_hints.get(intent.type.value, "")
        if intent_hint:
            context += f"\nIntent: {intent_hint}"
            context += f"\nRationale: {intent.rationale}"
        
        # For comparison intent, emphasize symmetric time scoping requirement
        if intent.requires_comparison or intent.type.value == "comparison":
            context += """

CRITICAL COMPARISON RULES:
- Create exactly TWO subquestions: SQ1_current and SQ2_comparison
- DEFAULT: NO group_by (produces two scalar totals for side-by-side comparison)
- ONLY add group_by if question explicitly requests time-series ("monthly", "by month", "trend", "over time")
- BOTH subquestions MUST have their OWN scoped time constraint
- EVERY time constraint MUST include applies_to field
- Example (SCALAR comparison - NO group_by):
  "this month vs last month revenue":
  "constraints": [
    {{"type": "time", "expression": "this_month", "applies_to": "SQ1_current"}},
    {{"type": "time", "expression": "previous_month", "applies_to": "SQ2_comparison"}}
  ]
  "subquestions": [
    {{"id": "SQ1_current", "tables": ["orders"], "aggregations": [{{"agg": "sum", "col": "revenue"}}]}},
    {{"id": "SQ2_comparison", "tables": ["orders"], "aggregations": [{{"agg": "sum", "col": "revenue"}}]}}
  ]
- NEVER create unscoped time constraints in comparison queries
"""
    
    # Add follow-up context
    if prev_plan and classification:
        followup_type = classification.get("type", "")
        if followup_type == "comparison":
            context += """
This is a COMPARISON follow-up. Generate:
- SQ1_current: original intent
- SQ2_comparison: same intent but time-shifted
- Add scoped time constraint: {{"type": "time", "expression": "previous_month", "applies_to": "SQ2_comparison"}}
"""
        else:
            context += f"\nThis is a follow-up of type: {followup_type}. Previous plan: {json.dumps(prev_plan, indent=2)}"
    
    # Initial prompt
    user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
        question=question,
        schema=schema,
        context=context,
    )
    
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # Retry loop
    for attempt in range(max_retries + 1):
        # Call planner LLM
        response = call_llm(messages, role="planner")
        
        # Try to parse JSON
        try:
            plan_dict = _parse_json_strict(response)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                # Repair: JSON parse error
                repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=response,
                    errors=f"JSON parse error: {e}",
                )
                messages = [
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                raise ValueError(
                    f"Failed to parse valid JSON after {max_retries} retries. "
                    f"Last error: {e}"
                ) from e
        
        # Try to validate plan
        try:
            validate_plan_or_raise(plan_dict)
            # Success!
            return plan_dict
        except ValueError as e:
            if attempt < max_retries:
                # Repair: validation error
                error_lines = str(e).split("\n")
                repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=json.dumps(plan_dict, indent=2),
                    errors="\n".join(error_lines),
                )
                messages = [
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                raise ValueError(
                    f"Plan validation failed after {max_retries} retries.\n{e}"
                ) from e
    
    # Should not reach here
    raise ValueError("Unexpected error in plan generation")


def _parse_json_strict(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    text = response.strip()
    
    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("```"):
                start_idx = i + 1
            elif i > 0 and line.strip().startswith("```"):
                end_idx = i
                break
        
        text = "\n".join(lines[start_idx:end_idx])
    
    return json.loads(text)
