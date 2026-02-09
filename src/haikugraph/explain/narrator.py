"""Narrator for converting execution results into user-facing explanations.

This module provides intent-aware narration that runs after SQL execution.
It converts structured data into human-readable text without changing planner
or executor logic.

Narration Rules:
- Never called if any subquestion failed (short-circuit to error)
- Intent-aware output (metric, grouped_metric, comparison, lookup, diagnostic)
- Only describes what exists in results (no speculation)
- No SQL, no schema, no column speculation
- Deterministic with repair loop for invalid JSON
"""

import json
import os
from typing import Any

from haikugraph.llm.router import call_llm
from haikugraph.planning.intent import Intent, IntentType


# Narrator prompt templates

NARRATOR_SYSTEM_PROMPT = """You are a data narrator. Convert query results into clear, user-facing explanations.

Rules:
- Output ONLY valid JSON: {"text": "<explanation>"}
- Describe ONLY what exists in results (no speculation)
- No SQL, no schema, no technical jargon
- No markdown, no formatting beyond plain text
- Be concise and direct"""

NARRATOR_USER_PROMPT_TEMPLATE = """Explain these query results to the user.

Original Question: {question}

Intent: {intent_type}
{intent_rationale}

Results:
{results_summary}

Instructions based on intent:

{intent_instructions}

General rules:
- Describe data generically ("revenue", "appointments") based on context
- For comparisons: explicit contrast (increase/decrease, delta if calculable)
- For groups: bullet list or short description
- For lookups: describe what was listed + count
- For diagnostics: cautious, descriptive, no causal claims
- Include row counts where relevant

Output format (JSON only):
{{
  "text": "<clear, 1-3 sentence explanation>"
}}

Generate the explanation now. Output ONLY the JSON object."""

NARRATOR_REPAIR_PROMPT_TEMPLATE = """The previous output had errors. Fix it.

Previous output:
{previous_output}

Errors:
{errors}

Rules:
- Output ONLY valid JSON (no markdown)
- Required: text field (string)
- No extra fields

Return ONLY the corrected JSON."""


# Intent-specific instructions
INTENT_INSTRUCTIONS = {
    IntentType.METRIC: """
METRIC: Single aggregated value
- One summary sentence with the value
- Example: "Total revenue is $X" or "There are X appointments"
""",
    IntentType.GROUPED_METRIC: """
GROUPED_METRIC: Aggregation by dimension
- Bullet list or short table description
- Mention top values if many rows
- Example: "Revenue by barber: Alice ($X), Bob ($Y), Charlie ($Z)"
""",
    IntentType.COMPARISON: """
COMPARISON: Contrast between periods/cohorts
- A11 REQUIREMENT: Use ONLY the normalized comparison structure provided
- NEVER compute delta or percentage yourself
- Use pre-computed current.value, comparison.value, delta, delta_pct, and direction
- Format: "<metric> <current_period> (<current_value>) vs <comparison_period> (<comparison_value>) - <direction> by <delta> (<delta_pct>%)"
- If delta_pct is null (division by zero), omit percentage
- Example: "Revenue this_year ($30,000) vs previous_year ($25,000) - up by $5,000 (20%)"
- Example with flat: "Revenue this_month ($1,000) vs previous_month ($1,000) - flat"
- Example with zero base: "Revenue this_month ($100) vs previous_month ($0) - up by $100"
""",
    IntentType.LOOKUP: """
LOOKUP: Raw rows listing
- Describe what was listed
- Include row count
- Example: "Found X recent appointments. First few: ..."
""",
    IntentType.DIAGNOSTIC: """
DIAGNOSTIC: Health/anomalies/gaps
- Cautious, descriptive tone
- NO causal claims (no "because", "due to")
- Describe patterns observed
- Example: "Revenue shows X pattern. Observed Y in period Z."
""",
    IntentType.UNKNOWN: """
UNKNOWN: Default explanation
- Simple description of results
- Mention row count and key values
""",
}


def narrate_results(
    original_question: str,
    intent: Intent | None,
    plan: dict,
    results: dict[str, dict],
    *,
    comparison: dict | None = None,
    max_retries: int = 1,
    _legacy_mode: bool = False,
) -> str:
    """Convert execution results into user-facing explanation.
    
    This function runs AFTER SQL execution and converts structured data
    into human-readable text. It is intent-aware and never invents data.
    
    A11 Update: For comparison queries, this function MUST receive a normalized
    comparison structure. The narrator is FORBIDDEN from computing deltas or
    inferring trends - all math is done before narration.
    
    Args:
        original_question: User's original question
        intent: Intent classification result (A8) or None
        plan: Validated plan dict
        results: Dict keyed by subquestion_id, each value contains:
                 - rows: list of dicts
                 - columns: list of column names
                 - row_count: int
                 - error: str (optional, if execution failed)
        comparison: A11 normalized comparison dict (if this is a comparison query)
                    Contains: metric, current, comparison, delta, delta_pct, direction
        max_retries: Maximum repair attempts (default 1)
    
    Returns:
        User-facing explanation text
    
    Raises:
        ValueError: If narration fails after all repair attempts
    """
    # Check if any subquestion failed - short-circuit if so
    failed_subquestions = []
    for sq_id, result in results.items():
        if result.get("error"):
            failed_subquestions.append({
                "id": sq_id,
                "error": result["error"][:200]  # Truncate long errors
            })
    
    # Use LLM to explain failures in user-friendly way with suggestions
    if failed_subquestions:
        return _narrate_failure(
            original_question=original_question,
            failed_subquestions=failed_subquestions,
            plan=plan,
            max_retries=max_retries
        )
    
    # Allow override from environment
    max_retries = int(os.environ.get("HG_NARRATOR_MAX_RETRIES", str(max_retries)))
    
    # A11: For comparison queries, use normalized comparison structure instead of raw results
    if comparison:
        # Narrator MUST use pre-computed comparison values - no math allowed
        results_summary = _build_comparison_summary(comparison)
    else:
        # Build results summary from raw SQL results
        results_summary = _build_results_summary(results)
    
    # Get intent-specific instructions
    intent_type = intent.type if intent else IntentType.UNKNOWN
    intent_instructions = INTENT_INSTRUCTIONS.get(intent_type, INTENT_INSTRUCTIONS[IntentType.UNKNOWN])
    
    intent_rationale = ""
    if intent:
        intent_rationale = f"Rationale: {intent.rationale}"
    
    # Build initial prompt
    user_prompt = NARRATOR_USER_PROMPT_TEMPLATE.format(
        question=original_question,
        intent_type=intent_type.value if intent else "unknown",
        intent_rationale=intent_rationale,
        results_summary=results_summary,
        intent_instructions=intent_instructions,
    )
    
    messages = [
        {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # Retry loop
    for attempt in range(max_retries + 1):
        # Call narrator LLM (uses temperature=0.2 for slight creativity)
        response = call_llm(messages, role="narrator")
        
        # Legacy mode: accept plain text (for backward compatibility)
        if _legacy_mode:
            return response.strip()
        
        # Try to parse JSON
        try:
            narration_dict = _parse_json_strict(response)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                # Repair: JSON parse error
                repair_prompt = NARRATOR_REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=response,
                    errors=f"JSON parse error: {e}",
                )
                messages = [
                    {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                raise ValueError(
                    f"Failed to parse valid JSON from narrator after {max_retries} retries. "
                    f"Last error: {e}"
                ) from e
        
        # Try to extract text
        try:
            text = narration_dict.get("text")
            if not text or not isinstance(text, str):
                raise ValueError("Missing or invalid 'text' field")
            # Success!
            return text.strip()
        except Exception as e:
            if attempt < max_retries:
                # Repair: validation error
                repair_prompt = NARRATOR_REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=json.dumps(narration_dict, indent=2),
                    errors=str(e),
                )
                messages = [
                    {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                raise ValueError(
                    f"Narration validation failed after {max_retries} retries.\n{e}"
                ) from e
    
    # Should not reach here
    raise ValueError("Unexpected error in narration")


def _narrate_failure(
    original_question: str,
    failed_subquestions: list[dict],
    plan: dict,
    max_retries: int = 1,
) -> str:
    """Use LLM to explain query failures in user-friendly language.
    
    This function converts technical error messages into plain English
    and suggests alternative questions based on available data.
    
    Args:
        original_question: User's original question
        failed_subquestions: List of failed subquestion dicts with id and error
        plan: The plan that was executed
        max_retries: Maximum repair attempts
    
    Returns:
        User-friendly error explanation with suggestions
    """
    # Extract table names from plan
    tables_used = []
    for sq in plan.get("subquestions", []):
        tables_used.extend(sq.get("tables", []))
    tables_used = list(set(tables_used))
    
    # Build error context
    error_details = []
    for failed in failed_subquestions:
        error_msg = failed["error"]
        error_details.append(f"- {error_msg}")
    
    error_summary = "\n".join(error_details)
    
    # Failure explanation prompt
    failure_prompt = f"""The user asked: "{original_question}"

The query failed with these errors:
{error_summary}

Tables involved: {', '.join(tables_used) if tables_used else 'unknown'}

Your task:
1. Explain what went wrong in simple, non-technical language
2. Identify the likely issue (missing column, wrong table, data type mismatch, etc.)
3. Suggest 2-3 alternative questions the user could try based on the tables available

Rules:
- Use plain English, avoid technical jargon like "column", "table", "SQL"
- Be helpful and constructive, not apologetic
- Frame suggestions as "You could try asking..."
- Keep it concise (2-3 sentences for explanation + bullet list of suggestions)

Output format (JSON only):
{{
  "text": "<explanation>\n\nYou could try asking:\n- <suggestion 1>\n- <suggestion 2>\n- <suggestion 3>"
}}

Generate the explanation now. Output ONLY the JSON object."""
    
    messages = [
        {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": failure_prompt},
    ]
    
    # Retry loop
    for attempt in range(max_retries + 1):
        try:
            response = call_llm(messages, role="narrator")
            narration_dict = _parse_json_strict(response)
            text = narration_dict.get("text")
            
            if not text or not isinstance(text, str):
                raise ValueError("Missing or invalid 'text' field")
            
            return text.strip()
            
        except Exception as e:
            if attempt < max_retries:
                # Repair attempt
                repair_prompt = NARRATOR_REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=response if 'response' in locals() else "<no output>",
                    errors=str(e),
                )
                messages = [
                    {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                # Fallback to basic error message if LLM fails
                error_lines = ["I couldn't answer your question because:"]
                for failed in failed_subquestions:
                    # Extract key part of error
                    error = failed["error"]
                    if "does not have a column" in error:
                        # Extract column name from error
                        parts = error.split('"')
                        if len(parts) >= 2:
                            col_name = parts[-2]
                            error_lines.append(f"- The data doesn't include information about '{col_name}'")
                    elif "Binder Error" in error:
                        error_lines.append("- The query referenced data that doesn't exist")
                    else:
                        error_lines.append(f"- Technical issue: {error[:100]}")
                
                error_lines.append("\nTry asking about different aspects of your data.")
                return "\n".join(error_lines)
    
    # Should not reach here
    return "I couldn't answer your question due to a data structure mismatch."


def _build_comparison_summary(comparison: dict) -> str:
    """Build a summary from normalized comparison structure (A11).
    
    This function formats the pre-computed comparison values for narrator.
    Narrator MUST NOT compute delta or delta_pct - only format what's provided.
    
    Args:
        comparison: Normalized comparison dict with:
                   - metric: str
                   - current: {value, period, subquestion_id, row_count}
                   - comparison: {value, period, subquestion_id, row_count}
                   - delta: float
                   - delta_pct: float | null
                   - direction: "up" | "down" | "flat"
    
    Returns:
        Formatted comparison summary
    """
    metric = comparison.get("metric", "unknown_metric")
    current = comparison.get("current", {})
    comp = comparison.get("comparison", {})
    delta = comparison.get("delta", 0)
    delta_pct = comparison.get("delta_pct")
    direction = comparison.get("direction", "unknown")
    
    current_value = current.get("value", 0)
    current_period = current.get("period", "current")
    comp_value = comp.get("value", 0)
    comp_period = comp.get("period", "previous")
    
    lines = [
        f"Metric: {metric}",
        f"Current period ({current_period}): {current_value}",
        f"Comparison period ({comp_period}): {comp_value}",
        f"Delta: {delta}",
    ]
    
    if delta_pct is not None:
        lines.append(f"Percentage change: {delta_pct:.2f}%")
    else:
        lines.append("Percentage change: N/A (division by zero)")
    
    lines.append(f"Direction: {direction}")
    
    return "\n".join(lines)


def _build_results_summary(results: dict[str, dict]) -> str:
    """Build a concise text summary of results for the narrator.
    
    Args:
        results: Dict keyed by subquestion_id
    
    Returns:
        Text summary of results
    """
    lines = []
    
    for sq_id, result in results.items():
        lines.append(f"\n{sq_id}:")
        
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        row_count = result.get("row_count", len(rows))
        
        lines.append(f"  Row count: {row_count}")
        
        if columns:
            lines.append(f"  Columns: {', '.join(columns)}")
        
        # Show first few rows for context
        if rows:
            lines.append("  Sample data:")
            for i, row in enumerate(rows[:5], 1):
                # Format row as key: value pairs
                if isinstance(row, dict):
                    row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                    lines.append(f"    {i}. {row_str}")
                else:
                    lines.append(f"    {i}. {row}")
            
            if len(rows) > 5:
                lines.append(f"    ... ({len(rows) - 5} more rows)")
    
    return "\n".join(lines) if lines else "No results"


def narrate(
    question: str,
    plan: dict,
    results: dict[str, Any],
    meta: dict[str, Any],
    subquestion_results: list[dict] | None = None,
) -> str:
    """Backward-compatible wrapper for narrate_results().
    
    This function provides compatibility with the old narrator interface.
    It converts the old format to the new format and calls narrate_results().
    
    Args:
        question: Original user question
        plan: Final validated plan dict
        results: Per-subquestion results
        meta: Per-subquestion metadata
        subquestion_results: Optional full subquestion result list (with status)
    
    Returns:
        Natural language explanation text
    """
    # Convert old format to new format
    new_results = {}
    
    if subquestion_results:
        # Check for failures using subquestion_results
        for sq_result in subquestion_results:
            sq_id = sq_result.get("id", "unknown")
            
            if sq_result.get("status") != "success":
                # Failed subquestion
                new_results[sq_id] = {
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "error": sq_result.get("error", "Unknown error")
                }
            else:
                # Successful subquestion
                new_results[sq_id] = {
                    "rows": results.get(sq_id, []),
                    "columns": list(results.get(sq_id, [{}])[0].keys()) if results.get(sq_id) else [],
                    "row_count": sq_result.get("row_count", 0)
                }
    else:
        # No subquestion_results - assume success
        for sq_id, result_data in results.items():
            if isinstance(result_data, list):
                new_results[sq_id] = {
                    "rows": result_data,
                    "columns": list(result_data[0].keys()) if result_data else [],
                    "row_count": len(result_data)
                }
            else:
                new_results[sq_id] = {
                    "rows": [result_data] if result_data else [],
                    "columns": [],
                    "row_count": 1 if result_data else 0
                }
    
    # Call new narrate_results (no intent in old API, legacy mode for plain text)
    return narrate_results(
        original_question=question,
        intent=None,
        plan=plan,
        results=new_results,
        _legacy_mode=True  # Accept plain text responses for backward compatibility
    )


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
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    return json.loads(text)
