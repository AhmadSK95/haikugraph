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

NARRATOR_SYSTEM_PROMPT = """You are a senior data analyst assistant. Your job is to present query results in a clear, detailed, and professional format that a business user can immediately understand and share.

You MUST:
- Present numbers with proper formatting (commas for thousands, 2 decimal places for currency)
- Use markdown tables when presenting multiple rows/columns of data
- Provide context and interpretation, not just raw numbers
- Respond as a knowledgeable assistant, not a chatbot
- Include ALL data rows in the results (do not truncate or summarize away data)
- Output ONLY valid JSON: {"text": "<explanation>"}
- Describe ONLY what exists in results (no speculation, no invented data)
- No SQL, no schema names, no technical database jargon"""

NARRATOR_USER_PROMPT_TEMPLATE = """Present these query results to a business user.

Original Question: {question}

Intent: {intent_type}
{intent_rationale}

Results:
{results_summary}

Instructions based on intent:

{intent_instructions}

Formatting rules:
- Start with a direct one-line answer to the question in **bold**
- For single values: state the value clearly, then add brief context (e.g. what period, what was counted)
- For grouped data: ALWAYS use a markdown table with headers. Include ALL rows.
- For comparisons: show both values, the delta, and percentage change clearly
- Format large numbers with commas (e.g. 1,962 not 1962)
- Format currency values appropriately (e.g. ₹83.14 Cr or ₹8,31,40,000)
- If results are empty or zero, say so explicitly and explain what it means
- Add a brief "Summary" line at the end with key takeaway
- Include row counts where relevant

Output format (JSON only):
{{
  "text": "<detailed explanation with tables and formatting>"
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
- State the answer in **bold** on the first line
- Include context: what metric, what time period, what entity
- If the value is 0 or NULL, explicitly state "No data found for this period" and explain why (e.g. no transactions recorded yet)
- Example: "**Total revenue in December 2025: ₹83.14 Cr**\n\nThis represents the sum of all transaction amounts recorded in December 2025 across 1,962 transactions."
""",
    IntentType.GROUPED_METRIC: """
GROUPED_METRIC: Aggregation by dimension
- State the overall finding in **bold** first
- ALWAYS present data in a markdown table with proper headers
- Include a total/summary row at the bottom if applicable
- Add brief interpretation (e.g. which segment dominates, notable patterns)
- Example:
  "**Revenue breakdown by platform (December 2025)**\n\n| Platform | Revenue | Transactions |\n|----------|---------|-------------|\n| B2C-APP | ₹7.2 Cr | 224 |\n| B2B | ₹33 L | 4 |\n\n**Summary:** B2C-APP dominates with 87% of total revenue."
""",
    IntentType.COMPARISON: """
COMPARISON: Contrast between periods/cohorts
- A11 REQUIREMENT: Use ONLY the normalized comparison structure provided
- NEVER compute delta or percentage yourself
- Present as a comparison table
- Use pre-computed current.value, comparison.value, delta, delta_pct, and direction
- Include a clear up/down indicator and percentage
- Example:
  "**Revenue Comparison: This Month vs Last Month**\n\n| Period | Revenue |\n|--------|---------|\n| This Month | ₹10.5 Cr |\n| Last Month | ₹8.2 Cr |\n| **Change** | **+₹2.3 Cr (+28%)** ⬆️ |\n\nRevenue increased by 28% compared to the previous month."
""",
    IntentType.LOOKUP: """
LOOKUP: Raw rows listing
- State what was found and the count
- Present data in a markdown table
- Show all available columns
- If many rows, show first 10-20 with a note about total count
- Example: "**Found 284 transactions**\n\n| Transaction ID | Amount | Date | Status |\n|...|...|...|...|\n\nShowing first 10 of 284 results."
""",
    IntentType.DIAGNOSTIC: """
DIAGNOSTIC: Health/anomalies/gaps
- Cautious, descriptive tone
- NO causal claims (no "because", "due to")
- Present observations in a table if multiple data points
- Describe patterns observed
""",
    IntentType.UNKNOWN: """
UNKNOWN: Default explanation
- Present data clearly with tables if multiple rows
- Include row count and key values
- Add brief interpretation
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
    
    # Deterministic short-circuit: never call LLM on execution failure.
    if failed_subquestions:
        return _narrate_failure(
            original_question=original_question,
            failed_subquestions=failed_subquestions,
            plan=plan,
            max_retries=max_retries,
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
    """Return a deterministic failure message without invoking an LLM."""
    _ = (original_question, plan, max_retries)
    lines = [
        "Query execution failed for one or more subquestions.",
        "I cannot provide a reliable answer until the failed steps are fixed.",
        "",
    ]
    for failed in failed_subquestions:
        sq_id = str(failed.get("id", "unknown"))
        error = str(failed.get("error", "Unknown error")).strip()
        if len(error) > 200:
            error = error[:200]
        lines.append(f"• {sq_id}: {error}")
    return "\n".join(lines).strip()


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
        
        # Show data rows (include more rows for table rendering)
        if rows:
            lines.append("  Data:")
            for i, row in enumerate(rows[:20], 1):
                # Format row as key: value pairs
                if isinstance(row, dict):
                    row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                    lines.append(f"    {i}. {row_str}")
                else:
                    lines.append(f"    {i}. {row}")
            
            if len(rows) > 20:
                lines.append(f"    ... ({len(rows) - 20} more rows)")
    
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
    """Parse JSON from LLM response, handling various formatting issues.
    
    Args:
        response: Raw LLM response text
    
    Returns:
        Parsed JSON dict
    
    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    import re
    
    text = response.strip()
    
    # Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object from text (LLM might add preamble/postamble)
    json_match = re.search(r'\{[^{}]*"text"\s*:\s*"[^"]*(?:\\.[^"]*)*"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Last resort: try the original text
    return json.loads(text)
