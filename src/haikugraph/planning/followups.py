"""Conversational continuity for HaikuGraph - handle follow-up questions with plan patching.

This module provides deterministic pattern matching and plan patching for common
follow-up question types, avoiding full plan regeneration.
"""

import copy
import re
from typing import Any


def classify_followup(new_question: str, prev_question: str, prev_plan: dict) -> dict:
    """
    Classify if new_question is a follow-up to prev_question.

    Args:
        new_question: The new question from the user
        prev_question: The previous question
        prev_plan: The previous plan dict

    Returns:
        Classification dict with:
        {
            "is_followup": bool,
            "type": "time_refine|filter_refine|groupby_change|limit_change|
                     metric_change|comparison|clarification|new_question",
            "confidence": float (0-1),
            "patches": [ ... ]  # structured patch intents
        }
    """
    new_lower = new_question.lower().strip()
    prev_lower = prev_question.lower().strip()

    # Check if it's clearly a new independent question
    if _is_new_question(new_lower, prev_lower):
        return {
            "is_followup": False,
            "type": "new_question",
            "confidence": 0.9,
            "patches": [],
        }

    # Try to detect follow-up patterns
    patches = []
    followup_type = None
    confidence = 0.0

    # 1. Time refine patterns
    time_match = _detect_time_refine(new_lower)
    if time_match:
        patches.append({"type": "time_refine", "value": time_match})
        followup_type = "time_refine"
        confidence = 0.95

    # 2. Filter refine patterns
    filter_match = _detect_filter_refine(new_lower)
    if filter_match:
        patches.append({"type": "filter_refine", "value": filter_match})
        if not followup_type:
            followup_type = "filter_refine"
        confidence = max(confidence, 0.9)

    # 3. Groupby change patterns
    groupby_match = _detect_groupby_change(new_lower)
    if groupby_match:
        patches.append({"type": "groupby_change", "value": groupby_match})
        if not followup_type:
            followup_type = "groupby_change"
        confidence = max(confidence, 0.85)

    # 4. Limit change patterns
    limit_match = _detect_limit_change(new_lower)
    if limit_match:
        patches.append({"type": "limit_change", "value": limit_match})
        if not followup_type:
            followup_type = "limit_change"
        confidence = max(confidence, 0.9)

    # 5. Comparison patterns
    comparison_match = _detect_comparison(new_lower)
    if comparison_match:
        patches.append({"type": "comparison", "value": comparison_match})
        if not followup_type:
            followup_type = "comparison"
        confidence = max(confidence, 0.85)

    # If no patterns matched, not a follow-up
    if not patches:
        return {
            "is_followup": False,
            "type": "new_question",
            "confidence": 0.8,
            "patches": [],
        }

    return {
        "is_followup": True,
        "type": followup_type or "clarification",
        "confidence": confidence,
        "patches": patches,
    }


def patch_plan(prev_plan: dict, classification: dict, new_question: str) -> dict:
    """
    Apply patches to previous plan based on classification.

    This function:
    - Preserves resolved ambiguities from A4
    - Preserves chosen tables and join_paths unless changed
    - Applies incremental updates to constraints, group_by, aggregations
    - Returns a plan that validates against schema.py

    Args:
        prev_plan: Previous plan dict
        classification: Classification from classify_followup
        new_question: The new question text

    Returns:
        Patched plan dict
    """
    if not classification.get("is_followup"):
        raise ValueError("Cannot patch plan - classification is not a follow-up")

    # Deep copy to avoid mutating original
    patched = copy.deepcopy(prev_plan)

    # Update original_question
    patched["original_question"] = merge_questions(
        prev_plan.get("original_question", ""),
        new_question,
        classification,
    )

    # Apply patches
    for patch in classification.get("patches", []):
        patch_type = patch.get("type")
        value = patch.get("value")

        if patch_type == "time_refine":
            _apply_time_refine(patched, value)
        elif patch_type == "filter_refine":
            _apply_filter_refine(patched, value)
        elif patch_type == "groupby_change":
            _apply_groupby_change(patched, value)
        elif patch_type == "limit_change":
            _apply_limit_change(patched, value)
        elif patch_type == "comparison":
            _apply_comparison(patched, value)

    return patched


def merge_questions(prev_question: str, new_question: str, classification: dict) -> str:
    """
    Produce an updated canonical question that reflects the refined query.

    Args:
        prev_question: Previous question
        new_question: New follow-up question
        classification: Classification dict

    Returns:
        Merged question string
    """
    followup_type = classification.get("type", "")

    # For time refines, append the time constraint
    if followup_type == "time_refine":
        patches = classification.get("patches", [])
        for patch in patches:
            if patch.get("type") == "time_refine":
                time_val = patch.get("value", {})
                period = time_val.get("period", "")
                if period:
                    # Remove old time phrases if present
                    cleaned = _remove_time_phrases(prev_question)
                    return f"{cleaned} ({period})"

    # For filter refines, append the filter
    if followup_type == "filter_refine":
        patches = classification.get("patches", [])
        for patch in patches:
            if patch.get("type") == "filter_refine":
                filter_val = patch.get("value", {})
                filter_expr = filter_val.get("expression", "")
                if filter_expr:
                    return f"{prev_question} where {filter_expr}"

    # For groupby changes, append grouping phrase
    if followup_type == "groupby_change":
        patches = classification.get("patches", [])
        for patch in patches:
            if patch.get("type") == "groupby_change":
                groupby_val = patch.get("value", {})
                column = groupby_val.get("column", "")
                if column:
                    return f"{prev_question} by {column}"

    # For limit changes, append limit
    if followup_type == "limit_change":
        patches = classification.get("patches", [])
        for patch in patches:
            if patch.get("type") == "limit_change":
                limit_val = patch.get("value", {})
                limit_n = limit_val.get("limit", 0)
                if limit_n:
                    return f"{prev_question} (top {limit_n})"

    # For comparison, describe both periods
    if followup_type == "comparison":
        return f"{prev_question} with comparison to previous period"

    # Default: concatenate
    return f"{prev_question} - {new_question}"


# Pattern detection helpers


def _is_new_question(new_lower: str, prev_lower: str) -> bool:
    """Check if question seems to be entirely new."""
    # Keywords that indicate a completely new question
    new_keywords = [
        "show me",
        "what is",
        "what are",
        "how many",
        "who",
        "when",
        "list",
        "find",
        "get",
        "give me",
    ]

    # If starts with these and shares few words with previous, likely new
    for keyword in new_keywords:
        if new_lower.startswith(keyword):
            # Check word overlap
            prev_words = set(prev_lower.split())
            new_words = set(new_lower.split())
            overlap = len(prev_words & new_words) / max(len(new_words), 1)
            if overlap < 0.3:  # Less than 30% overlap
                return True

    return False


def _detect_time_refine(text: str) -> dict[str, Any] | None:
    """Detect time refinement patterns."""
    # Pattern: "last N days|weeks|months|years"
    match = re.search(r"last\s+(\d+)\s+(day|week|month|year)s?", text)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        return {"period": f"last_{count}_{unit}s", "count": count, "unit": unit}

    # Pattern: "yesterday", "today", "this week", "this month", "this year"
    simple_periods = {
        "yesterday": {"period": "yesterday"},
        "today": {"period": "today"},
        "this week": {"period": "this_week"},
        "this month": {"period": "this_month"},
        "this year": {"period": "this_year"},
    }
    for phrase, value in simple_periods.items():
        if phrase in text:
            return value

    return None


def _detect_filter_refine(text: str) -> dict[str, Any] | None:
    """Detect filter refinement patterns."""
    # Pattern: "only <column> = <value>" or "only <value>"
    match = re.search(r"only\s+(\w+)\s*=\s*['\"]?(\w+)['\"]?", text)
    if match:
        column = match.group(1)
        value = match.group(2)
        return {"expression": f"{column} = '{value}'", "column": column, "value": value}

    # Pattern: "only <value>" (when column is implicit)
    match = re.search(r"only\s+(\w+)", text)
    if match:
        value = match.group(1)
        return {"expression": f"status = '{value}'", "column": "status", "value": value}

    # Pattern: "where <expression>"
    if "where" in text:
        parts = text.split("where", 1)
        if len(parts) > 1:
            expr = parts[1].strip()
            return {"expression": expr, "raw": True}

    return None


def _detect_groupby_change(text: str) -> dict[str, Any] | None:
    """Detect groupby change patterns."""
    # Pattern: "by <column>", "group by <column>", "breakdown by <column>"
    patterns = [
        r"breakdown\s+by\s+(\w+)",
        r"group\s+by\s+(\w+)",
        r"\bby\s+(\w+)$",  # "by month" at end
        r"\bby\s+(\w+)\b",  # "by month" anywhere
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            column = match.group(1)
            return {"column": column}

    return None


def _detect_limit_change(text: str) -> dict[str, Any] | None:
    """Detect limit/top-N change patterns."""
    # Pattern: "top N", "first N", "show N", "limit N"
    patterns = [
        r"top\s+(\d+)",
        r"first\s+(\d+)",
        r"show\s+(?:me\s+)?(\d+)",
        r"limit\s+(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            limit = int(match.group(1))
            return {"limit": limit}

    return None


def _detect_comparison(text: str) -> dict[str, Any] | None:
    """Detect comparison patterns."""
    # Keywords for comparison
    comparison_keywords = [
        "compare",
        "comparison",
        "vs",
        "versus",
        "compared to",
        "previous",
        "prior",
    ]

    for keyword in comparison_keywords:
        if keyword in text:
            # Try to detect what periods to compare
            if "previous month" in text or "last month" in text:
                return {"compare_to": "previous_month"}
            elif "previous year" in text or "last year" in text:
                return {"compare_to": "previous_year"}
            elif "previous week" in text or "last week" in text:
                return {"compare_to": "previous_week"}
            else:
                return {"compare_to": "previous_period"}

    return None


# Patch application helpers


def _apply_time_refine(plan: dict, value: dict) -> None:
    """Apply time constraint refinement to plan."""
    period = value.get("period", "")
    if not period:
        return

    # Get primary table from first subquestion for table-qualified constraint
    subquestions = plan.get("subquestions", [])
    if not subquestions:
        return

    primary_table = subquestions[0].get("tables", [None])[0]
    if not primary_table:
        return

    # Remove existing time constraints
    constraints = plan.get("constraints", [])
    plan["constraints"] = [c for c in constraints if c.get("type") != "time"]

    # Add new time constraint with table-qualified expression
    # Use a symbolic expression that executor will translate
    # Try to infer time column from common names or use generic
    time_col = _infer_time_column(subquestions[0])
    expression = f"{primary_table}.{time_col} in {period}"
    plan["constraints"].append({"type": "time", "expression": expression})


def _apply_filter_refine(plan: dict, value: dict) -> None:
    """Apply filter constraint to plan."""
    expression = value.get("expression", "")
    if not expression:
        return

    # Get primary table for table-qualified constraint
    subquestions = plan.get("subquestions", [])
    if not subquestions:
        return

    primary_table = subquestions[0].get("tables", [None])[0]
    if not primary_table:
        return

    # Add filter constraint (additive unless contradictory)
    if "constraints" not in plan:
        plan["constraints"] = []

    # Check if contradicts existing filter on same column
    column = value.get("column")
    if column:
        # Remove conflicting filters on same column
        plan["constraints"] = [
            c
            for c in plan["constraints"]
            if c.get("type") == "time" or column not in c.get("expression", "")
        ]

    # Ensure expression is table-qualified
    if "." not in expression:
        # Simple format like "status = 'value'" -> "table.status = 'value'"
        expression = f"{primary_table}.{expression}"

    plan["constraints"].append({"type": "filter", "expression": expression})


def _apply_groupby_change(plan: dict, value: dict) -> None:
    """Apply groupby change to plan subquestions."""
    column = value.get("column", "")
    if not column:
        return

    # Update first subquestion's group_by
    subquestions = plan.get("subquestions", [])
    if not subquestions:
        return

    sq = subquestions[0]
    sq["group_by"] = [column]

    # Ensure aggregations exist
    if not sq.get("aggregations"):
        # Add default count aggregation
        sq["aggregations"] = [{"agg": "count", "col": "*"}]


def _apply_limit_change(plan: dict, value: dict) -> None:
    """Apply row limit to plan."""
    limit = value.get("limit", 0)
    if limit <= 0:
        return

    # Store as top-level extra field (allowed by schema extra="allow")
    plan["row_limit"] = limit


def _apply_comparison(plan: dict, value: dict) -> None:
    """Apply comparison by adding a second subquestion for comparison period."""
    compare_to = value.get("compare_to", "previous_period")

    subquestions = plan.get("subquestions", [])
    if not subquestions:
        return

    # Get primary table for table-qualified constraints
    primary_table = subquestions[0].get("tables", [None])[0]
    if not primary_table:
        return

    # Clone first subquestion for comparison
    current_sq = subquestions[0]
    comparison_sq = copy.deepcopy(current_sq)

    # Update IDs and descriptions
    current_sq["id"] = "SQ1_current"
    current_sq["description"] = f"{current_sq.get('description', '')} (current period)"

    comparison_sq["id"] = "SQ2_comparison"
    comparison_sq["description"] = f"{comparison_sq.get('description', '')} ({compare_to})"

    # Add time constraint for comparison period
    if "constraints" not in plan:
        plan["constraints"] = []

    # Infer time column from subquestion
    time_col = _infer_time_column(current_sq)

    # Note: Currently executor does not support applies_to scoping
    # This is a forward-compatibility marker for future enhancement
    # For now, comparison queries should be handled by separate execution
    plan["constraints"].append(
        {
            "type": "time",
            "expression": f"{primary_table}.{time_col} in {compare_to}",
            "applies_to": "SQ2_comparison",
        }
    )

    # Add comparison subquestion if not already present
    if len(subquestions) == 1:
        plan["subquestions"].append(comparison_sq)


def _remove_time_phrases(text: str) -> str:
    """Remove common time phrases from text."""
    patterns = [
        r"\(last \d+ \w+\)",
        r"\(yesterday\)",
        r"\(today\)",
        r"\(this \w+\)",
        r"last \d+ \w+",
        r"yesterday",
        r"today",
        r"this \w+",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def _infer_time_column(subquestion: dict) -> str:
    """Infer time column name from subquestion columns.

    Args:
        subquestion: Subquestion dict

    Returns:
        Inferred time column name (defaults to 'created_at')
    """
    columns = subquestion.get("columns", [])

    # Common time column names
    time_column_names = [
        "created_at",
        "updated_at",
        "date",
        "timestamp",
        "time",
        "datetime",
        "order_date",
        "transaction_date",
    ]

    # Check if any column matches known time column names
    for col in columns:
        col_lower = col.lower()
        if col_lower in time_column_names or "date" in col_lower or "time" in col_lower:
            return col

    # Default to created_at
    return "created_at"
