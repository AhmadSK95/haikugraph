"""Interactive ambiguity resolution for HaikuGraph plans.

This module provides functions to detect unresolved ambiguities in plans,
generate human-readable questions, and apply user resolutions back to plans.
"""

import copy


def get_unresolved_ambiguities(plan: dict, *, confidence_threshold: float = 0.7) -> list[dict]:
    """
    Get ambiguities that need user resolution.

    An ambiguity is considered unresolved if:
    - recommended is None, OR
    - confidence < confidence_threshold

    Args:
        plan: Plan dictionary with optional 'ambiguities' field
        confidence_threshold: Minimum confidence for auto-resolution (default: 0.7)

    Returns:
        List of unresolved ambiguity dicts
    """
    ambiguities = plan.get("ambiguities", [])
    if not ambiguities:
        return []

    unresolved = []
    for amb in ambiguities:
        recommended = amb.get("recommended")
        confidence = amb.get("confidence", 0.0)

        # Unresolved if no recommendation or low confidence
        if recommended is None or confidence < confidence_threshold:
            unresolved.append(amb)

    return unresolved


def ambiguity_to_question(ambiguity: dict) -> dict:
    """
    Convert an ambiguity into a human-readable question.

    Args:
        ambiguity: Ambiguity dict with 'issue' and 'options'

    Returns:
        Question dict with format:
        {
            "issue": "original issue string",
            "question": "human-readable question",
            "options": ["option1", "option2", ...],
            "type": "single_choice"
        }
    """
    issue = ambiguity.get("issue", "")
    options = ambiguity.get("options", [])

    # Generate human-readable question from issue
    question = _issue_to_question(issue)

    return {
        "issue": issue,
        "question": question,
        "options": options,
        "type": "single_choice",
    }


def _issue_to_question(issue: str) -> str:
    """
    Convert an issue string into a user-friendly question.

    Args:
        issue: Issue description from ambiguity

    Returns:
        Human-readable question string
    """
    # Common patterns to convert
    issue_lower = issue.lower()

    # Pattern: "Entity 'X' found in multiple tables"
    if "entity" in issue_lower and "found in multiple tables" in issue_lower:
        # Extract entity name if quoted
        if "'" in issue or '"' in issue:
            # Simple extraction between quotes
            parts = issue.split("'")
            if len(parts) >= 2:
                entity = parts[1]
                return f"Which table should be used for '{entity}'?"

    # Pattern: "Multiple tables contain column" or "Multiple tables contain table.column"
    if "multiple tables contain" in issue_lower:
        if "column" in issue_lower:
            # Extract column name if present
            parts = issue.split("contain")
            if len(parts) >= 2:
                col_part = parts[1].strip()
                return f"Which table should be used for {col_part}?"

    # Pattern: "Ambiguous column" or "Column X found in"
    if "column" in issue_lower and ("ambiguous" in issue_lower or "found in" in issue_lower):
        return "Which column should be used?"

    # Pattern: "Multiple possible joins between"
    if "multiple possible joins" in issue_lower or "join path" in issue_lower:
        return "Which join path should be used?"

    # Pattern: "Time constraint" or "Time filter"
    if "time" in issue_lower and ("constraint" in issue_lower or "filter" in issue_lower):
        return "Which time period should be used?"

    # Default: Convert issue to question form
    # If it already ends with '?', use as-is
    if issue.endswith("?"):
        return issue

    # Otherwise, prepend "Which" and make it a question
    return f"Which option should be used for: {issue}?"


def apply_user_resolution(plan: dict, issue: str, chosen: str) -> dict:
    """
    Apply a user's resolution to a plan by updating the matching ambiguity.

    This function:
    1. Finds the ambiguity with matching issue
    2. Sets recommended = chosen
    3. Sets confidence = 1.0
    4. Returns updated plan (does not modify in-place)

    Args:
        plan: Plan dictionary
        issue: Issue string to match
        chosen: User's chosen option

    Returns:
        Updated plan dict with resolution applied

    Raises:
        ValueError: If issue not found or chosen not in options
    """
    # Deep copy to avoid modifying original
    updated_plan = copy.deepcopy(plan)

    ambiguities = updated_plan.get("ambiguities", [])
    if not ambiguities:
        raise ValueError("No ambiguities found in plan")

    # Find matching ambiguity
    found = False
    for amb in ambiguities:
        if amb.get("issue") == issue:
            found = True

            # Validate chosen is in options
            options = amb.get("options", [])
            if chosen not in options:
                raise ValueError(f"Chosen option '{chosen}' not in available options: {options}")

            # Apply resolution
            amb["recommended"] = chosen
            amb["confidence"] = 1.0
            break

    if not found:
        raise ValueError(f"Ambiguity with issue '{issue}' not found in plan")

    return updated_plan


def validate_no_unresolved_ambiguities(plan: dict, *, confidence_threshold: float = 0.7) -> None:
    """
    Validate that a plan has no unresolved ambiguities.

    Raises:
        ValueError: If unresolved ambiguities exist, with details
    """
    unresolved = get_unresolved_ambiguities(plan, confidence_threshold=confidence_threshold)

    if unresolved:
        issues = [amb.get("issue", "unknown") for amb in unresolved]
        raise ValueError(
            "Unresolved ambiguities remain. Run with --interactive to resolve them.\n"
            "Unresolved issues:\n" + "\n".join(f"  - {issue}" for issue in issues)
        )
