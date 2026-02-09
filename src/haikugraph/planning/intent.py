"""Intent classification for natural language questions.

This module provides schema and classification for determining user intent
before plan generation, enabling better context for the planner LLM.
"""

import json
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from haikugraph.llm.router import call_llm


class IntentType(str, Enum):
    """Types of question intents recognized by the system.
    
    A8 Intent Taxonomy (fixed):
    - metric: single aggregated value
    - grouped_metric: aggregated values by dimension
    - comparison: same metric across time or cohorts
    - lookup: raw rows / listings
    - diagnostic: health, gaps, anomalies, missing data
    - unknown: cannot confidently classify
    """
    
    METRIC = "metric"  # Single aggregated value: "What is total revenue?"
    GROUPED_METRIC = "grouped_metric"  # Aggregation by dimension: "Revenue by barber"
    COMPARISON = "comparison"  # Temporal/cohort comparison: "This vs last month"
    LOOKUP = "lookup"  # Raw rows: "Show me recent appointments"
    DIAGNOSTIC = "diagnostic"  # Health/anomalies: "Why did revenue drop?"
    UNKNOWN = "unknown"  # Cannot confidently classify


class Intent(BaseModel):
    """Structured intent classification for a user question.
    
    A8 Output Format (machine-reliable JSON):
    {
      "type": "<intent>",
      "confidence": 0.0-1.0,
      "rationale": "<short explanation>",
      "requires_comparison": true|false
    }
    
    Intent classification runs BEFORE plan generation and provides
    context to the planner LLM without constraining execution logic.
    """
    
    type: IntentType = Field(
        ...,
        description="Primary intent type from A8 taxonomy"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for classification (0.0-1.0)"
    )
    
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short explanation for the classification (1-200 chars)"
    )
    
    requires_comparison: bool = Field(
        ...,
        description="Whether query requires comparing multiple subquestions"
    )
    
    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Ensure rationale is non-empty and trimmed."""
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Rationale cannot be empty")
        return trimmed


# Prompt templates for intent classification

INTENT_SYSTEM_PROMPT = """You are an intent classifier. Classify the user's question intent. Output ONLY valid JSON. No markdown. No commentary."""

INTENT_USER_PROMPT_TEMPLATE = """Classify the intent of this question:

Question: {question}

A8 Intent Taxonomy (fixed):

1. "metric" - Single aggregated value
   Examples: "What is total revenue?", "How many appointments?"
   Signals: "total", "sum", "count", "average" WITHOUT "by" grouping

2. "grouped_metric" - Aggregated values BY dimension
   Examples: "Revenue by barber", "Appointments per customer", "Sales by region"
   Signals: "by", "per", "breakdown", "each" + aggregation words

3. "comparison" - Same metric across time or cohorts
   Examples: "Revenue this month vs last month", "Compare Q1 to Q2"
   Signals: "vs", "versus", "compare", "compared to", temporal phrases

4. "lookup" - Raw rows / listings (no aggregation)
   Examples: "Show me recent appointments", "List all customers"
   Signals: "show", "list", "display", "get", no aggregation words

5. "diagnostic" - Health, gaps, anomalies, missing data
   Examples: "Why did revenue drop?", "What's wrong with sales?"
   Signals: "why", "what happened", "problem", "issue", "missing"

6. "unknown" - Cannot confidently classify
   Use when ambiguous, contradictory signals, or unclear intent

Classification rules:
- Look ONLY at language signals (keywords, question structure)
- DO NOT infer database schema, tables, or columns
- DO NOT generate SQL or plans
- Distinguish "total revenue" (metric) from "revenue by X" (grouped_metric)
- Set requires_comparison=true ONLY for comparison intent
- Set confidence: 0.9+ clear, 0.7-0.9 moderate, 0.5-0.7 weak, <0.5 unknown
- Provide brief rationale (1 sentence, <200 chars)

Output format (JSON only):
{{
  "type": "metric",
  "confidence": 0.95,
  "rationale": "Single aggregation without grouping dimension",
  "requires_comparison": false
}}

Classify the question now. Output ONLY the JSON object."""

INTENT_REPAIR_PROMPT_TEMPLATE = """The previous JSON output had errors. Fix the JSON.

Previous output:
{previous_output}

Errors:
{errors}

Rules:
- Output ONLY valid JSON (no markdown)
- Required: type (one of: metric, grouped_metric, comparison, lookup, diagnostic, unknown)
- Required: confidence (float between 0.0 and 1.0)
- Required: rationale (string, 1-200 characters)
- Required: requires_comparison (boolean)
- No extra fields

Return ONLY the corrected JSON."""


def classify_intent(question: str, *, max_retries: int = 1) -> Intent:
    """Classify the intent of a natural language question.
    
    This function:
    1. Calls LLM with intent-only prompt (no schema/tables)
    2. Parses JSON response
    3. Validates against Intent schema
    4. On failure, attempts one repair
    
    Args:
        question: User's natural language question
        max_retries: Maximum repair attempts (default 1 for fast failure)
    
    Returns:
        Validated Intent object
    
    Raises:
        ValueError: If intent classification fails after repair attempts
    """
    # Allow override from environment
    max_retries = int(os.environ.get("HG_INTENT_MAX_RETRIES", str(max_retries)))
    
    # Initial prompt
    user_prompt = INTENT_USER_PROMPT_TEMPLATE.format(question=question)
    
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # Retry loop
    for attempt in range(max_retries + 1):
        # Call intent classifier (uses same model as planner)
        response = call_llm(messages, role="intent")
        
        # Try to parse JSON
        try:
            intent_dict = _parse_json_strict(response)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                # Repair: JSON parse error
                repair_prompt = INTENT_REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=response,
                    errors=f"JSON parse error: {e}",
                )
                messages = [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                raise ValueError(
                    f"Failed to parse valid JSON from intent classifier after {max_retries} retries. "
                    f"Last error: {e}"
                ) from e
        
        # Try to validate intent
        try:
            intent = Intent(**intent_dict)
            # Success!
            return intent
        except Exception as e:
            if attempt < max_retries:
                # Repair: validation error
                repair_prompt = INTENT_REPAIR_PROMPT_TEMPLATE.format(
                    previous_output=json.dumps(intent_dict, indent=2),
                    errors=str(e),
                )
                messages = [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ]
                continue
            else:
                raise ValueError(
                    f"Intent validation failed after {max_retries} retries.\n{e}"
                ) from e
    
    # Should not reach here
    raise ValueError("Unexpected error in intent classification")


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
