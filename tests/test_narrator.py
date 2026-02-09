"""Tests for intent-aware narrator.

This module validates that the narrator:
- Never called if execution failed (short-circuits)
- Produces intent-aware explanations
- Outputs structured JSON
- Has repair loop for invalid JSON
- Never invents data or shows SQL
"""

import json
from unittest.mock import patch

import pytest

from haikugraph.explain.narrator import narrate_results
from haikugraph.planning.intent import Intent, IntentType


# ============================================================================
# Helper Functions
# ============================================================================

def _fake_narration_return(text: str) -> str:
    """Convert text to narrator JSON format for mocking."""
    return json.dumps({"text": text})


def _mock_intent(intent_type: IntentType, requires_comparison: bool = False) -> Intent:
    """Create mock intent for testing."""
    return Intent(
        type=intent_type,
        confidence=0.95,
        rationale=f"Mock {intent_type.value} for testing",
        requires_comparison=requires_comparison
    )


# ============================================================================
# Failure Handling Tests
# ============================================================================

def test_narrator_not_called_when_execution_failed():
    """A1) Narrator never called if any subquestion failed - short-circuits to error."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    # Execution failed
    results = {
        "SQ1": {
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": "DuckDB parser error: syntax error at or near SELECT"
        }
    }
    
    # Should NOT call LLM - short-circuit to error
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        explanation = narrate_results(question, intent, plan, results)
        
        # Verify LLM was NOT called
        assert mock_llm.call_count == 0, "Narrator should not call LLM when execution failed"
        
        # Verify error message returned
        assert "Query execution failed" in explanation
        assert "SQ1" in explanation
        assert "parser error" in explanation.lower()


def test_narrator_short_circuits_with_multiple_failures():
    """A2) Multiple subquestion failures all listed in error."""
    question = "Revenue this month vs last month"
    intent = _mock_intent(IntentType.COMPARISON, requires_comparison=True)
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1_current", "tables": ["orders"]},
            {"id": "SQ2_comparison", "tables": ["orders"]}
        ]
    }
    
    results = {
        "SQ1_current": {
            "error": "Table 'orders' not found"
        },
        "SQ2_comparison": {
            "error": "Connection timeout"
        }
    }
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        explanation = narrate_results(question, intent, plan, results)
        
        assert mock_llm.call_count == 0
        assert "SQ1_current" in explanation
        assert "SQ2_comparison" in explanation
        assert "Table 'orders' not found" in explanation
        assert "Connection timeout" in explanation


# ============================================================================
# Intent-Aware Narration Tests
# ============================================================================

def test_narrator_metric_single_aggregation():
    """B1) METRIC intent → single summary sentence."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"total_revenue": 25000.50}],
            "columns": ["total_revenue"],
            "row_count": 1
        }
    }
    
    expected_text = "Total revenue is $25,000.50"
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = _fake_narration_return(expected_text)
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert explanation == expected_text
        assert mock_llm.call_count == 1
        
        # Verify prompt contains intent info
        call_messages = mock_llm.call_args[0][0]
        full_prompt = " ".join(msg["content"] for msg in call_messages)
        assert "metric" in full_prompt.lower()
        assert "single aggregated value" in full_prompt.lower()


def test_narrator_grouped_metric_by_dimension():
    """B2) GROUPED_METRIC intent → bullet list or short description."""
    question = "What is revenue by barber?"
    intent = _mock_intent(IntentType.GROUPED_METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["appointments"]}]}
    
    results = {
        "SQ1": {
            "rows": [
                {"barber": "Alice", "revenue": 10000},
                {"barber": "Bob", "revenue": 8500},
                {"barber": "Charlie", "revenue": 6500}
            ],
            "columns": ["barber", "revenue"],
            "row_count": 3
        }
    }
    
    expected_text = "Revenue by barber: Alice ($10,000), Bob ($8,500), Charlie ($6,500)"
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = _fake_narration_return(expected_text)
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert explanation == expected_text
        
        # Verify intent instructions
        full_prompt = " ".join(msg["content"] for msg in mock_llm.call_args[0][0])
        assert "grouped_metric" in full_prompt.lower()
        assert "bullet list" in full_prompt.lower() or "dimension" in full_prompt.lower()


def test_narrator_comparison_with_delta():
    """B3) COMPARISON intent → explicit increase/decrease with delta."""
    question = "Revenue this month vs last month"
    intent = _mock_intent(IntentType.COMPARISON, requires_comparison=True)
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1_current", "tables": ["orders"]},
            {"id": "SQ2_comparison", "tables": ["orders"]}
        ]
    }
    
    results = {
        "SQ1_current": {
            "rows": [{"revenue": 30000}],
            "columns": ["revenue"],
            "row_count": 1
        },
        "SQ2_comparison": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    expected_text = "Revenue this month ($30,000) vs last month ($25,000) - increased by $5,000 (20%)"
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = _fake_narration_return(expected_text)
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert "increased" in explanation.lower() or "decreased" in explanation.lower()
        assert "30,000" in explanation or "30000" in explanation
        
        # Verify comparison instructions
        full_prompt = " ".join(msg["content"] for msg in mock_llm.call_args[0][0])
        assert "comparison" in full_prompt.lower()
        assert "delta" in full_prompt.lower()


def test_narrator_lookup_raw_rows():
    """B4) LOOKUP intent → describe what was listed + row count."""
    question = "Show me recent appointments"
    intent = _mock_intent(IntentType.LOOKUP)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["appointments"]}]}
    
    results = {
        "SQ1": {
            "rows": [
                {"customer": "John", "service": "Haircut", "time": "10:00 AM"},
                {"customer": "Jane", "service": "Coloring", "time": "11:30 AM"},
                {"customer": "Bob", "service": "Trim", "time": "2:00 PM"},
            ],
            "columns": ["customer", "service", "time"],
            "row_count": 3
        }
    }
    
    expected_text = "Found 3 recent appointments. First few: John (Haircut at 10:00 AM), Jane (Coloring at 11:30 AM), Bob (Trim at 2:00 PM)"
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = _fake_narration_return(expected_text)
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert "3" in explanation or "three" in explanation.lower()
        assert "appointments" in explanation.lower()
        
        # Verify lookup instructions
        full_prompt = " ".join(msg["content"] for msg in mock_llm.call_args[0][0])
        assert "lookup" in full_prompt.lower()
        assert "row count" in full_prompt.lower()


def test_narrator_diagnostic_no_causal_claims():
    """B5) DIAGNOSTIC intent → cautious, no causal claims."""
    question = "Why did revenue drop?"
    intent = _mock_intent(IntentType.DIAGNOSTIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [
                {"month": "January", "revenue": 30000},
                {"month": "February", "revenue": 28000},
                {"month": "March", "revenue": 22000}
            ],
            "columns": ["month", "revenue"],
            "row_count": 3
        }
    }
    
    expected_text = "Revenue shows declining pattern. Observed decrease from $30,000 (January) to $22,000 (March)."
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = _fake_narration_return(expected_text)
        
        explanation = narrate_results(question, intent, plan, results)
        
        # Should NOT contain causal words
        causal_words = ["because", "due to", "caused by", "reason is"]
        for word in causal_words:
            assert word not in explanation.lower(), f"Diagnostic should not contain causal claim: {word}"
        
        # Verify diagnostic instructions
        full_prompt = " ".join(msg["content"] for msg in mock_llm.call_args[0][0])
        assert "diagnostic" in full_prompt.lower()
        assert "no causal claims" in full_prompt.lower()


def test_narrator_unknown_intent_default():
    """B6) UNKNOWN intent → simple description."""
    question = "Tell me things"
    intent = _mock_intent(IntentType.UNKNOWN)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["data"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"value": 100}],
            "columns": ["value"],
            "row_count": 1
        }
    }
    
    expected_text = "Found 1 row with value 100"
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = _fake_narration_return(expected_text)
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert explanation == expected_text


# ============================================================================
# JSON Output & Repair Loop Tests
# ============================================================================

def test_narrator_outputs_structured_json():
    """C1) Narrator outputs structured JSON with 'text' field."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        # LLM returns proper JSON
        mock_llm.return_value = '{"text": "Total revenue is $25,000"}'
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert explanation == "Total revenue is $25,000"


def test_narrator_handles_markdown_wrapped_json():
    """C2) Narrator strips markdown code blocks from JSON."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    markdown_response = """```json
{
  "text": "Total revenue is $25,000"
}
```"""
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = markdown_response
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert explanation == "Total revenue is $25,000"


def test_narrator_repair_loop_missing_field():
    """C3) Repair loop fixes missing 'text' field."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    # First: missing text field
    invalid_json = '{"explanation": "Total revenue is $25,000"}'
    
    # Second: valid
    valid_json = '{"text": "Total revenue is $25,000"}'
    
    call_count = {"count": 0}
    def side_effect(messages, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return invalid_json
        else:
            return valid_json
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.side_effect = side_effect
        
        explanation = narrate_results(question, intent, plan, results)
        
        # Should repair and succeed
        assert mock_llm.call_count == 2
        assert explanation == "Total revenue is $25,000"


def test_narrator_repair_loop_invalid_json():
    """C4) Repair loop fixes invalid JSON."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    # First: invalid JSON
    invalid_json = 'This is not JSON'
    
    # Second: valid
    valid_json = '{"text": "Total revenue is $25,000"}'
    
    call_count = {"count": 0}
    def side_effect(messages, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return invalid_json
        else:
            return valid_json
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.side_effect = side_effect
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert mock_llm.call_count == 2
        assert explanation == "Total revenue is $25,000"


def test_narrator_fails_after_max_retries():
    """C5) Narrator raises ValueError after exhausting retries."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    # Always return invalid JSON
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = "Not JSON at all"
        
        with pytest.raises(ValueError) as exc_info:
            narrate_results(question, intent, plan, results)
        
        error_msg = str(exc_info.value)
        assert "retries" in error_msg.lower()
        assert "json" in error_msg.lower()
        
        # Default max_retries=1, so 2 calls total
        assert mock_llm.call_count == 2


# ============================================================================
# No Speculation Tests
# ============================================================================

def test_narrator_no_sql_in_prompt():
    """D1) Narrator prompt does NOT contain SQL."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = '{"text": "Total revenue is $25,000"}'
        
        narrate_results(question, intent, plan, results)
        
        # Check prompt doesn't contain SQL
        call_messages = mock_llm.call_args[0][0]
        full_prompt = " ".join(msg["content"] for msg in call_messages)
        
        assert "SELECT" not in full_prompt.upper()
        assert "CREATE TABLE" not in full_prompt.upper()


def test_narrator_no_schema_speculation():
    """D2) Narrator prompt says 'no schema speculation'."""
    question = "What is total revenue?"
    intent = _mock_intent(IntentType.METRIC)
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = '{"text": "Total revenue is $25,000"}'
        
        narrate_results(question, intent, plan, results)
        
        # Check prompt says no speculation
        call_messages = mock_llm.call_args[0][0]
        full_prompt = " ".join(msg["content"] for msg in call_messages)
        
        assert "no speculation" in full_prompt.lower() or "only what exists" in full_prompt.lower()


def test_narrator_works_without_intent():
    """D3) Narrator works when intent is None (graceful fallback)."""
    question = "What is total revenue?"
    intent = None  # No intent provided
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["orders"]}]}
    
    results = {
        "SQ1": {
            "rows": [{"revenue": 25000}],
            "columns": ["revenue"],
            "row_count": 1
        }
    }
    
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = '{"text": "Total revenue is $25,000"}'
        
        explanation = narrate_results(question, intent, plan, results)
        
        assert explanation == "Total revenue is $25,000"
        assert mock_llm.call_count == 1
