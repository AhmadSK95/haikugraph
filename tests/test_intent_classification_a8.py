"""A8: Intent Classification Tests.

This module validates the A8 intent classifier with deterministic LLM mocking.

A8 Intent Taxonomy:
- metric: single aggregated value
- grouped_metric: aggregated values by dimension
- comparison: same metric across time or cohorts
- lookup: raw rows / listings
- diagnostic: health, gaps, anomalies, missing data
- unknown: cannot confidently classify
"""

import json
from unittest.mock import patch

import pytest

from haikugraph.planning.intent import (
    classify_intent,
    Intent,
    IntentType,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _fake_intent_return(intent_dict: dict) -> str:
    """Convert intent dict to JSON string for mocking LLM responses."""
    return json.dumps(intent_dict, indent=2)


# ============================================================================
# A8 Intent Classification Tests - Happy Paths
# ============================================================================

def test_intent_metric_single_aggregation():
    """A1) Metric: 'What is total revenue?' → single aggregated value."""
    question = "What is total revenue?"
    
    valid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Single aggregation without grouping dimension",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        # Validate A8 schema
        assert isinstance(intent, Intent)
        assert intent.type == IntentType.METRIC
        assert 0.9 <= intent.confidence <= 1.0
        assert len(intent.rationale) > 0
        assert intent.requires_comparison is False
        
        # Verify LLM called once
        assert mock_llm.call_count == 1
        call_messages = mock_llm.call_args[0][0]
        question_in_prompt = any(question in msg["content"] for msg in call_messages)
        assert question_in_prompt


def test_intent_grouped_metric_aggregation_by_dimension():
    """A2) Grouped Metric: 'Revenue by barber' → aggregation WITH dimension."""
    question = "What is revenue by barber?"
    
    valid_intent = {
        "type": "grouped_metric",
        "confidence": 0.92,
        "rationale": "Aggregation with 'by' grouping dimension",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.GROUPED_METRIC
        assert intent.confidence >= 0.8
        assert "by" in intent.rationale.lower() or "group" in intent.rationale.lower()
        assert intent.requires_comparison is False


def test_intent_comparison_temporal():
    """A3) Comparison: 'Revenue this month vs last month' → temporal comparison."""
    question = "Revenue this month vs last month"
    
    valid_intent = {
        "type": "comparison",
        "confidence": 0.94,
        "rationale": "Temporal comparison using 'vs' signal",
        "requires_comparison": True
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.COMPARISON
        assert intent.confidence >= 0.8
        assert intent.requires_comparison is True  # Must be True for comparison


def test_intent_lookup_raw_rows():
    """A4) Lookup: 'Show me recent appointments' → raw rows without aggregation."""
    question = "Show me recent appointments"
    
    valid_intent = {
        "type": "lookup",
        "confidence": 0.91,
        "rationale": "Raw data listing without aggregation",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.LOOKUP
        assert intent.confidence >= 0.8
        assert intent.requires_comparison is False


def test_intent_diagnostic_anomaly():
    """A5) Diagnostic: 'Why did revenue drop?' → health/anomaly query."""
    question = "Why did revenue drop?"
    
    valid_intent = {
        "type": "diagnostic",
        "confidence": 0.93,
        "rationale": "Causal question about anomaly",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.DIAGNOSTIC
        assert intent.confidence >= 0.8
        assert "why" in question.lower()  # Diagnostic signal present


def test_intent_unknown_ambiguous():
    """A6) Unknown: 'Tell me things' → cannot confidently classify."""
    question = "Tell me things"
    
    valid_intent = {
        "type": "unknown",
        "confidence": 0.3,
        "rationale": "Ambiguous request without clear intent",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.UNKNOWN
        assert intent.confidence < 0.6  # Unknown typically low confidence


# ============================================================================
# Edge Cases - Distinguishing Similar Intents
# ============================================================================

def test_intent_metric_vs_grouped_metric():
    """B1) Distinguish metric from grouped_metric: 'total' vs 'by'."""
    # Metric: no grouping
    question_metric = "What is total revenue?"
    intent_metric = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Total without grouping",
        "requires_comparison": False
    }
    
    # Grouped metric: has grouping
    question_grouped = "What is revenue by barber?"
    intent_grouped = {
        "type": "grouped_metric",
        "confidence": 0.93,
        "rationale": "Aggregation with 'by' dimension",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        # Test metric
        mock_llm.return_value = _fake_intent_return(intent_metric)
        result_metric = classify_intent(question_metric)
        assert result_metric.type == IntentType.METRIC
        
        # Test grouped_metric
        mock_llm.return_value = _fake_intent_return(intent_grouped)
        result_grouped = classify_intent(question_grouped)
        assert result_grouped.type == IntentType.GROUPED_METRIC


def test_intent_comparison_requires_flag():
    """B2) Comparison intent MUST have requires_comparison=True."""
    question = "Compare revenue to last year"
    
    valid_intent = {
        "type": "comparison",
        "confidence": 0.92,
        "rationale": "Temporal comparison signal",
        "requires_comparison": True  # Must be True
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.COMPARISON
        assert intent.requires_comparison is True


def test_intent_ambiguous_mixed_signals():
    """B3) Mixed signals: 'Show total revenue by barber vs last month'."""
    question = "Show total revenue by barber vs last month"
    
    # Could be grouped_metric + comparison, classifier should pick dominant
    valid_intent = {
        "type": "comparison",  # Dominant: vs signal
        "confidence": 0.75,  # Moderate confidence due to mixed signals
        "rationale": "Comparison dominates despite grouping",
        "requires_comparison": True
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.COMPARISON
        assert 0.6 <= intent.confidence <= 0.9  # Moderate confidence for mixed


# ============================================================================
# Repair Loop Tests
# ============================================================================

def test_intent_repair_loop_missing_field():
    """C1) Repair loop: missing required field → repairs to valid."""
    question = "What is total revenue?"
    
    # First: missing rationale
    invalid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "requires_comparison": False
        # Missing rationale!
    }
    
    # Second: valid
    valid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Single aggregation",
        "requires_comparison": False
    }
    
    call_count = {"count": 0}
    def side_effect(messages, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return _fake_intent_return(invalid_intent)
        else:
            return _fake_intent_return(valid_intent)
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.side_effect = side_effect
        
        intent = classify_intent(question)
        
        # Should repair and succeed
        assert mock_llm.call_count == 2
        assert intent.type == IntentType.METRIC
        assert len(intent.rationale) > 0


def test_intent_handles_markdown_wrapped_json():
    """C2) Handles markdown-wrapped JSON responses."""
    question = "What is total revenue?"
    
    markdown_response = """```json
{
  "type": "metric",
  "confidence": 0.95,
  "rationale": "Single aggregation",
  "requires_comparison": false
}
```"""
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = markdown_response
        
        intent = classify_intent(question)
        
        assert intent.type == IntentType.METRIC
        assert intent.confidence == 0.95


def test_intent_validates_confidence_bounds():
    """C3) Validates confidence is within 0.0-1.0."""
    question = "What is total revenue?"
    
    # Confidence > 1.0 should fail
    invalid_intent = {
        "type": "metric",
        "confidence": 1.5,  # Invalid!
        "rationale": "Test",
        "requires_comparison": False
    }
    
    # After repair
    valid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Test",
        "requires_comparison": False
    }
    
    call_count = {"count": 0}
    def side_effect(messages, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return _fake_intent_return(invalid_intent)
        else:
            return _fake_intent_return(valid_intent)
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.side_effect = side_effect
        
        intent = classify_intent(question)
        
        assert mock_llm.call_count == 2
        assert 0.0 <= intent.confidence <= 1.0


def test_intent_validates_rationale_not_empty():
    """C4) Validates rationale is non-empty."""
    question = "What is total revenue?"
    
    # Empty rationale should fail
    invalid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "",  # Invalid!
        "requires_comparison": False
    }
    
    # After repair
    valid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Single aggregation",
        "requires_comparison": False
    }
    
    call_count = {"count": 0}
    def side_effect(messages, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return _fake_intent_return(invalid_intent)
        else:
            return _fake_intent_return(valid_intent)
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.side_effect = side_effect
        
        intent = classify_intent(question)
        
        assert mock_llm.call_count == 2
        assert len(intent.rationale) > 0


# ============================================================================
# Negative Tests - Failure After Max Retries
# ============================================================================

def test_intent_fails_with_junk_after_retries():
    """D1) Raises ValueError when LLM returns junk after retries."""
    question = "What is total revenue?"
    
    junk_response = "This is not JSON at all!"
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = junk_response
        
        with pytest.raises(ValueError) as exc_info:
            classify_intent(question)
        
        error_msg = str(exc_info.value)
        assert "retries" in error_msg.lower()
        assert "json" in error_msg.lower() or "parse" in error_msg.lower()
        
        # Default max_retries=1, so 2 calls total
        assert mock_llm.call_count == 2


def test_intent_fails_with_invalid_type_after_retries():
    """D2) Raises ValueError when type is invalid after retries."""
    question = "What is total revenue?"
    
    invalid_intent = {
        "type": "invalid_type",  # Not in A8 taxonomy!
        "confidence": 0.95,
        "rationale": "Test",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(invalid_intent)
        
        with pytest.raises(ValueError) as exc_info:
            classify_intent(question)
        
        error_msg = str(exc_info.value)
        assert "validation" in error_msg.lower() or "retries" in error_msg.lower()
        assert mock_llm.call_count == 2


# ============================================================================
# Schema Validation Tests
# ============================================================================

def test_intent_no_schema_in_prompt():
    """E1) Intent prompt does NOT contain database schema."""
    question = "What is total revenue?"
    
    valid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Single aggregation",
        "requires_comparison": False
    }
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.return_value = _fake_intent_return(valid_intent)
        
        classify_intent(question)
        
        # Check prompt doesn't contain schema keywords
        call_messages = mock_llm.call_args[0][0]
        full_prompt = " ".join(msg["content"] for msg in call_messages)
        
        # Should NOT pass actual database schema
        assert "Database schema:" not in full_prompt
        assert "CREATE TABLE" not in full_prompt.upper()


def test_intent_all_types_valid():
    """E2) All A8 taxonomy types are valid enum values."""
    # Verify IntentType enum has exactly 6 types from A8 spec
    expected_types = {"metric", "grouped_metric", "comparison", "lookup", "diagnostic", "unknown"}
    actual_types = {t.value for t in IntentType}
    
    assert actual_types == expected_types, f"Intent taxonomy mismatch. Expected {expected_types}, got {actual_types}"


def test_intent_rationale_length_validation():
    """E3) Rationale length must be 1-200 characters."""
    question = "What is total revenue?"
    
    # Rationale too long (>200 chars)
    invalid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "A" * 201,  # 201 chars - invalid!
        "requires_comparison": False
    }
    
    # After repair
    valid_intent = {
        "type": "metric",
        "confidence": 0.95,
        "rationale": "Single aggregation",
        "requires_comparison": False
    }
    
    call_count = {"count": 0}
    def side_effect(messages, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return _fake_intent_return(invalid_intent)
        else:
            return _fake_intent_return(valid_intent)
    
    with patch("haikugraph.planning.intent.call_llm") as mock_llm:
        mock_llm.side_effect = side_effect
        
        intent = classify_intent(question)
        
        # Should repair
        assert mock_llm.call_count == 2
        assert 1 <= len(intent.rationale) <= 200
