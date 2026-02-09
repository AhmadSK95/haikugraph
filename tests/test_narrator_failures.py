"""Tests for narrator behavior when subquestions fail.

This module tests that the narrator correctly refuses to provide numeric
answers when any subquestion execution fails.
"""

import pytest
from unittest.mock import patch

from haikugraph.explain.narrator import narrate


def test_narrator_refuses_answer_on_single_failure():
    """Test that narrator refuses to answer when one subquestion fails."""
    question = "What is total revenue?"
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
        ],
    }
    
    # Simulate failed subquestion result
    subquestion_results = [
        {
            "id": "SQ1",
            "status": "failed",
            "error": "Parser Error: syntax error at or near 'FROM'",
            "row_count": 0,
            "preview_rows": [],
            "metadata": {},
        }
    ]
    
    results = {"SQ1": []}
    meta = {"SQ1": {}}
    
    explanation = narrate(
        question=question,
        plan=plan,
        results=results,
        meta=meta,
        subquestion_results=subquestion_results,
    )
    
    # Should refuse to answer
    assert "failed" in explanation.lower()
    assert "Query execution failed" in explanation
    # Old format had "Cannot provide numeric answer", new format is simpler
    # Just verify it refuses to give an answer
    assert "SQ1" in explanation
    assert "Parser Error" in explanation or "syntax error" in explanation


def test_narrator_refuses_answer_on_multiple_failures():
    """Test that narrator lists all failed subquestions."""
    question = "Compare revenue this month vs last month"
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1_current", "tables": ["orders"]},
            {"id": "SQ2_comparison", "tables": ["orders"]},
        ],
    }
    
    # Both subquestions failed
    subquestion_results = [
        {
            "id": "SQ1_current",
            "status": "failed",
            "error": "Parser Error: double FROM clause",
            "row_count": 0,
            "preview_rows": [],
            "metadata": {},
        },
        {
            "id": "SQ2_comparison",
            "status": "failed",
            "error": "Parser Error: double FROM clause",
            "row_count": 0,
            "preview_rows": [],
            "metadata": {},
        },
    ]
    
    results = {"SQ1_current": [], "SQ2_comparison": []}
    meta = {"SQ1_current": {}, "SQ2_comparison": {}}
    
    explanation = narrate(
        question=question,
        plan=plan,
        results=results,
        meta=meta,
        subquestion_results=subquestion_results,
    )
    
    # Should list both failed subquestions
    assert "Query execution failed" in explanation
    assert "SQ1_current" in explanation
    assert "SQ2_comparison" in explanation
    assert explanation.count("•") >= 2  # At least 2 bullet points


def test_narrator_works_normally_when_all_succeed():
    """Test that narrator works normally when all subquestions succeed.
    
    Uses mock to avoid calling real LLM.
    """
    question = "What is total revenue?"
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
        ],
    }
    
    # All subquestions succeeded
    subquestion_results = [
        {
            "id": "SQ1",
            "status": "success",
            "row_count": 10,
            "preview_rows": [{"sum_amount": 1000.0}],
            "metadata": {},
        }
    ]
    
    results = {"SQ1": [{"sum_amount": 1000.0}]}
    meta = {"SQ1": {}}
    
    # Mock the LLM call to avoid actually calling Ollama
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = "The total revenue is $1000."
        
        explanation = narrate(
            question=question,
            plan=plan,
            results=results,
            meta=meta,
            subquestion_results=subquestion_results,
        )
        
        # The LLM should have been called (not short-circuited by failure check)
        mock_llm.assert_called_once()
        
        # The explanation should not contain failure messages
        assert "Query execution failed" not in explanation
        assert explanation == "The total revenue is $1000."


def test_narrator_partial_failure_refuses_answer():
    """Test that narrator refuses even if only one of multiple subquestions fails."""
    question = "Compare revenue by product"
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
            {"id": "SQ2", "tables": ["products"]},
        ],
    }
    
    # One success, one failure
    subquestion_results = [
        {
            "id": "SQ1",
            "status": "success",
            "row_count": 10,
            "preview_rows": [{"product_id": 1, "revenue": 1000.0}],
            "metadata": {},
        },
        {
            "id": "SQ2",
            "status": "failed",
            "error": "Table 'products' does not exist",
            "row_count": 0,
            "preview_rows": [],
            "metadata": {},
        },
    ]
    
    results = {"SQ1": [{"product_id": 1, "revenue": 1000.0}], "SQ2": []}
    meta = {"SQ1": {}, "SQ2": {}}
    
    explanation = narrate(
        question=question,
        plan=plan,
        results=results,
        meta=meta,
        subquestion_results=subquestion_results,
    )
    
    # Should refuse to answer due to partial failure
    assert "Query execution failed" in explanation
    assert "SQ2" in explanation
    assert "does not exist" in explanation


def test_llm_not_called_when_failures_exist():
    """Ensure narrator short-circuits and does not call LLM when failures exist."""
    question = "Any question"
    plan = {"original_question": question, "subquestions": [{"id": "SQ1", "tables": ["t"]}]}
    # One failed subquestion
    subquestion_results = [{"id": "SQ1", "status": "failed", "error": "boom", "row_count": 0, "preview_rows": [], "metadata": {}}]
    results = {"SQ1": []}
    meta = {"SQ1": {}}
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        explanation = narrate(question, plan, results, meta, subquestion_results=subquestion_results)
        mock_llm.assert_not_called()
        assert "Query execution failed" in explanation


def test_narrator_truncates_long_error_messages():
    """Test that long error messages are truncated to avoid overwhelming output."""
    question = "What is total revenue?"
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
        ],
    }
    
    # Create a very long error message
    long_error = "Parser Error: " + "x" * 500
    
    subquestion_results = [
        {
            "id": "SQ1",
            "status": "failed",
            "error": long_error,
            "row_count": 0,
            "preview_rows": [],
            "metadata": {},
        }
    ]
    
    results = {"SQ1": []}
    meta = {"SQ1": {}}
    
    explanation = narrate(
        question=question,
        plan=plan,
        results=results,
        meta=meta,
        subquestion_results=subquestion_results,
    )
    
    # Error should be truncated (current limit is 200 chars)
    # The full 500-char error should not appear
    assert long_error not in explanation
    # But should still mention the error
    assert "Parser Error" in explanation


def test_narrator_handles_missing_subquestion_results():
    """Test that narrator handles case when subquestion_results is None."""
    question = "What is total revenue?"
    plan = {
        "original_question": question,
        "subquestions": [
            {"id": "SQ1", "tables": ["orders"]},
        ],
    }
    
    results = {"SQ1": [{"sum_amount": 1000.0}]}
    meta = {"SQ1": {}}
    
    # Mock the LLM call to avoid actually calling Ollama
    with patch("haikugraph.explain.narrator.call_llm") as mock_llm:
        mock_llm.return_value = "The total revenue is $1000."
        
        # Call without subquestion_results (backward compatibility)
        explanation = narrate(
            question=question,
            plan=plan,
            results=results,
            meta=meta,
            subquestion_results=None,
        )
        
        # Should call LLM normally (no failure check triggered)
        mock_llm.assert_called_once()
        assert "Query execution failed" not in explanation
