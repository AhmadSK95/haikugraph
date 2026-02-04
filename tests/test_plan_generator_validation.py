"""Tests for LLM-powered plan generator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from haikugraph.llm.plan_generator import (
    create_initial_plan_prompt,
    create_repair_prompt,
    generate_plan,
)


def test_create_initial_plan_prompt():
    """Test that initial prompt contains all required elements."""
    schema_text = "Table: users\nColumns:\n  - id (INTEGER)\n  - name (VARCHAR)"
    question = "How many users are there?"

    prompt = create_initial_plan_prompt(schema_text, question)

    assert "original_question" in prompt
    assert "subquestions" in prompt
    assert schema_text in prompt
    assert question in prompt
    assert "CRITICAL RULES" in prompt


def test_create_repair_prompt():
    """Test that repair prompt includes errors and previous attempt."""
    schema_text = "Table: users"
    question = "Count users"
    invalid_plan = '{"original_question": "Count users"}'
    errors = ["subquestions: Field required", "tables: List should have at least 1 item"]

    prompt = create_repair_prompt(schema_text, question, invalid_plan, errors)

    assert "VALIDATION ERRORS" in prompt
    assert all(err in prompt for err in errors)
    assert invalid_plan in prompt
    assert question in prompt


@pytest.fixture
def valid_plan_json():
    """Return a valid plan JSON string."""
    plan = {
        "original_question": "How many users are there?",
        "subquestions": [
            {
                "id": "SQ1",
                "description": "Count total users",
                "tables": ["users"],
                "columns": ["id"],
                "aggregations": [{"agg": "count", "col": "id"}],
            }
        ],
        "intent": {"type": "metric", "confidence": 0.9},
        "plan_confidence": 0.9,
    }
    return json.dumps(plan)


@pytest.fixture
def invalid_plan_json_missing_tables():
    """Return an invalid plan JSON string (missing tables)."""
    plan = {
        "original_question": "How many users?",
        "subquestions": [
            {
                "id": "SQ1",
                "description": "Count users",
                "tables": [],  # Invalid: empty tables list
                "columns": ["id"],
            }
        ],
    }
    return json.dumps(plan)


@pytest.fixture
def invalid_plan_json_missing_subquestions():
    """Return an invalid plan JSON string (missing subquestions)."""
    plan = {
        "original_question": "How many users?",
        # Missing subquestions field
    }
    return json.dumps(plan)


@patch("haikugraph.llm.plan_generator.call_openai")
@patch("haikugraph.llm.plan_generator.introspect_schema")
def test_generate_plan_success_first_try(
    mock_introspect, mock_call_openai, valid_plan_json, tmp_path
):
    """Test successful plan generation on first attempt."""
    # Create a temporary test database file
    db_path = tmp_path / "test.duckdb"
    db_path.touch()

    # Mock schema introspection
    mock_introspect.return_value = "Table: users\nColumns:\n  - id (INTEGER)"

    # Mock LLM response with valid plan
    mock_call_openai.return_value = valid_plan_json

    # Generate plan
    plan = generate_plan("How many users are there?", db_path)

    # Assertions
    assert plan["original_question"] == "How many users are there?"
    assert len(plan["subquestions"]) == 1
    assert plan["subquestions"][0]["tables"] == ["users"]
    assert mock_call_openai.call_count == 1


@patch("haikugraph.llm.plan_generator.call_openai")
@patch("haikugraph.llm.plan_generator.introspect_schema")
def test_generate_plan_repair_after_validation_error(
    mock_introspect, mock_call_openai, invalid_plan_json_missing_tables, valid_plan_json, tmp_path
):
    """Test that plan generator attempts repair after validation error."""
    db_path = tmp_path / "test.duckdb"
    db_path.touch()

    mock_introspect.return_value = "Table: users"

    # First call returns invalid plan, second call returns valid plan
    mock_call_openai.side_effect = [
        invalid_plan_json_missing_tables,  # First attempt: invalid
        valid_plan_json,  # Repair attempt: valid
    ]

    plan = generate_plan("How many users?", db_path)

    # Should succeed after repair
    assert plan["original_question"] == "How many users are there?"
    assert len(plan["subquestions"]) == 1
    # Should have called LLM twice (initial + 1 repair)
    assert mock_call_openai.call_count == 2


@patch("haikugraph.llm.plan_generator.call_openai")
@patch("haikugraph.llm.plan_generator.introspect_schema")
def test_generate_plan_fails_after_max_retries(
    mock_introspect, mock_call_openai, invalid_plan_json_missing_subquestions, tmp_path
):
    """Test that plan generator fails after max retries."""
    db_path = tmp_path / "test.duckdb"
    db_path.touch()

    mock_introspect.return_value = "Table: users"

    # Always return invalid plan
    mock_call_openai.return_value = invalid_plan_json_missing_subquestions

    with pytest.raises(ValueError, match="Plan validation failed after .* retries"):
        generate_plan("How many users?", db_path)

    # Should have called LLM 3 times (initial + 2 repairs)
    assert mock_call_openai.call_count == 3


@patch("haikugraph.llm.plan_generator.call_openai")
@patch("haikugraph.llm.plan_generator.introspect_schema")
def test_generate_plan_handles_json_parse_error(
    mock_introspect, mock_call_openai, valid_plan_json, tmp_path
):
    """Test that plan generator handles JSON parse errors with repair."""
    db_path = tmp_path / "test.duckdb"
    db_path.touch()

    mock_introspect.return_value = "Table: users"

    # First call returns invalid JSON, second returns valid JSON
    mock_call_openai.side_effect = [
        "This is not valid JSON {{{",  # Invalid JSON
        valid_plan_json,  # Valid JSON after repair
    ]

    plan = generate_plan("Count users", db_path)

    assert plan["original_question"] == "How many users are there?"
    assert mock_call_openai.call_count == 2


@patch("haikugraph.llm.plan_generator.call_openai")
@patch("haikugraph.llm.plan_generator.introspect_schema")
def test_generate_plan_with_markdown_code_blocks(
    mock_introspect, mock_call_openai, valid_plan_json, tmp_path
):
    """Test that plan generator handles markdown code blocks in LLM response."""
    db_path = tmp_path / "test.duckdb"
    db_path.touch()

    mock_introspect.return_value = "Table: users"

    # LLM response with markdown code blocks
    markdown_response = f"```json\n{valid_plan_json}\n```"
    mock_call_openai.return_value = markdown_response

    plan = generate_plan("Count users", db_path)

    assert plan["original_question"] == "How many users are there?"
    assert len(plan["subquestions"]) == 1


def test_generate_plan_database_not_found():
    """Test that generate_plan raises FileNotFoundError for missing database."""
    db_path = Path("/nonexistent/database.duckdb")

    with pytest.raises(FileNotFoundError, match="Database not found"):
        generate_plan("How many users?", db_path)
