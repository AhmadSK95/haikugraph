"""Tests for interactive ambiguity resolution."""

import pytest

from haikugraph.planning.ambiguity import (
    ambiguity_to_question,
    apply_user_resolution,
    get_unresolved_ambiguities,
    validate_no_unresolved_ambiguities,
)


def test_get_unresolved_ambiguities_empty_plan():
    """Test with plan that has no ambiguities."""
    plan = {"original_question": "test", "subquestions": []}
    result = get_unresolved_ambiguities(plan)
    assert result == []


def test_get_unresolved_ambiguities_no_recommendation():
    """Test ambiguity with no recommendation is unresolved."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {"issue": "Test issue", "options": ["a", "b"], "recommended": None, "confidence": 0.0}
        ],
    }
    result = get_unresolved_ambiguities(plan)
    assert len(result) == 1
    assert result[0]["issue"] == "Test issue"


def test_get_unresolved_ambiguities_low_confidence():
    """Test ambiguity with low confidence is unresolved."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {
                "issue": "Test issue",
                "options": ["a", "b"],
                "recommended": "a",
                "confidence": 0.5,  # Below default threshold of 0.7
            }
        ],
    }
    result = get_unresolved_ambiguities(plan)
    assert len(result) == 1


def test_get_unresolved_ambiguities_high_confidence():
    """Test ambiguity with high confidence is not unresolved."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {
                "issue": "Test issue",
                "options": ["a", "b"],
                "recommended": "a",
                "confidence": 0.9,  # Above threshold
            }
        ],
    }
    result = get_unresolved_ambiguities(plan)
    assert len(result) == 0


def test_get_unresolved_ambiguities_custom_threshold():
    """Test with custom confidence threshold."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {
                "issue": "Test issue",
                "options": ["a", "b"],
                "recommended": "a",
                "confidence": 0.75,
            }
        ],
    }
    # With threshold 0.8, confidence 0.75 is unresolved
    result = get_unresolved_ambiguities(plan, confidence_threshold=0.8)
    assert len(result) == 1

    # With threshold 0.7, confidence 0.75 is resolved
    result = get_unresolved_ambiguities(plan, confidence_threshold=0.7)
    assert len(result) == 0


def test_ambiguity_to_question_entity_pattern():
    """Test question generation for entity ambiguity."""
    ambiguity = {
        "issue": "Entity 'customer' found in multiple tables",
        "options": ["customers", "orders"],
        "confidence": 0.5,
    }
    result = ambiguity_to_question(ambiguity)

    assert result["type"] == "single_choice"
    assert result["issue"] == ambiguity["issue"]
    assert result["options"] == ["customers", "orders"]
    assert "customer" in result["question"]
    assert result["question"].endswith("?")


def test_ambiguity_to_question_column_pattern():
    """Test question generation for column ambiguity."""
    ambiguity = {
        "issue": "Multiple tables contain column name",
        "options": ["users.name", "customers.name"],
        "confidence": 0.3,
    }
    result = ambiguity_to_question(ambiguity)

    assert result["type"] == "single_choice"
    assert "column" in result["question"].lower() or "name" in result["question"]


def test_ambiguity_to_question_already_question():
    """Test issue that is already a question."""
    ambiguity = {
        "issue": "Which time period should be used?",
        "options": ["last_30_days", "last_7_days"],
        "confidence": 0.4,
    }
    result = ambiguity_to_question(ambiguity)

    assert result["question"] == "Which time period should be used?"


def test_ambiguity_to_question_default_fallback():
    """Test default question generation for unknown pattern."""
    ambiguity = {
        "issue": "Some custom ambiguity",
        "options": ["option1", "option2"],
        "confidence": 0.2,
    }
    result = ambiguity_to_question(ambiguity)

    assert result["question"].endswith("?")
    assert "Some custom ambiguity" in result["question"]


def test_apply_user_resolution_success():
    """Test successful application of user resolution."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [{"issue": "Test issue", "options": ["a", "b", "c"], "recommended": None}],
    }

    updated_plan = apply_user_resolution(plan, "Test issue", "b")

    # Original plan should be unchanged
    assert plan["ambiguities"][0]["recommended"] is None

    # Updated plan should have resolution
    assert updated_plan["ambiguities"][0]["recommended"] == "b"
    assert updated_plan["ambiguities"][0]["confidence"] == 1.0


def test_apply_user_resolution_invalid_choice():
    """Test error when chosen option not in options list."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [{"issue": "Test issue", "options": ["a", "b"], "recommended": None}],
    }

    with pytest.raises(ValueError, match="not in available options"):
        apply_user_resolution(plan, "Test issue", "invalid_option")


def test_apply_user_resolution_issue_not_found():
    """Test error when issue not found in plan."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [{"issue": "Test issue", "options": ["a", "b"], "recommended": None}],
    }

    with pytest.raises(ValueError, match="not found in plan"):
        apply_user_resolution(plan, "Nonexistent issue", "a")


def test_apply_user_resolution_no_ambiguities():
    """Test error when plan has no ambiguities."""
    plan = {"original_question": "test", "subquestions": []}

    with pytest.raises(ValueError, match="No ambiguities found"):
        apply_user_resolution(plan, "Test issue", "a")


def test_apply_user_resolution_multiple_ambiguities():
    """Test resolution of one ambiguity doesn't affect others."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {"issue": "Issue 1", "options": ["a", "b"], "recommended": None},
            {"issue": "Issue 2", "options": ["x", "y"], "recommended": None},
        ],
    }

    updated_plan = apply_user_resolution(plan, "Issue 1", "b")

    # Only first ambiguity should be resolved
    assert updated_plan["ambiguities"][0]["recommended"] == "b"
    assert updated_plan["ambiguities"][0]["confidence"] == 1.0
    assert updated_plan["ambiguities"][1]["recommended"] is None


def test_validate_no_unresolved_ambiguities_success():
    """Test validation passes with all ambiguities resolved."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {"issue": "Issue 1", "options": ["a", "b"], "recommended": "a", "confidence": 0.9}
        ],
    }

    # Should not raise
    validate_no_unresolved_ambiguities(plan)


def test_validate_no_unresolved_ambiguities_no_ambiguities():
    """Test validation passes with no ambiguities."""
    plan = {"original_question": "test", "subquestions": []}

    # Should not raise
    validate_no_unresolved_ambiguities(plan)


def test_validate_no_unresolved_ambiguities_unresolved():
    """Test validation fails with unresolved ambiguities."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [{"issue": "Unresolved issue", "options": ["a", "b"], "recommended": None}],
    }

    with pytest.raises(ValueError, match="Unresolved ambiguities remain"):
        validate_no_unresolved_ambiguities(plan)


def test_validate_no_unresolved_ambiguities_low_confidence():
    """Test validation fails with low confidence ambiguities."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {
                "issue": "Low confidence issue",
                "options": ["a", "b"],
                "recommended": "a",
                "confidence": 0.5,
            }
        ],
    }

    with pytest.raises(ValueError, match="Unresolved ambiguities remain"):
        validate_no_unresolved_ambiguities(plan)


def test_validate_no_unresolved_ambiguities_custom_threshold():
    """Test validation with custom threshold."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {
                "issue": "Test issue",
                "options": ["a", "b"],
                "recommended": "a",
                "confidence": 0.75,
            }
        ],
    }

    # Should pass with threshold 0.7
    validate_no_unresolved_ambiguities(plan, confidence_threshold=0.7)

    # Should fail with threshold 0.8
    with pytest.raises(ValueError, match="Unresolved ambiguities remain"):
        validate_no_unresolved_ambiguities(plan, confidence_threshold=0.8)


def test_sequential_resolution():
    """Test resolving multiple ambiguities sequentially."""
    plan = {
        "original_question": "test",
        "subquestions": [],
        "ambiguities": [
            {"issue": "Issue 1", "options": ["a", "b"], "recommended": None},
            {"issue": "Issue 2", "options": ["x", "y"], "recommended": None},
            {"issue": "Issue 3", "options": ["p", "q"], "recommended": None},
        ],
    }

    # Resolve first
    plan = apply_user_resolution(plan, "Issue 1", "a")
    assert plan["ambiguities"][0]["recommended"] == "a"
    assert get_unresolved_ambiguities(plan) == [
        plan["ambiguities"][1],
        plan["ambiguities"][2],
    ]

    # Resolve second
    plan = apply_user_resolution(plan, "Issue 2", "y")
    assert plan["ambiguities"][1]["recommended"] == "y"
    assert get_unresolved_ambiguities(plan) == [plan["ambiguities"][2]]

    # Resolve third
    plan = apply_user_resolution(plan, "Issue 3", "p")
    assert plan["ambiguities"][2]["recommended"] == "p"
    assert get_unresolved_ambiguities(plan) == []

    # Validation should pass
    validate_no_unresolved_ambiguities(plan)
