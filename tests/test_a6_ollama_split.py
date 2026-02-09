"""Tests for A6 POC: Ollama-based split LLM (Planner + Narrator)."""

import json
from unittest.mock import MagicMock

import pytest

from haikugraph.explain.narrator import narrate
from haikugraph.planning.llm_planner import generate_or_patch_plan
from haikugraph.planning.schema import validate_plan_or_raise


class TestOllamaPlanner:
    """Tests for Ollama planner with strict JSON validation and repair."""

    def test_planner_generates_valid_plan(self, monkeypatch):
        """Test that planner generates a valid plan on first try."""
        # Mock call_llm to return valid JSON
        valid_plan = {
            "original_question": "What is the total revenue?",
            "subquestions": [
                {
                    "id": "SQ1",
                    "description": "Get total revenue",
                    "tables": ["orders"],
                    "columns": ["revenue"],
                    "aggregations": [{"agg": "sum", "col": "revenue"}],
                }
            ],
        }

        def mock_call_llm(messages, role="planner", **kwargs):
            return json.dumps(valid_plan)

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)

        # Generate plan
        schema = "Table: orders\nColumns:\n  - revenue (DOUBLE)"
        plan = generate_or_patch_plan(
            question="What is the total revenue?",
            schema=schema,
        )

        # Validate plan passes schema validation
        validate_plan_or_raise(plan)
        assert plan["original_question"] == "What is the total revenue?"
        assert len(plan["subquestions"]) == 1
        assert plan["subquestions"][0]["id"] == "SQ1"

    def test_planner_repair_loop_json_parse_error(self, monkeypatch):
        """Test that planner repairs JSON parse errors."""
        call_count = [0]

        def mock_call_llm(messages, role="planner", **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: invalid JSON
                return "{ invalid json"
            else:
                # Second call: valid JSON after repair
                return json.dumps({
                    "original_question": "Test question",
                    "subquestions": [
                        {"id": "SQ1", "tables": ["test_table"]}
                    ],
                })

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)

        schema = "Table: test_table\nColumns:\n  - col1 (VARCHAR)"
        plan = generate_or_patch_plan(question="Test question", schema=schema)

        # Should have repaired and returned valid plan
        assert call_count[0] == 2  # Initial + 1 repair
        validate_plan_or_raise(plan)

    def test_planner_repair_loop_validation_error(self, monkeypatch):
        """Test that planner repairs validation errors."""
        call_count = [0]

        def mock_call_llm(messages, role="planner", **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: missing required field
                return json.dumps({
                    "subquestions": [{"id": "SQ1", "tables": ["test_table"]}]
                })
            else:
                # Second call: fixed validation error
                return json.dumps({
                    "original_question": "Fixed question",
                    "subquestions": [{"id": "SQ1", "tables": ["test_table"]}],
                })

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)

        schema = "Table: test_table\nColumns:\n  - col1 (VARCHAR)"
        plan = generate_or_patch_plan(question="Test question", schema=schema)

        # Should have repaired validation error
        assert call_count[0] == 2
        validate_plan_or_raise(plan)
        assert plan["original_question"] == "Fixed question"

    def test_planner_fails_after_max_retries(self, monkeypatch):
        """Test that planner raises error after max retries."""
        def mock_call_llm(messages, role="planner", **kwargs):
            # Always return invalid JSON
            return "{ invalid json"

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)
        monkeypatch.setenv("HG_MAX_RETRIES", "1")

        schema = "Table: test_table"
        with pytest.raises(ValueError, match="Failed to parse valid JSON"):
            generate_or_patch_plan(question="Test", schema=schema)

    def test_planner_comparison_followup_scoped_constraint(self, monkeypatch):
        """Test that comparison followup generates symmetric scoped time constraints."""
        def mock_call_llm(messages, role="planner", **kwargs):
            # Return plan with scoped constraints for BOTH comparison subquestions
            return json.dumps({
                "original_question": "Compare revenue vs last month",
                "subquestions": [
                    {"id": "SQ1_current", "tables": ["orders"]},
                    {"id": "SQ2_comparison", "tables": ["orders"]},
                ],
                "constraints": [
                    {
                        "type": "time",
                        "expression": "this_month",
                        "applies_to": "SQ1_current",
                    },
                    {
                        "type": "time",
                        "expression": "previous_month",
                        "applies_to": "SQ2_comparison",
                    }
                ],
            })

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)

        schema = "Table: orders\nColumns:\n  - date (DATE)"
        prev_plan = {"original_question": "What is revenue?", "subquestions": []}
        classification = {"type": "comparison", "is_followup": True}

        plan = generate_or_patch_plan(
            question="Compare vs last month",
            schema=schema,
            prev_plan=prev_plan,
            classification=classification,
        )

        # Validate plan
        validate_plan_or_raise(plan)

        # Assert symmetric scoped time constraints exist
        assert "constraints" in plan
        time_constraints = [c for c in plan["constraints"] if c["type"] == "time"]
        assert len(time_constraints) == 2, "Should have 2 scoped time constraints"

        sq1_constraint = next(c for c in time_constraints if c["applies_to"] == "SQ1_current")
        sq2_constraint = next(c for c in time_constraints if c["applies_to"] == "SQ2_comparison")
        
        assert "this_month" in sq1_constraint["expression"] or "current" in sq1_constraint["expression"]
        assert "previous_month" in sq2_constraint["expression"]

    def test_planner_requires_unique_subquestion_ids(self, monkeypatch):
        """Test that planner validates unique subquestion IDs."""
        call_count = [0]

        def mock_call_llm(messages, role="planner", **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First: duplicate IDs
                return json.dumps({
                    "original_question": "Test",
                    "subquestions": [
                        {"id": "SQ1", "tables": ["t1"]},
                        {"id": "SQ1", "tables": ["t2"]},  # Duplicate!
                    ],
                })
            else:
                # Second: fixed unique IDs
                return json.dumps({
                    "original_question": "Test",
                    "subquestions": [
                        {"id": "SQ1", "tables": ["t1"]},
                        {"id": "SQ2", "tables": ["t2"]},
                    ],
                })

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)

        schema = "Table: t1\nTable: t2"
        plan = generate_or_patch_plan(question="Test", schema=schema)

        assert call_count[0] == 2  # Repair was triggered
        validate_plan_or_raise(plan)
        assert plan["subquestions"][0]["id"] != plan["subquestions"][1]["id"]


class TestOllamaNarrator:
    """Tests for Ollama narrator (results explanation)."""

    def test_narrator_called_after_execution(self, monkeypatch):
        """Test that narrator is called after execution, not before."""
        narrator_called = [False]

        def mock_call_llm(messages, role="narrator", **kwargs):
            narrator_called[0] = True
            assert role == "narrator"  # Must be narrator role
            return "This is a test explanation."

        monkeypatch.setattr("haikugraph.explain.narrator.call_llm", mock_call_llm)

        # Simulate calling narrator with results
        plan = {
            "original_question": "What is total revenue?",
            "subquestions": [{"id": "SQ1", "tables": ["orders"]}],
        }
        results = {"SQ1": [{"sum_revenue": 1000}]}
        meta = {"SQ1": {"constraints_applied": []}}

        explanation = narrate(
            question="What is total revenue?",
            plan=plan,
            results=results,
            meta=meta,
        )

        assert narrator_called[0]
        assert "explanation" in explanation.lower()

    def test_narrator_does_not_output_sql(self, monkeypatch):
        """Test that narrator never outputs SQL in response."""
        def mock_call_llm(messages, role="narrator", **kwargs):
            # Narrator should be instructed to NOT output SQL
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            assert system_msg is not None
            # New narrator says "No SQL" instead of "NEVER output SQL"
            assert "no sql" in system_msg["content"].lower()
            return "Total revenue is $1000. No SQL here."

        monkeypatch.setattr("haikugraph.explain.narrator.call_llm", mock_call_llm)

        plan = {"original_question": "Revenue?", "subquestions": [{"id": "SQ1"}]}
        results = {"SQ1": [{"revenue": 1000}]}
        meta = {"SQ1": {}}

        explanation = narrate("Revenue?", plan, results, meta)

        # Ensure no SQL-like patterns in output
        assert "SELECT" not in explanation.upper()
        assert "FROM" not in explanation.upper()

    def test_narrator_includes_comparison_delta(self, monkeypatch):
        """Test that narrator handles comparison results with delta."""
        def mock_call_llm(messages, role="narrator", **kwargs):
            # Check that results include comparison data
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            assert "SQ1_current" in user_msg["content"]
            assert "SQ2_comparison" in user_msg["content"]
            return "Current: $1000, Previous: $800, Delta: +$200 (+25%)"

        monkeypatch.setattr("haikugraph.explain.narrator.call_llm", mock_call_llm)

        plan = {
            "original_question": "Compare revenue",
            "subquestions": [
                {"id": "SQ1_current"},
                {"id": "SQ2_comparison"},
            ],
        }
        results = {
            "SQ1_current": [{"revenue": 1000}],
            "SQ2_comparison": [{"revenue": 800}],
        }
        meta = {"SQ1_current": {}, "SQ2_comparison": {}}

        explanation = narrate("Compare revenue", plan, results, meta)

        # Check explanation mentions comparison
        assert "current" in explanation.lower() or "comparison" in explanation.lower()


class TestA6Integration:
    """Integration tests for A6 flow."""

    def test_planner_before_narrator(self, monkeypatch):
        """Test that planner is always called before narrator."""
        call_order = []

        def mock_call_llm(messages, role="planner", **kwargs):
            call_order.append(role)
            if role == "planner":
                return json.dumps({
                    "original_question": "Test",
                    "subquestions": [{"id": "SQ1", "tables": ["t"]}],
                })
            else:
                return "Explanation text"

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)
        monkeypatch.setattr("haikugraph.explain.narrator.call_llm", mock_call_llm)

        # Generate plan (planner)
        schema = "Table: t"
        plan = generate_or_patch_plan("Test", schema)

        # Then narrate (narrator)
        explanation = narrate("Test", plan, {"SQ1": []}, {"SQ1": {}})

        # Verify order: planner first, narrator second
        assert call_order == ["planner", "narrator"]

    def test_no_openai_dependencies(self):
        """Test that A6 modules don't import OpenAI."""
        import haikugraph.llm.router as router_module
        import haikugraph.llm.ollama_client as ollama_module
        import haikugraph.planning.llm_planner as planner_module
        import haikugraph.explain.narrator as narrator_module

        # Check that none of these modules import openai
        for module in [router_module, ollama_module, planner_module, narrator_module]:
            source = str(module.__file__)
            with open(source) as f:
                content = f.read()
                assert "import openai" not in content
                assert "from openai" not in content

    def test_constraint_applies_to_validation(self, monkeypatch):
        """Test that constraints with applies_to reference valid subquestion IDs."""
        call_count = [0]

        def mock_call_llm(messages, role="planner", **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Invalid applies_to
                return json.dumps({
                    "original_question": "Test",
                    "subquestions": [{"id": "SQ1", "tables": ["t"]}],
                    "constraints": [
                        {"type": "time", "expression": "last_month", "applies_to": "SQ999"}
                    ],
                })
            else:
                # Fixed applies_to
                return json.dumps({
                    "original_question": "Test",
                    "subquestions": [{"id": "SQ1", "tables": ["t"]}],
                    "constraints": [
                        {"type": "time", "expression": "last_month", "applies_to": "SQ1"}
                    ],
                })

        monkeypatch.setattr("haikugraph.planning.llm_planner.call_llm", mock_call_llm)

        schema = "Table: t"
        plan = generate_or_patch_plan("Test", schema)

        # Should have repaired invalid applies_to
        assert call_count[0] == 2
        validate_plan_or_raise(plan)
        assert plan["constraints"][0]["applies_to"] == "SQ1"
