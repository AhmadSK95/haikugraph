from __future__ import annotations

from haikugraph.qa.fast_round_runner import DEFAULT_FAST_SUITES, select_fast_suites


def test_select_fast_suites_defaults_when_no_changes():
    suites = select_fast_suites([])
    assert suites == DEFAULT_FAST_SUITES


def test_select_fast_suites_uses_path_rules():
    suites = select_fast_suites(
        [
            "src/haikugraph/poc/agentic_team.py",
            "src/haikugraph/analytics/advanced_packs.py",
        ]
    )
    assert "tests/test_api_runtime_features.py" in suites
    assert "tests/test_brd_explainability.py" in suites
    assert "tests/test_advanced_analytics_packs.py" in suites


def test_select_fast_suites_falls_back_to_default_when_no_rule_matches():
    suites = select_fast_suites(["README.md"])
    assert suites == DEFAULT_FAST_SUITES

