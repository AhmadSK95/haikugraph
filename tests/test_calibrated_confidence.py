"""Tests for calibrated confidence scoring.

Verifies that the confidence scoring module:
- Decomposes confidence into measurable factors
- Penalizes contract violations, fallback mappings, and missing coverage
- Enforces a minimum-factor floor so no single weak signal is hidden
- Returns proper confidence levels (HIGH, MEDIUM, LOW, UNCERTAIN)
- Produces a fully auditable decomposition structure
"""

from __future__ import annotations

import pytest

from haikugraph.scoring.calibrated_confidence import (
    CalibratedConfidence,
    ConfidenceFactor,
    compute_calibrated_confidence,
    _compute_goal_term_coverage,
    _compute_replay_score,
    _level_from_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strong_contract() -> dict:
    """Contract validation that passes all checks."""
    return {"valid": True, "violations": []}


def _weak_contract(n_violations: int = 2) -> dict:
    """Contract validation with violations."""
    return {
        "valid": False,
        "violations": [f"violation_{i}" for i in range(n_violations)],
    }


def _strong_audit() -> dict:
    """Audit result where all checks pass."""
    return {"passed": 7, "warned": 0, "failed": 0}


def _weak_audit(failures: int = 2) -> dict:
    """Audit result with failures."""
    return {"passed": 5, "warned": 0, "failed": failures}


def _good_replay() -> list[dict]:
    """Replay history with all successes and high confidence."""
    return [
        {"success": True, "confidence": 0.9},
        {"success": True, "confidence": 0.85},
        {"success": True, "confidence": 0.95},
    ]


def _bad_replay() -> list[dict]:
    """Replay history with all failures and low confidence."""
    return [
        {"success": False, "confidence": 0.2},
        {"success": False, "confidence": 0.15},
    ]


# ---------------------------------------------------------------------------
# Test 1: High confidence when all factors are strong
# ---------------------------------------------------------------------------

class TestHighConfidenceAllFactorsStrong:
    def test_all_factors_strong_yields_high_confidence(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.95,
            goal_text="Show total payment amount by platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        assert result.level == "HIGH"
        assert result.overall >= 0.8
        assert len(result.penalties) == 0

    def test_high_confidence_has_no_penalties(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment amount platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        assert result.penalties == []


# ---------------------------------------------------------------------------
# Test 2: Low confidence with contract violations
# ---------------------------------------------------------------------------

class TestLowConfidenceContractViolations:
    def test_many_violations_yield_low_confidence(self):
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=4),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment amount platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
        )
        # 4 violations => contract_score = max(0.1, 1.0 - 1.0) = 0.1
        # min factor floor: overall <= 0.1 * 1.5 = 0.15
        assert result.overall < 0.5
        assert result.level in ("LOW", "UNCERTAIN")
        assert any("Contract drift" in p for p in result.penalties)

    def test_single_violation_reduces_contract_score(self):
        # First compute the perfect baseline (no violations)
        perfect = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment amount platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
        )
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=1),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment amount platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
        )
        # 1 violation => contract_score = 0.75
        # Overall should be less than or equal to a perfect run
        assert result.overall <= perfect.overall
        assert any("Contract drift" in p for p in result.penalties)
        # Verify the contract factor score is reduced
        contract_factor = next(f for f in result.factors if f.name == "contract_alignment")
        assert contract_factor.score == 0.75


# ---------------------------------------------------------------------------
# Test 3: Medium confidence with fallback used
# ---------------------------------------------------------------------------

class TestMediumConfidenceFallbackUsed:
    def test_fallback_used_reduces_confidence(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment amount platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=True,
            replay_history=_good_replay(),
        )
        # Fallback score = 0.5 (weight 0.10)
        # min factor floor: overall <= 0.5 * 1.5 = 0.75
        assert result.overall <= 0.75
        assert result.level in ("MEDIUM", "HIGH")
        assert any("Fallback" in p for p in result.penalties)

    def test_no_fallback_does_not_penalize(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment amount platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
        )
        assert not any("Fallback" in p for p in result.penalties)


# ---------------------------------------------------------------------------
# Test 4: Confidence penalized for missing goal-term coverage
# ---------------------------------------------------------------------------

class TestConfidencePenalizedForMissingCoverage:
    def test_low_coverage_adds_penalty(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="Show refund transactions by university customers in December",
            sql_text="SELECT * FROM test_1_1_merged",  # SQL misses many goal terms
            fallback_used=False,
        )
        # Many goal tokens (refund, transactions, university, customers, december)
        # will not appear in the simple SELECT * SQL
        goal_factor = next(f for f in result.factors if f.name == "goal_term_coverage")
        assert goal_factor.score < 0.8

    def test_high_coverage_no_penalty(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
        )
        goal_factor = next(f for f in result.factors if f.name == "goal_term_coverage")
        assert goal_factor.score >= 0.5


# ---------------------------------------------------------------------------
# Test 5: Confidence floor from weakest factor
# ---------------------------------------------------------------------------

class TestConfidenceFloorFromWeakestFactor:
    def test_floor_caps_overall_at_1_5x_weakest(self):
        # Create scenario: one very weak factor (contract with 4 violations = 0.1)
        # and everything else strong
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=4),
            audit_result=_strong_audit(),
            intake_confidence=1.0,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        min_factor_score = min(f.score for f in result.factors)
        # overall should be <= min_factor * 1.5
        assert result.overall <= min_factor_score * 1.5 + 0.0001  # small epsilon

    def test_no_factor_below_zero(self):
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=10),
            audit_result=_weak_audit(failures=5),
            intake_confidence=0.0,
            goal_text="xyzzy foobar baz",
            sql_text="SELECT 1",
            fallback_used=True,
            replay_history=_bad_replay(),
        )
        assert result.overall >= 0.0
        for factor in result.factors:
            assert factor.score >= 0.0


# ---------------------------------------------------------------------------
# Test 6: Goal-term coverage calculation
# ---------------------------------------------------------------------------

class TestGoalTermCoverage:
    def test_all_terms_present(self):
        coverage = _compute_goal_term_coverage(
            "payment platform",
            "SELECT platform_name, SUM(payment_amount) FROM t",
        )
        assert coverage == 1.0

    def test_no_terms_present(self):
        coverage = _compute_goal_term_coverage(
            "refund university december",
            "SELECT 1",
        )
        assert coverage == 0.0

    def test_partial_coverage(self):
        coverage = _compute_goal_term_coverage(
            "payment refund platform",
            "SELECT platform_name, SUM(payment_amount) FROM t",
        )
        # "payment" and "platform" match, "refund" does not -> 2/3
        assert abs(coverage - 2 / 3) < 0.01

    def test_empty_goal_returns_default(self):
        coverage = _compute_goal_term_coverage("", "SELECT 1")
        assert coverage == 0.5

    def test_empty_sql_returns_default(self):
        coverage = _compute_goal_term_coverage("payment", "")
        assert coverage == 0.5

    def test_only_stop_words_returns_high(self):
        coverage = _compute_goal_term_coverage("what is the total", "SELECT 1")
        assert coverage == 0.8  # All tokens are stop words


# ---------------------------------------------------------------------------
# Test 7: Replay score with history
# ---------------------------------------------------------------------------

class TestReplayScoreWithHistory:
    def test_all_successes_high_confidence(self):
        score = _compute_replay_score(_good_replay())
        assert score >= 0.8

    def test_all_failures_low_confidence(self):
        score = _compute_replay_score(_bad_replay())
        assert score < 0.3

    def test_mixed_history(self):
        history = [
            {"success": True, "confidence": 0.8},
            {"success": False, "confidence": 0.3},
        ]
        score = _compute_replay_score(history)
        # success_rate = 0.5, avg_confidence = 0.55
        # blended = 0.6*0.5 + 0.4*0.55 = 0.52
        assert 0.4 < score < 0.7


# ---------------------------------------------------------------------------
# Test 8: Replay score with no history
# ---------------------------------------------------------------------------

class TestReplayScoreNoHistory:
    def test_none_returns_neutral(self):
        score = _compute_replay_score(None)
        assert score == 0.7

    def test_empty_list_returns_neutral(self):
        # Empty list has total=0, falls to the total==0 branch
        score = _compute_replay_score([])
        assert score == 0.7


# ---------------------------------------------------------------------------
# Test 9: Level thresholds (HIGH / MEDIUM / LOW / UNCERTAIN)
# ---------------------------------------------------------------------------

class TestLevelThresholds:
    @pytest.mark.parametrize(
        "score, expected_level",
        [
            (1.0, "HIGH"),
            (0.8, "HIGH"),
            (0.85, "HIGH"),
            (0.79, "MEDIUM"),
            (0.5, "MEDIUM"),
            (0.6, "MEDIUM"),
            (0.49, "LOW"),
            (0.3, "LOW"),
            (0.35, "LOW"),
            (0.29, "UNCERTAIN"),
            (0.0, "UNCERTAIN"),
            (0.1, "UNCERTAIN"),
        ],
    )
    def test_level_from_score(self, score, expected_level):
        assert _level_from_score(score) == expected_level


# ---------------------------------------------------------------------------
# Test 10: Confidence decomposition structure
# ---------------------------------------------------------------------------

class TestConfidenceDecompositionStructure:
    def test_result_has_all_expected_factors(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM t",
            fallback_used=False,
        )
        factor_names = {f.name for f in result.factors}
        expected = {
            "contract_alignment",
            "audit_quality",
            "goal_term_coverage",
            "intake_clarity",
            "mapping_directness",
            "replay_strength",
        }
        assert factor_names == expected

    def test_weights_sum_to_one(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.9,
        )
        total_weight = sum(f.weight for f in result.factors)
        assert abs(total_weight - 1.0) < 0.001

    def test_to_dict_has_required_keys(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
        )
        d = result.to_dict()
        assert "overall" in d
        assert "level" in d
        assert "factors" in d
        assert "penalties" in d
        assert "reasoning" in d
        assert isinstance(d["factors"], list)
        assert isinstance(d["overall"], float)

    def test_factor_to_dict_has_required_keys(self):
        factor = ConfidenceFactor(
            name="test",
            score=0.8,
            weight=0.25,
            reason="test reason",
            penalty_applied=0.2,
        )
        d = factor.to_dict()
        assert d["name"] == "test"
        assert d["score"] == 0.8
        assert d["weight"] == 0.25
        assert d["reason"] == "test reason"
        assert d["penalty_applied"] == 0.2


# ---------------------------------------------------------------------------
# Test 11: High confidence with contract violations is impossible
# ---------------------------------------------------------------------------

class TestHighConfidenceWrongImpossible:
    def test_contract_violations_prevent_high_confidence(self):
        """Demonstrate that contract violations + HIGH confidence is impossible.

        With 2 violations, contract_score = max(0.1, 1.0 - 0.50) = 0.50.
        The min-factor floor limits overall to 0.50 * 1.5 = 0.75, which is
        below the 0.80 threshold for HIGH confidence. This is a design
        invariant: the system cannot claim HIGH confidence when the SQL
        drifts from the semantic contract.
        """
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=2),
            audit_result=_strong_audit(),
            intake_confidence=1.0,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        assert result.level != "HIGH"
        assert result.overall < 0.8

    def test_audit_failures_prevent_high_confidence(self):
        """Audit failures cap audit_score at 0.4, and the min-factor floor
        limits overall to 0.4 * 1.5 = 0.60 -- below HIGH threshold."""
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_weak_audit(failures=3),
            intake_confidence=1.0,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        assert result.level != "HIGH"
        assert result.overall < 0.8

    def test_fallback_used_prevents_high_confidence(self):
        """Fallback score = 0.5 => floor = 0.75 => cannot reach HIGH."""
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=1.0,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=True,
            replay_history=_good_replay(),
        )
        assert result.level != "HIGH"
        assert result.overall <= 0.75


# ---------------------------------------------------------------------------
# Test 12: Penalties list populated
# ---------------------------------------------------------------------------

class TestPenaltiesListPopulated:
    def test_contract_violation_adds_penalty(self):
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=3),
        )
        assert any("Contract drift" in p for p in result.penalties)

    def test_no_contract_validation_adds_penalty(self):
        result = compute_calibrated_confidence(
            contract_validation=None,
        )
        assert any("No contract validation" in p for p in result.penalties)

    def test_audit_failure_adds_penalty(self):
        result = compute_calibrated_confidence(
            audit_result=_weak_audit(failures=1),
        )
        assert any("Audit failures" in p for p in result.penalties)

    def test_fallback_adds_penalty(self):
        result = compute_calibrated_confidence(
            fallback_used=True,
        )
        assert any("Fallback" in p for p in result.penalties)

    def test_low_goal_coverage_adds_penalty(self):
        result = compute_calibrated_confidence(
            goal_text="refund university december special custom workflow",
            sql_text="SELECT 1",
        )
        assert any("goal-term coverage" in p for p in result.penalties)

    def test_weak_replay_adds_penalty(self):
        result = compute_calibrated_confidence(
            replay_history=_bad_replay(),
        )
        assert any("Weak replay" in p for p in result.penalties)


# ---------------------------------------------------------------------------
# Test 13: Overall score is bounded [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestOverallScoreBounded:
    def test_minimum_bound(self):
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=10),
            audit_result=_weak_audit(failures=10),
            intake_confidence=0.0,
            goal_text="xyzzy foobar baz quux widget nonce",
            sql_text="SELECT 1",
            fallback_used=True,
            replay_history=_bad_replay(),
        )
        assert 0.0 <= result.overall <= 1.0

    def test_maximum_bound(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=1.0,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        assert 0.0 <= result.overall <= 1.0


# ---------------------------------------------------------------------------
# Test 14: Reasoning string contains factor info
# ---------------------------------------------------------------------------

class TestReasoningString:
    def test_reasoning_contains_factor_names(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
        )
        for factor in result.factors:
            assert factor.name in result.reasoning

    def test_reasoning_contains_level(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
        )
        assert result.level in result.reasoning

    def test_reasoning_contains_penalties_when_present(self):
        result = compute_calibrated_confidence(
            contract_validation=_weak_contract(n_violations=1),
            fallback_used=True,
        )
        assert "Penalties:" in result.reasoning
        assert "Contract drift" in result.reasoning
        assert "Fallback" in result.reasoning


# ---------------------------------------------------------------------------
# Test 15: Intake confidence clamping
# ---------------------------------------------------------------------------

class TestIntakeConfidenceClamping:
    def test_intake_above_one_clamped(self):
        result = compute_calibrated_confidence(
            intake_confidence=1.5,
        )
        intake_factor = next(f for f in result.factors if f.name == "intake_clarity")
        assert intake_factor.score == 1.0

    def test_intake_below_zero_clamped(self):
        result = compute_calibrated_confidence(
            intake_confidence=-0.3,
        )
        intake_factor = next(f for f in result.factors if f.name == "intake_clarity")
        assert intake_factor.score == 0.0


# ---------------------------------------------------------------------------
# Test 16: Default arguments produce valid result
# ---------------------------------------------------------------------------

class TestDefaultArguments:
    def test_all_defaults(self):
        """compute_calibrated_confidence() with no args should not crash."""
        result = compute_calibrated_confidence()
        assert isinstance(result, CalibratedConfidence)
        assert 0.0 <= result.overall <= 1.0
        assert result.level in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN")
        assert len(result.factors) == 6

    def test_default_contract_penalty(self):
        """No contract_validation passed => 'No contract validation' penalty."""
        result = compute_calibrated_confidence()
        assert "No contract validation performed" in result.penalties


# ---------------------------------------------------------------------------
# Test 17: CalibratedConfidence dataclass round-trip
# ---------------------------------------------------------------------------

class TestCalibratedConfidenceDataclass:
    def test_to_dict_round_trip(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
            intake_confidence=0.85,
            goal_text="payment platform",
            sql_text="SELECT platform_name, SUM(payment_amount) FROM t",
            fallback_used=False,
            replay_history=_good_replay(),
        )
        d = result.to_dict()

        # Ensure serializable (no dataclass objects in the dict)
        assert isinstance(d["overall"], float)
        assert isinstance(d["level"], str)
        for factor_dict in d["factors"]:
            assert isinstance(factor_dict, dict)
            assert "name" in factor_dict
            assert "score" in factor_dict
            assert "weight" in factor_dict
            assert "reason" in factor_dict

    def test_overall_is_rounded_in_dict(self):
        result = compute_calibrated_confidence(
            contract_validation=_strong_contract(),
            audit_result=_strong_audit(),
        )
        d = result.to_dict()
        # overall should have at most 4 decimal places
        overall_str = str(d["overall"])
        if "." in overall_str:
            decimals = len(overall_str.split(".")[1])
            assert decimals <= 4


# ---------------------------------------------------------------------------
# Test 18: Monotonicity -- adding violations never increases confidence
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_more_violations_lower_confidence(self):
        scores = []
        for n in range(5):
            result = compute_calibrated_confidence(
                contract_validation=_weak_contract(n_violations=n) if n > 0 else _strong_contract(),
                audit_result=_strong_audit(),
                intake_confidence=0.9,
                goal_text="payment platform",
                sql_text="SELECT platform_name, SUM(payment_amount) FROM test_1_1_merged GROUP BY platform_name",
                fallback_used=False,
                replay_history=_good_replay(),
            )
            scores.append(result.overall)

        # Each additional violation should result in equal or lower confidence
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1], (
                f"Adding violation {i} increased confidence: {scores[i - 1]} -> {scores[i]}"
            )
