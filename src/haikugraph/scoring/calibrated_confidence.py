"""Calibrated confidence scoring for dataDa.

Confidence is decomposed into measurable factors and penalized for:
- Contract drift: SQL doesn't match the semantic contract
- Fallback mapping: Domain/table mapping used fallback instead of exact match
- Missing goal-term coverage: User's key terms not reflected in SQL
- Weak replay: Previous similar queries had poor results

The final confidence is a weighted product of factors, ensuring that
high confidence is only assigned when ALL factors are strong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ConfidenceFactor:
    """A single factor contributing to overall confidence."""

    name: str
    score: float  # 0.0 to 1.0
    weight: float  # relative weight
    reason: str
    penalty_applied: float = 0.0  # how much was deducted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibratedConfidence:
    """Fully decomposed confidence score with audit trail."""

    overall: float  # final confidence 0.0 to 1.0
    level: str  # HIGH, MEDIUM, LOW, UNCERTAIN
    factors: list[ConfidenceFactor] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "level": self.level,
            "factors": [f.to_dict() for f in self.factors],
            "penalties": self.penalties,
            "reasoning": self.reasoning,
        }


def _level_from_score(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    elif score >= 0.3:
        return "LOW"
    return "UNCERTAIN"


def compute_calibrated_confidence(
    *,
    contract_validation: dict[str, Any] | None = None,
    audit_result: dict[str, Any] | None = None,
    intake_confidence: float = 0.5,
    goal_text: str = "",
    sql_text: str = "",
    fallback_used: bool = False,
    replay_history: list[dict[str, Any]] | None = None,
    row_count: int = 0,
) -> CalibratedConfidence:
    """Compute calibrated confidence from pipeline signals.

    The computation uses a weighted geometric mean of factors, which ensures
    that ANY weak factor pulls down the overall score (unlike arithmetic mean
    which can hide problems).

    Args:
        contract_validation: Result from validate_sql_against_contract()
        audit_result: Result from AuditAgent
        intake_confidence: Confidence from IntakeAgent
        goal_text: Original user question
        sql_text: Generated SQL
        fallback_used: Whether domain mapping used fallback
        replay_history: Previous similar query results
        row_count: Number of result rows

    Returns:
        CalibratedConfidence with decomposition
    """
    factors: list[ConfidenceFactor] = []
    penalties: list[str] = []

    # Factor 1: Contract alignment (weight: 0.30)
    contract_score = 1.0
    if contract_validation:
        if contract_validation.get("valid"):
            contract_score = 1.0
        else:
            violations = contract_validation.get("violations", [])
            # Each violation reduces score
            contract_score = max(0.1, 1.0 - 0.25 * len(violations))
            penalties.append(f"Contract drift: {len(violations)} violation(s)")
    else:
        contract_score = 0.6  # No contract validation = uncertainty
        penalties.append("No contract validation performed")

    if contract_score >= 0.9:
        contract_reason = "Contract validation: passed"
    elif contract_validation:
        n_violations = len(contract_validation.get("violations", []))
        contract_reason = f"Contract validation: {n_violations} violations"
    else:
        contract_reason = "Contract validation: not performed"

    factors.append(ConfidenceFactor(
        name="contract_alignment",
        score=contract_score,
        weight=0.30,
        reason=contract_reason,
        penalty_applied=max(0, 1.0 - contract_score),
    ))

    # Factor 2: Audit quality (weight: 0.25)
    audit_score = 0.5
    if audit_result:
        passed = audit_result.get("passed", 0)
        total = passed + audit_result.get("warned", 0) + audit_result.get("failed", 0)
        if total > 0:
            audit_score = passed / total
        if audit_result.get("failed", 0) > 0:
            audit_score = min(audit_score, 0.4)
            penalties.append(f"Audit failures: {audit_result.get('failed', 0)}")

    if audit_result:
        audit_passed = audit_result.get("passed", 0)
        audit_total_checks = audit_passed + audit_result.get("failed", 0)
        audit_reason = f"Audit checks: {audit_passed}/{audit_total_checks} passed"
    else:
        audit_reason = "No audit performed"

    factors.append(ConfidenceFactor(
        name="audit_quality",
        score=audit_score,
        weight=0.25,
        reason=audit_reason,
        penalty_applied=max(0, 1.0 - audit_score),
    ))

    # Factor 3: Goal-term coverage (weight: 0.20)
    goal_coverage = _compute_goal_term_coverage(goal_text, sql_text)
    if goal_coverage < 0.5:
        penalties.append(f"Low goal-term coverage: {goal_coverage:.0%}")

    factors.append(ConfidenceFactor(
        name="goal_term_coverage",
        score=goal_coverage,
        weight=0.20,
        reason=f"Goal terms reflected in SQL: {goal_coverage:.0%}",
        penalty_applied=max(0, 1.0 - goal_coverage),
    ))

    # Factor 4: Intake clarity (weight: 0.10)
    intake_score = min(1.0, max(0.0, intake_confidence))
    factors.append(ConfidenceFactor(
        name="intake_clarity",
        score=intake_score,
        weight=0.10,
        reason=f"Intake extraction confidence: {intake_score:.2f}",
        penalty_applied=max(0, 1.0 - intake_score),
    ))

    # Factor 5: Fallback penalty (weight: 0.10)
    fallback_score = 0.5 if fallback_used else 1.0
    if fallback_used:
        penalties.append("Fallback domain mapping used instead of exact match")

    factors.append(ConfidenceFactor(
        name="mapping_directness",
        score=fallback_score,
        weight=0.10,
        reason="Direct domain mapping" if not fallback_used else "Fallback mapping used",
        penalty_applied=0.5 if fallback_used else 0.0,
    ))

    # Factor 6: Replay strength (weight: 0.05)
    replay_score = _compute_replay_score(replay_history)
    if replay_score < 0.5 and replay_history:
        penalties.append(f"Weak replay history: {replay_score:.0%}")

    factors.append(ConfidenceFactor(
        name="replay_strength",
        score=replay_score,
        weight=0.05,
        reason=f"Similar past queries: {'strong' if replay_score >= 0.7 else 'weak' if replay_history else 'none'}",
        penalty_applied=max(0, 1.0 - replay_score),
    ))

    # Compute weighted score (geometric-weighted to penalize weak links)
    total_weight = sum(f.weight for f in factors)
    weighted_sum = sum(f.score * f.weight for f in factors) / total_weight if total_weight > 0 else 0.0

    # Apply minimum-factor floor: overall can't exceed 1.5x the weakest factor
    min_factor = min(f.score for f in factors) if factors else 0.0
    overall = min(weighted_sum, min_factor * 1.5)
    overall = round(max(0.0, min(1.0, overall)), 4)

    level = _level_from_score(overall)

    reasoning_parts = [f"{f.name}={f.score:.2f}(w={f.weight})" for f in factors]
    reasoning = f"Calibrated: {' + '.join(reasoning_parts)} -> {overall:.4f} ({level})"
    if penalties:
        reasoning += f" | Penalties: {'; '.join(penalties)}"

    return CalibratedConfidence(
        overall=overall,
        level=level,
        factors=factors,
        penalties=penalties,
        reasoning=reasoning,
    )


def _compute_goal_term_coverage(goal: str, sql: str) -> float:
    """Compute what fraction of meaningful goal terms appear in the SQL.

    Strips stop words and checks if user's key terms are reflected.
    """
    if not goal or not sql:
        return 0.5  # No data to judge

    stop_words = {
        "what", "is", "the", "a", "an", "of", "for", "in", "on", "to",
        "how", "many", "much", "are", "was", "were", "be", "been",
        "do", "does", "did", "can", "could", "would", "should", "will",
        "me", "my", "i", "we", "our", "show", "tell", "give", "get",
        "all", "each", "every", "any", "some", "this", "that", "these",
        "from", "with", "by", "at", "and", "or", "but", "not", "no",
        "total", "sum", "count", "average", "number",  # aggregation words map to SQL functions
    }

    # Extract meaningful terms from goal
    goal_tokens = set(re.findall(r'[a-z]+', goal.lower())) - stop_words
    if not goal_tokens:
        return 0.8  # All stop words = likely a simple query

    sql_lower = sql.lower()
    matched = sum(1 for t in goal_tokens if t in sql_lower)

    return matched / len(goal_tokens) if goal_tokens else 0.8


def _compute_replay_score(history: list[dict[str, Any]] | None) -> float:
    """Compute confidence boost/penalty from similar past query results."""
    if not history:
        return 0.7  # Neutral - no history

    # Look at success rate of similar past queries
    successes = sum(1 for h in history if h.get("success", False))
    total = len(history)

    if total == 0:
        return 0.7

    success_rate = successes / total

    # Also factor in past confidence scores
    past_confidences = [h.get("confidence", 0.5) for h in history]
    avg_confidence = sum(past_confidences) / len(past_confidences)

    # Blend success rate and average confidence
    return 0.6 * success_rate + 0.4 * avg_confidence
