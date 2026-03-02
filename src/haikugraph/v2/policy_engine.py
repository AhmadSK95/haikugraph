"""Centralized policy engine for safety, governance, and quality gates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from haikugraph.v2.exceptions import PolicyViolationError


@dataclass(frozen=True)
class PolicyDecision:
    allow: bool
    action: str = "allow"
    reason: str = ""
    quality_flags: list[str] = field(default_factory=list)


_BLOCK_SQL_PATTERNS = (
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r"\btruncate\b",
    r"\binsert\s+into\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\balter\s+table\b",
)
_UNSUPPORTED_PATTERNS = (
    r"\bsentiment\b",
    r"\bstock price\b",
    r"\bcrypto price\b",
    r"\bexternal market\b",
    r"\bpredict\b",
    r"\bforecast\b",
    r"\bprojection\b",
)


def evaluate_stage_policy(
    *,
    stage: str,
    goal: str,
    intent: Any | None = None,
    query_plan: Any | None = None,
) -> PolicyDecision:
    """Evaluate policy once per stage and return a deterministic decision."""
    text = str(goal or "").strip()
    lower = text.lower()
    if not text:
        return PolicyDecision(allow=False, action="refuse", reason="Goal is empty.")

    for pattern in _BLOCK_SQL_PATTERNS:
        if re.search(pattern, lower):
            return PolicyDecision(
                allow=False,
                action="refuse",
                reason="Request contains a destructive operation. Read-only analytics only.",
                quality_flags=["policy_refusal", "destructive_intent"],
            )

    for pattern in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, lower):
            return PolicyDecision(
                allow=False,
                action="refuse",
                reason="Request is outside supported analytics scope for this dataset.",
                quality_flags=["policy_refusal", "unsupported_domain"],
            )

    if stage == "query_compiler" and query_plan is not None:
        sql = str(getattr(query_plan, "sql", "") or "").lower()
        for pattern in _BLOCK_SQL_PATTERNS:
            if re.search(pattern, sql):
                return PolicyDecision(
                    allow=False,
                    action="refuse",
                    reason="Compiled SQL failed read-only guardrails.",
                    quality_flags=["policy_refusal", "compiled_sql_blocked"],
                )

    if intent is not None and bool(getattr(intent, "requires_validity_guard", False)):
        return PolicyDecision(
            allow=True,
            action="allow",
            reason="Validity guard required by intent.",
            quality_flags=["semantic_guard_required"],
        )
    return PolicyDecision(allow=True)


def enforce_policy(
    *,
    stage: str,
    goal: str,
    intent: Any | None = None,
    query_plan: Any | None = None,
) -> PolicyDecision:
    decision = evaluate_stage_policy(stage=stage, goal=goal, intent=intent, query_plan=query_plan)
    if not decision.allow:
        raise PolicyViolationError(decision.reason or "Policy gate blocked the request.")
    return decision
