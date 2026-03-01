"""Intent parser for v2 using operation-oriented follow-up interpretation."""

from __future__ import annotations

import re

from haikugraph.v2.types import ConversationStateV2, IntentSpecV2


def parse_intent(goal: str, state: ConversationStateV2) -> IntentSpecV2:
    lower = str(goal or "").lower().strip()
    followup_markers = [
        "same slice",
        "same scope",
        "keep that",
        "keep same",
        "now",
        "also",
        "add",
        "switch metric",
    ]
    is_followup = bool(state.turn_index > 1 and any(m in lower for m in followup_markers))
    ops: list[str] = []
    if is_followup:
        ops.append("carry_scope")
        if str(state.prior_slice_signature or "").strip():
            ops.append("carry_slice_signature")
    if "switch metric" in lower:
        ops.append("switch_metric")
    if re.search(r"\b(add|include|also|plus)\b.*\b(total\s+)?(amount|value|revenue)\b", lower):
        ops.append("add_secondary_metric")
    if "same grouped output" in lower or "same grouping" in lower:
        ops.append("preserve_grouping")
    if re.search(r"\bvalid transaction(s)?\b", lower) or "mt103" in lower:
        ops.append("enforce_validity_guard")
    if re.search(r"\b(spend|amount|revenue|value)\b.*\btransaction", lower):
        ops.append("enforce_validity_guard")
    requires_validity_guard = "enforce_validity_guard" in ops
    strategy = "comparison" if any(k in lower for k in ["compare", "versus", " vs "]) else "metric"
    return IntentSpecV2(
        goal=goal,
        is_followup=is_followup,
        operations=sorted(set(ops)),
        requires_validity_guard=requires_validity_guard,
        strategy=strategy,
    )
