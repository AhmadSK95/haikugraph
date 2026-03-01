"""Query compiler hints for v2 stage diagnostics."""

from __future__ import annotations

from haikugraph.v2.types import IntentSpecV2, PlanSetV2, QueryPlanV2


def compile_query_hint(intent: IntentSpecV2, plans: PlanSetV2) -> QueryPlanV2:
    del plans
    guardrails = ["read_only_sql", "bounded_result_size"]
    if intent.requires_validity_guard:
        guardrails.append("validity_guard_mt103")
    if intent.is_followup:
        guardrails.append("followup_scope_carryover")
    return QueryPlanV2(sql_hint="delegate_to_v1_engine", guardrails=guardrails)

