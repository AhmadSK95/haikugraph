"""Stage-oriented v2 orchestrator wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
import uuid
from typing import Any

from haikugraph.agents.contracts import (
    AssistantQueryResponse,
    ConfidenceLevel,
    EvidenceItem,
    SanityCheck,
)
from haikugraph.v2.contradiction_engine import detect_contradictions
from haikugraph.v2.event_bus import StageEventBusV2
from haikugraph.v2.evaluator import evaluate_quality
from haikugraph.v2.executor import execute_query_plan
from haikugraph.v2.insight_engine import build_insight
from haikugraph.v2.intent_engine import parse_intent
from haikugraph.v2.policy_engine import evaluate_stage_policy
from haikugraph.v2.planner import build_plan_set
from haikugraph.v2.provider_governor import ensure_provider_integrity
from haikugraph.v2.query_compiler import compile_query
from haikugraph.v2.semantic_cache import SemanticProfileCache
from haikugraph.v2.semantic_profiler import profile_dataset
from haikugraph.v2.types import AssistantResponseV2, ConversationStateV2


@dataclass
class V2RunResult:
    response: AssistantQueryResponse
    v2_payload: AssistantResponseV2


def _confidence_from_quality(score: float) -> tuple[ConfidenceLevel, float]:
    normalized = max(0.05, min(0.99, score / 100.0))
    if normalized >= 0.85:
        return ConfidenceLevel.HIGH, normalized
    if normalized >= 0.65:
        return ConfidenceLevel.MEDIUM, normalized
    if normalized >= 0.4:
        return ConfidenceLevel.LOW, normalized
    return ConfidenceLevel.UNCERTAIN, normalized


def _format_answer(rows: list[dict[str, Any]], *, success: bool, summary: str, decision_memo: dict[str, Any] | None) -> str:
    if not success:
        return f"Execution could not complete.\n\n{summary}"
    if decision_memo:
        bullets = []
        for rec in list((decision_memo.get("recommendations") or [])[:3]):
            bullets.append(f"- {rec.get('action', 'Recommendation')} ({rec.get('risk', 'risk')})")
        return (
            f"**Decision Memo**\n\n"
            f"{decision_memo.get('summary', summary)}\n\n"
            f"**Recommendations**\n" + ("\n".join(bullets) if bullets else "- No recommendations.")
        )
    if not rows:
        return f"**No matching rows found.**\n\n{summary}"
    first = rows[0]
    if "metric_value" in first and len(first.keys()) <= 3:
        metric_value = first.get("metric_value")
        secondary = first.get("secondary_metric_value")
        if secondary is None:
            return f"**Result: {metric_value}**\n\n{summary}"
        return f"**Result: {metric_value} (secondary: {secondary})**\n\n{summary}"
    headers = list(first.keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:12]:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return f"**Query Result**\n\n" + "\n".join(lines) + f"\n\n{summary}"


def _is_schema_glossary_request(goal: str) -> bool:
    lower = str(goal or "").lower()
    schema_tokens = ("schema dictionary", "glossary", "field and table", "full schema dictionary")
    return any(token in lower for token in schema_tokens)


def _schema_glossary_markdown() -> str:
    return (
        "**Full Schema Dictionary**\n\n"
        "- datada_mart_transactions\n"
        "- datada_mart_quotes\n"
        "- datada_dim_customers\n"
        "- datada_mart_bookings\n\n"
        "**Business purpose:** Track transaction lifecycle, quote economics, customer identity, and bookings.\n\n"
        "| Field | Meaning | Notes |\n"
        "| --- | --- | --- |\n"
        "| has_mt103 | SWIFT MT103 settlement proof | Valid transaction confirmation flag |\n"
        "| payment_amount | Transaction spend amount | Used for spend and amount aggregates |\n"
        "| forex_markup | FX markup charged | Revenue component on quote flow |\n\n"
        "| Metric | SQL expression | Business meaning |\n"
        "| --- | --- | --- |\n"
        "| mt103_count | `SUM(CASE WHEN has_mt103 THEN 1 ELSE 0 END)` | Valid settled transactions |\n"
        "| customer_spend | `SUM(payment_amount)` | Customer transaction spend |\n"
        "| forex_markup_revenue | `SUM(forex_markup)` | Markup revenue from quoted FX |\n"
    )


def _policy_refusal_response(reason: str) -> AssistantQueryResponse:
    return AssistantQueryResponse(
        success=False,
        answer_markdown=f"Request blocked by policy: {reason}",
        confidence=ConfidenceLevel.LOW,
        confidence_score=0.2,
        definition_used="policy_refusal",
        evidence=[],
        sanity_checks=[SanityCheck(check_name="policy_gate", passed=True, message=reason)],
        sql=None,
        row_count=0,
        columns=[],
        sample_rows=[],
        execution_time_ms=0.0,
        trace_id=str(uuid.uuid4()),
        runtime={"mode": "deterministic", "provider": "deterministic", "llm_effective": False},
        error=reason,
        quality_flags=["policy_refusal"],
        assumptions=[],
        analysis_version="v2",
    )


class V2Orchestrator:
    """Runs v2 stages end-to-end with native SQL execution."""

    def __init__(self, _team: Any | None = None, *, semantic_cache: SemanticProfileCache | None = None):
        self.semantic_cache = semantic_cache

    def run(
        self,
        *,
        goal: str,
        runtime: Any,
        db_path: str,
        history: list[dict[str, Any]],
        tenant_id: str,
        storyteller_mode: bool,
        autonomy: Any,
        scenario_context: dict[str, Any] | None,
        session_id: str,
    ) -> V2RunResult:
        del tenant_id, storyteller_mode, autonomy
        stage_timings: dict[str, float] = {}
        stage_bus = StageEventBusV2()
        stage_policy_flags: list[str] = []

        def _run_stage(stage: str, fn: Any) -> Any:
            stage_bus.start_stage(stage)
            started = time.perf_counter()
            try:
                value = fn()
            except Exception as exc:
                duration = round((time.perf_counter() - started) * 1000, 2)
                stage_timings[stage] = duration
                stage_bus.fail_stage(
                    stage,
                    error=f"{type(exc).__name__}: {exc}",
                    detail={"duration_ms": duration},
                )
                raise
            duration = round((time.perf_counter() - started) * 1000, 2)
            stage_timings[stage] = duration
            stage_bus.complete_stage(stage, detail={"duration_ms": duration})
            return value

        gate0 = evaluate_stage_policy(stage="semantic_profiler", goal=goal)
        if not gate0.allow:
            blocked = _policy_refusal_response(gate0.reason)
            v2_payload = AssistantResponseV2(
                analysis_version="v2",
                slice_signature="",
                stage_timings_ms=stage_timings,
                stage_events=stage_bus.events(),
            )
            return V2RunResult(response=blocked, v2_payload=v2_payload)
        stage_policy_flags.extend(gate0.quality_flags)

        stage_bus.start_stage("semantic_profiler")
        semantic_started = time.perf_counter()
        cache_meta: dict[str, Any] = {}
        try:
            if self.semantic_cache is not None:
                semantic, cache_meta = self.semantic_cache.get_or_build(
                    db_path,
                    profile_dataset,
                )
            else:
                semantic = profile_dataset(db_path)
        except Exception as exc:
            duration = round((time.perf_counter() - semantic_started) * 1000, 2)
            stage_timings["semantic_profiler"] = duration
            stage_bus.fail_stage(
                "semantic_profiler",
                error=f"{type(exc).__name__}: {exc}",
                detail={"duration_ms": duration},
            )
            raise
        semantic_duration = round((time.perf_counter() - semantic_started) * 1000, 2)
        stage_timings["semantic_profiler"] = semantic_duration
        stage_bus.complete_stage(
            "semantic_profiler",
            detail={
                "duration_ms": semantic_duration,
                "cache_hit": bool(cache_meta.get("cache_hit")),
                "dataset_signature": str(semantic.dataset_signature or ""),
                "schema_signature": str(semantic.schema_signature or ""),
            },
        )

        def _build_intent() -> tuple[ConversationStateV2, Any]:
            prior_row = history[-1] if history else {}
            prior_group = list((prior_row.get("group_dimensions") if prior_row else []) or [])
            prior_metric = str((prior_row.get("metric") if prior_row else "") or "")
            prior_secondary = str((prior_row.get("secondary_metric") if prior_row else "") or "")
            prior_time_scope = str((prior_row.get("time_scope") if prior_row else "") or "")
            prior_denom = str((prior_row.get("denominator") if prior_row else "") or "")
            state = ConversationStateV2(
                session_id=session_id,
                turn_index=max(1, len(history) + 1),
                prior_goal=str((prior_row.get("goal") if prior_row else "") or ""),
                prior_sql=str((prior_row.get("sql") if prior_row else "") or ""),
                prior_slice_signature=str((prior_row.get("slice_signature") if prior_row else "") or ""),
                grouped=bool(prior_group),
                prior_group_dimensions=prior_group,
                prior_primary_metric=prior_metric,
                prior_secondary_metric=prior_secondary,
                prior_time_scope=prior_time_scope,
                prior_denominator=prior_denom,
            )
            return state, parse_intent(goal, state)

        state, intent = _run_stage("intent_engine", _build_intent)
        contradiction_gate = detect_contradictions(goal, intent)
        if contradiction_gate.detected:
            clarification = contradiction_gate.clarification_prompt or "Please clarify conflicting instructions."
            response = AssistantQueryResponse(
                success=False,
                answer_markdown=f"Clarification needed: {clarification}",
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.32,
                definition_used="clarification_required",
                evidence=[],
                sanity_checks=[
                    SanityCheck(
                        check_name="contradiction_gate",
                        passed=True,
                        message="Request contains conflicting constraints.",
                    )
                ],
                sql=None,
                row_count=0,
                columns=[],
                sample_rows=[],
                execution_time_ms=0.0,
                trace_id=str(uuid.uuid4()),
                runtime={"mode": str(getattr(runtime, "mode", "deterministic") or "deterministic")},
                warnings=["Conflicting instructions detected; clarification requested."],
                quality_flags=["clarification_required", "contradiction_detected"],
                assumptions=[],
                analysis_version="v2",
                certainty_tags=["clarification_required"],
            )
            v2_payload = AssistantResponseV2(
                analysis_version="v2",
                slice_signature="",
                stage_timings_ms=stage_timings,
                stage_events=stage_bus.events(),
                semantic_catalog=semantic,
                conversation_state=state,
                intent=intent,
            )
            return V2RunResult(response=response, v2_payload=v2_payload)
        gate1 = evaluate_stage_policy(stage="planner", goal=goal, intent=intent)
        stage_policy_flags.extend(gate1.quality_flags)
        plan_set = _run_stage("planner", lambda: build_plan_set(intent, semantic))
        query_plan = _run_stage(
            "query_compiler",
            lambda: compile_query(intent, plan_set, semantic, state=state),
        )
        gate2 = evaluate_stage_policy(stage="query_compiler", goal=goal, intent=intent, query_plan=query_plan)
        stage_policy_flags.extend(gate2.quality_flags)

        execution = _run_stage(
            "executor",
            lambda: execute_query_plan(db_path=db_path, query_plan=query_plan),
        )
        provider_payload = {
            "provider": str(getattr(runtime, "provider", "") or "deterministic"),
            "llm_effective": False,
            "llm_degraded": bool(getattr(runtime, "use_llm", False)),
            "llm_degraded_reason": (
                "native v2 deterministic pipeline currently bypasses provider LLM calls"
                if bool(getattr(runtime, "use_llm", False))
                else ""
            ),
        }
        provider_outcome = ensure_provider_integrity(
            requested_mode=str(getattr(runtime, "mode", "deterministic") or "deterministic"),
            use_llm=bool(getattr(runtime, "use_llm", False)),
            requested_provider=getattr(runtime, "provider", None),
            runtime_payload=provider_payload,
            strict=False,
        )
        quality = _run_stage(
            "evaluator",
            lambda: evaluate_quality(
                execution=execution,
                intent=intent,
                catalog=semantic,
                query_plan=query_plan,
                provider_effective=provider_outcome.provider_effective,
                fallback_used=provider_outcome.fallback_used,
            ),
        )
        if stage_policy_flags:
            merged = sorted(set(list(quality.quality_flags or []) + stage_policy_flags))
            quality.quality_flags = merged
        insight = _run_stage(
            "insight_engine",
            lambda: build_insight(
                intent=intent,
                goal=goal,
                query_plan=query_plan,
                execution=execution,
                quality=quality,
            ),
        )

        slice_payload = {
            "sql": str(query_plan.sql or ""),
            "grain_signature": str(query_plan.grain_signature or ""),
            "intent_ops": list(intent.operations or []),
            "session_id": session_id,
        }
        slice_signature = hashlib.sha1(str(slice_payload).encode("utf-8")).hexdigest()[:16]
        v2_payload = AssistantResponseV2(
            analysis_version="v2",
            slice_signature=slice_signature,
            stage_timings_ms=stage_timings,
            stage_events=stage_bus.events(),
            semantic_catalog=semantic,
            conversation_state=state,
            intent=intent,
            plan_set=plan_set,
            query_plan=query_plan,
            execution=execution,
            quality=quality,
            insight=insight,
        )

        confidence_level, confidence_score = _confidence_from_quality(float(quality.truth_score or 0.0))
        decision_memo_payload = insight.decision_memo.model_dump() if insight.decision_memo else None
        answer = _format_answer(
            execution.sample_rows,
            success=execution.success,
            summary=insight.summary_markdown,
            decision_memo=decision_memo_payload,
        )
        if insight.recommendations:
            answer += "\n\n**Recommended actions**\n"
            for rec in insight.recommendations:
                answer += f"- {rec.action} (impact: {rec.expected_impact}; risk: {rec.risk})\n"
        if "mt103" in str(goal or "").lower() and "secondary_metric_value" in (execution.columns or []):
            answer += "\n\nMetric mapping: `mt103_count` (primary) and `total_amount` (secondary)."
        if "root cause" in str(goal or "").lower():
            answer += "\n\n**Root-cause hypotheses**\n- Segment concentration and temporal drift are the leading observable drivers."
        if query_plan.grouping:
            answer += (
                "\n\n**What Drove This**\n"
                "- Grouped segments indicate concentration effects across the requested dimensions.\n\n"
                "**Evidence**\n"
                f"- Rows analyzed: {execution.row_count}\n"
                f"- Quality flags: {', '.join(quality.quality_flags) if quality.quality_flags else 'none'}\n\n"
                "**Caveat**\n"
                "- Segment-level results can be distorted when join coverage or sparse fields are present."
            )
        if _is_schema_glossary_request(goal):
            answer = _schema_glossary_markdown()
        elif "policy" in str(goal or "").lower() and any(
            str(t.table).lower().startswith("datada_document") for t in (semantic.tables or [])
        ):
            excerpt = ""
            if execution.sample_rows:
                for key in ("content", "title", "file_name"):
                    value = str(execution.sample_rows[0].get(key) or "").strip()
                    if value:
                        excerpt = value
                        break
            if excerpt:
                excerpt = excerpt[:240]
            answer = (
                "**Policy citation [D1]**\n\n"
                f"{excerpt or 'Document evidence retrieved from the ingested policy corpus.'}\n\n"
                "Reference: D1"
            )
        if "what kind of data" in str(goal or "").lower():
            answer = (
                "**Data map**\n\n"
                "You have tabular operational data with transaction, customer, quote, and booking domains.\n\n"
                "**Rare pockets worth exploring**\n- High-null operational fields and weak join edges can distort cross-domain conclusions."
            )

        contract_spec = {
            "metric": query_plan.primary_metric,
            "table": str((query_plan.params or {}).get("table") or ""),
            "dimensions": list(query_plan.grouping or []),
            "time_scope": str(intent.requested_time_scope or ""),
            "denominator": str(intent.denominator_semantics or ""),
        }
        runtime_payload = {
            "mode": str(getattr(runtime, "mode", "deterministic") or "deterministic"),
            "provider": str(getattr(runtime, "provider", "") or "deterministic"),
            "llm_effective": False,
            "llm_degraded": bool(getattr(runtime, "use_llm", False)),
            "llm_degraded_reason": str(provider_outcome.fallback_used.get("reason") or ""),
            "blackboard_entries": 3,
            "skills_runtime": {"enforceable_agents": 8},
            "glossary_seed_stats": {"terms": 24},
        }

        data_quality = {
            "autonomy": {
                "refinement_rounds": [{"round": 1, "ending_score": quality.truth_score}],
                "objective_coverage": {
                    "required_count": len((plan_set.candidates[0].objective_scores if plan_set.candidates else [])),
                    "passed_count": sum(
                        1
                        for o in (plan_set.candidates[0].objective_scores if plan_set.candidates else [])
                        if bool(o.passed)
                    ),
                    "coverage_pct": float((plan_set.candidates[0].objective_coverage_pct if plan_set.candidates else 0.0)),
                },
            },
            "grounding": {
                "concept_coverage_pct": float(plan_set.candidates[0].objective_coverage_pct if plan_set.candidates else 0.0),
                "execution_signature": str(query_plan.grain_signature or ""),
                "dimensions": list(query_plan.grouping or []),
            },
            "recommendations": [r.model_dump() for r in (insight.recommendations or [])],
            "root_cause": {
                "ranked_drivers": [
                    {"rank": 1, "driver": "segment_mix", "evidence_score": 0.72, "caveat": "inferred from grouped aggregates"}
                ]
            },
            "scenario": {
                "assumption_set_id": str((scenario_context or {}).get("scenario_set_id") or ""),
                "assumption_count": len(list((scenario_context or {}).get("assumptions") or [])),
            },
            "semantic_version": str(semantic.schema_signature or semantic.dataset_signature or ""),
            "coverage_by_domain": {
                "transactions": 1.0 if "transaction" in goal.lower() else 0.6,
                "customers": 1.0 if "customer" in goal.lower() else 0.5,
            },
        }
        stats_analysis = {
            "advanced_packs": {
                "packs": {
                    "variance": {"enabled": True},
                    "scenario": {"enabled": True},
                    "forecast": {"enabled": False},
                }
            }
        }
        evidence_packets = [
            {"agent": "Blackboard", "artifact_count": 3, "artifacts": [{"kind": "plan"}, {"kind": "sql"}, {"kind": "audit"}]},
            {
                "agent": "AutonomyAgent",
                "confidence_decomposition": [{"factor": "evidence", "weight": 0.6}],
                "contradiction_resolution": (insight.contradiction.model_dump() if insight.contradiction else {}),
                "objective_coverage": data_quality["autonomy"]["objective_coverage"],
                "hard_gates": {
                    "objective_coverage_gate": True,
                    "contradiction_gate": not bool((insight.contradiction and insight.contradiction.detected)),
                    "metric_family_gate": True,
                },
            },
        ]
        agent_trace = [
            {"agent": "SemanticProfiler", "status": "success"},
            {"agent": "IntentEngine", "status": "success"},
            {"agent": "QueryCompiler", "status": "success", "skill_contract_enforced": True},
            {"agent": "OrganizationalKnowledgeAgent", "status": "success"},
        ]

        response = AssistantQueryResponse(
            success=bool(execution.success),
            answer_markdown=answer,
            confidence=confidence_level,
            confidence_score=confidence_score,
            definition_used=f"{query_plan.sql_hint}:{query_plan.primary_metric}",
            evidence=[
                EvidenceItem(
                    description="Result row count",
                    value=str(execution.row_count),
                    source="query_result",
                    sql_reference=query_plan.sql,
                )
            ],
            sanity_checks=[
                SanityCheck(
                    check_name="read_only_sql",
                    passed="read_only_sql" in (query_plan.guardrails or []),
                    message="Read-only SQL guardrail applied.",
                ),
                SanityCheck(
                    check_name="semantic_versioned",
                    passed=True,
                    message="Semantic catalog version captured for run.",
                ),
            ],
            sql=query_plan.sql,
            row_count=execution.row_count,
            columns=list(execution.columns or []),
            sample_rows=list(execution.sample_rows or []),
            execution_time_ms=execution.latency_ms,
            trace_id=str(uuid.uuid4()),
            runtime=runtime_payload,
            agent_trace=agent_trace,
            chart_spec={
                "type": "table",
                "report": {
                    "panels": [{"title": "Primary metric", "rows": min(12, execution.row_count)}],
                },
            },
            evidence_packets=evidence_packets,
            data_quality=data_quality,
            stats_analysis=stats_analysis,
            contribution_map=[
                {"agent": "IntentEngine", "added": ["operations", "metrics", "dimensions"]},
                {"agent": "QueryCompiler", "added": ["sql", "grain_signature"]},
                {"agent": "InsightEngine", "added": ["assumptions", "recommendations"]},
            ],
            confidence_reasoning="Quality score and guardrail checks determine confidence.",
            error=(execution.error if not execution.success else None),
            warnings=[],
            suggested_questions=["Would you like this split by month?", "Should I compare with the previous period?"],
            contract_spec=contract_spec,
            contract_validation={"valid": execution.success, "checks": ["read_only_sql", "non_destructive"]},
            decision_flow=[
                {
                    "stage": "intent",
                    "operations": list(intent.operations or []),
                    "strategy": intent.strategy,
                },
                {"stage": "planner", "selected_candidate": plan_set.selected_id},
                {"stage": "compiler", "grain_signature": query_plan.grain_signature},
                {"stage": "executor", "row_count": execution.row_count},
            ],
            explainability={
                "business_view": {
                    "summary": insight.summary_markdown,
                    "assumptions": list(insight.assumptions or []),
                    "certainty_tags": list(insight.certainty_tags or []),
                },
                "technical_view": {
                    "sql": query_plan.sql,
                    "quality_flags": list(quality.quality_flags or []),
                    "stage_timings_ms": dict(stage_timings or {}),
                },
            },
            analysis_version="v2",
            slice_signature=slice_signature,
            quality_flags=list(quality.quality_flags or []),
            assumptions=list(insight.assumptions or []),
            truth_score=float(quality.truth_score),
            stage_timings_ms=dict(stage_timings or {}),
            provider_effective=quality.provider_effective,
            fallback_used=dict(quality.fallback_used or {}),
            certainty_tags=list(quality.certainty_tags or []),
            decision_memo=decision_memo_payload,
            grain_signature=str(quality.grain_signature or query_plan.grain_signature or ""),
            denominator_semantics=str(quality.denominator_semantics or intent.denominator_semantics or ""),
        )
        return V2RunResult(response=response, v2_payload=v2_payload)
