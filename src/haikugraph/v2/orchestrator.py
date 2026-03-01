"""Stage-oriented v2 orchestrator wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

from haikugraph.agents.contracts import AssistantQueryResponse
from haikugraph.v2.event_bus import StageEventBusV2
from haikugraph.v2.evaluator import evaluate_quality
from haikugraph.v2.executor import summarize_execution
from haikugraph.v2.insight_engine import build_insight
from haikugraph.v2.intent_engine import parse_intent
from haikugraph.v2.planner import build_plan_set
from haikugraph.v2.query_compiler import compile_query_hint
from haikugraph.v2.semantic_cache import SemanticProfileCache
from haikugraph.v2.semantic_profiler import profile_dataset
from haikugraph.v2.types import AssistantResponseV2, ConversationStateV2


@dataclass
class V2RunResult:
    response: AssistantQueryResponse
    v2_payload: AssistantResponseV2


class V2Orchestrator:
    """Runs v2 stages and delegates SQL answer generation to the existing engine."""

    def __init__(self, team: Any, *, semantic_cache: SemanticProfileCache | None = None):
        self.team = team
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
        stage_timings: dict[str, float] = {}
        stage_bus = StageEventBusV2()

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
            state = ConversationStateV2(
                session_id=session_id,
                turn_index=max(1, len(history) + 1),
                prior_goal=str((prior_row.get("goal") if prior_row else "") or ""),
                prior_sql=str((prior_row.get("sql") if prior_row else "") or ""),
                prior_slice_signature=str((prior_row.get("slice_signature") if prior_row else "") or ""),
                grouped=False,
            )
            return state, parse_intent(goal, state)

        _state, intent = _run_stage("intent_engine", _build_intent)
        plan_set = _run_stage("planner", lambda: build_plan_set(intent, semantic))
        query_hint = _run_stage("query_compiler", lambda: compile_query_hint(intent, plan_set))

        response = _run_stage(
            "executor_delegate",
            lambda: self.team.run(
                goal,
                runtime,
                tenant_id=tenant_id,
                conversation_context=history,
                storyteller_mode=storyteller_mode,
                autonomy=autonomy,
                scenario_context=scenario_context,
            ),
        )

        def _evaluate() -> tuple[Any, Any, Any]:
            exec_summary = summarize_execution(response)
            quality = evaluate_quality(response, intent, semantic)
            insight = build_insight(intent, goal)
            return exec_summary, quality, insight

        exec_summary, quality, insight = _run_stage("evaluator_insight", _evaluate)

        slice_payload = {
            "sql": str(response.sql or ""),
            "contract": dict(response.contract_spec or {}),
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
            intent=intent,
            plan_set=plan_set,
            query_plan=query_hint,
            execution=exec_summary,
            quality=quality,
            insight=insight,
        )
        return V2RunResult(response=response, v2_payload=v2_payload)
