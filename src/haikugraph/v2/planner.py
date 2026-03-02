"""Plan candidate generation with objective matrix and hard gates."""

from __future__ import annotations

import concurrent.futures

from haikugraph.v2.types import IntentSpecV2, ObjectiveScoreV2, PlanCandidateV2, PlanSetV2, SemanticCatalogV2


def _objective_matrix(intent: IntentSpecV2, catalog: SemanticCatalogV2, *, fallback: bool) -> list[ObjectiveScoreV2]:
    high_risk = int(catalog.quality_summary.get("high_risk_join_edges") or 0)
    sparse_tables = int(catalog.quality_summary.get("sparse_table_count") or 0)
    objectives = [
        ObjectiveScoreV2(
            objective_id="semantic_alignment",
            score=1.0 if intent.goal.strip() else 0.0,
            passed=bool(intent.goal.strip()),
            weight=0.35,
            detail="Goal parsed and aligned with semantic catalog." if intent.goal.strip() else "Empty goal.",
        ),
        ObjectiveScoreV2(
            objective_id="continuity_alignment",
            score=1.0 if (not intent.is_followup or "carry_scope" in (intent.operations or [])) else 0.0,
            passed=not intent.is_followup or "carry_scope" in (intent.operations or []),
            weight=0.25,
            detail="Follow-up operation model carries scope.",
        ),
        ObjectiveScoreV2(
            objective_id="join_quality",
            score=max(0.0, 1.0 - min(1.0, (high_risk * 0.25))),
            passed=high_risk <= 2,
            weight=0.2,
            detail=f"High-risk join edges: {high_risk}.",
        ),
        ObjectiveScoreV2(
            objective_id="data_sparsity",
            score=max(0.0, 1.0 - min(1.0, (sparse_tables * 0.15))),
            passed=sparse_tables <= 4,
            weight=0.1,
            detail=f"Sparse tables detected: {sparse_tables}.",
        ),
        ObjectiveScoreV2(
            objective_id="fallback_penalty",
            score=0.75 if fallback else 1.0,
            passed=True,
            weight=0.1,
            detail="Fallback candidate penalty applied." if fallback else "Primary candidate.",
        ),
    ]
    if intent.requires_validity_guard:
        objectives.append(
            ObjectiveScoreV2(
                objective_id="validity_guard_gate",
                score=1.0,
                passed=True,
                weight=0.15,
                detail="Semantic validity guard required and enforced at compile time.",
            )
        )
    return objectives


def _build_candidate(
    *,
    candidate_id: str,
    intent: IntentSpecV2,
    catalog: SemanticCatalogV2,
    strategy: str,
    fallback: bool,
) -> PlanCandidateV2:
    objectives = _objective_matrix(intent, catalog, fallback=fallback)
    weighted = [(o.score, o.weight) for o in objectives]
    denom = sum(weight for _, weight in weighted) or 1.0
    coverage = round((sum(score * weight for score, weight in weighted) / denom) * 100.0, 2)

    risk_flags: list[str] = []
    hard_failures: list[str] = []
    if int(catalog.quality_summary.get("high_risk_join_edges") or 0) > 0:
        risk_flags.append("join_fragility")
    if intent.requires_validity_guard:
        risk_flags.append("validity_guard_required")
    if intent.is_followup and "carry_scope" not in (intent.operations or []):
        risk_flags.append("continuity_risk")
        hard_failures.append("continuity_alignment")
    if coverage < 55.0:
        hard_failures.append("objective_coverage_floor")
    hard_gate_pass = not hard_failures
    return PlanCandidateV2(
        candidate_id=candidate_id,
        strategy=strategy,
        objective_coverage_pct=coverage,
        risk_flags=sorted(set(risk_flags + (["fallback"] if fallback else []))),
        objective_scores=objectives,
        hard_gate_pass=hard_gate_pass,
        hard_gate_failures=sorted(set(hard_failures)),
        sql_complexity_score=0.45 if strategy == "metric" else 0.62,
    )


def build_plan_set(intent: IntentSpecV2, catalog: SemanticCatalogV2) -> PlanSetV2:
    specs = [
        ("c1_primary", intent.strategy, False),
        ("c2_fallback", "metric", True),
    ]
    candidates: list[PlanCandidateV2] = []
    # Parallel candidate scoring keeps planner latency bounded for larger candidate sets.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(specs))) as pool:
        future_map = {
            pool.submit(
                _build_candidate,
                candidate_id=candidate_id,
                intent=intent,
                catalog=catalog,
                strategy=strategy,
                fallback=fallback,
            ): candidate_id
            for candidate_id, strategy, fallback in specs
        }
        for future in concurrent.futures.as_completed(future_map):
            candidate = future.result()
            candidates.append(candidate)
            # Early stop: if primary candidate has strong objective coverage, skip waiting on slower fallbacks.
            if candidate.candidate_id == "c1_primary" and candidate.objective_coverage_pct >= 95.0:
                break
    seen = {c.candidate_id for c in candidates}
    for candidate_id, strategy, fallback in specs:
        if candidate_id in seen:
            continue
        candidates.append(
            _build_candidate(
                candidate_id=candidate_id,
                intent=intent,
                catalog=catalog,
                strategy=strategy,
                fallback=fallback,
            )
        )
    candidates.sort(key=lambda c: c.candidate_id)
    eligible = [c for c in candidates if c.hard_gate_pass]
    selected = eligible[0] if eligible else candidates[0]
    return PlanSetV2(selected_id=selected.candidate_id, candidates=candidates)
