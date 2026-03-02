# Unified Runtime Decommission Postmortem (Interim)

Last Updated: 2026-03-02  
Owner: Principal Engineering

## Scope
This document captures the decommission progress from multi-runtime (`v1`/`v2`/`shadow`) query execution to a single unified v2 query runtime while preserving public API compatibility.

## What Changed
1. `/api/assistant/query` now executes only the v2 orchestration pipeline.
2. Query-time delegation to legacy runtime was removed (`delegate_to_v1_engine` removed).
3. Runtime switching behavior was normalized to `v2` semantics only.
4. Compatibility alias added: `/api/assistant/runtime/readiness` (with `/runtime/cutover/readiness` retained).
5. Non-query legacy operations were moved behind service-layer adapters (`services/*`) to isolate compatibility debt.

## Quality Evidence (This Pass)
Reference artifact: `reports/v2_unified_rewrite_validation_20260302_103649.json`

Targeted validation slices passed for:
1. v2 stage orchestration and adapter projection.
2. Core API sample and deterministic quality checks.
3. v2 additive diagnostics + readiness endpoints.
4. Semantic validity guard and grouped dual-metric behavior.
5. Corrections/rules APIs through service-layer wrappers.

## Residual Risks
1. Full release-tier recertification is pending after this rewrite pass.
2. Some legacy modules remain for non-query compatibility endpoints and source-truth utilities.
3. UI accessibility/performance hardening remains in progress.

## Follow-up Actions
1. Run full release-tier QA certification twice consecutively.
2. Retire or archive remaining legacy modules not needed for compatibility.
3. Close final governance tracker items (G6/G7/G8) with artifacts only.
