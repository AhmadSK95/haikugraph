# `agentic_team.py` Refactor Plan

Current state:
- `src/haikugraph/poc/agentic_team.py` is a monolith with orchestration, planning, SQL generation, audit logic, narrative logic, and UI-facing payload assembly.
- Risk grows with file size: slower reviews, merge conflicts, harder latency tuning, and fragile change surface.

## Why This Is A Real Issue

1. Change blast radius is high for every edit.
2. Profiling and optimization are harder when all stages are interleaved.
3. Testing speed suffers because isolated unit testing of stages is limited.

## Recommended Decomposition (Incremental)

1. `poc/orchestration/`
   - run loop, stage sequencing, retry/correction policy.
2. `poc/contracts/`
   - contract build/validation + canonical binding helpers.
3. `poc/explainability/`
   - business/technical explain payload assembly.
4. `poc/narrative/`
   - deterministic/LLM narrative builders and policy disclosures.
5. `poc/query_planning/`
   - intent normalization, decomposition, SQL strategy selection.

## Migration Strategy

1. Extract pure helper functions first (no runtime behavior changes).
2. Keep old call sites and add adapter wrappers until tests are green.
3. Move one subsystem per round with targeted tests.
4. Track file size and per-module test runtime after each extraction.

## Round R10 Start (Implemented)

1. Extracted dual-level explainability payload builder to:
   - `src/haikugraph/poc/explainability_payload.py`
2. `agentic_team.py` now consumes the extracted builder.
3. Functional behavior preserved via existing explainability tests.

## Round R12 (Implemented)

1. Extracted KPI decomposition ownership/target logic to:
   - `src/haikugraph/poc/kpi_decomposition.py`
2. Extracted candidate contradiction resolution logic to:
   - `src/haikugraph/poc/contradiction_resolution.py`
3. Extracted visualization/report composition logic to:
   - `src/haikugraph/poc/visualization_report.py`
4. Extracted recommendation generation (action/impact/risk) to:
   - `src/haikugraph/poc/recommendations.py`
5. `agentic_team.py` now routes these responsibilities through wrappers and integration points.
6. Validation:
   - `tests/test_kpi_decomposition_catalog.py`
   - `tests/test_contradiction_resolution.py`
   - `tests/test_visualization_report.py`
   - `tests/test_api_runtime_features.py`
   - `tests/test_brd_explainability.py`

## Round R13 (Implemented)

1. Extracted blackboard contract and edge helpers to:
   - `src/haikugraph/poc/blackboard.py`
2. `agentic_team.py` now delegates artifact posting/query/edge-building to module helpers.
3. Added focused regression coverage:
   - `tests/test_blackboard_contracts.py`
4. Fast-round script hardening for standalone execution:
   - `scripts/run_fast_round_tests.py` now bootstraps `src` path directly.

## Alternative decompositions if growth continues

1. Stage registry architecture (recommended next):
   - define stage interfaces (`plan`, `execute`, `audit`, `narrate`) and run them through a common orchestrator.
2. Event bus plus immutable state snapshots:
   - each stage emits events; payload assembly is done by reducers for clearer replay/debug.
3. Vertical slice modules by intent family:
   - separate transaction/quote/customer strategy modules with shared planner core.

Trigger threshold:
- If `agentic_team.py` exceeds 13k lines or per-change diff touches >3 stage zones, prioritize Stage Registry extraction before adding new feature logic.
