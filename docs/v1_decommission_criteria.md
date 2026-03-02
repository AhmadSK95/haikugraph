# v1 Runtime Decommission Criteria

Last Updated: 2026-03-02  
Owner: Platform Engineering

## Unified Runtime Criteria (Artifact-Only)
1. Query API routes only through v2 (no `v1` or `shadow` query branch reachable).
2. `HG_RUNTIME_VERSION` accepts `v2` only for runtime selection semantics.
3. Two consecutive unified-v2 certification passes are recorded in artifacts.
4. No unresolved `high` or `critical` incidents attributed to unified-v2 runtime.
5. Postmortem and rollback runbook are published and linked from trackers.

## Current Status
- `DONE`: Query runtime branch removal (`/api/assistant/query` now always executes v2).
- `DONE`: Compatibility endpoint alias retained (`/runtime/cutover/readiness` + `/runtime/readiness`).
- `DONE`: Service-layer wrappers isolate remaining legacy-backed non-query APIs.
- `IN_PROGRESS`: Full release-tier recertification (targeted post-rewrite validation is complete).
- `IN_PROGRESS`: Final archival/removal of legacy modules not used by non-query compatibility endpoints.

## Decommission Steps
1. Keep v1 modules isolated behind services only until full recert is complete.
2. Remove deprecated runtime-switch docs that imply `v1/shadow` query execution.
3. Archive v1-only query tests/scripts and publish mapping to unified-v2 equivalents.
4. Publish decommission postmortem with regressions, mitigations, and operator guidance.
