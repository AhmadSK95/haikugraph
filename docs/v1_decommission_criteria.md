# v1 Runtime Decommission Criteria

Last Updated: 2026-03-01
Owner: Platform Engineering

## Required Before v1 Retirement
1. `HG_RUNTIME_VERSION=v2` at 100% production traffic for 14 consecutive days.
2. No hard-floor truth-score violations during that period.
3. No unresolved `high` or `critical` incidents linked to v2 runtime.
4. Rollback drill completed successfully in the last 30 days.
5. Shadow diff mismatch rate below 2% for representative workloads.

## Decommission Steps
1. Freeze v1 branch to security-only changes.
2. Remove v1 default routing path from API runtime selection.
3. Archive v1-only modules and tests with migration note.
4. Remove deprecated environment variables tied only to v1.
5. Publish decommission postmortem with observed deltas.
