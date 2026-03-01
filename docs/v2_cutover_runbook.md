# dataDa v2 Cutover and Rollback Runbook

Last Updated: 2026-03-01
Owner: Principal Engineering

## Scope
This runbook defines canary promotion, automatic rollback triggers, and operator actions for moving default runtime from `v1` to `v2`.

## Preconditions
1. Latest release-tier truth report exists in `reports/v2_qa_truth_report_*.json`.
2. Latest release report `summary.release_gate_passed` is `true`.
3. No hard-floor violations in Q1-Q8.
4. `GET /api/assistant/quality/latest` returns `composite_truth_score >= 98`.
5. Stage SLO snapshot has no sustained budget breaches for deterministic and auto paths over the last 24h.

## Canary Plan
1. Set `HG_RUNTIME_VERSION=shadow` in canary environment.
2. Route 10% read traffic for 30 minutes.
3. Validate `runtime.shadow_diff.sql_equal` and confidence deltas from trust dashboard samples.
4. Promote to `HG_RUNTIME_VERSION=v2` for the canary only.
5. Hold 60 minutes at 10% traffic and watch:
   - Query success rate >= 0.95
   - Warning rate <= 0.15
   - p95 execution <= configured mode target
6. If all checks pass, increase to 25%, 50%, and 100% with the same checks at each step.

## Automatic Rollback Conditions
Rollback to `HG_RUNTIME_VERSION=v1` if any condition persists for 10 consecutive minutes:
1. Success rate < 0.95.
2. p95 latency exceeds target by >25%.
3. Warning rate > 0.20.
4. Stage SLO breaches on `executor_delegate` or `planner` exceed 20% of sampled runs.
5. Explicit provider mode integrity check fails (hidden fallback detected).

## Rollback Procedure
1. Set `HG_RUNTIME_VERSION=v1`.
2. Restart API service.
3. Confirm `/api/assistant/health` and run Q0 smoke via truth runner.
4. Open incident with root-cause category:
   - `continuity_regression`
   - `semantic_guard_regression`
   - `latency_regression`
   - `provider_integrity_regression`
5. Attach latest truth report and trust dashboard snapshots to incident metadata.

## Post-Rollback Follow-up
1. Run merge-tier truth suite on patch candidate.
2. Re-run release-tier truth suite before next canary attempt.
3. Update trackers from generated artifacts only.
