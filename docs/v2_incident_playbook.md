# dataDa v2 Incident Playbook

Last Updated: 2026-03-01
Owner: Reliability Engineering

## Incident Classes
1. `query_failure`: query request returns `success=false`.
2. `slo_monitor`: aggregate SLO breach from trust dashboard.
3. `stage_slo_monitor`: per-stage budget breach from stage timings.
4. `schema_drift`: dataset schema signature changed.

## Severity Policy
1. `critical`: data corruption or tenant isolation breach.
2. `high`: sustained query failures or policy violations.
3. `medium`: sustained SLO breach or schema drift impacting production analytics.
4. `low`: transient stage SLO breaches without user-visible correctness impact.

## Response Workflow
1. Acknowledge incident in <15 minutes.
2. Classify scope:
   - single tenant
   - single provider mode
   - global
3. Capture evidence:
   - `trace_id`
   - latest truth report id
   - trust dashboard snapshot
   - stage timing snapshot
4. Execute mitigation:
   - provider integrity issue: pin deterministic/auto mode
   - semantic/schema issue: invalidate semantic cache and rerun `/datasets/profile`
   - runtime regression: switch `HG_RUNTIME_VERSION=v1`
5. Resolve with corrective task ids and validation artifacts.

## Evidence Requirements for Closure
1. Reproduction query and expected/actual behavior.
2. Fix commit reference.
3. Post-fix merge-tier truth report.
4. If rollback happened, next successful canary report.
