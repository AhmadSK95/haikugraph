# dataDa v2 Weekly Quality Review Cadence

Last Updated: 2026-03-01
Owner: Principal Engineering

## Cadence
- Frequency: weekly (Monday 10:00 America/New_York)
- Duration: 45 minutes
- Required attendees: principal engineer, QA owner, runtime owner, product owner

## Inputs (artifact-only)
1. Latest PR, merge, and release truth reports.
2. `GET /api/assistant/quality/latest` output.
3. `GET /api/assistant/runtime/stage-slo` output.
4. Trust dashboard and open incidents over last 7 days.

## Required Review Sections
1. Gate status (G0-G8) with artifact references.
2. Hard-floor violations and category drift.
3. Follow-up continuity and semantic guard regressions.
4. Provider integrity and fallback transparency.
5. Latency and stage budget trends.
6. Top 5 warning-noise contributors and remediation owner.

## Exit Criteria
1. Each open blocker has owner + due milestone.
2. Tracker updates include report paths and run ids.
3. No manual score overrides without artifact references.
