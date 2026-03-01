# V2 Latency SLO Tracker

Last Updated: 2026-03-01

## Mode SLO Targets
| Mode | p50 Target (ms) | p95 Target (ms) | Status |
|---|---:|---:|---|
| deterministic | <=3,500 | <=8,000 | PASS (`p50=75.81`, `p95=75.81`, run `20260301_202407`) |
| auto | <=4,500 | <=8,000 | PASS (`p50=73.49`, `p95=73.49`, run `20260301_202407`) |
| openai | <=7,000 | <=12,000 | IN_PROGRESS (release tier run pending) |
| anthropic | <=7,000 | <=12,000 | IN_PROGRESS (release tier run pending) |
| local | <=9,000 | <=15,000 | IN_PROGRESS (release tier run pending) |

## Stage Budgets (ms)
| Stage | Budget |
|---|---:|
| semantic_profiler | 900 |
| intent_engine | 900 |
| planner | 1,500 |
| query_compiler | 1,200 |
| executor_delegate | 6,000 |
| evaluator_insight | 1,200 |

## Baseline Evidence
- Fresh multimode p95 observed: ~24,982 ms (fails target).
- Runtime now captures `stage_timings_ms` and exposes `/api/assistant/runtime/stage-slo`.
- Latest merge truth run (`v2_qa_truth_report_20260301_202407.json`) shows deterministic+auto SLO pass.

## Optimization Backlog
1. DONE: semantic profiling overhead reduced with dataset-signature cache (`SemanticProfileCache`).
2. Parallel candidate scoring with early-stop.
3. Avoid unnecessary LLM calls for deterministic follow-up patches.
4. Improve SQL path for common grouped dual-metric cases.
5. Add warmup + provider-aware queue shaping in runtime.
6. DONE: QA latency probes now use fail-fast stop on repeated provider failures.

## Reporting Cadence
- Update after each merge-gate QA run.
- Promotion requires two consecutive gate-passing runs.

## Latest Stage-SLO Snapshot Note
- PR-tier pass does not replace merge/release certification.
- Merge tier (`deterministic+auto`) is now certified.
- Release tier (`20260301_214634`) failed before multimode scoring due blackbox startup health read-timeout; latency bars were not the failing floor in that run.
