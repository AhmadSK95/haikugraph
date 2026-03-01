# V2 Latency SLO Tracker

Last Updated: 2026-03-01

## Mode SLO Targets
| Mode | p50 Target (ms) | p95 Target (ms) | Status |
|---|---:|---:|---|
| deterministic | <=3,500 | <=8,000 | PASS (`p50=77.91`, `p95=77.91`, run `20260301_224543`) |
| auto | <=4,500 | <=8,000 | PASS (`p50=95.08`, `p95=95.08`, run `20260301_224543`) |
| openai | <=7,000 | <=12,000 | PASS (`p50=78.29`, `p95=78.29`, run `20260301_224543`) |
| anthropic | <=7,000 | <=12,000 | PASS (`p50=95.64`, `p95=95.64`, run `20260301_224543`) |
| local | <=9,000 | <=15,000 | IN_PROGRESS (not yet covered by Q7 probe matrix) |

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
- Latest release truth run (`v2_qa_truth_report_20260301_224543.json`) shows deterministic/auto/openai/anthropic SLO pass.
- Release blackbox wall-clock improved to `1113.89s` with provider-aware heavy-mode scheduling (`--heavy-cloud-workers 2`).

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
- Merge tier (`deterministic+auto`) is certified.
- Release tier (`20260301_224543`) is certified with no latency floor violations.
