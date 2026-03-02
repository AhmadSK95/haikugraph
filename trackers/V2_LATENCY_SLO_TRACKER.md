# V2 Latency SLO Tracker

Last Updated: 2026-03-02

## Mode SLO Targets
| Mode | p95 Target (ms) | Latest Observed p95 (ms) | Status |
|---|---:|---:|---|
| deterministic | <=8,000 | 77.01 | PASS |
| auto | <=8,000 | 93.24 | PASS |
| openai | <=12,000 | 76.32 | PASS |
| anthropic | <=12,000 | 96.27 | PASS |
| local | <=15,000 | covered via release blackbox pass; not in Q7 probe matrix | IN_PROGRESS |

Reference: `reports/v2_qa_truth_report_20260302_135446.json`

## Stage Budget Targets (ms)
| Stage | Budget |
|---|---:|
| semantic_profiler | 900 |
| intent_engine | 900 |
| planner | 1,500 |
| query_compiler | 1,200 |
| executor_delegate | 6,000 |
| evaluator_insight | 1,200 |

## Instrumentation Status
1. `stage_timings_ms` is emitted in query responses.
2. `/api/assistant/runtime/stage-slo` exposes recent SLO breaches by stage.
3. Stage breach incidents are deduped in runtime store.

## Remaining Latency Work
1. Add local-mode probes into Q7 latency matrix.
2. Complete early-stop candidate scoring in planner path.
3. Tighten deterministic follow-up short path to reduce runtime variance.
