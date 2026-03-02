# V2 Latency SLO Tracker

Last Updated: 2026-03-02

## Mode SLO Targets
| Mode | p95 Target (ms) | Latest Observed p95 (ms) | Status |
|---|---:|---:|---|
| deterministic | <=8,000 | 77.01 | PASS |
| auto | <=8,000 | 93.24 | PASS |
| openai | <=12,000 | 76.32 | PASS |
| anthropic | <=12,000 | 96.27 | PASS |
| local | <=15,000 | 575.69 (`load_stress`) | PASS |

References: `reports/v2_qa_truth_report_20260302_170607.json`, `reports/v2_load_stress_trend_20260302_171152.json`

## Stage Budget Targets (ms)
| Stage | Budget |
|---|---:|
| semantic_profiler | 900 |
| intent_engine | 900 |
| planner | 1,500 |
| query_compiler | 1,200 |
| executor | 6,000 |
| evaluator | 1,200 |
| insight_engine | 1,200 |

## Instrumentation Status
1. `stage_timings_ms` is emitted in query responses.
2. `/api/assistant/runtime/stage-slo` exposes recent SLO breaches by stage (legacy aliases retained for compatibility).
3. Stage breach incidents are deduped in runtime store.

## Remaining Latency Work
1. None. All current mode SLO bars are artifact-passed.
