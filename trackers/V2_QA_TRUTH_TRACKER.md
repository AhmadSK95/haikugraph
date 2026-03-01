# V2 QA Truth Tracker

Last Updated: 2026-03-01
Source: `scripts/run_v2_truth_qa.py` outputs (`reports/v2_qa_truth_report_*.json`)

## Suite Matrix
| Suite | Focus | Gate |
|---|---|---|
| Q0 | Smoke/runtime sanity | 100% |
| Q1 | Factual correctness vs SQL truth | >=99% |
| Q2 | Follow-up continuity + dual-metric carry-forward | >=98% |
| Q3 | Business semantic guard consistency | >=98% |
| Q4 | Strategic insight rubric | >=95% |
| Q5 | Safety/refusal integrity | 100% |
| Q6 | Portability on unseen data profile | >=95% |
| Q7 | Latency/SLO by mode and stage | meet SLO |
| Q8 | Confidence calibration + warning hygiene | pass thresholds |

## Truth Score Policy
- Composite `truth_score` uses weighted category scoring.
- Hard-floor veto applies: overall score cannot exceed floor compliance.
- Release authority uses `truth_score`, not tracker completion.

### Weights
| Category | Weight |
|---|---:|
| Q1 Factual | 0.22 |
| Q2 Follow-up | 0.18 |
| Q3 Semantics | 0.16 |
| Q4 Strategic | 0.12 |
| Q5 Safety | 0.12 |
| Q6 Portability | 0.08 |
| Q7 Latency | 0.07 |
| Q8 Calibration/Warnings | 0.05 |

## Runner Modes
| Tier | Purpose | Execution |
|---|---|---|
| `pr` | Fast guardrail | deterministic+auto quick subset |
| `merge` | Main branch gate | deterministic+auto full + semantic probes + latency |
| `release` | Certification | multimode + strict integrity + full suites |

## Parallelization Policy
| Dimension | Policy |
|---|---|
| Atomic case workers | deterministic/auto up to 6 |
| Local workers | up to 2 |
| OpenAI/Anthropic workers | up to 2 each with backoff |
| Follow-up chains | sequential inside chain; chains parallelized |
| Truth runs | cache disabled |
| Latency runs | cache labeled and enabled |

## Run Commands
1. `python scripts/run_v2_truth_qa.py --tier pr --base-url http://127.0.0.1:8000`
2. `python scripts/run_v2_truth_qa.py --tier merge --base-url http://127.0.0.1:8000`
3. `python scripts/run_v2_truth_qa.py --tier release --base-url http://127.0.0.1:8000 --strict-provider-integrity`

## Latest Status
- Latest run: `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/v2_qa_truth_report_20260301_214634.json`
- Tier: `release`
- Weighted score: `36.0`
- Composite truth score after hard-floor policy: `0.0`
- Floor violations: `Q1,Q2,Q4,Q5`

## Recent Run History
| Timestamp (UTC) | Tier | Weighted | Composite | Gate Result | Notes |
|---|---|---:|---:|---|---|
| `2026-03-01T21:48:25Z` | `release` | 36.0 | 0.0 | FAIL | `qa_round11_blackbox_fresh` startup health read-timeout under multimode load; Q1/Q2/Q4/Q5 floored to 0 |
| `2026-03-01T21:36:08Z` | `pr` | 100.0 | 100.0 | PASS | Fresh PR-tier run after runtime/cache/governance updates |
| `2026-03-01T20:24:07Z` | `merge` | 100.0 | 100.0 | PASS | Merge-tier gates all green (Q1-Q8) |
| `2026-03-01T20:22:09Z` | `merge` | 36.0 | 0.0 | FAIL | Black-box health timeout collapse before runner hardening |
| `2026-03-01T19:42:26Z` | `pr` | 100.0 | 100.0 | PASS | Q2/Q3/Q8 gaps closed; no floor violations |
| `2026-03-01T19:40:51Z` | `pr` | 97.5 | 50.0 | FAIL | Q8 Brier miss (0.3962) |
| `2026-03-01T19:37:33Z` | `pr` | 97.5 | 50.0 | FAIL | Q8 Brier miss (0.3962) |
| `2026-03-01T19:00:09Z` | `pr` | 93.4 | 0.0 | FAIL | Q3 + Q8 floor violations |

## Open QA Risks
1. Release-tier certification currently fails due `qa_round11_blackbox_fresh` startup health read-timeout in multimode path, collapsing Q1/Q2/Q4/Q5 to zero.
2. Semantic suite rows still show `analysis_version=v1` under default runtime configuration; v2/shadow truth runs must be promoted in governance gates.
3. Runner is now hardened with provider-aware mode selection, bounded blackbox subprocess timeouts, and fail-fast latency probes, but release-tier still needs a passing multimode artifact.
