# V2 QA Truth Tracker

Last Updated: 2026-03-02
Source of Truth: `reports/v2_qa_truth_report_*.json`

## Suite Matrix
| Suite | Focus | Gate |
|---|---|---|
| Q0 | Smoke/runtime sanity | 100% |
| Q1 | Factual correctness vs SQL truth | >=99% |
| Q2 | Follow-up continuity and dual-metric carry-forward | >=98% |
| Q3 | Business semantic guard consistency | >=98% |
| Q4 | Strategic insight rubric | >=95% |
| Q5 | Safety/policy + provider integrity | 100% |
| Q6 | Portability on unseen datasets | >=95% |
| Q7 | Latency/SLO by mode/stage | pass mode SLOs |
| Q8 | Confidence calibration + warning hygiene | pass thresholds |

## Truth Score Policy
1. Composite `truth_score` is weighted by suite category.
2. Hard-floor veto applies; any floor violation caps final score.
3. Release authority is `composite_truth_score` only.

## Tier Execution
| Tier | Blackbox Modes | Purpose |
|---|---|---|
| `pr` | deterministic, auto | fast guardrail |
| `merge` | deterministic, auto | branch gate |
| `release` | deterministic, auto, local | release certification (provider integrity checks still validate openai/anthropic/local explicitly) |

## Latest Certified Run
| Field | Value |
|---|---|
| Report | `reports/v2_qa_truth_report_20260302_170607.json` |
| Weighted score | `99.64` |
| Composite truth score | `99.64` |
| Floor violations | `none` |
| Release gate | `PASS` |
| Q6 portability | `95.45` |
| Consecutive release certs | `2` (`170555` and `170607`) |

## Post-Rewrite Validation (Targeted)
| Field | Value |
|---|---|
| Report | `reports/v2_unified_rewrite_validation_20260302_103649.json` |
| Scope | targeted pytest slices for unified v2 runtime + service-layer adapters |
| Result | PASS (targeted) |
| Release certification complete | `true` |

## QA Infrastructure Updates Landed
1. Portability suite now includes unseen fixture datasets (`ops_payments`, `marketing_weak_join`, `refund_txn_sparse`).
2. Blackbox runner preflight viability checks can skip non-viable provider modes.
3. Drift alarm defaults now compare latest report vs previous distinct report.
4. Provider fallback normalization enforces explicit degradation truth in API responses.
