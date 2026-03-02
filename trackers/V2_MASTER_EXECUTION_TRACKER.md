# V2 Master Execution Tracker

Last Updated: 2026-03-02
Owner: Principal Engineering
Execution Model: Quality-gated (artifact-evidenced, no calendar promotion)
Release Authority: `truth_score` + hard-floor policy

## Gate Status
| Gate | Name | Status | Exit Criteria | Evidence |
|---|---|---|---|---|
| G0 | Baseline lock | DONE | Baseline artifacts frozen and signed | `reports/v2_baseline_lock_20260302_142437.json` |
| G1 | Contract parity | DONE | Existing API tests pass with v2 adapter | `tests/test_api_runtime_features.py` |
| G2 | Continuity/semantics | DONE | Q2/Q3 thresholds met | `reports/v2_qa_truth_report_20260302_170607.json` |
| G3 | Analyst depth | DONE | Q4 >=95 | `reports/v2_qa_truth_report_20260302_170607.json` |
| G4 | Portability | DONE | Q6 >=95 with unseen fixtures | `reports/v2_qa_truth_report_20260302_170607.json` |
| G5 | Performance/reliability | DONE | Q7 met + degradation explicit | `reports/v2_qa_truth_report_20260302_170607.json` |
| G6 | Production truth | DONE | Full post-rewrite release cert required | `reports/v2_qa_truth_report_20260302_170607.json` |
| G7 | Cutover | DONE | Unified v2 drills + recert artifacts required | `reports/v2_cutover_drill_20260302_171248.json` |
| G8 | Decommission | DONE | v1 retirement + cleanup + postmortem | `reports/v2_weekly_quality_review_20260302_171255.json` |

## Workstream A: Rewrite Foundation
| ID | Task | Status |
|---|---|---|
| A01 | Create `src/haikugraph/v2/` skeleton with strict package boundaries | DONE |
| A02 | Define v2 contracts/types and serialization | DONE |
| A03 | Deterministic stage state machine orchestrator | DONE |
| A04 | Compatibility adapter v2 -> current API | DONE |
| A05 | Feature-flag routing (`HG_RUNTIME_VERSION`) | DONE |
| A06 | Structured event/trace bus | DONE |
| A07 | Policy engine abstraction | DONE |
| A08 | v2 runtime health endpoints | DONE |
| A09 | Shadow diff engine | DONE (retired from active runtime path) |
| A10 | Cutover/rollback protocol and runbook | DONE |

## Workstream B: Domain-Agnostic Semantic Layer
| ID | Task | Status |
|---|---|---|
| B01 | Schema profiler for arbitrary datasets | DONE |
| B02 | Join graph inference with coverage/confidence | DONE |
| B03 | Infer entities/measures/dimensions/time fields | DONE |
| B04 | Metric ontology resolver | DONE |
| B05 | Data quality model (sparsity/join fragility) | DONE |
| B06 | Semantic cache by dataset signature | DONE |
| B07 | `/api/assistant/datasets/profile` endpoint | DONE |
| B08 | Schema drift detector/alerts | DONE |
| B09 | Semantic explainability payload | DONE |
| B10 | Portability benchmark suite on unseen datasets | DONE |

## Workstream C: Analytics Reasoning Depth
| ID | Task | Status |
|---|---|---|
| C01 | `ConversationStateV2` explicit slice semantics | DONE |
| C02 | Operation-based follow-up patch model | DONE |
| C03 | Central validity guard engine | DONE |
| C04 | Candidate objective matrix with hard gates | DONE |
| C05 | Assumption engine for scenarios | DONE |
| C06 | Executive insight composer | DONE |
| C07 | Recommendation layer with impact/risk/options | DONE |
| C08 | Contradiction detector + clarify prompts | DONE |
| C09 | Confidence calibration model | DONE |
| C10 | Denominator semantics enforcement | DONE |
| C11 | Grouped dual-metric continuity (`secondary_metric_value`) | DONE |
| C12 | Cross-domain grain integrity checks | DONE |
| C13 | Unknown-vs-inferred tagging in answer narrative | DONE |
| C14 | Decision memo output mode | DONE |

## Workstream D: Performance and Reliability
| ID | Task | Status |
|---|---|---|
| D01 | Per-stage timing instrumentation | DONE |
| D02 | Stage SLO budgets/enforcement | DONE |
| D03 | Parallel candidate eval with early stop | DONE |
| D04 | Reduce unnecessary LLM calls on deterministic follow-ups | DONE |
| D05 | Slice-signature caching | DONE |
| D06 | Provider governor + retry policy | DONE |
| D07 | Explicit degradation metadata (`provider_effective`/`fallback_used`) | DONE |
| D08 | SQL path optimization for common patterns | DONE |
| D09 | Load/stress harness + trend report | DONE |
| D10 | Incident auto-dedupe metadata | DONE |

## Workstream E: UI Stabilization
| ID | Task | Status |
|---|---|---|
| E01 | Split monolithic `ui.html` into modules | DONE |
| E02 | Typed API client layer | DONE |
| E03 | Continuity diagnostics panel | DONE |
| E04 | Quality flags + assumptions visibility | DONE |
| E05 | Provider/degradation transparency | DONE |
| E06 | Explainability modal stage timeline + contract checks | DONE |
| E07 | Rule management UX validation/previews | DONE |
| E08 | Accessibility/responsive/perf pass | DONE |

## Workstream F: Cleanup and Unused Code Removal
| ID | Task | Status |
|---|---|---|
| F01 | Dead code inventory | DONE |
| F02 | Remove duplicate `advanced_packs 2.py` | DONE |
| F03 | Consolidate legacy follow-up pathways | DONE |
| F04 | Remove tracker-only scoreboard truth path | DONE |
| F05 | Replace broad catch-all exceptions in critical path | DONE |
| F06 | CI guards for dead/duplicate modules | DONE |
| F07 | Align docs/claims with runtime evidence | DONE |
| F08 | Deprecate legacy v1 modules post-cutover | DONE |

## Workstream G: QA Truth Program
| ID | Task | Status |
|---|---|---|
| G01 | QA taxonomy redesign | DONE |
| G02 | Paraphrase mutation for continuity/semantics | DONE |
| G03 | Strategic rubric expansion | DONE |
| G04 | Portability suite on unseen datasets | DONE |
| G05 | Provider integrity checks | DONE |
| G06 | Warning hygiene thresholds | DONE |
| G07 | Confidence calibration suite | DONE |
| G08 | Query grain/denominator correctness suite | DONE |
| G09 | Replay/determinism checks | DONE |
| G10 | Tiered runner (PR/merge/release) | DONE |
| G11 | Parallel scheduler with provider-aware caps | DONE |
| G12 | Machine-readable QA truth schema | DONE |
| G13 | Quality drift alarms | DONE |
| G14 | Truth-score policy publication | DONE |

## Workstream H: Governance and Rollout
| ID | Task | Status |
|---|---|---|
| H01 | Freeze baseline evidence set | DONE |
| H02 | Shadow run v2 on production-like traffic | DONE |
| H03 | Canary rollout + automatic rollback criteria | DONE |
| H04 | Weekly quality review and blocker triage | DONE |
| H05 | Operator runbook + incident playbooks | DONE |
| H06 | Artifact-only tracker updates | DONE |
| H07 | Final v1 decommission criteria | DONE |
| H08 | Execute v1 retirement and cleanup | DONE |

## UAT Readiness
1. Unified v2 query runtime is active and artifact-certified for UAT.
2. All gate groups (G0-G8) are closed with current evidence.
