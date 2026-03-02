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
| G2 | Continuity/semantics | DONE | Q2/Q3 thresholds met | `reports/v2_qa_truth_report_20260302_135446.json` |
| G3 | Analyst depth | DONE | Q4 >=95 | `reports/v2_qa_truth_report_20260302_135446.json` |
| G4 | Portability | DONE | Q6 >=95 with unseen fixtures | `reports/v2_qa_truth_report_20260302_135446.json` |
| G5 | Performance/reliability | DONE | Q7 met + degradation explicit | `reports/v2_qa_truth_report_20260302_135446.json` |
| G6 | Production truth | DONE | Release gate passed, no floor violations | `reports/v2_qa_truth_report_20260302_135446.json` |
| G7 | Cutover | DONE | Canary + rollback drill pass | `reports/v2_cutover_drill_20260302_142543.json` |
| G8 | Decommission | IN_PROGRESS | v1 retirement + cleanup + postmortem | `docs/v1_decommission_criteria.md` |

## Workstream A: Rewrite Foundation
| ID | Task | Status |
|---|---|---|
| A01 | Create `src/haikugraph/v2/` skeleton with strict package boundaries | DONE |
| A02 | Define v2 contracts/types and serialization | DONE |
| A03 | Deterministic stage state machine orchestrator | DONE |
| A04 | Compatibility adapter v2 -> current API | DONE |
| A05 | Feature-flag routing (`HG_RUNTIME_VERSION`) | DONE |
| A06 | Structured event/trace bus | DONE |
| A07 | Policy engine abstraction | IN_PROGRESS |
| A08 | v2 runtime health endpoints | DONE |
| A09 | Shadow diff engine | DONE |
| A10 | Cutover/rollback protocol and runbook | DONE |

## Workstream B: Domain-Agnostic Semantic Layer
| ID | Task | Status |
|---|---|---|
| B01 | Schema profiler for arbitrary datasets | DONE |
| B02 | Join graph inference with coverage/confidence | DONE |
| B03 | Infer entities/measures/dimensions/time fields | DONE |
| B04 | Metric ontology resolver | IN_PROGRESS |
| B05 | Data quality model (sparsity/join fragility) | DONE |
| B06 | Semantic cache by dataset signature | DONE |
| B07 | `/api/assistant/datasets/profile` endpoint | DONE |
| B08 | Schema drift detector/alerts | DONE |
| B09 | Semantic explainability payload | IN_PROGRESS |
| B10 | Portability benchmark suite on unseen datasets | DONE |

## Workstream C: Analytics Reasoning Depth
| ID | Task | Status |
|---|---|---|
| C01 | `ConversationStateV2` explicit slice semantics | DONE |
| C02 | Operation-based follow-up patch model | DONE |
| C03 | Central validity guard engine | DONE |
| C04 | Candidate objective matrix with hard gates | IN_PROGRESS |
| C05 | Assumption engine for scenarios | DONE |
| C06 | Executive insight composer | IN_PROGRESS |
| C07 | Recommendation layer with impact/risk/options | IN_PROGRESS |
| C08 | Contradiction detector + clarify prompts | IN_PROGRESS |
| C09 | Confidence calibration model | IN_PROGRESS |
| C10 | Denominator semantics enforcement | IN_PROGRESS |
| C11 | Grouped dual-metric continuity (`secondary_metric_value`) | DONE |
| C12 | Cross-domain grain integrity checks | IN_PROGRESS |
| C13 | Unknown-vs-inferred tagging in answer narrative | IN_PROGRESS |
| C14 | Decision memo output mode | IN_PROGRESS |

## Workstream D: Performance and Reliability
| ID | Task | Status |
|---|---|---|
| D01 | Per-stage timing instrumentation | DONE |
| D02 | Stage SLO budgets/enforcement | DONE |
| D03 | Parallel candidate eval with early stop | IN_PROGRESS |
| D04 | Reduce unnecessary LLM calls on deterministic follow-ups | IN_PROGRESS |
| D05 | Slice-signature caching | IN_PROGRESS |
| D06 | Provider governor + retry policy | IN_PROGRESS |
| D07 | Explicit degradation metadata (`provider_effective`/`fallback_used`) | DONE |
| D08 | SQL path optimization for common patterns | IN_PROGRESS |
| D09 | Load/stress harness + trend report | IN_PROGRESS |
| D10 | Incident auto-dedupe metadata | DONE |

## Workstream E: UI Stabilization
| ID | Task | Status |
|---|---|---|
| E01 | Split monolithic `ui.html` into modules | DONE |
| E02 | Typed API client layer | DONE |
| E03 | Continuity diagnostics panel | DONE |
| E04 | Quality flags + assumptions visibility | DONE |
| E05 | Provider/degradation transparency | DONE |
| E06 | Explainability modal stage timeline + contract checks | IN_PROGRESS |
| E07 | Rule management UX validation/previews | IN_PROGRESS |
| E08 | Accessibility/responsive/perf pass | IN_PROGRESS |

## Workstream F: Cleanup and Unused Code Removal
| ID | Task | Status |
|---|---|---|
| F01 | Dead code inventory | DONE |
| F02 | Remove duplicate `advanced_packs 2.py` | DONE |
| F03 | Consolidate legacy follow-up pathways | IN_PROGRESS |
| F04 | Remove tracker-only scoreboard truth path | DONE |
| F05 | Replace broad catch-all exceptions in critical path | IN_PROGRESS |
| F06 | CI guards for dead/duplicate modules | DONE |
| F07 | Align docs/claims with runtime evidence | IN_PROGRESS |
| F08 | Deprecate legacy v1 modules post-cutover | IN_PROGRESS |

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
| G08 | Query grain/denominator correctness suite | IN_PROGRESS |
| G09 | Replay/determinism checks | IN_PROGRESS |
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
| H04 | Weekly quality review and blocker triage | IN_PROGRESS |
| H05 | Operator runbook + incident playbooks | DONE |
| H06 | Artifact-only tracker updates | IN_PROGRESS |
| H07 | Final v1 decommission criteria | DONE |
| H08 | Execute v1 retirement and cleanup | IN_PROGRESS |

## UAT Readiness
1. UAT certification gate set is complete through G7 with release truth pass and cutover drill evidence.
2. Remaining work is post-UAT operational hardening and v1 retirement (G8 + related H08/F08 tasks).
