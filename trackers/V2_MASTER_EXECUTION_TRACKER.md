# V2 Master Execution Tracker

Last Updated: 2026-03-01
Owner: Principal Engineering
Execution Model: Quality-gated (no date-based promotion)
Runtime Authority: `truth_score` + hard-floor gates

## Gate Status
| Gate | Name | Status | Exit Criteria |
|---|---|---|---|
| G0 | Baseline lock | IN_PROGRESS | Evidence artifacts frozen and signed |
| G1 | Contract parity | DONE | Existing API tests pass with v2 adapter |
| G2 | Continuity/semantics | DONE | Q2/Q3 thresholds met in merge-tier truth run |
| G3 | Analyst depth | DONE | Q4 threshold met in merge-tier truth run |
| G4 | Portability | NOT_STARTED | Q6 >= 95 on unseen datasets |
| G5 | Performance/reliability | DONE | Q7 SLO met + explicit degradation |
| G6 | Production truth | DONE | Composite truth score meets policy at release tier |
| G7 | Cutover | NOT_STARTED | Canary + rollback drills pass |
| G8 | Decommission | NOT_STARTED | v1 retired + postmortem complete |

## Workstream A: Rewrite Foundation
| ID | Task | Status | Evidence |
|---|---|---|---|
| A01 | Create `src/haikugraph/v2/` skeleton with strict package boundaries | DONE | `src/haikugraph/v2/*` added |
| A02 | Define v2 contracts/types and serialization | DONE | `src/haikugraph/v2/types.py` contracts + stage event types |
| A03 | Deterministic stage state machine orchestrator | DONE | `StageEventBusV2` + ordered transition tests |
| A04 | Compatibility adapter v2 -> current API | DONE | `src/haikugraph/v2/compat_adapter.py` |
| A05 | Feature-flag routing (`HG_RUNTIME_VERSION`) | DONE | `server.py` v1/v2/shadow routing |
| A06 | Structured event/trace bus | DONE | `src/haikugraph/v2/event_bus.py` + adapter projection |
| A07 | Policy engine abstraction | IN_PROGRESS | quality/guard hooks centralized in runtime path |
| A08 | v2 runtime health endpoints | DONE | quality/latest, quality/run detail, datasets/profile, runtime/stage-slo |
| A09 | Shadow diff engine | DONE | `runtime.shadow_diff` in shadow mode |
| A10 | Cutover/rollback protocol and runbook | IN_PROGRESS | `docs/v2_cutover_runbook.md` + `/api/assistant/runtime/cutover/readiness` |

## Workstream B: Domain-Agnostic Semantic Layer
| ID | Task | Status | Evidence |
|---|---|---|---|
| B01 | Schema profiler for arbitrary datasets | DONE | `profile_dataset` in `semantic_profiler.py` |
| B02 | Join graph inference with coverage/confidence | DONE | `JoinEdgeV2` coverage + risk |
| B03 | Infer entities/measures/dimensions/time fields | DONE | semantic catalog extraction |
| B04 | Metric ontology resolver | IN_PROGRESS | heuristic resolver present; full ontology pending |
| B05 | Data quality model (sparsity/join fragility) | DONE | quality summary + join risk flags |
| B06 | Semantic cache by dataset signature | DONE | `src/haikugraph/v2/semantic_cache.py` + cache-hit metadata in API/runtime |
| B07 | `/api/assistant/datasets/profile` endpoint | DONE | server endpoint implemented |
| B08 | Schema-drift detector/alerts | DONE | `record_schema_signature` + `schema_drift` incidents/webhooks |
| B09 | Semantic explainability payload | IN_PROGRESS | v2 payload includes provenance scaffolding |
| B10 | Portability benchmark suite | NOT_STARTED | queued in QA program |

## Workstream C: Analytics Reasoning Depth
| ID | Task | Status | Evidence |
|---|---|---|---|
| C01 | `ConversationStateV2` explicit slice semantics | IN_PROGRESS | v2 state model + slice signatures |
| C02 | Operation-based follow-up patch model | IN_PROGRESS | `intent_engine` follow-up ops |
| C03 | Central validity guard engine | IN_PROGRESS | MT103 semantic variants patched in v1 + v2 intent ops |
| C04 | Candidate objective matrix with hard gates | IN_PROGRESS | planner objective coverage/risk flags |
| C05 | Assumption engine for scenarios | DONE | `insight_engine` assumptions output |
| C06 | Executive insight composer | IN_PROGRESS | insight scaffolding active, rubric hardening pending |
| C07 | Recommendation layer with impact/risk/options | IN_PROGRESS | existing recommendations integrated |
| C08 | Contradiction detector + clarify prompts | IN_PROGRESS | existing contradiction tests; v2 alignment pending |
| C09 | Confidence calibration model | IN_PROGRESS | calibration remains tracked in QA |
| C10 | Denominator semantics enforcement | IN_PROGRESS | existing suites present; v2 hooks pending |
| C11 | Grouped dual-metric continuity | IN_PROGRESS | follow-up split suppression + ops model |
| C12 | Cross-domain grain integrity checks | IN_PROGRESS | existing grain checks in v1 |
| C13 | Unknown vs inferred tagging | IN_PROGRESS | assumptions + quality flags exposed |
| C14 | Decision memo mode | NOT_STARTED | pending formatter + QA rubric |

## Workstream D: Performance and Reliability
| ID | Task | Status | Evidence |
|---|---|---|---|
| D01 | Per-stage timing instrumentation | DONE | `stage_timings_ms` in response + run metadata |
| D02 | Stage SLO budgets/enforcement | DONE | stage breach detection + incident emission (`stage_slo_monitor`) |
| D03 | Parallel candidate eval with early stop | NOT_STARTED | pending planner optimization |
| D04 | Reduce unnecessary LLM calls on follow-ups | IN_PROGRESS | follow-up deterministic controls improved |
| D05 | Slice-signature caching | IN_PROGRESS | runtime cache includes runtime version; slice caching pending |
| D06 | Provider governor + retry policy | IN_PROGRESS | existing retry paths and explicit mode checks |
| D07 | Explicit degradation metadata | DONE | `provider_effective` + `fallback_used` + UI badges |
| D08 | SQL path optimization for common patterns | IN_PROGRESS | latency tasks active |
| D09 | Load/stress harness and trends | IN_PROGRESS | existing stress tests; trend report pending |
| D10 | Incident auto-dedupe metadata | DONE | runtime incident dedupe in `RuntimeStore` |

## Workstream E: UI Stabilization
| ID | Task | Status | Evidence |
|---|---|---|---|
| E01 | Split `ui.html` into modules | DONE | `ui.css` + `ui.js` externalized and served via `/ui/assets/*` |
| E02 | Typed API client layer | DONE | centralized `apiClient` with shape validation and request timeouts |
| E03 | Continuity diagnostics panel | IN_PROGRESS | slice signature visible in result cards |
| E04 | Quality flags + assumptions visibility | DONE | result cards and explain modal updated |
| E05 | Provider/degradation transparency | DONE | provider/fallback chips in UI |
| E06 | Explainability modal stage timeline + contract checks | IN_PROGRESS | diagnostics payload expanded |
| E07 | Rule management UX validation/previews | IN_PROGRESS | existing rule UI; validation hardening pending |
| E08 | Accessibility/responsive/perf pass | NOT_STARTED | pending lighthouse regression |

## Workstream F: Cleanup and Unused Code Removal
| ID | Task | Status | Evidence |
|---|---|---|---|
| F01 | Dead code inventory | DONE | `repo_hygiene_check.sh` + unit test in CI path |
| F02 | Remove `advanced_packs 2.py` duplicate | DONE | duplicate file removed |
| F03 | Consolidate follow-up pathways | IN_PROGRESS | split suppression/continuity patches |
| F04 | Remove tracker-only scoreboard truth path | DONE | scoreboard now includes QA-derived truth score |
| F05 | Replace broad catch-all exceptions | IN_PROGRESS | partial hardening complete |
| F06 | CI guards for dead/duplicate modules | DONE | repo hygiene script + test added |
| F07 | Align docs/claims with runtime evidence | IN_PROGRESS | v2 trackers introduced |
| F08 | Deprecate legacy v1 post-cutover | NOT_STARTED | pending G7/G8 |

## Workstream G: QA Truth Program
| ID | Task | Status | Evidence |
|---|---|---|---|
| G01 | QA taxonomy redesign | DONE | `scripts/run_v2_truth_qa.py` suite taxonomy |
| G02 | Paraphrase mutation for continuity/semantics | IN_PROGRESS | integrated with round11 fresh mutation |
| G03 | Strategic rubric expansion | IN_PROGRESS | truth runner rubric hooks |
| G04 | Portability suite on unseen datasets | IN_PROGRESS | dataset profile checks in truth runner |
| G05 | Provider integrity checks | DONE | explicit mode fallback surfaced and scored |
| G06 | Warning hygiene thresholds | DONE | truth runner + scoreboard include warning metrics |
| G07 | Confidence calibration suite | DONE | Q8 pass with explicit Brier + warning checks |
| G08 | Query grain/denominator suite | IN_PROGRESS | mapped in test matrix |
| G09 | Replay/determinism checks | IN_PROGRESS | shadow and deterministic controls in place |
| G10 | Tiered runner (PR/merge/release) | DONE | tiered execution in truth runner |
| G11 | Parallel scheduler with provider caps | DONE | provider-aware worker caps in truth runner |
| G12 | Machine-readable QA truth schema | DONE | `v2_qa_truth_report_*.json` schema |
| G13 | Quality drift alarms | IN_PROGRESS | score drift endpoint + pending alert automation |
| G14 | Truth-score policy publication | DONE | policy embedded in tracker + runner output |

## Workstream H: Governance and Rollout
| ID | Task | Status | Evidence |
|---|---|---|---|
| H01 | Freeze baseline evidence set | IN_PROGRESS | baseline refs tracked in scorecard |
| H02 | Shadow run v2 on production-like traffic | IN_PROGRESS | `HG_RUNTIME_VERSION=shadow` diff path |
| H03 | Canary + auto rollback criteria | IN_PROGRESS | `docs/v2_cutover_runbook.md` criteria published; canary drill pending |
| H04 | Weekly quality review cadence | IN_PROGRESS | `docs/v2_quality_review_cadence.md` published; operational rollout pending |
| H05 | Operator runbook + incident playbooks | IN_PROGRESS | `docs/v2_incident_playbook.md` + cutover runbook published |
| H06 | Artifact-only tracker update policy | IN_PROGRESS | truth endpoints artifact-backed |
| H07 | v1 decommission criteria | IN_PROGRESS | `docs/v1_decommission_criteria.md` drafted; approval pending |
| H08 | v1 retirement and cleanup | NOT_STARTED | gated after cutover |

## Latest Certification Evidence
1. Release-tier certification passed with no floor violations: `reports/v2_qa_truth_report_20260301_224543.json` (`composite_truth_score=99.4`, `release_gate_passed=true`).
2. Multimode blackbox completed successfully: `reports/qa_round11_blackbox_fresh_20260301_230422.json` (`overall_pass_rate=98.26`, no transport timeout).

## Current Rollout Blockers
1. Default serving path is still `v1` unless `HG_RUNTIME_VERSION` is switched (`v2`/`shadow`).
2. Canary drills and rollback simulation evidence remain open before production cutover.
3. v1 retirement criteria approval and execution remain open.
