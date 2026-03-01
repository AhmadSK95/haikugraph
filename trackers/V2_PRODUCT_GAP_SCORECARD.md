# V2 Product Gap Scorecard

Last Updated: 2026-03-01
Truth Authority: QA artifacts + runtime telemetry (not tracker completion)

## Baseline Snapshot
| Metric | Baseline (2026-03-01 pre-fix) | Latest Merge Truth (2026-03-01 20:24 UTC) | Target | Status |
|---|---:|---:|---|---|
| Multimode overall pass | 95.65% | 100.0% (`merge`) | >=98% | IN_PROGRESS (release multimode pending) |
| Release-tier gate | N/A | FAIL (`release` run `20260301_214634`) | PASS | IN_PROGRESS |
| Follow-up continuity | 66.67% | 100.0% (`Q2`) | >=98% | CLOSED (PR-validated) |
| Business semantic probes | 90.0% | 100.0% (`Q3`) | >=98% | CLOSED (PR-validated) |
| Factual correctness | 100% | 100.0% (`Q1`) | >=99% | PASS |
| Safety/refusal | 100% | 100.0% (`Q5`) | 100% | PASS |
| Warning hygiene | 3.69 to 14.2 avg | 0.8917 blended (`Q8`) | <=1.0 avg non-actionable | CLOSED (PR-validated) |
| Latency p95 | ~24,982 ms | 76.67 ms deterministic (`Q7` PR) | <=12,000 cloud; <=8,000 deterministic/auto | IN_PROGRESS |

## Gap Register
| Gap ID | Problem Statement | Impact | Current Mitigation | Closure Criteria | Status |
|---|---|---|---|---|---|
| G01 | Follow-up continuity failure on FU dual-metric asks | Scope drift, analyst mistrust | Operation-based follow-up + split suppression patches | Q2 >=98% across paraphrases | CLOSED (PR-validated) |
| G02 | Validity guard miss on spend semantic variant | Business-rule inconsistency | MT103 variant regex + centralized intent guard | 100% guard compliance in semantic suite | CLOSED (PR-validated) |
| G03 | SkillContract warning noise dominates output | Diagnostic trust degradation | Per-run dedupe + warning tracking | <=1.0 avg non-actionable warnings/case | CLOSED (PR-validated) |
| G04 | Scoreboard drift vs reality | Misreported readiness | Truth score computed from artifacts + drift exposed | Scoreboard authority = truth policy | IN_PROGRESS |
| G05 | Monolithic orchestration blast radius | Regression risk | v2 stage package scaffolded | Stage modules fully isolated + test coverage | IN_PROGRESS |
| G06 | High p95 latency | Analyst workflow friction | Stage timings + stage SLO endpoint added | Q7 SLO gates pass across merge/release modes | IN_PROGRESS |
| G07 | UI monolith and limited diagnostics | Low maintainability/observability | Modular shell + typed API client implemented | Modular shell + typed API client | CLOSED |
| G08 | Sparse data + weak joins not surfaced strongly | Overconfident cross-domain inference | Join risk flags + schema drift detector + runtime caveat flags | Confidence/caveat enforcement in evaluator | IN_PROGRESS |
| G09 | Hidden degradation fallback behavior | Mode ambiguity | `provider_effective` + `fallback_used` surfaced API+UI | Explicit mode integrity checks pass | IN_PROGRESS |
| G10 | Duplicate/stale code artifacts | Hygiene debt | Duplicate module removed + hygiene guard added | CI blocks duplicate/stale artifacts | CLOSED (initial) |
| G11 | Governance rollout artifacts incomplete | Cutover risk and rollback ambiguity | Cutover runbook + incident playbook + quality cadence docs + readiness endpoint | Canary drill evidence + release cert | IN_PROGRESS |
| G12 | Release QA runner instability under multimode startup | False-negative gate failures | Tiered timeout controls + fail-fast latency + provider availability checks | Stable release-tier multimode pass with artifact | IN_PROGRESS |

## Evidence References
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/qa_round11_blackbox_fresh_20260301_174452.json`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/blackbox_semantic_probe_20260301_172928.json`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/v2_qa_truth_report_20260301_194226.json`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/v2_qa_truth_report_20260301_202407.json`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/input_data_profile_20260301_175046.json`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/src/haikugraph/api/server.py`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/src/haikugraph/v2/`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/docs/v2_cutover_runbook.md`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/docs/v2_incident_playbook.md`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/docs/v2_quality_review_cadence.md`

## Promotion Rule
- Release promotion blocked unless all hard-floor categories pass.
- Composite truth score is vetoed by any floor violation.
- Merge/release authority requires stable black-box execution (no transport-timeout-induced suite collapse).
