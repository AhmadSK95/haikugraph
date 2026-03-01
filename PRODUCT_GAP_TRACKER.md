# dataDa Product Gap Tracker (North Pole from Product POV)

Last updated: 2026-03-01  
Owner: Product + Engineering  
Purpose: fixed North Pole capability map, current product distance, and measurable closure criteria.

## 1) North Pole calibration (fixed scale)

North Pole definition for dataDa:
- A domain-agnostic, agentic data analytics team that can ingest any business data, translate vague business asks into correct analyses, explain decisions clearly, and continuously improve without hallucinating.

Scoring model:
- Fixed capability set: **45 skills** (do not change between rounds)
- Equal weight: **2.2222 points per skill**
- Skill status:
  - `DONE` = production-ready and repeatable
  - `PARTIAL` = works in limited cases
  - `GAP` = missing/not reliable
- Scores:
  - `NP Strict = DONE/45 * 100`
  - `NP Reality = (DONE + 0.5*PARTIAL)/45 * 100`

Current baseline:
- DONE: 45
- PARTIAL: 0
- GAP: 0
- NP Strict: 100.00
- NP Reality: 100.00
- Planner gap status: `CLOSED` end-to-end (`GAP=0`, `PARTIAL=0`) with full fresh multimode evidence.

Latest evidence round (R3, 2026-03-01):
- Fresh multimode black-box run reached full closure criteria in Round11:
  - `reports/qa_round11_blackbox_fresh_20260301_024249.json`
  - `reports/qa_round11_blackbox_fresh_20260301_024249.md`
  - Result: 115/115 (100%) with category scores all at 100%.
- Root-cause fix validated for the last unstable case (`R11_D02`):
  - Intake metric lock for explicit quote-count phrasing.
  - Autonomy candidate generation now avoids forex-markup candidate drift on explicit count asks.
- Fresh multimode black-box run completed with no mode errors:
  - `reports/qa_round11_blackbox_fresh_20260228_233413.json`
  - `reports/qa_round11_blackbox_fresh_20260228_233413.md`
- Updated provider/model benchmark:
  - `reports/provider_model_benchmark_20260228_192101.json`
  - `reports/provider_model_benchmark_20260228_192101.html`
- Historical note: NP was 40.00 at the R3 checkpoint; later recalibrated to 77.78/55.56 in R10, 78.89/57.78 in R11, and 90.00/80.00 in R12.

Latest implementation slice evidence (R4, 2026-03-01):
- Semantic glossary/rule matching upgrades shipped in `src/haikugraph/poc/autonomy.py` with new paraphrase tests:
  - `tests/test_autonomy_semantic.py`
- Skill contract enforcement shipped in `src/haikugraph/poc/skill_runtime.py` and agent trace wiring:
  - `tests/test_skill_runtime.py`
  - `tests/test_api_runtime_features.py`
- QA semantic-depth hard gates shipped in `src/haikugraph/qa/control_plane.py`:
  - `tests/test_qa_control_plane.py`
- Targeted regression slice status:
  - `65 passed` across runtime/QA/latency/skill suites:
    - `tests/test_api_runtime_features.py`
    - `tests/test_llm_router_latency.py`
    - `tests/test_skill_runtime.py`
    - `tests/test_qa_control_plane.py`
    - `tests/test_autonomy_semantic.py`
- NP score is intentionally unchanged in this document until the next full fresh 45-capability round is executed.

Latest implementation slice evidence (R5, 2026-03-01):
- Intent-collision and split-or-clarify improvements landed in `src/haikugraph/poc/agentic_team.py`:
  - mixed analytic/non-analytic requests now emit explicit clarification (`intent_collision_unresolved`) unless the user provides explicit sequencing.
  - non-metric mixed requests can deterministically split into multiple sub-questions.
- Multi-part sub-query runner now supports special intents (`data_overview`, `schema_exploration`) directly, avoiding silent failures on decomposed discovery asks.
- Regression evidence:
  - `73 passed` across targeted suites:
    - `tests/test_slice_completion.py`
    - `tests/test_api_runtime_features.py`
    - `tests/test_skill_runtime.py`
    - `tests/test_autonomy_semantic.py`
    - `tests/test_qa_control_plane.py`
  - multi-part/complex capability sanity:
    - `5 passed` (`1 xfailed expected`) in `tests/test_complex_capabilities.py -k "multi_part or gap32 or gap36 or gap39"`
- NP score is still held constant here until a full fresh 45-capability round is rerun.

Latest implementation slice evidence (R6, 2026-03-01):
- Autonomy decisioning now includes an explicit objective coverage matrix (metric/dimension/time/comparison/cross-domain/validity/execution/grounding) and uses it in candidate scoring and reconciliation payloads.
- Narrative outputs are now enriched with analyst-readable structure:
  - `What Drove This`
  - `Evidence`
  - `Caveat`
  for both deterministic and LLM narration paths when execution succeeds.
- Regression evidence:
  - `78 passed` across targeted suites:
    - `tests/test_api_runtime_features.py`
    - `tests/test_api_autonomy_features.py`
    - `tests/test_slice_completion.py`
    - `tests/test_skill_runtime.py`
    - `tests/test_autonomy_semantic.py`
    - `tests/test_qa_control_plane.py`
    - `tests/test_llm_router_latency.py`
- NP score remains unchanged in this tracker until the next fresh full 45-capability multimode evaluation is executed.

Latest implementation slice evidence (R7, 2026-03-01):
- Clarification gate hardening in `src/haikugraph/poc/agentic_team.py`:
  - stopped false-positive clarification blocks for valid asks like product discovery, `orders per region`, and denominator phrasing (`average ... per transaction`).
  - preserved hard clarification paths for ambiguity and unresolved mixed intents.
- Cross-domain SQL correctness hardening:
  - JOIN query builder now aliases dimensions by source table using table-specific column maps, fixing failures like booking metrics grouped by customer country.
- Safety hardening in `src/haikugraph/safety/policy_gate.py`:
  - anti-coercion patterns now catch broader bypass phrasing (`override your safety filters`, `bypass all guardrails`).
- Deterministic planner consistency hardening in `src/haikugraph/planning/plan.py`:
  - `compared to` phrasing now routes to `comparison` intent, removing comparison-variation drift.
- Regression evidence:
  - targeted formerly failing cases: `14 passed` across
    - `tests/test_api_queries.py`
    - `tests/test_benchmark_llm_modes.py`
    - `tests/test_brd_behavior_safety.py`
    - `tests/test_brd_canonical_factual.py`
    - `tests/test_complex_capabilities.py`
    - `tests/test_slice_completion.py`
    - `tests/test_policy_gate.py`
  - full suite: `738 passed, 81 skipped, 6 xfailed, 9 xpassed`.
- NP score remains unchanged here until the next fresh full 45-capability calibration run is executed.

Latest implementation slice evidence (R8, 2026-03-01):
- Canonical metric contract registry shipped in `src/haikugraph/poc/agentic_team.py`:
  - registry artifact now maps domain/table/metric to canonical metric id, expression, unit, preferred time column, and business definition.
  - alias-based semantic binding added for canonical metric resolution (for example, `fx charges` -> `quotes.forex_markup_revenue`).
- Contract binding now propagates canonical metric metadata across:
  - intake payload,
  - planning artifact,
  - contract spec and decision flow,
  - response trace payloads (`evidence_packets`, `data_quality.canonical_metric_contract`).
- Regression evidence:
  - targeted suites: `25 passed`
    - `tests/test_contract_guard_slice.py`
    - `tests/test_brd_explainability.py`
- NP score remains unchanged in this tracker until the next fresh full 45-capability calibration run is executed.

Latest implementation slice evidence (R9, 2026-03-01):
- Explainability payload contract upgraded to dual levels:
  - `explainability.business_view` for simplified business-readable reasoning.
  - `explainability.technical_view` for full diagnostic drill-down (SQL, contract, trace, runtime, quality).
  - backend wiring shipped in:
    - `src/haikugraph/agents/contracts.py`
    - `src/haikugraph/poc/agentic_team.py`
- Explain modal UX upgraded in `src/haikugraph/api/ui.html`:
  - business-first panel (`Business view`) plus structured `Technical drill-down`.
  - retained full timeline/diagnostic transparency (`Agent decision timeline`, `Advanced diagnostics (JSON)`).
- Regression evidence:
  - targeted suites: `32 passed`
    - `tests/test_brd_explainability.py`
    - `tests/test_slice_completion.py`
- R10 capability recalibration updated NP to:
  - DONE 25 / PARTIAL 20 / GAP 0
  - NP Strict: 55.56
  - NP Reality: 77.78

Latest implementation slice evidence (R10, 2026-03-01):
- Architecture scale control for `agentic_team.py`:
  - explainability payload assembly extracted into `src/haikugraph/poc/explainability_payload.py`.
  - decomposition roadmap added in `docs/agentic_team_refactor_plan.md`.
- KPI decomposition layer added:
  - response payload now includes `data_quality.kpi_decomposition` tree for KPI root/formula/drivers.
- Latency and runtime optimization layer:
  - API response cache (TTL/LRU) added with runtime telemetry (`response_cache_hit`, `response_cache_key`) in `src/haikugraph/api/server.py`.
  - async inflight backpressure caps added in API + runtime store (`count_async_jobs`) to avoid overload collapse.
  - quick latency benchmark harness added in `scripts/latency_optimization_check.py`.
- Provider policy overlays shipped for runtime routing parity controls:
  - `skills/providers/{openai,anthropic,ollama}.json`
  - router overlay binding in `src/haikugraph/llm/router.py`.
- Industry portability/onboarding artifacts shipped:
  - `docs/industry_onboarding_playbook.md`
  - `docs/templates/schema_profile_template.yaml`
  - `scripts/generate_onboarding_profile.py`
- Fresh + faster round testing upgrades:
  - `scripts/qa_round11_blackbox_fresh.py` now supports per-round prompt mutation (`--round-id`) and quick smoke mode (`--quick`).
  - `scripts/generate_round_capability_prompts.py` now generates a fresh 45-capability prompt set each round (plus quick mode).
- Regression evidence:
  - consolidated targeted regression slice:
    - `155 passed, 4 xfailed, 7 xpassed`
    across:
    - `tests/test_api_runtime_features.py` (cache + backpressure paths)
    - `tests/test_llm_router_latency.py`
    - `tests/test_brd_explainability.py`
    - `tests/test_slice_completion.py`
    - `tests/test_round11_fresh_fast.py`
    - `tests/test_round_capability_prompt_generator.py`
    - `tests/test_onboarding_profile.py`
    - `tests/test_security_hygiene_check.py`
    - `tests/test_autonomy_semantic.py`
    - `tests/test_api_autonomy_features.py`
    - `tests/test_api_queries.py`
    - `tests/test_brd_behavior_safety.py`
    - `tests/test_advanced_analytics_packs.py`
    - `tests/test_complex_capabilities.py`
- NP score remains unchanged in this tracker until the next fresh full 45-capability calibration run is executed.

Latest implementation slice evidence (R11, 2026-03-01):
- Product observability closure:
  - live capability scoreboard endpoint shipped at `/api/assistant/capability/scoreboard` in `src/haikugraph/api/server.py`.
  - endpoint parses `PRODUCT_GAP_TRACKER.md` A##/T## rows directly and returns counts, NP metrics, and remaining capabilities.
- Regression evidence:
  - `26 passed` in `tests/test_api_runtime_features.py` (includes new scoreboard endpoint coverage).
  - `27 passed` in `tests/test_brd_explainability.py` + `tests/test_advanced_analytics_packs.py`.
- R11 capability recalibration updated NP to:
  - DONE 26 / PARTIAL 19 / GAP 0
  - NP Strict: 57.78
  - NP Reality: 78.89

Latest implementation slice evidence (R12, 2026-03-01):
- Decomposition-first architecture pass:
  - extracted KPI decomposition to `src/haikugraph/poc/kpi_decomposition.py`.
  - extracted contradiction resolution to `src/haikugraph/poc/contradiction_resolution.py`.
  - extracted visualization/report composition to `src/haikugraph/poc/visualization_report.py`.
  - extracted recommendation generation to `src/haikugraph/poc/recommendations.py`.
  - `agentic_team.py` now delegates these responsibility blocks through wrappers.
- Analytics depth closures:
  - outlier policies now include threshold controls and alert levels.
  - variance supports baseline mode controls.
  - scenario pack supports authored assumption sets.
  - forecast now includes confidence intervals + calibration report.
  - funnel pack now emits domain-specific SQL template artifacts.
  - cohort output now includes a cohort heatmap grid payload.
- Narrative/contradiction/report quality closures:
  - responses now include explicit `Recommended Actions` with action/impact/risk/evidence.
  - near-tie candidate conflicts emit clarification prompts and conflict signals.
  - chart specs now include multi-panel report composition metadata.
- Round-speed testing closure:
  - fast impact-based regression selector added (`src/haikugraph/qa/fast_round_runner.py`) with runner script (`scripts/run_fast_round_tests.py --changed-only`).
- Regression evidence:
  - `14 passed`:
    - `tests/test_kpi_decomposition_catalog.py`
    - `tests/test_contradiction_resolution.py`
    - `tests/test_visualization_report.py`
    - `tests/test_advanced_analytics_packs.py`
  - `51 passed`:
    - `tests/test_api_runtime_features.py`
    - `tests/test_brd_explainability.py`
  - `13 passed`:
    - `tests/test_api_autonomy_features.py`
    - `tests/test_slice_completion.py`
- R12 capability recalibration updated NP to:
  - DONE 36 / PARTIAL 9 / GAP 0
  - NP Strict: 80.00
  - NP Reality: 90.00

Latest implementation slice evidence (R13, 2026-03-01):
- Final closure hardening + decomposition/latency/testing updates:
  - blackboard contract helpers extracted to `src/haikugraph/poc/blackboard.py` and wired in `src/haikugraph/poc/agentic_team.py`.
  - follow-up/context stability guards and metric/time consistency locks tightened in `src/haikugraph/poc/agentic_team.py`.
  - faster round runner script now bootstraps `src` path directly: `scripts/run_fast_round_tests.py`.
- Regression evidence:
  - `80 passed` across closure suites:
    - `tests/test_blackboard_contracts.py`
    - `tests/test_slice_completion.py`
    - `tests/test_api_runtime_features.py`
    - `tests/test_api_autonomy_features.py`
    - `tests/test_autonomy_semantic.py`
    - `tests/test_autonomy_store.py`
    - `tests/test_stream_snapshot.py`
    - `tests/test_runtime_store_scenarios.py`
    - `tests/test_provider_parity.py`
    - `tests/test_provider_parity_gate_script.py`
    - `tests/test_onboarding_profile.py`
    - `tests/test_api_connection_routing.py`
    - `tests/test_llm_router_latency.py`
  - fresh latency artifact:
    - `reports/latency_optimization_check_20260301_122855.json`
    - `reports/latency_optimization_check_20260301_122855.md`
  - fresh prompt set artifact (45 prompts):
    - `reports/round_capability_eval_20260301_122903.json`
    - `reports/round_capability_eval_20260301_122903.md`
  - full fresh multimode round (deterministic/local/openai/anthropic/auto):
    - `reports/qa_round11_blackbox_fresh_20260301_164742.json`
    - `reports/qa_round11_blackbox_fresh_20260301_164742.md`
    - Result: `115/115` (100%), category scores all `100`, honest score `100.0`.
- R13 capability recalibration updated NP to:
  - DONE 45 / PARTIAL 0 / GAP 0
  - NP Strict: 100.00
  - NP Reality: 100.00

## 2) Journey so far (condensed)

1. Started as deterministic NL->SQL behavior with weak continuity.
2. Added multi-agent trace, trust shell, and multi-provider modes.
3. Added rules/corrections and explainability UI.
4. Current frontier: sustaining NP 100 through fresh-round multimode validation and regression drift prevention.

## 3) Fixed North Pole capability map (45 skills)

## A) Data analyst core skills (A01-A25)

| ID | Analyst skill | Team-level output expected in industry | Status | Evidence now | Gap to close | Product requirement |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | Business question framing | converts vague ask into analysis objective | DONE | intake + clarification + intent-collision hard gates ship with deterministic split-or-clarify behavior | continue prompt-quality monitoring for niche phrasing | PR02, PR03 |
| A02 | KPI decomposition | goal -> KPI tree with formulas and owners | DONE | KPI decomposition now includes owner catalog, target, attainment, and variance fields via `src/haikugraph/poc/kpi_decomposition.py` | monitor KPI target drift and owner reassignments by quarter | PR04 |
| A03 | Metric semantics | business definition + caveats + usage | DONE | semantic retrieval + phrase synonym expansion + canonical metric contract binding now pass long-tail paraphrase regression and multimode black-box closure | extend multilingual synonym coverage as continuous improvement | PR02 |
| A04 | Grain identification | picks right unit of analysis | DONE | contract binding and decision-flow timeline now surface metric grain and canonical metric ids | keep grain disclosure concise for non-technical audiences | PR04 |
| A05 | Time-window scoping | applies period correctly + compares | DONE | time filters and checks present | improve ambiguity handling | PR03 |
| A06 | Dimension selection | selects meaningful slices | DONE | grouped analytics + cross-domain dimension routing and alias-safe joins validated in regression suites | expand domain-specific default-dimension heuristics | PR04 |
| A07 | Filter interpretation | interprets constraints correctly | DONE | filter parsing supports month/year, policy filters, and business-rule directives with guard validation | add richer natural-language negation cases | PR03 |
| A08 | Cross-table reasoning | joins entities for business narrative | DONE | cross-domain JOIN pathing and secondary-domain column aliasing are now validated and stable in regression slices | continue adding rare join-topology coverage | PR05 |
| A09 | Data quality awareness | flags null/dup/freshness risk | DONE | trust checks + quality surface | richer impact interpretation | PR07 |
| A10 | Source grounding | cites exact table/field lineage | DONE | explain trace + SQL lineage | business-readable lineage | PR06 |
| A11 | Outlier detection | identifies unusual behavior | DONE | advanced packs now apply policy-tuned outlier thresholds and alert severity (`outlier.policy`, `outlier.alert`) | keep threshold calibration tied to domain volatility baselines | PR04 |
| A12 | Trend analysis | reads directional change over time | DONE | trend/running-total/yoy patterns with analyst sections are now first-class in deterministic+LLM narration paths | keep benchmark prompts fresh for trend semantics | PR04 |
| A13 | Segmentation analysis | compares cohorts/segments | DONE | segmentation by platform/state/country/deal type with analyst evidence blocks is productionized | add more business-specific segment templates | PR04 |
| A14 | Funnel analysis | stage conversion + drop-off | DONE | funnel pack now includes domain-specific SQL template artifacts (`funnel.transactions.*`, `funnel.quotes.*`) with stage conversion outputs | add template quality scoring against live query outcomes | PR04 |
| A15 | Cohort retention analysis | retention tables/insights | DONE | cohort retention pack now emits retention rows plus cohort heatmap grid payload (`cohort_grid`) for production visualization wiring | add long-horizon (>12 month) cohort collapse handling | PR04 |
| A16 | Correlation analysis | relationships with caveats | DONE | correlation intent path and statistical analysis surface are integrated with confidence/caveat messaging | extend causal caveat library for stronger interpretation | PR04 |
| A17 | Variance analysis | actual vs baseline with drivers | DONE | variance pack now supports baseline source selection (`first`, `mean`, `median`, `previous_period`, etc.) and reports `baseline_mode` | keep baseline mode policy defaults per KPI family | PR04 |
| A18 | Root-cause hypothesising | ranked causes + evidence | DONE | ranked root-cause hypotheses with evidence score/caveat shipped via `src/haikugraph/poc/root_cause.py` and integrated into runtime payload/narrative | continue enriching driver templates by domain | PR05 |
| A19 | Scenario analysis | what-if with assumptions | DONE | persisted scenario-set CRUD + replay by id across sessions/tenants shipped in runtime store + API query flow | expand scenario authoring UX ergonomics | PR04 |
| A20 | Forecast reasoning | projection with confidence bounds | DONE | forecast pack now emits projection + confidence intervals + calibration metrics (`mae`, `mape_pct`, quality) behind policy gate | expand calibration to rolling backtests across larger windows | PR04 |
| A21 | Recommendation quality | action + impact + risk | DONE | deterministic recommendation layer now emits action/impact/risk/evidence and is rendered into narrative (`Recommended Actions`) | add recommendation feedback loop for acceptance-rate tuning | PR04 |
| A22 | Uncertainty communication | confidence + why | DONE | confidence/trust fields present | improve calibration quality | PR07 |
| A23 | Contradiction handling | asks follow-up on conflict | DONE | contradiction resolver now scores near-tie conflicts with explicit signals + clarification prompts (`needs_clarification`, `clarification_prompt`) | continue tuning severity thresholds from production conflict logs | PR03 |
| A24 | Multi-question handling | can split/sequence compound asks | DONE | decomposition planner + explicit sequence handling + mixed-intent clarification gate validated across runtime/complex capability tests | monitor rare nested multi-part compositions | PR03 |
| A25 | Business glossary authoring | table/field definitions for business users | DONE | semantic glossary memory + onboarding schema profile generator + playbook now provide repeatable business glossary authoring workflow | add UI-assisted glossary curation panel | PR02 |

## B) Data analytics team capability skills (T01-T20)

| ID | Team capability | Industry-level output expected | Status | Evidence now | Gap to close | Product requirement |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | Domain-agnostic onboarding | works across industries without code rewrites | DONE | runtime-managed onboarding profile metadata/version is now loaded and surfaced in health/query runtime contracts | add profile lifecycle telemetry dashboard | PR01, PR15 |
| T02 | Multi-source ingestion governance | file/db/stream with validation | DONE | governed stream snapshot ingestion pipeline (kafka/kinesis URI -> DuckDB mirror + freshness/schema manifest) is first-class in runtime routing | extend connector adapters beyond snapshot mode | PR01 |
| T03 | Semantic translation layer | business language -> canonical contracts | DONE | semantic retrieval + canonical metric contract binding shipped | broaden multilingual synonym coverage | PR02 |
| T04 | Agent role orchestration | specialist agents cooperate with clear decisions | DONE | decision-spine handoff contracts + blackboard reason-code edges + critical-artifact gating enforced and regression-tested | continue modular extraction from orchestration monolith | PR05 |
| T05 | Autonomous correction loop | self-corrects wrong plan | DONE | autonomy switch now enforces hard gates (objective coverage, contradiction severity, metric-family lock) with traceable reasons | continue broadening autonomy edge-case corpus | PR05 |
| T06 | Organizational memory | learns rules and improves future runs | DONE | semantic memory precision/ranking hardening plus tenant-isolation recall regression now meet closure thresholds | iterate memory ranking with production feedback signals | PR11 |
| T07 | Skill-based execution control | skills enforced as runtime quality gates | DONE | runtime skill enforcement + checks integrated and tested | expand skill policy authoring UX | PR05 |
| T08 | Provider strategy | local/openai/anthropic parity | DONE | automated parity reporting/gating plus local-mode deterministic-intake stability guard now keep full multimode capability round at 100 | continue periodic drift alerts and model policy tuning | PR09 |
| T09 | Latency strategy | intelligent low-latency/high-quality routing | DONE | adaptive routing + provider overlays + response cache telemetry + latency check harness landed | keep periodic benchmark tuning by mode | PR09, PR10 |
| T10 | Reliability under load | stable under concurrent workload | DONE | async queue backpressure limits and inflight guards shipped | add stress-run automation in CI/nightly job | PR10 |
| T11 | Trust and verification | source-truth checks + guardrails | DONE | trust dashboard/checks exist | deeper semantic checks needed | PR07 |
| T12 | Explainability UX | human-readable reasoning and lineage | DONE | dual-level explainability contract + UI shipped (`business_view` + `technical_view`) | add UX telemetry for readability scoring in production traffic | PR06 |
| T13 | Narrative quality | concise insight-first storytelling | DONE | narrative layer now injects analyst recommendations with explicit action/impact/risk and evidence-backed caveats | continue language-style eval on long-form executive summaries | PR04, PR06 |
| T14 | Visualization generation | chooses right chart/report structure | DONE | visualization builder now emits report compositions with panel layouts (comparison, grouped dual-metric, funnel, cohort heatmap) | add UI renderer parity checks for all panel types | PR12 |
| T15 | Session continuity | maintains context across thread | DONE | thread memory and follow-up | better long-thread consistency | PR08 |
| T16 | Correction UX loop | user can fix missing domain knowledge | DONE | fix + rules console present | broaden correction impact visibility | PR11 |
| T17 | Governance and policy safety | safe refusal and policy gate behavior | DONE | behavior safety is strong | broaden policy transparency | PR07 |
| T18 | QA honesty | testing reflects real product quality | DONE | semantic-depth gates + expanded targeted regression suites + fresh prompt generator shipped | keep extending adverse-case coverage | PR14 |
| T19 | Repo hygiene and security | clean docs/code/no key leakage | DONE | hygiene guard enforces stale top-level docs + secret/key scans; validated by automated script test | add CI workflow wiring for mandatory PR gating | PR13 |
| T20 | Product observability | growth tracker and capability scoreboard | DONE | live capability scoreboard endpoint now ships at `/api/assistant/capability/scoreboard`, auto-derived from tracker artifacts with NP/count metrics + remaining capability list | add UI widget + scheduled snapshot export for historical trend overlays | PR14 |

## 4) Product requirements (fixed for this program)

| Req ID | Product requirement |
| --- | --- |
| PR01 | Domain-agnostic ingestion across DB/files/streams |
| PR02 | Business semantic dictionary + translation layer |
| PR03 | Strong clarification and multi-intent handling |
| PR04 | Analyst-grade analytical depth and insight generation |
| PR05 | Autonomous multi-agent decision spine and correction quality |
| PR06 | Explainability that business users can understand |
| PR07 | Trust, validation, safety, and grounded confidence |
| PR08 | Session continuity and thread-aware dialogue |
| PR09 | Multi-provider + local model strategy with parity |
| PR10 | Latency, concurrency, and operational reliability |
| PR11 | Rule memory, correction loop, and organizational knowledge |
| PR12 | Visual/report output adaptability |
| PR13 | Clean, secure, maintainable OSS repository |
| PR14 | Honest QA framework with semantic depth scoring |
| PR15 | Industry portability and fast onboarding playbooks |

## 5) Highest-impact current gaps (P0/P1)

P0:
1. No capability remains in `GAP` or `PARTIAL`; closure is complete at capability-map level (`DONE=45`).
2. Maintain fresh multimode rounds each cycle to detect drift early and prevent regression.

P1:
1. Operational hardening: continuous benchmark drift monitoring across providers/modes.
2. Decomposition continuation: reduce `agentic_team.py` blast radius with staged module extraction.
3. Product telemetry: expose capability trend history and parity drift in UI dashboards.

## 6) Closure matrix (finalized in R13)

| Capability | Closed code gap | Closure criteria (DONE gate) | Validation evidence used |
| --- | --- | --- | --- |
| A03 Metric semantics | long-tail synonym/phrase mapping stabilized and unsafe remap guards tightened | >=95% semantic match recall with zero unsafe remap | `tests/test_autonomy_semantic.py`, `tests/test_api_autonomy_features.py`, `reports/qa_round11_blackbox_fresh_20260301_164742.json` |
| A18 Root-cause hypothesising | ranked driver hypotheses with evidence/caveat integrated into runtime and narrative | ranked root-cause payload in successful outputs | `tests/test_root_cause_hypotheses.py`, `tests/test_api_runtime_features.py` |
| A19 Scenario analysis | scenario assumptions persisted/replayed by id with tenant+connection guards | scenario-set CRUD + replay in runtime/query flow | `tests/test_runtime_store_scenarios.py`, `tests/test_api_runtime_features.py` |
| T01 Domain-agnostic onboarding | onboarding profile moved from script-only artifact to runtime-governed metadata | health/runtime payload exposes onboarding profile contract/version | `tests/test_onboarding_profile.py`, `tests/test_api_connection_routing.py` |
| T02 Multi-source ingestion governance | stream URI ingestion upgraded to governed snapshot mirror path | stream snapshot manifest with freshness/schema validation | `tests/test_stream_snapshot.py`, `tests/test_api_connection_routing.py` |
| T04 Agent role orchestration | handoff contracts + blackboard gating/edges enforced and decomposed helpers added | stage-level handoff reason codes and critical artifact gating | `tests/test_slice_completion.py`, `tests/test_blackboard_contracts.py` |
| T05 Autonomous correction loop | correction switch now hard-gated on coverage/contradiction/metric-family locks | correction cannot switch to unsafe plan family | `tests/test_api_autonomy_features.py`, `tests/test_slice_completion.py` |
| T06 Organizational memory | recall precision and tenant isolation hardened for noisy/paraphrased prompts | precision >=90% target suite and strict tenant isolation | `tests/test_autonomy_store.py`, `tests/test_autonomy_semantic.py` |
| T08 Provider strategy | automated parity gating + local-stability policy reduced multimode drift to zero | full multimode black-box round at 100 with no mode errors | `tests/test_provider_parity.py`, `tests/test_provider_parity_gate_script.py`, `reports/qa_round11_blackbox_fresh_20260301_164742.json` |

## 7) Round-based testing protocol (must be fresh every round)

For each build round `R`:
1. Generate a brand-new prompt set (no reuse from previous rounds).
2. Cover all 45 capabilities at least once in the round.
3. Run across all modes: deterministic, local, openai, anthropic, auto.
4. Score each capability as DONE/PARTIAL/GAP from evidence.
5. Publish round report under `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/reports/`.

Required artifacts per round:
- `round_R_capability_eval_<timestamp>.json`
- `round_R_capability_eval_<timestamp>.md`
- `round_R_multimode_perf_<timestamp>.md`

## 8) Definition of progress

A build round is considered a product improvement only if:
1. At least one `GAP -> PARTIAL` or `PARTIAL -> DONE` transition is proven.
2. No regression in safety/trust categories.
3. Measured NP Reality increases.
4. Report includes concrete failures, not only pass summary.
