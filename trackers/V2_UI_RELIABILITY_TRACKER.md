# V2 UI Reliability Tracker

Last Updated: 2026-03-01
Scope: Stabilize current UI shell while backend v2 matures.

## Reliability Objectives
1. Surface continuity and quality diagnostics in primary workflow.
2. Make provider degradation explicit and inspectable.
3. Keep explainability available without opening raw backend logs.
4. Prepare for modularization without blocking current UX.

## Task Tracker
| ID | Task | Status | Evidence |
|---|---|---|---|
| E01 | Split monolithic `ui.html` into modules | DONE | `ui.html` now references `/ui/assets/ui.css` + `/ui/assets/ui.js`; assets served by API |
| E02 | Add typed API client | DONE | `ui.js` now uses centralized `apiClient` with payload validation + timeout handling |
| E03 | Continuity diagnostics panel | IN_PROGRESS | slice signature chip displayed |
| E04 | Quality flags + assumptions in result cards | DONE | new chips + assumptions block |
| E05 | Provider/degradation transparency | DONE | provider/fallback chips visible |
| E06 | Explainability modal stage timeline and contract checks | IN_PROGRESS | diagnostics payload expanded + runtime stage events emitted from v2 adapter |
| E07 | Rule UX validation and preview | IN_PROGRESS | existing controls; stricter validation pending |
| E08 | Accessibility/responsive/perf pass | NOT_STARTED | lighthouse baseline pending |

## Defect Watchlist
| Priority | Item | Status |
|---|---|---|
| P1 | UI still single-file and difficult to safely evolve | CLOSED |
| P1 | No strict runtime schema validation in browser | CLOSED (core endpoints now validated through typed client layer) |
| P2 | Large explain payload rendering cost on long traces | OPEN |
| P2 | Missing stage timeline visual budget indicators | OPEN |

## Acceptance Criteria (Pre-Cutover)
- Diagnostics present for every response (`analysis_version`, `slice_signature`, `quality_flags`, `truth_score`).
- Explicit provider degradation visible for all fallback scenarios.
- Explain modal shows stage diagnostics without errors.
- No regression in async query flow and fix workflow.
