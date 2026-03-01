# Skills Integration Tracker (T0-T10)

## Goal
Build a layered, skill-driven autonomous analytics team using a project-local skill base in `haikugraph/skills/`.

## Task Board

| Task | Description | Status | Exit Criteria |
|---|---|---|---|
| T0 | Reset structure to `skills/` | completed | `agent_skills/` removed; `skills/` created |
| T1 | Seed skill base using `npx skills` | completed | required skill set downloaded to `skills/base/` + manifest captured |
| T2 | Create layer skill contracts | completed | `layer*.md` files created in `skills/` |
| T3 | Create agent skill contracts | completed | `agent-*.md` files created for all runtime agents |
| T4 | Publish skill inventory | completed | `skills/skill_inventory.yaml` lists all external + internal contracts |
| T5 | Publish skill routing matrix | completed | `skills/skill_router.md` maps intent/domain->layer->agent->skills |
| T6 | Runtime integration | completed | agent trace includes selected skills and policy rationale |
| T7 | Explainability integration | completed | Explain view shows per-agent skills used |
| T8 | Cleanup stale code/docs | completed | wrong scaffolding removed; dead references cleaned |
| T9 | Security and leak checks | completed | secret scan + ignore policy + pre-commit checks pass |
| T10 | QA + performance report | completed | benchmark report generated post-integration |

## Required skill set (lean)
1. `find-skills`
2. `spreadsheet`
3. `pdf`
4. `doc`
5. `figma-implement-design`
6. `csv-data-wrangler`
7. `query-expert`
8. `sql-optimization-patterns`
9. `dispatching-parallel-agents`
10. `verification-before-completion`
11. `systematic-debugging`

## Closure Gate
- T0-T10 all completed.
- Report published under `skills/reports/` and `reports/`.
- Latest benchmark artifacts: `reports/provider_model_benchmark_20260228_192101.{json,html}`.
