# Skill Catalog Seed (Lean v1)

Installed via `npx skills add ... --agent codex --copy` into this project:

1. `vercel-labs/skills` -> `find-skills`
2. `openai/skills` -> `spreadsheet`
3. `openai/skills` -> `pdf`
4. `openai/skills` -> `doc`
5. `404kidwiz/claude-supercode-skills` -> `csv-data-wrangler`
6. `jamesrochabrun/skills` -> `query-expert`
7. `rmyndharis/antigravity-skills` -> `sql-optimization-patterns`
8. `obra/superpowers` -> `dispatching-parallel-agents`
9. `obra/superpowers` -> `systematic-debugging`
10. `obra/superpowers` -> `verification-before-completion`

Reason for lean base:
- keep dependencies auditable
- keep startup/install time bounded
- expand only on measured quality or latency benefit
