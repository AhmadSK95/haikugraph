# Skill Router Matrix

## Runtime routing principles
1. Data-first: collect evidence before narration.
2. Contract-first: compile SQL that satisfies metric/time/dimension contract.
3. Safety-first: unresolved ambiguity must clarify or lower confidence.

## Intent/domain to skill mapping

| Intent | Domain | Layers | Primary agents | Skills |
|---|---|---|---|---|
| data_overview | any | L1,L2,L4,L6 | DiscoveryPlanner, CatalogProfiler, Narrative | spreadsheet, query-expert, verification-before-completion |
| grouped_metric | transactions | L2,L4,L5 | SemanticRetrieval, Planning, QueryEngineer, Audit | query-expert, sql-optimization-patterns, verification-before-completion |
| grouped_metric | quotes | L2,L4,L5 | SemanticRetrieval, Planning, QueryEngineer, Audit | query-expert, sql-optimization-patterns, verification-before-completion |
| comparison | cross-domain | L3,L4,L5 | Dispatcher, Planning, Autonomy, Audit | dispatching-parallel-agents, query-expert, systematic-debugging |
| vague_question | any | L3,L4,L6 | Intake, ChiefAnalyst, Narrative | find-skills, dispatching-parallel-agents, verification-before-completion |
| document_question | docs/pdf | L1,L4,L6 | DocumentRetrieval, Narrative | pdf, doc, verification-before-completion |

## Auto mode policy
- Start with provider priority policy.
- Escalate model strength when ambiguity score or contradiction score is high.
- Keep deterministic fallback for contract replay and dispute cases.

## Trace requirement
Each agent trace row must include:
- selected_skills
- skill_policy_reason
- skill_contract_file
