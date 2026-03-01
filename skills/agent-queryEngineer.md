# agent-queryEngineer

Agent: queryEngineer
Layer: layer4-queryEngineer

Use skills:
- query-expert
- sql-optimization-patterns

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- final SQL
- safe predicates
- grouping correctness
