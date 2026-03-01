# agent-semanticRetrieval

Agent: semanticRetrieval
Layer: layer4-semantic

Use skills:
- query-expert
- sql-optimization-patterns

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- table/column mapping
- metric grounding
