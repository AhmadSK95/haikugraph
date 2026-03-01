# agent-planning

Agent: planning
Layer: layer4-planning

Use skills:
- dispatching-parallel-agents
- query-expert

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- executable graph
- specialist handoffs
