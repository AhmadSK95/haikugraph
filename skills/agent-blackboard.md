# agent-blackboard

Agent: blackboard
Layer: layer4-planning

Use skills:
- dispatching-parallel-agents

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- artifact registry
- producer/consumer edges
