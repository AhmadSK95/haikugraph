# agent-intake

Agent: intake
Layer: layer4-context

Use skills:
- query-expert

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- structured goal
- ambiguity flags
- clarifying needs
