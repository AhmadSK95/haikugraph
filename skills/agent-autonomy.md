# agent-autonomy

Agent: autonomy
Layer: layer5-trustValidation

Use skills:
- dispatching-parallel-agents
- verification-before-completion

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- candidate ranking
- correction rationale
