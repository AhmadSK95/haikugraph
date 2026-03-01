# agent-execution

Agent: execution
Layer: layer4-execution

Use skills:
- verification-before-completion

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- execution packet
- row sample
- timing packet
