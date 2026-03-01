# agent-trust

Agent: trust
Layer: layer5-trustValidation

Use skills:
- verification-before-completion

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- reliability telemetry
- trust summaries
