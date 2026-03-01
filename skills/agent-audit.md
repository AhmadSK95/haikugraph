# agent-audit

Agent: audit
Layer: layer5-trustValidation

Use skills:
- verification-before-completion
- systematic-debugging

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- check results
- score
- failure causes
