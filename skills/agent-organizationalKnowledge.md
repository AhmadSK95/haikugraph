# agent-organizationalKnowledge

Agent: organizationalKnowledge
Layer: layer4-semantic

Use skills:
- query-expert
- verification-before-completion

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- domain directives
- required policy notes
