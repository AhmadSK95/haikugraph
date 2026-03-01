# agent-memory

Agent: memory
Layer: layer4-context

Use skills:
- systematic-debugging

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- episodic hints
- correction candidates
