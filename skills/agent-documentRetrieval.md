# agent-documentRetrieval

Agent: documentRetrieval
Layer: layer1-ingest

Use skills:
- pdf
- doc

Execution contract:
- Apply only listed skills unless explicit override is provided by dispatcher policy.
- Keep output grounded in observed data and contract checks.

Must produce:
- relevant chunks
- citation metadata
