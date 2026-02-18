# dataDa Enterprise Autonomy Blueprint

## Positioning

dataDa is a **verifiable autonomous analytics engine**:

- autonomous in reasoning, decomposition, and self-correction
- bounded in side-effects, security scope, and resource usage
- transparent in every answer (trace, SQL, checks, confidence)

## Differentiator vs Generic BI Chat

1. Truth-first runtime:
   - candidate plan reconciliation
   - replay checks + concept alignment checks
   - confidence tied to evidence quality, not model fluency
2. Persistent learning:
   - successful runs and correction actions are persisted
   - reusable correction rules are recalled on future queries
3. Human + machine correction loop:
   - user feedback endpoint stores explicit correction signals
   - optional correction rules can be registered immediately
4. Hybrid runtime choice:
   - deterministic, local LLM, OpenAI, or auto routing

## Bounded Autonomy Definition

Bounds do not limit cognition. Bounds limit risk.

- Cognitive autonomy: planning, decomposition, and correction are autonomous.
- Risk bounds: read-only execution, policy gates, iteration caps, and safe probes.

Current runtime controls:

- `autonomy_mode`
- `auto_correction`
- `strict_truth`
- `max_refinement_rounds`
- `max_candidate_plans`

## Current Components

1. MemoryAgent
   - episodic recall from prior successful runs
   - correction rule recall
2. AutonomyAgent
   - plan variant generation (keyword + memory + correction rules)
   - candidate execution and scoring
   - autonomous switch to better grounded candidate
3. Toolsmith probes (bounded)
   - targeted probe SQL for concept mismatch diagnosis

## Enterprise Buildout Path

1. Data fabric:
   - direct DB connectors (Postgres, Snowflake, BigQuery)
   - streaming append connectors (Kafka/Kinesis)
2. Governance:
   - tenant-aware policy scopes, RBAC, and approval gates
3. Scale:
   - warehouse pushdown for billion-row workloads
   - partition-aware semantic marts and materialized aggregates
4. Reliability:
   - queue-based execution, retries, and distributed trace storage

## APIs Added

- `POST /api/assistant/query` now supports autonomy controls.
- `POST /api/assistant/feedback` persists quality feedback and optional correction rules.
