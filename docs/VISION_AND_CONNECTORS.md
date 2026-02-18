# dataDa Vision and Connector Strategy

## Why This Project Exists

ChatGPT is strong at general reasoning. dataDa is built for a narrower but high-value problem:

- grounded analytics over private enterprise data
- reproducible SQL, not just prose answers
- transparent failure/debug path (trace, checks, confidence)
- controllable deployment (local, hybrid, or hosted LLM)

In short:
- **ChatGPT**: general intelligence
- **dataDa**: accountable analytics teammate on your data stack

## Who Benefits

- data teams that need explainable answers with SQL lineage
- operations/business teams that want natural language analytics without BI authoring
- privacy-sensitive environments where local models are required

## Ingestion Policy (Current)

The project now uses one ingestion mechanism:

- `haikugraph ingest` -> unified smart ingest pipeline
  - scans Excel files
  - detects related files
  - merges on key overlap
  - writes DuckDB marts source tables

Direct DB usage is supported:

- `haikugraph use-db --db-path /path/to/existing.duckdb_or_db`
  - validates file
  - sets `HG_DB_PATH` in `.env`
  - API uses that DB directly

## Connector Roadmap

### Structured DB connectors

- Postgres
- MySQL
- SQL Server
- Snowflake
- BigQuery

Pattern:
- introspect schema -> normalize semantic marts -> register source metadata

### Stream connectors

- Kafka / Kinesis / PubSub
- CDC from OLTP systems

Pattern:
- append-only landing tables -> periodic compaction -> incremental marts

### Document connectors (PDF/DOCX/MD/Text)

For text-heavy content, do not force SQL-only answers.

Pattern:
- parse documents into `documents_raw` and `documents_chunks`
- embed chunks for retrieval
- return dual evidence:
  - metric evidence (SQL)
  - narrative evidence (retrieved text citations)

## Guardrails for Open Source Value

- all claims tied to evidence (SQL rows or retrieved docs)
- confidence reflects audit checks, not model style
- deterministic fallback always available
- benchmark scripts included for reproducibility

