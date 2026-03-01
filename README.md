# dataDa

`dataDa` is an agentic data analytics product-in-progress: a conversational analytics team that can ingest business data, answer questions, explain decisions, and improve via correction rules.

This repository now uses a **minimal top-level documentation set**:

1. `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/README.md` (how to run/use)
2. `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/PRODUCT_GAP_TRACKER.md` (North Pole from product POV)
3. `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/TECHNICAL_BUILD_TRACKER.md` (implementation plan + growth tracking)

## Current Product Baseline

- North Pole baseline (product reality): **40/100**
- Primary blocker: semantic intelligence depth (business definition quality, analyst-level interpretation, autonomous decision quality)
- Existing strengths: safety, trace transparency, multi-provider runtime, trust shell

## Quick Start

## 1) Setup

```bash
cd /Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2) Prepare data

### Ingest local files into DuckDB

```bash
PYTHONPATH=src python -m haikugraph.cli ingest \
  --data-dir ./data \
  --db-path ./data/haikugraph.db \
  --force
```

### Or attach an existing DuckDB directly

```bash
PYTHONPATH=src python -m haikugraph.cli use-db \
  --db-path /absolute/path/to/your.duckdb
```

## 3) Provider setup (optional)

Set keys in environment:

```bash
HG_OPENAI_API_KEY=...
HG_ANTHROPIC_API_KEY=...
HG_OLLAMA_BASE_URL=http://localhost:11434
```

## 4) Run

```bash
./run.sh
```

Open:
- UI: [http://localhost:8000](http://localhost:8000)
- Health: [http://localhost:8000/api/assistant/health](http://localhost:8000/api/assistant/health)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## 5) Validate quickly

```bash
curl -s http://127.0.0.1:8000/api/assistant/health
curl -s http://127.0.0.1:8000/api/assistant/providers
```

## Core APIs

- `POST /api/assistant/query`
- `POST /api/assistant/query/async`
- `GET /api/assistant/query/async/{job_id}`
- `POST /api/assistant/fix`
- `GET /api/assistant/rules`
- `GET /api/assistant/providers`
- `GET /api/assistant/capability/scoreboard`
- `GET /api/assistant/models/local`
- `GET /api/assistant/models/openai`
- `GET /api/assistant/models/anthropic`
- `POST /api/assistant/models/local/pull`

## Test and benchmark

Run regression:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -q
```

Run fast per-round regression slice (impact-based):

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_fast_round_tests.py --changed-only
```

Run black-box QA:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/qa_round11_blackbox_fresh.py \
  --base-url http://127.0.0.1:8000 \
  --db-path data/haikugraph.db \
  --atomic-workers 6 \
  --local-atomic-workers 1 \
  --followup-workers 2 \
  --local-followup-workers 1 \
  --mode-workers 2 \
  --retry-count 2 \
  --retry-backoff-seconds 0.7
```

Run provider/model benchmark:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_provider_model_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --db-path data/haikugraph.db \
  --max-cases 6 \
  --max-models-per-provider 1 \
  --workers 2 \
  --local-workers 1 \
  --request-timeout 90 \
  --request-retries 1
```

Notes:
- QA and benchmark runs now auto-generate an isolated tenant id unless `--tenant-id` is provided.
- This avoids cross-run query-budget throttling (`HTTP 429`) contaminating results.
- Use `--all-advertised-models` for exhaustive model sweeps (slower).

Runtime latency knobs:

```bash
# cache provider availability checks for N seconds (default 8)
HG_PROVIDER_SNAPSHOT_TTL_SECONDS=8

# in auto mode, route simple metric asks to deterministic for lower latency
HG_AUTO_DETERMINISTIC_FAST_PATH=true
```

## Product direction

The fixed North Pole definition, capability list, and progress logic are in:
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/PRODUCT_GAP_TRACKER.md`
- `/Users/moenuddeenahmadshaik/Desktop/dataAssistantGenAI/haikugraph/TECHNICAL_BUILD_TRACKER.md`
