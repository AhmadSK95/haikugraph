# haikugraph

Data assistant with graph-based analysis.

## Installation

```bash
# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Ingest Excel Files

Ingest Excel files from the `data/` directory into a DuckDB database:

```bash
# Basic usage - ingest all Excel files from ./data
haikugraph ingest

# Specify custom data directory
haikugraph ingest --data-dir ./my-data

# Specify custom database path
haikugraph ingest --db-path ./output/my-database.duckdb

# Read a specific sheet by name or index
haikugraph ingest --sheet "Sheet2"
haikugraph ingest --sheet 1

# Combine options
haikugraph ingest --data-dir ./data --db-path ./data/main.duckdb --sheet 0
```

**Behavior:**
- Finds all `.xlsx` and `.xls` files in the data directory
- Reads the first sheet by default (or specify with `--sheet`)
- Creates tables with sanitized names from filenames
- Overwrites existing tables by default (controlled by `--force`)
- Creates DuckDB file at `./data/haikugraph.duckdb`
- Prints summary of tables, rows, and columns
- **Auto-fallback to string mode**: If a file has mixed types or dtype issues, automatically retries with all columns as strings to ensure ingestion succeeds

**Table Naming:**
- Filename → lowercase → non-alphanumeric → underscore → collapse
- Example: `Sales Data 2024.xlsx` → `sales_data_2024`
- Files starting with digits get `t_` prefix: `2024-data.xlsx` → `t_2024_data`

### Profile Tables

Generate a detailed JSON profile of all tables in the database:

```bash
# Basic usage - profile all tables
haikugraph profile

# Specify custom database path
haikugraph profile --db-path ./data/main.duckdb

# Specify custom output path
haikugraph profile --out ./reports/profile.json

# Control sampling for large tables
haikugraph profile --sample-rows 50000 --top-k 20
```

**What it profiles:**
- **Per table**: row count, column count, sample rows
- **Per column**: 
  - Data type, null count/percentage
  - Distinct count (approximate for large tables)
  - Sample values (up to 5)
  - For numeric: min, max, mean
  - For dates: min_date, max_date
  - For strings: top values with counts and percentages

**Output**: JSON file at `./data/profile.json` (default)

**View the profile:**
```bash
# Pretty-print first part
cat data/profile.json | head -100

# View with Python
python3 -m json.tool data/profile.json | less

# Quick peek at structure
jq 'keys' data/profile.json
```

### Generate Data Cards

Create semantic data cards from the profile:

```bash
# Build cards from profile
haikugraph cards build

# List all cards
haikugraph cards list

# Show a specific card
haikugraph cards show "table:test_1_1"
haikugraph cards show "column:test_1_1.customer_id"

# View table card with top columns
haikugraph cards table test_3_1
```

**What are Data Cards?**

Data cards are semantic annotations that help understand your data:

- **TableCard**: Grain, primary keys, time/money/status columns, gotchas, suggested metrics
- **ColumnCard**: Type, nullability, distinctness, semantic hints (identifier, timestamp, money, geography)
- **RelationCard**: Potential joins between tables with confidence scores

**Output:** Individual JSON files in `./data/cards/` plus an `index.json`

### Check Environment

Verify your haikugraph setup:

```bash
haikugraph doctor
```

This shows:
- Python executable path
- haikugraph and DuckDB versions
- Data directory status and Excel file count
- Database status and table count

### Other Commands

```bash
haikugraph --help        # Show all commands
haikugraph --version     # Show version
```

## Development

```bash
# Format code
ruff format .

# Lint code
ruff check .
```
