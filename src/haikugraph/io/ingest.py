"""Excel to DuckDB ingestion utilities."""

import re
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def sanitize_table_name(filename: str) -> str:
    """
    Sanitize filename into a valid SQL table name.

    Rules:
    - Lowercase
    - Non-alphanumeric -> underscore
    - Collapse multiple underscores
    - Prefix with t_ if starts with digit
    """
    # Remove extension and convert to lowercase
    name = Path(filename).stem.lower()

    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name)

    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)

    # Strip leading/trailing underscores
    name = name.strip("_")

    # Prefix with t_ if starts with digit
    if name and name[0].isdigit():
        name = f"t_{name}"

    return name or "table"


def ingest_excel_to_duckdb(
    data_dir: Path,
    db_path: Path,
    sheet: Optional[str | int] = None,
    force: bool = True,
) -> dict:
    """
    Ingest Excel files from data_dir into DuckDB database.

    Args:
        data_dir: Directory containing Excel files
        db_path: Path to DuckDB database file
        sheet: Sheet name or index (default: first sheet)
        force: Overwrite existing tables if True

    Returns:
        Dictionary with ingestion results
    """
    data_dir = Path(data_dir)
    db_path = Path(db_path)

    # Find Excel files
    excel_patterns = ["*.xlsx", "*.xls"]
    excel_files = []
    for pattern in excel_patterns:
        excel_files.extend(data_dir.glob(pattern))

    if not excel_files:
        return {
            "status": "no_files",
            "message": f"No Excel files found in {data_dir}",
            "tables": [],
        }

    # Process files
    results = {
        "status": "success",
        "db_path": str(db_path.absolute()),
        "tables": [],
        "errors": [],
    }

    # Create database connection
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    for excel_file in sorted(excel_files):
        try:
            # Read Excel file
            sheet_arg = sheet if sheet is not None else 0
            df = None
            read_mode = "normal"

            # First attempt: normal read
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_arg)
            except (ValueError, TypeError, OverflowError) as e:
                # Retry with string fallback for problematic files
                try:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_arg,
                        dtype=str,
                        keep_default_na=False,
                    )
                    # Ensure all object columns are strings
                    for col in df.columns:
                        if df[col].dtype == "object":
                            df[col] = df[col].astype(str)
                    # Replace empty strings with None for cleaner NULL handling
                    df = df.replace("", None)
                    read_mode = "string_fallback"
                except Exception:
                    # Re-raise original error if fallback also fails
                    raise e

            # Sanitize table name
            table_name = sanitize_table_name(excel_file.name)

            # Drop table if exists and force is True
            if force:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Write to DuckDB - try normal first, then string fallback if it fails
            try:
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            except Exception as duck_error:
                # DuckDB type inference failed, convert all to strings
                if read_mode != "string_fallback":
                    # Convert all columns to string for DuckDB compatibility
                    for col in df.columns:
                        df[col] = df[col].astype(str)
                    df = df.replace("nan", None).replace("None", None)
                    read_mode = "string_fallback"
                    # Retry with string dataframe
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                else:
                    # Already in string mode, re-raise
                    raise duck_error

            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            table_info = {
                "file": excel_file.name,
                "table": table_name,
                "rows": row_count,
                "columns": len(df.columns),
            }
            if read_mode == "string_fallback":
                table_info["mode"] = "string_fallback"

            results["tables"].append(table_info)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            # Truncate error message to 200 chars
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."

            results["errors"].append(
                {
                    "file": excel_file.name,
                    "error_type": error_type,
                    "error": error_msg,
                }
            )

    if results["errors"] and not results["tables"]:
        results["status"] = "failed"
    elif results["errors"]:
        results["status"] = "partial"

    return results


def format_ingestion_summary(results: dict) -> str:
    """Format ingestion results as a human-readable summary."""
    lines = []

    if results["status"] == "no_files":
        lines.append(f"❌ {results['message']}")
        lines.append("\nPlace Excel files (.xlsx, .xls) in the data directory to ingest.")
        return "\n".join(lines)

    lines.append("✅ Ingestion complete\n")
    lines.append(f"Database: {results['db_path']}")
    lines.append(f"Tables created: {len(results['tables'])}")

    if results["tables"]:
        lines.append("\nTables:")
        for table in results["tables"]:
            mode_indicator = " [string mode]" if table.get("mode") == "string_fallback" else ""
            lines.append(
                f"  • {table['table']:20s} "
                f"({table['rows']:,} rows × {table['columns']} columns){mode_indicator} "
                f"← {table['file']}"
            )

    if results["errors"]:
        lines.append(f"\n⚠️  Failures ({len(results['errors'])}):")
        for error in results["errors"]:
            error_type = error.get("error_type", "Error")
            lines.append(f"  • {error['file']} [{error_type}]: {error['error']}")

    return "\n".join(lines)
