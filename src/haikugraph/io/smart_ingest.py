"""Smart Excel ingestion with intelligent file merging.

This module detects when multiple Excel files represent the same dataset
split across different columns and merges them into a single table.
"""

import re
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def _read_file(file_path: Path, sheet=None) -> pd.DataFrame:
    """Read a data file into a DataFrame based on its extension."""
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path, sheet_name=sheet if sheet is not None else 0)
    elif suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def detect_file_groups(excel_files: list[Path]) -> dict[str, list[Path]]:
    """
    Analyze Excel files to detect groups that should be merged.
    
    Files are grouped together if they:
    1. Share common key columns (transaction_id, customer_id, etc.)
    2. Have significant overlap in key values (>1000 shared IDs)
    
    Args:
        excel_files: List of Excel file paths
        
    Returns:
        Dictionary mapping group_name -> list of files to merge
    """
    # Load all files and analyze
    file_data = {}
    for f in excel_files:
        try:
            # Load full file - need complete key columns for overlap analysis
            df = _read_file(f)
            file_data[f] = {
                'df': df,
                'columns': set(df.columns),
                'key_columns': set(df.columns) & {
                    'transaction_id', 'customer_id', 'payee_id', 
                    'quote_id', 'deal_id', 'sha_id'
                }
            }
        except Exception:
            # Skip files that can't be read
            continue
    
    # Group files by key overlap
    groups = {}
    processed = set()
    
    for f1 in sorted(file_data.keys()):
        if f1 in processed:
            continue
            
        # Start a new group with this file
        group = [f1]
        d1 = file_data[f1]
        
        # Find files that should merge with f1
        for f2 in sorted(file_data.keys()):
            if f2 == f1 or f2 in processed:
                continue
                
            d2 = file_data[f2]
            
            # Check if they share key columns
            common_keys = d1['key_columns'] & d2['key_columns']
            if not common_keys:
                continue
            
            # Check if key values overlap significantly
            # Use the most reliable key column available
            key_col = None
            max_overlap = 0
            
            for key in common_keys:
                try:
                    ids1 = set(d1['df'][key].dropna())
                    ids2 = set(d2['df'][key].dropna())
                    overlap = len(ids1 & ids2)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        key_col = key
                except Exception:
                    continue
            
            # If significant overlap (>1000 IDs), merge these files
            if max_overlap > 1000:
                group.append(f2)
                processed.add(f2)
        
        # Create group name based on first file
        group_name = _generate_group_name(f1, len(group))
        groups[group_name] = group
        processed.add(f1)
    
    return groups


def _generate_group_name(primary_file: Path, group_size: int) -> str:
    """Generate a table name for a merged group."""
    # Use sanitized name from primary file
    name = primary_file.stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    
    if name and name[0].isdigit():
        name = f"t_{name}"
    
    # Add indicator if merged from multiple files
    if group_size > 1:
        name = f"{name}_merged"
    
    return name or "table"


def merge_files_on_key(files: list[Path], key_column: str = None) -> pd.DataFrame:
    """
    Merge multiple Excel files into a single DataFrame.
    
    Strategy:
    - If files have duplicate rows per key, union all rows (preserving duplicates)
    - Add missing columns with NULL values
    
    Args:
        files: List of file paths to merge
        key_column: Join key (auto-detected if None)
        
    Returns:
        Merged DataFrame with all columns from all files
    """
    if not files:
        raise ValueError("No files to merge")
    
    if len(files) == 1:
        return _read_file(files[0])
    
    # Load all files
    dfs = []
    all_columns = set()
    
    for file_path in files:
        try:
            df = _read_file(file_path)
            dfs.append(df)
            all_columns.update(df.columns)
        except Exception:
            continue
    
    if not dfs:
        raise ValueError("No files could be loaded")
    
    # Standardize columns across all dataframes
    # Add missing columns as NULL
    standardized_dfs = []
    for df in dfs:
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
        # Reorder columns consistently
        standardized_dfs.append(df[sorted(all_columns)])
    
    # Concatenate all dataframes
    # This preserves all rows including duplicates within each file
    merged_df = pd.concat(standardized_dfs, axis=0, ignore_index=True)
    
    # Remove exact duplicate rows (same values across ALL columns)
    # This removes true duplicates while keeping different info for same keys
    merged_df = merged_df.drop_duplicates()
    
    return merged_df


def smart_ingest_excel_to_duckdb(
    data_dir: Path,
    db_path: Path,
    sheet: Optional[str | int] = None,
    force: bool = True,
) -> dict:
    """
    Intelligently ingest Excel files with automatic merging.
    
    This function:
    1. Analyzes all Excel files to detect related datasets
    2. Merges files that represent the same data with different columns
    3. Loads merged data into DuckDB tables
    
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
    
    # Find data files
    supported_patterns = ["*.xlsx", "*.xls", "*.csv", "*.parquet", "*.pq"]
    data_files = []
    for pattern in supported_patterns:
        data_files.extend(data_dir.glob(pattern))

    if not data_files:
        return {
            "status": "no_files",
            "message": f"No data files found in {data_dir} (supported: .xlsx, .xls, .csv, .parquet)",
            "tables": [],
        }

    # Detect file groups
    print("Analyzing data files for relationships...")
    file_groups = detect_file_groups(data_files)
    
    # Process results
    results = {
        "status": "success",
        "db_path": str(db_path.absolute()),
        "tables": [],
        "errors": [],
        "merges": [],
    }
    
    # Create database connection
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    
    for table_name, files in sorted(file_groups.items()):
        try:
            if len(files) > 1:
                # Merge multiple files
                source_files = [f.name for f in files]
                results["merges"].append({
                    "table": table_name,
                    "source_files": source_files,
                    "file_count": len(files)
                })
                print(f"Merging {len(files)} files into '{table_name}': {source_files}")
                
                df = merge_files_on_key(files)
            else:
                # Single file, no merge needed
                df = _read_file(files[0], sheet=sheet)
            
            # Drop table if exists and force is True
            if force:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Write to DuckDB - try normal first, then string fallback if it fails
            read_mode = "normal"
            try:
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            except Exception as duck_error:
                # DuckDB type inference failed, convert all to strings
                for col in df.columns:
                    df[col] = df[col].astype(str)
                df = df.replace("nan", None).replace("None", None)
                read_mode = "string_fallback"
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            table_info = {
                "file": files[0].name if len(files) == 1 else f"{len(files)} merged files",
                "table": table_name,
                "rows": row_count,
                "columns": len(df.columns),
            }
            if read_mode == "string_fallback":
                table_info["mode"] = "string_fallback"
            if len(files) > 1:
                table_info["merged_from"] = [f.name for f in files]
            
            results["tables"].append(table_info)
        
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            
            results["errors"].append({
                "files": [f.name for f in files],
                "error_type": error_type,
                "error": error_msg,
            })
    
    conn.close()
    
    if results["errors"] and not results["tables"]:
        results["status"] = "failed"
    elif results["errors"]:
        results["status"] = "partial"
    
    return results


def format_smart_ingestion_summary(results: dict) -> str:
    """Format smart ingestion results as a human-readable summary."""
    lines = []
    
    if results["status"] == "no_files":
        lines.append(f"❌ {results['message']}")
        lines.append("\nPlace data files (.xlsx, .xls, .csv, .parquet) in the data directory to ingest.")
        return "\n".join(lines)
    
    lines.append("✅ Smart ingestion complete\n")
    lines.append(f"Database: {results['db_path']}")
    lines.append(f"Tables created: {len(results['tables'])}")
    
    if results["merges"]:
        lines.append(f"\n🔗 Merged file groups: {len(results['merges'])}")
        for merge in results["merges"]:
            lines.append(f"  • {merge['table']}: {merge['file_count']} files combined")
            for src in merge['source_files']:
                lines.append(f"    - {src}")
    
    if results["tables"]:
        lines.append("\nTables:")
        for table in results["tables"]:
            mode_indicator = " [string mode]" if table.get("mode") == "string_fallback" else ""
            merge_indicator = " [MERGED]" if "merged_from" in table else ""
            lines.append(
                f"  • {table['table']:20s} "
                f"({table['rows']:,} rows × {table['columns']} columns){mode_indicator}{merge_indicator}"
            )
            if "merged_from" in table:
                lines.append(f"    ← {', '.join(table['merged_from'][:2])}")
                if len(table['merged_from']) > 2:
                    lines.append(f"      + {len(table['merged_from']) - 2} more")
    
    if results["errors"]:
        lines.append(f"\n⚠️  Failures ({len(results['errors'])}):")
        for error in results["errors"]:
            error_type = error.get("error_type", "Error")
            files_str = ", ".join(error['files'][:2])
            lines.append(f"  • {files_str} [{error_type}]: {error['error']}")
    
    return "\n".join(lines)
