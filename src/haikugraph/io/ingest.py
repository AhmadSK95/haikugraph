"""Deprecated ingestion wrapper.

This module is kept for backward compatibility only.
The project now uses a single ingestion mechanism implemented in
`haikugraph.io.smart_ingest`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from haikugraph.io.smart_ingest import (
    format_smart_ingestion_summary,
    smart_ingest_excel_to_duckdb,
)


def ingest_excel_to_duckdb(
    data_dir: Path,
    db_path: Path,
    sheet: Optional[str | int] = None,
    force: bool = True,
) -> dict:
    """Backward-compatible alias to the unified smart ingest pipeline."""
    return smart_ingest_excel_to_duckdb(
        data_dir=Path(data_dir),
        db_path=Path(db_path),
        sheet=sheet,
        force=force,
    )


def format_ingestion_summary(results: dict) -> str:
    """Backward-compatible alias to smart ingest summary formatting."""
    return format_smart_ingestion_summary(results)

