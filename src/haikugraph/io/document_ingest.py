"""Document ingestion into DuckDB evidence table for text-heavy sources."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".log", ".rst", ".pdf", ".docx"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        import pypdf  # type: ignore
    except Exception:
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception:
        return ""
    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".markdown", ".log", ".rst"}:
        return _read_text_file(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    return ""


def _token_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9_]+", text))


def ingest_documents_to_duckdb(
    *,
    docs_dir: Path | str,
    db_path: Path | str,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(docs_dir).expanduser()
    db = Path(db_path).expanduser()
    if not root.exists():
        return {"success": False, "message": f"Document directory not found at {root}", "ingested": 0}

    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        return {"success": False, "message": f"No supported documents in {root}", "ingested": 0}

    conn = duckdb.connect(str(db), read_only=False)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datada_documents (
                doc_id VARCHAR,
                source_path VARCHAR,
                file_name VARCHAR,
                file_type VARCHAR,
                title VARCHAR,
                content VARCHAR,
                token_count BIGINT,
                ingested_at TIMESTAMP
            )
            """
        )
        if force:
            conn.execute("DELETE FROM datada_documents")

        inserted = 0
        skipped = 0
        for file_path in files:
            content = _extract_text(file_path).strip()
            if not content:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO datada_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid.uuid4()),
                    str(file_path),
                    file_path.name,
                    file_path.suffix.lower().lstrip("."),
                    file_path.stem,
                    content,
                    _token_count(content),
                    datetime.utcnow(),
                ],
            )
            inserted += 1
    finally:
        conn.close()

    return {
        "success": True,
        "message": f"Ingested {inserted} document(s); skipped {skipped}.",
        "ingested": inserted,
        "skipped": skipped,
        "db_path": str(db),
    }

