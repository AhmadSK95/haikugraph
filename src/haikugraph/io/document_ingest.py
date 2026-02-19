"""Document ingestion into DuckDB evidence table for text-heavy sources."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".log", ".rst", ".pdf", ".docx"}
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP_CHARS = 180


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


def _chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    clean = (text or "").strip()
    if not clean:
        return []
    cap = max(220, int(chunk_chars))
    overlap = max(0, min(cap // 2, int(overlap_chars)))
    out: list[dict[str, Any]] = []
    start = 0
    idx = 0
    n = len(clean)
    while start < n:
        end = min(n, start + cap)
        # Prefer cutting at sentence/newline boundaries when possible.
        if end < n:
            boundary_window = clean[start:end]
            boundary = max(
                boundary_window.rfind("\n\n"),
                boundary_window.rfind(". "),
                boundary_window.rfind("; "),
            )
            if boundary > 180:
                end = start + boundary + 1
        piece = clean[start:end].strip()
        if piece:
            out.append(
                {
                    "chunk_index": idx,
                    "char_start": start,
                    "char_end": end,
                    "content": piece,
                    "token_count": _token_count(piece),
                }
            )
            idx += 1
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return out


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datada_document_chunks (
                chunk_id VARCHAR,
                doc_id VARCHAR,
                source_path VARCHAR,
                file_name VARCHAR,
                title VARCHAR,
                chunk_index BIGINT,
                char_start BIGINT,
                char_end BIGINT,
                content VARCHAR,
                token_count BIGINT,
                ingested_at TIMESTAMP
            )
            """
        )
        if force:
            conn.execute("DELETE FROM datada_documents")
            conn.execute("DELETE FROM datada_document_chunks")

        inserted = 0
        skipped = 0
        chunk_inserted = 0
        for file_path in files:
            content = _extract_text(file_path).strip()
            if not content:
                skipped += 1
                continue
            source_path = str(file_path)
            conn.execute("DELETE FROM datada_document_chunks WHERE source_path = ?", [source_path])
            conn.execute("DELETE FROM datada_documents WHERE source_path = ?", [source_path])
            doc_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO datada_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    doc_id,
                    source_path,
                    file_path.name,
                    file_path.suffix.lower().lstrip("."),
                    file_path.stem,
                    content,
                    _token_count(content),
                    datetime.utcnow(),
                ],
            )
            for chunk in _chunk_text(content):
                conn.execute(
                    """
                    INSERT INTO datada_document_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        doc_id,
                        source_path,
                        file_path.name,
                        file_path.stem,
                        int(chunk["chunk_index"]),
                        int(chunk["char_start"]),
                        int(chunk["char_end"]),
                        str(chunk["content"]),
                        int(chunk["token_count"]),
                        datetime.utcnow(),
                    ],
                )
                chunk_inserted += 1
            inserted += 1
    finally:
        conn.close()

    return {
        "success": True,
        "message": (
            f"Ingested {inserted} document(s), {chunk_inserted} chunks; "
            f"skipped {skipped}."
        ),
        "ingested": inserted,
        "chunks": chunk_inserted,
        "skipped": skipped,
        "db_path": str(db),
    }
