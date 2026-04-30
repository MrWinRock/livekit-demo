"""Read-only access to the RAG-UI backend's documents store.

Reads the backend's SQLite database (read-only) to list documents the user
has already uploaded, then reads the corresponding PDF files from the
backend's storage directory and extracts text. The backend itself is never
written to and its code is untouched.

Configuration via env:
- BACKEND_DB_PATH: absolute path to the backend's app.db (default: relative
  guess from this project's location).
- BACKEND_STORAGE_PATH: absolute path to the backend's storage directory.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BACKEND_DB_PATH = Path(
    os.environ.get(
        "BACKEND_DB_PATH",
        Path(__file__).parent.parent.parent.parent
        / "ai"
        / "backend"
        / "data"
        / "app.db",
    )
)
DEFAULT_BACKEND_STORAGE_PATH = Path(
    os.environ.get(
        "BACKEND_STORAGE_PATH",
        Path(__file__).parent.parent.parent.parent / "ai" / "backend" / "storage",
    )
)


@dataclass(frozen=True)
class Document:
    id: str
    name: str
    type: str
    size_bytes: int
    uploaded_at: str
    status: str
    chunk_count: int | None
    storage_path: str

    def to_summary(self) -> str:
        size_kb = self.size_bytes // 1024
        return (
            f"{self.name} (id={self.id[:8]}, type={self.type}, "
            f"{size_kb} KB, status={self.status})"
        )


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the backend SQLite in read-only mode so we can never corrupt it."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        name=row["name"],
        type=row["type"],
        size_bytes=row["size_bytes"],
        uploaded_at=row["uploaded_at"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        storage_path=row["storage_path"],
    )


def list_ready_documents(
    db_path: str | Path = DEFAULT_BACKEND_DB_PATH,
) -> list[Document]:
    """Return only documents the backend has finished processing."""
    with _connect_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE status = 'ready' ORDER BY uploaded_at DESC"
        ).fetchall()
    return [_row_to_document(r) for r in rows]


def get_document(
    document_id: str,
    db_path: str | Path = DEFAULT_BACKEND_DB_PATH,
) -> Document | None:
    with _connect_readonly(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return _row_to_document(row) if row else None


def find_documents_by_name(
    keyword: str,
    db_path: str | Path = DEFAULT_BACKEND_DB_PATH,
) -> list[Document]:
    """Case-insensitive substring search on document name. Ready docs only."""
    pattern = f"%{keyword.strip()}%"
    with _connect_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE status = 'ready' "
            "AND name LIKE ? COLLATE NOCASE "
            "ORDER BY uploaded_at DESC",
            (pattern,),
        ).fetchall()
    return [_row_to_document(r) for r in rows]


def _resolve_file_path(
    document: Document,
    storage_root: str | Path = DEFAULT_BACKEND_STORAGE_PATH,
) -> Path:
    """Resolve the on-disk PDF path for a document.

    Prefers the storage_path column when it points at an existing file.
    Falls back to `<storage_root>/<id>.<type>` (the backend's naming convention).
    """
    candidate = Path(document.storage_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    relative = Path(storage_root) / candidate.name
    if relative.exists():
        return relative
    return Path(storage_root) / f"{document.id}.{document.type}"


def read_document_text(
    document: Document,
    storage_root: str | Path = DEFAULT_BACKEND_STORAGE_PATH,
    max_chars: int = 8000,
) -> str:
    """Extract plain text from a document's underlying file.

    Currently supports PDF (via pypdf) and plain text. Output is truncated
    to `max_chars` to keep the LLM context bounded — voice replies don't
    need 100 pages of input.
    """
    path = _resolve_file_path(document, storage_root)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found at {path}")

    if document.type == "pdf":
        text = _extract_pdf_text(path)
    elif document.type in ("txt", "md"):
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported document type for text extraction: {document.type}"
        )

    if len(text) > max_chars:
        return (
            text[:max_chars]
            + f"\n\n[...truncated. Document is {len(text)} characters total.]"
        )
    return text


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()
