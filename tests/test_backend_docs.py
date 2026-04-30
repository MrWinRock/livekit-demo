"""Tests for the backend documents reader.

These tests build a tiny SQLite that mimics the backend's `documents` table,
plus tiny PDF/txt files on disk, so we never depend on the real backend
project being present at the expected path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend_docs import (
    Document,
    find_documents_by_name,
    get_document,
    list_ready_documents,
    read_document_text,
)

SCHEMA = """
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER,
    error_reason TEXT,
    storage_path TEXT NOT NULL
)
"""


def _seed_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    name: str,
    doc_type: str,
    storage_path: str,
    status: str = "ready",
    size_bytes: int = 1024,
    uploaded_at: str = "2026-04-30T10:00:00.000Z",
    chunk_count: int | None = 5,
) -> None:
    conn.execute(
        "INSERT INTO documents "
        "(id, name, type, size_bytes, uploaded_at, status, chunk_count, "
        "error_reason, storage_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            doc_id,
            name,
            doc_type,
            size_bytes,
            uploaded_at,
            status,
            chunk_count,
            None,
            storage_path,
        ),
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "app.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        _seed_document(
            conn,
            doc_id="doc-ready-1",
            name="Q3_Report.pdf",
            doc_type="pdf",
            storage_path="doc-ready-1.pdf",
            uploaded_at="2026-04-30T10:00:00.000Z",
        )
        _seed_document(
            conn,
            doc_id="doc-ready-2",
            name="Annual_Plan.txt",
            doc_type="txt",
            storage_path="doc-ready-2.txt",
            uploaded_at="2026-04-29T10:00:00.000Z",
        )
        _seed_document(
            conn,
            doc_id="doc-processing",
            name="In_Progress.pdf",
            doc_type="pdf",
            storage_path="doc-processing.pdf",
            status="processing",
            chunk_count=None,
        )
    return path


def test_list_ready_documents_excludes_processing(db_path: Path) -> None:
    docs = list_ready_documents(db_path)
    assert {d.id for d in docs} == {"doc-ready-1", "doc-ready-2"}


def test_list_ready_documents_orders_newest_first(db_path: Path) -> None:
    docs = list_ready_documents(db_path)
    assert docs[0].id == "doc-ready-1"
    assert docs[1].id == "doc-ready-2"


def test_get_document_returns_row(db_path: Path) -> None:
    doc = get_document("doc-ready-1", db_path)
    assert doc is not None
    assert doc.name == "Q3_Report.pdf"
    assert doc.type == "pdf"


def test_get_document_returns_none_for_missing_id(db_path: Path) -> None:
    assert get_document("nope", db_path) is None


def test_find_documents_by_name_case_insensitive(db_path: Path) -> None:
    results = find_documents_by_name("REPORT", db_path)
    assert len(results) == 1
    assert results[0].id == "doc-ready-1"


def test_find_documents_by_name_skips_non_ready(db_path: Path) -> None:
    results = find_documents_by_name("In_Progress", db_path)
    assert results == []


def test_read_document_text_for_txt(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "doc-1.txt").write_text("Hello from a plain text document.")

    doc = Document(
        id="doc-1",
        name="Hello.txt",
        type="txt",
        size_bytes=10,
        uploaded_at="2026-04-30T10:00:00.000Z",
        status="ready",
        chunk_count=1,
        storage_path="doc-1.txt",
    )
    text = read_document_text(doc, storage_root=storage)
    assert "Hello from a plain text document." in text


def test_read_document_text_truncates_long_content(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    body = "A" * 20000
    (storage / "doc-1.txt").write_text(body)

    doc = Document(
        id="doc-1",
        name="Long.txt",
        type="txt",
        size_bytes=20000,
        uploaded_at="2026-04-30T10:00:00.000Z",
        status="ready",
        chunk_count=1,
        storage_path="doc-1.txt",
    )
    text = read_document_text(doc, storage_root=storage, max_chars=500)
    assert len(text) < 20000
    assert "truncated" in text


def test_read_document_text_raises_for_missing_file(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    doc = Document(
        id="missing",
        name="Missing.pdf",
        type="pdf",
        size_bytes=0,
        uploaded_at="2026-04-30T10:00:00.000Z",
        status="ready",
        chunk_count=0,
        storage_path="missing.pdf",
    )
    with pytest.raises(FileNotFoundError):
        read_document_text(doc, storage_root=storage)


def test_read_document_text_rejects_unsupported_type(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "doc-1.docx").write_bytes(b"not actually a docx")

    doc = Document(
        id="doc-1",
        name="Thing.docx",
        type="docx",
        size_bytes=10,
        uploaded_at="2026-04-30T10:00:00.000Z",
        status="ready",
        chunk_count=1,
        storage_path="doc-1.docx",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        read_document_text(doc, storage_root=storage)


def test_document_summary_includes_key_fields() -> None:
    doc = Document(
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        name="Q3_Report.pdf",
        type="pdf",
        size_bytes=204800,
        uploaded_at="2026-04-30T10:00:00.000Z",
        status="ready",
        chunk_count=82,
        storage_path="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.pdf",
    )
    summary = doc.to_summary()
    assert "Q3_Report.pdf" in summary
    assert "aaaaaaaa" in summary  # truncated id
    assert "pdf" in summary
    assert "200 KB" in summary
