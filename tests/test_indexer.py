"""Unit tests for the indexing orchestrator.

Tests cover:
  - No files discovered -> empty result
  - All files up to date -> all skipped
  - New files -> full pipeline executes
  - Mixed: some files need indexing, some skipped
  - Permission errors handled gracefully
  - Embedding failure propagates as RuntimeError
  - BM25 index updated after indexing
  - Result statistics are accurate
  - Integration: real ChromaDB + fake embeddings
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_rag.bm25_index import BM25Index
from knowledge_rag.indexer import IndexingResult, _read_file_with_retry, index_documents
from knowledge_rag.ingestion import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake 768-dim embeddings."""
    return [[0.1] * 768 for _ in texts]


def _make_chunk(
    content: str = "test content",
    source_file: str = "test.md",
    chunk_index: int = 0,
    total_chunks: int = 1,
    frontmatter: dict | None = None,
) -> DocumentChunk:
    """Create a DocumentChunk for testing."""
    return DocumentChunk(
        content=content,
        source_file=source_file,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(content),
        total_chunks=total_chunks,
        frontmatter=frontmatter or {},
    )


# ---------------------------------------------------------------------------
# Unit tests (all dependencies mocked)
# ---------------------------------------------------------------------------


@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_no_files_discovered(mock_discover: MagicMock, tmp_path: Path) -> None:
    """No files discovered -> empty result."""
    mock_discover.return_value = []

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test")
    bm25 = BM25Index()
    config = RagConfig()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result.files_indexed == 0
    assert result.chunks_indexed == 0
    assert result.files_skipped == 0
    assert result.files_failed == 0


@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_all_files_up_to_date(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    tmp_path: Path,
) -> None:
    """All files up to date -> all skipped."""
    # Create real files so read_text works
    f1 = tmp_path / "a.md"
    f2 = tmp_path / "b.md"
    f1.write_text("content a", encoding="utf-8")
    f2.write_text("content b", encoding="utf-8")

    mock_discover.return_value = [f1, f2]
    mock_needs_reindex.return_value = False

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_skip")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result.files_indexed == 0
    assert result.files_skipped == 2
    assert result.files_failed == 0


@patch("knowledge_rag.indexer.replace_file_chunks")
@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_new_files(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    mock_replace: MagicMock,
    tmp_path: Path,
) -> None:
    """Two new files -> ingest, embed, store called correctly."""
    f1 = tmp_path / "doc1.md"
    f2 = tmp_path / "doc2.md"
    f1.write_text("doc one content", encoding="utf-8")
    f2.write_text("doc two content", encoding="utf-8")

    mock_discover.return_value = [f1, f2]
    mock_needs_reindex.return_value = True

    chunks1 = [_make_chunk("chunk1a", "doc1.md", 0, 2), _make_chunk("chunk1b", "doc1.md", 1, 2)]
    chunks2 = [_make_chunk("chunk2a", "doc2.md", 0, 1)]
    mock_ingest.side_effect = [chunks1, chunks2]
    mock_embed_batch.return_value = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_new")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result.files_indexed == 2
    assert result.chunks_indexed == 3
    assert mock_ingest.call_count == 2


@patch("knowledge_rag.indexer.replace_file_chunks")
@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_skips_unchanged(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    mock_replace: MagicMock,
    tmp_path: Path,
) -> None:
    """3 files, 1 needs reindex, 2 skipped."""
    f1 = tmp_path / "changed.md"
    f2 = tmp_path / "same1.md"
    f3 = tmp_path / "same2.md"
    f1.write_text("changed", encoding="utf-8")
    f2.write_text("same", encoding="utf-8")
    f3.write_text("same", encoding="utf-8")

    mock_discover.return_value = [f1, f2, f3]
    mock_needs_reindex.side_effect = [True, False, False]

    chunks = [_make_chunk("new chunk", "changed.md", 0, 1)]
    mock_ingest.return_value = chunks
    mock_embed_batch.return_value = [[0.1] * 768]

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_mixed")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result.files_indexed == 1
    assert result.files_skipped == 2
    assert result.files_failed == 0


@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_handles_permission_error(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    tmp_path: Path,
) -> None:
    """PermissionError on file read -> counted as failed, continues."""
    good_file = tmp_path / "good.md"
    bad_file = tmp_path / "bad.md"
    good_file.write_text("good content", encoding="utf-8")
    bad_file.write_text("bad content", encoding="utf-8")
    # Make bad_file unreadable by mocking Path.read_text
    mock_discover.return_value = [bad_file, good_file]

    # First call (bad_file) raises, second call (good_file) returns False (skip)
    original_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        if self == bad_file:
            raise PermissionError("Access denied")
        return original_read_text(self, *args, **kwargs)

    mock_needs_reindex.return_value = False

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_perm")
    config = RagConfig()
    bm25 = BM25Index()

    with patch.object(Path, "read_text", mock_read_text):
        result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result.files_failed == 1
    assert result.files_skipped == 1


@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_embedding_failure_propagates(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    tmp_path: Path,
) -> None:
    """Embedding RuntimeError propagates (abort semantics)."""
    f1 = tmp_path / "doc.md"
    f1.write_text("content", encoding="utf-8")

    mock_discover.return_value = [f1]
    mock_needs_reindex.return_value = True
    mock_ingest.return_value = [_make_chunk("text", "doc.md", 0, 1)]
    mock_embed_batch.side_effect = RuntimeError("Batch 1 failed after 3 retries")

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_fail")
    config = RagConfig()
    bm25 = BM25Index()

    with pytest.raises(RuntimeError, match="(?i)batch"):
        index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)


@patch("knowledge_rag.indexer.replace_file_chunks")
@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_bm25_updated(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    mock_replace: MagicMock,
    tmp_path: Path,
) -> None:
    """BM25 index is updated after indexing."""
    f1 = tmp_path / "doc.md"
    f1.write_text("content", encoding="utf-8")

    mock_discover.return_value = [f1]
    mock_needs_reindex.return_value = True
    mock_ingest.return_value = [
        _make_chunk("hello world", "doc.md", 0, 2),
        _make_chunk("foo bar", "doc.md", 1, 2),
    ]
    mock_embed_batch.return_value = [[0.1] * 768, [0.2] * 768]

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_bm25")
    config = RagConfig()
    bm25 = BM25Index()

    assert len(bm25) == 0
    index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
    assert len(bm25) == 2


@patch("knowledge_rag.indexer.replace_file_chunks")
@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_result_statistics(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    mock_replace: MagicMock,
    tmp_path: Path,
) -> None:
    """IndexingResult fields match expected counts."""
    f1 = tmp_path / "new.md"
    f2 = tmp_path / "old.md"
    f1.write_text("new", encoding="utf-8")
    f2.write_text("old", encoding="utf-8")

    mock_discover.return_value = [f1, f2]
    mock_needs_reindex.side_effect = [True, False]
    mock_ingest.return_value = [
        _make_chunk("c1", "new.md", 0, 3),
        _make_chunk("c2", "new.md", 1, 3),
        _make_chunk("c3", "new.md", 2, 3),
    ]
    mock_embed_batch.return_value = [[0.1] * 768] * 3

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_stats")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result == IndexingResult(
        files_indexed=1,
        chunks_indexed=3,
        files_skipped=1,
        files_failed=0,
    )


# ---------------------------------------------------------------------------
# Integration test with real ChromaDB
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Partial reindex (target_files)
# ---------------------------------------------------------------------------


@patch("knowledge_rag.indexer.replace_file_chunks")
@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
def test_index_partial_reindex_uses_target_files(
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    mock_replace: MagicMock,
    tmp_path: Path,
) -> None:
    """When target_files is set, only those files are processed."""
    f1 = tmp_path / "target.md"
    f2 = tmp_path / "other.md"
    f1.write_text("target content", encoding="utf-8")
    f2.write_text("other content", encoding="utf-8")

    mock_ingest.return_value = [_make_chunk("target chunk", "target.md", 0, 1)]
    mock_embed_batch.return_value = [[0.1] * 768]

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_partial")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(
        tmp_path, config, collection, bm25, _fake_embed_fn,
        target_files=[f1],
    )

    assert result.files_indexed == 1
    assert mock_ingest.call_count == 1
    # Verify only target file was passed to ingest
    call_path = mock_ingest.call_args[0][0]
    assert call_path == f1


@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_partial_skips_removed_file_cleanup(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    tmp_path: Path,
) -> None:
    """Partial reindex skips removed-file cleanup."""
    f1 = tmp_path / "existing.md"
    f1.write_text("content", encoding="utf-8")

    mock_needs_reindex.return_value = False

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_partial_skip")
    # Pre-add a chunk for a "removed" file
    collection.add(
        ids=["removed.md::chunk_0"],
        documents=["old content"],
        metadatas=[{"source_file": "removed.md", "chunk_index": 0}],
        embeddings=[[0.1] * 768],
    )
    assert collection.count() == 1

    config = RagConfig()
    bm25 = BM25Index()

    # Partial reindex with just f1 — should NOT clean up removed.md
    result = index_documents(
        tmp_path, config, collection, bm25, _fake_embed_fn,
        target_files=[f1],
    )

    # The "removed" file's chunk should still be there
    assert collection.count() == 1
    assert result.files_skipped == 1


# ---------------------------------------------------------------------------
# Force reindex
# ---------------------------------------------------------------------------


@patch("knowledge_rag.indexer.replace_file_chunks")
@patch("knowledge_rag.indexer.embed_chunks_batched")
@patch("knowledge_rag.indexer.ingest_file")
@patch("knowledge_rag.indexer.file_needs_reindex")
@patch("knowledge_rag.indexer.discover_markdown_files")
def test_index_force_reindexes_unchanged(
    mock_discover: MagicMock,
    mock_needs_reindex: MagicMock,
    mock_ingest: MagicMock,
    mock_embed_batch: MagicMock,
    mock_replace: MagicMock,
    tmp_path: Path,
) -> None:
    """force=True bypasses change detection and reindexes unchanged files."""
    f1 = tmp_path / "unchanged.md"
    f1.write_text("same content", encoding="utf-8")

    mock_discover.return_value = [f1]
    mock_needs_reindex.return_value = False  # Would normally skip

    mock_ingest.return_value = [_make_chunk("chunk", "unchanged.md", 0, 1)]
    mock_embed_batch.return_value = [[0.1] * 768]

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_force")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(
        tmp_path, config, collection, bm25, _fake_embed_fn,
        force=True,
    )

    # file_needs_reindex returned False, but force=True overrides
    assert result.files_indexed == 1
    assert result.files_skipped == 0


# ---------------------------------------------------------------------------
# _read_file_with_retry tests (PLT-05 Windows file locking)
# ---------------------------------------------------------------------------


def test_read_file_with_retry_success(tmp_path: Path) -> None:
    """Normal read succeeds on first attempt without retries."""
    f = tmp_path / "ok.md"
    f.write_text("hello world", encoding="utf-8")

    result = _read_file_with_retry(f)

    assert result == "hello world"


@patch("knowledge_rag.indexer.time.sleep")
def test_read_file_with_retry_permission_error_then_success(
    mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """PermissionError on first attempt, success on second -> returns content."""
    f = tmp_path / "locked.md"
    f.write_text("recovered content", encoding="utf-8")

    original_read_text = Path.read_text
    call_count = 0

    def mock_read_text(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise PermissionError("File locked by antivirus")
        return original_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", mock_read_text):
        result = _read_file_with_retry(f, max_retries=3, base_delay=0.5)

    assert result == "recovered content"
    assert call_count == 2
    # Verify sleep was called with correct backoff delay (0.5 * 2^0 = 0.5)
    mock_sleep.assert_called_once_with(0.5)


@patch("knowledge_rag.indexer.time.sleep")
def test_read_file_with_retry_all_fail(
    mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """All retries exhausted -> raises PermissionError."""
    f = tmp_path / "always_locked.md"
    f.write_text("content", encoding="utf-8")

    with patch.object(Path, "read_text", side_effect=PermissionError("Locked")):
        with pytest.raises(PermissionError, match="Locked"):
            _read_file_with_retry(f, max_retries=2, base_delay=0.1)

    # Should have slept twice (attempt 0 and 1 retry, then attempt 2 fails)
    assert mock_sleep.call_count == 2


def test_read_file_with_retry_unicode_error_not_retried(tmp_path: Path) -> None:
    """UnicodeDecodeError is not retried -- raises immediately."""
    f = tmp_path / "binary.dat"
    f.write_bytes(b"\xff\xfe\x00\x01")  # Invalid UTF-8

    with pytest.raises(UnicodeDecodeError):
        _read_file_with_retry(f, max_retries=3)


# ---------------------------------------------------------------------------
# indexed_at timestamp
# ---------------------------------------------------------------------------


def test_index_stores_indexed_at(tmp_path: Path) -> None:
    """Integration: chunks stored by index_documents have indexed_at metadata."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "ts.md").write_text(
        "---\ntitle: Timestamp Test\n---\n\nContent for timestamp test.",
        encoding="utf-8",
    )

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        "documents", metadata={"hnsw:space": "cosine"}
    )
    config = RagConfig()
    config.chunk_size = 500
    config.overlap = 50
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
    assert result.files_indexed >= 1

    # Verify indexed_at is in metadata
    stored = collection.get(include=["metadatas"])
    for meta in stored["metadatas"]:
        assert "indexed_at" in meta
        assert isinstance(meta["indexed_at"], float)
        assert meta["indexed_at"] > 0


# ---------------------------------------------------------------------------
# Integration test with real ChromaDB
# ---------------------------------------------------------------------------


def test_index_stores_in_chromadb(tmp_path: Path) -> None:
    """Integration: real ChromaDB + fake embeddings -> chunks stored."""
    # Create markdown files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "readme.md").write_text(
        "---\ntitle: README\n---\n\nThis is a test document with enough content to create chunks.",
        encoding="utf-8",
    )

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        "documents", metadata={"hnsw:space": "cosine"}
    )
    config = RagConfig()
    config.chunk_size = 500  # small chunks
    config.overlap = 50
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    assert result.files_indexed >= 1
    assert result.chunks_indexed >= 1
    assert collection.count() == result.chunks_indexed
    assert len(bm25) == result.chunks_indexed


# ---------------------------------------------------------------------------
# Mode router tests (09-06)
# ---------------------------------------------------------------------------

from knowledge_rag.indexer import _discover_for_mode, _ingest_file_routed


def test_discover_for_mode_docs(tmp_path: Path) -> None:
    """content_mode='docs' discovers only .md files."""
    (tmp_path / "readme.md").write_text("# Hello")
    (tmp_path / "main.py").write_text("print('hi')")

    files = _discover_for_mode(tmp_path, "docs")
    extensions = {f.suffix for f in files}
    assert ".md" in extensions
    assert ".py" not in extensions


def test_discover_for_mode_code(tmp_path: Path) -> None:
    """content_mode='code' discovers only code files."""
    (tmp_path / "readme.md").write_text("# Hello")
    (tmp_path / "main.py").write_text("print('hi')")

    files = _discover_for_mode(tmp_path, "code")
    extensions = {f.suffix for f in files}
    assert ".py" in extensions
    assert ".md" not in extensions


def test_discover_for_mode_both(tmp_path: Path) -> None:
    """content_mode='both' discovers .md + code files."""
    (tmp_path / "readme.md").write_text("# Hello")
    (tmp_path / "main.py").write_text("print('hi')")

    files = _discover_for_mode(tmp_path, "both")
    extensions = {f.suffix for f in files}
    assert ".md" in extensions
    assert ".py" in extensions


def test_discover_for_mode_unknown_defaults_docs(tmp_path: Path) -> None:
    """Unknown content_mode falls back to 'docs'."""
    (tmp_path / "readme.md").write_text("# Hello")
    (tmp_path / "main.py").write_text("print('hi')")

    files = _discover_for_mode(tmp_path, "invalid")
    extensions = {f.suffix for f in files}
    assert ".md" in extensions
    assert ".py" not in extensions


def test_ingest_routed_markdown(tmp_path: Path) -> None:
    """.md file routes to ingest_file (markdown pipeline)."""
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Title\n\nContent here.")

    chunks = _ingest_file_routed(md_file, tmp_path, 1000, 200)
    assert len(chunks) >= 1
    assert chunks[0].source_file == "doc.md"


def test_ingest_routed_python(tmp_path: Path) -> None:
    """.py file routes to ingest_code_file."""
    py_file = tmp_path / "main.py"
    py_file.write_text("def hello():\n    return 'world'\n")

    chunks = _ingest_file_routed(py_file, tmp_path, 1000, 200)
    assert len(chunks) >= 1
    assert chunks[0].language == "python"


def test_ingest_routed_unsupported(tmp_path: Path) -> None:
    """.txt file returns empty list (unsupported)."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("some notes")

    chunks = _ingest_file_routed(txt_file, tmp_path, 1000, 200)
    assert chunks == []


def test_index_docs_mode_only_markdown(tmp_path: Path) -> None:
    """content_mode='docs' only indexes .md files end-to-end."""
    (tmp_path / "readme.md").write_text("# Hello\n\nWorld.")
    (tmp_path / "main.py").write_text("def foo():\n    pass\n")

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_docs_mode")
    config = RagConfig()
    config.content_mode = "docs"
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
    assert result.files_indexed >= 1

    # Check that only .md content is indexed
    stored = collection.get(include=["metadatas"])
    source_files = {m["source_file"] for m in stored["metadatas"]}
    assert all(sf.endswith(".md") for sf in source_files)


def test_index_code_mode_only_code(tmp_path: Path) -> None:
    """content_mode='code' only indexes code files end-to-end."""
    (tmp_path / "readme.md").write_text("# Hello\n\nWorld.")
    (tmp_path / "main.py").write_text("def foo():\n    return 42\n")

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_code_mode")
    config = RagConfig()
    config.content_mode = "code"
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
    assert result.files_indexed >= 1

    stored = collection.get(include=["metadatas"])
    source_files = {m["source_file"] for m in stored["metadatas"]}
    assert all(sf.endswith(".py") for sf in source_files)


def test_index_both_mode_all_files(tmp_path: Path) -> None:
    """content_mode='both' indexes both .md and code files."""
    (tmp_path / "readme.md").write_text("# Hello\n\nWorld.")
    (tmp_path / "main.py").write_text("def foo():\n    return 42\n")

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_both_mode")
    config = RagConfig()
    config.content_mode = "both"
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

    stored = collection.get(include=["metadatas"])
    source_files = {m["source_file"] for m in stored["metadatas"]}
    extensions = {sf.split(".")[-1] for sf in source_files}
    assert "md" in extensions
    assert "py" in extensions


def test_index_stores_heading_hierarchy(tmp_path: Path) -> None:
    """Integration: heading_hierarchy flows through indexer to ChromaDB metadata."""
    (tmp_path / "structured.md").write_text(
        "# Config\n\nIntro.\n\n## Database\n\nDB details.\n\n## Cache\n\nCache details."
    )

    from knowledge_rag.config import RagConfig
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_hierarchy")
    config = RagConfig()
    bm25 = BM25Index()

    result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
    assert result.files_indexed >= 1

    stored = collection.get(include=["metadatas"])
    hierarchies = [m.get("heading_hierarchy", "") for m in stored["metadatas"]]
    # At least one chunk should have a heading hierarchy
    assert any(h for h in hierarchies)
