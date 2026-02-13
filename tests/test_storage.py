"""Comprehensive tests for knowledge_rag.storage module.

Covers: ChromaDB client creation, collection management, chunk ID generation,
file chunk replacement with empty-guard, content hashing, stored hash retrieval,
and file reindex detection.

All tests use REAL ChromaDB at tmp_path (not mocked).
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from knowledge_rag.ingestion import DocumentChunk
from knowledge_rag.storage import (
    compute_file_hash,
    create_chroma_client,
    file_needs_reindex,
    get_chunk_ids_for_file,
    get_or_create_collection,
    get_source_file_from_id,
    get_stored_hash,
    make_chunk_id,
    replace_file_chunks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "test content",
    source_file: str = "test.md",
    chunk_index: int = 0,
    title: str = "",
    tags: list[str] | None = None,
) -> DocumentChunk:
    """Create a DocumentChunk with minimal required fields."""
    return DocumentChunk(
        content=content,
        source_file=source_file,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(content),
        total_chunks=1,
        frontmatter={"title": title, "tags": tags or []},
    )


def _fake_embedding(dim: int = 384) -> list[float]:
    """Return a fixed-dimension embedding vector."""
    return [0.1] * dim


# ---------------------------------------------------------------------------
# create_chroma_client
# ---------------------------------------------------------------------------


class TestCreateChromaClient:
    def test_creates_directory(self, tmp_path: Path) -> None:
        chroma_dir = tmp_path / "chroma_data"
        assert not chroma_dir.exists()
        create_chroma_client(chroma_dir)
        assert chroma_dir.exists()

    def test_returns_persistent_client(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        assert isinstance(client, chromadb.ClientAPI)


# ---------------------------------------------------------------------------
# get_or_create_collection
# ---------------------------------------------------------------------------


class TestGetOrCreateCollection:
    def test_default_name(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        assert coll.name == "documents"

    def test_custom_name(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client, name="custom")
        assert coll.name == "custom"

    def test_cosine_space(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        assert coll.metadata is not None
        assert coll.metadata.get("hnsw:space") == "cosine"


# ---------------------------------------------------------------------------
# make_chunk_id / get_source_file_from_id
# ---------------------------------------------------------------------------


class TestChunkIds:
    def test_make_chunk_id_posix_path(self) -> None:
        result = make_chunk_id("docs/readme.md", 0)
        assert result == "docs/readme.md::chunk_0"

    def test_make_chunk_id_windows_backslash(self) -> None:
        result = make_chunk_id("docs\\readme.md", 0)
        assert result == "docs/readme.md::chunk_0"

    def test_get_source_file_from_id(self) -> None:
        result = get_source_file_from_id("docs/readme.md::chunk_0")
        assert result == "docs/readme.md"


# ---------------------------------------------------------------------------
# get_chunk_ids_for_file
# ---------------------------------------------------------------------------


class TestGetChunkIdsForFile:
    def test_empty_collection(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        ids = get_chunk_ids_for_file(coll, "nonexistent.md")
        assert ids == []

    def test_with_chunks(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        # Manually add two chunks for one file
        coll.add(
            ids=["test.md::chunk_0", "test.md::chunk_1"],
            documents=["content 0", "content 1"],
            metadatas=[
                {"source_file": "test.md", "chunk_index": 0},
                {"source_file": "test.md", "chunk_index": 1},
            ],
            embeddings=[_fake_embedding(), _fake_embedding()],
        )
        ids = get_chunk_ids_for_file(coll, "test.md")
        assert sorted(ids) == ["test.md::chunk_0", "test.md::chunk_1"]


# ---------------------------------------------------------------------------
# replace_file_chunks
# ---------------------------------------------------------------------------


class TestReplaceFileChunks:
    def test_adds_new(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [_make_chunk(content="hello world", chunk_index=0)]
        count = replace_file_chunks(coll, "test.md", chunks, [_fake_embedding()])
        assert count == 1
        assert coll.count() == 1

    def test_replaces_existing(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        # Add initial version
        chunks_v1 = [_make_chunk(content="version 1", chunk_index=0)]
        replace_file_chunks(coll, "test.md", chunks_v1, [_fake_embedding()])
        assert coll.count() == 1
        # Replace with new version (two chunks)
        chunks_v2 = [
            _make_chunk(content="version 2a", chunk_index=0),
            _make_chunk(content="version 2b", chunk_index=1),
        ]
        count = replace_file_chunks(
            coll, "test.md", chunks_v2, [_fake_embedding(), _fake_embedding()]
        )
        assert count == 2
        assert coll.count() == 2

    def test_empty_guard(self, tmp_path: Path) -> None:
        """Ensure empty stale_ids doesn't delete all documents (ChromaDB bug)."""
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        # Add chunk for file A
        chunks_a = [_make_chunk(content="file A", source_file="a.md", chunk_index=0)]
        replace_file_chunks(coll, "a.md", chunks_a, [_fake_embedding()])
        assert coll.count() == 1
        # Add chunk for file B (file B has no stale_ids -- empty guard matters)
        chunks_b = [_make_chunk(content="file B", source_file="b.md", chunk_index=0)]
        replace_file_chunks(coll, "b.md", chunks_b, [_fake_embedding()])
        # Both files should exist
        assert coll.count() == 2

    def test_metadata(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [
            _make_chunk(
                content="metadata test",
                source_file="docs/guide.md",
                chunk_index=0,
                title="Guide",
                tags=["tutorial", "intro"],
            )
        ]
        replace_file_chunks(coll, "docs/guide.md", chunks, [_fake_embedding()])
        result = coll.get(ids=["docs/guide.md::chunk_0"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["source_file"] == "docs/guide.md"
        assert meta["chunk_index"] == 0
        assert meta["title"] == "Guide"
        assert meta["tags"] == "tutorial,intro"
        assert "content_hash" in meta

    def test_indexed_at_stored(self, tmp_path: Path) -> None:
        """indexed_at timestamp is included in metadata when provided."""
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [_make_chunk(content="timestamp test", chunk_index=0)]
        replace_file_chunks(
            coll, "test.md", chunks, [_fake_embedding()], indexed_at=1700000000.0
        )
        result = coll.get(ids=["test.md::chunk_0"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["indexed_at"] == 1700000000.0

    def test_indexed_at_omitted_when_none(self, tmp_path: Path) -> None:
        """indexed_at is NOT in metadata when not provided."""
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [_make_chunk(content="no timestamp", chunk_index=0)]
        replace_file_chunks(coll, "test.md", chunks, [_fake_embedding()])
        result = coll.get(ids=["test.md::chunk_0"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert "indexed_at" not in meta


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_deterministic(self) -> None:
        h1 = compute_file_hash("hello")
        h2 = compute_file_hash("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_different_content(self) -> None:
        h1 = compute_file_hash("hello")
        h2 = compute_file_hash("world")
        assert h1 != h2


# ---------------------------------------------------------------------------
# get_stored_hash / file_needs_reindex
# ---------------------------------------------------------------------------


class TestStoredHashAndReindex:
    def test_get_stored_hash_not_found(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        assert get_stored_hash(coll, "nonexistent.md") is None

    def test_get_stored_hash_found(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [_make_chunk(content="hash test", chunk_index=0)]
        replace_file_chunks(coll, "test.md", chunks, [_fake_embedding()])
        stored = get_stored_hash(coll, "test.md")
        assert stored == compute_file_hash("hash test")

    def test_file_needs_reindex_new_file(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        assert file_needs_reindex(coll, "new.md", "some content") is True

    def test_file_needs_reindex_unchanged(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [_make_chunk(content="stable content", chunk_index=0)]
        replace_file_chunks(coll, "test.md", chunks, [_fake_embedding()])
        assert file_needs_reindex(coll, "test.md", "stable content") is False

    def test_file_needs_reindex_changed(self, tmp_path: Path) -> None:
        client = create_chroma_client(tmp_path / "chroma")
        coll = get_or_create_collection(client)
        chunks = [_make_chunk(content="old content", chunk_index=0)]
        replace_file_chunks(coll, "test.md", chunks, [_fake_embedding()])
        assert file_needs_reindex(coll, "test.md", "new content") is True
