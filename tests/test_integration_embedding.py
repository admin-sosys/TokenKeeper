"""Integration tests for Phase 4: Embedding & Storage.

All tests use real ChromaDB (at tmp_path) with fake 768-dim embeddings.
No Ollama needed. Tests validate the complete pipeline: indexing, persistence,
BM25 rebuild, per-project isolation, and change detection.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from knowledge_rag.bm25_index import BM25Index
from knowledge_rag.config import RagConfig
from knowledge_rag.indexer import index_documents
from knowledge_rag.storage import get_or_create_collection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Produce deterministic, unique 768-dim vectors per text."""
    result: list[list[float]] = []
    for text in texts:
        seed = hash(text) % 10000
        vec = [(seed + i) / 10000.0 for i in range(768)]
        result.append(vec)
    return result


def _create_md(path: Path, title: str, body: str, tags: list[str] | None = None) -> None:
    """Create a markdown file with optional YAML frontmatter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_lines = ["---", f"title: {title}"]
    if tags:
        fm_lines.append(f"tags: [{', '.join(tags)}]")
    fm_lines.append("---\n")
    path.write_text("\n".join(fm_lines) + body, encoding="utf-8")


def _setup_project(root: Path) -> tuple[chromadb.PersistentClient, chromadb.Collection, RagConfig, BM25Index]:
    """Create ChromaDB client, collection, config, and BM25 for a project root."""
    chroma_path = root / ".rag" / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = get_or_create_collection(client)
    config = RagConfig()
    config.chunk_size = 200
    config.overlap = 50
    bm25 = BM25Index()
    return client, collection, config, bm25


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end indexing pipeline tests."""

    def test_index_documents_end_to_end(self, tmp_path: Path) -> None:
        """Index 3 markdown files, verify chunks stored correctly."""
        _create_md(tmp_path / "docs" / "readme.md", "README", "This is a readme with enough content to create at least one chunk for testing.")
        _create_md(tmp_path / "docs" / "guide.md", "Guide", "A longer guide document. " * 20, tags=["alpha", "beta"])
        _create_md(tmp_path / "notes.md", "Notes", "Some notes about the project.")

        _client, collection, config, bm25 = _setup_project(tmp_path)

        result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

        assert result.files_indexed == 3
        assert result.chunks_indexed > 0
        assert collection.count() == result.chunks_indexed
        assert len(bm25) == result.chunks_indexed

    def test_chunk_ids_format(self, tmp_path: Path) -> None:
        """Stored chunk IDs follow 'source_file::chunk_N' format."""
        _create_md(tmp_path / "test.md", "Test", "Test content for ID format.")

        _client, collection, config, bm25 = _setup_project(tmp_path)
        index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

        stored = collection.get()
        for chunk_id in stored["ids"]:
            assert "::" in chunk_id, f"ID {chunk_id} missing '::' separator"
            assert "::chunk_" in chunk_id, f"ID {chunk_id} missing '::chunk_' pattern"


class TestPersistence:
    """ChromaDB persistence across simulated restarts."""

    def test_persistence_survives_restart(self, tmp_path: Path) -> None:
        """Data persists after closing and reopening ChromaDB client."""
        _create_md(tmp_path / "doc.md", "Doc", "Persistent document content here.")

        client, collection, config, bm25 = _setup_project(tmp_path)
        result = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        original_count = collection.count()
        assert original_count > 0

        # Simulate restart: delete client reference
        chroma_path = str(tmp_path / ".rag" / "chroma")
        del client
        del collection

        # Reopen
        client2 = chromadb.PersistentClient(path=chroma_path)
        collection2 = get_or_create_collection(client2)
        assert collection2.count() == original_count


class TestBM25Rebuild:
    """BM25 index rebuild from ChromaDB metadata."""

    def test_bm25_rebuild_from_metadata(self, tmp_path: Path) -> None:
        """Fresh BM25 index can be rebuilt from stored ChromaDB metadata."""
        # Need multiple docs for BM25 IDF to produce nonzero scores
        _create_md(tmp_path / "python.md", "Python", "python programming language for data science and machine learning.")
        _create_md(tmp_path / "java.md", "Java", "java enterprise software development and web applications.")
        _create_md(tmp_path / "rust.md", "Rust", "rust systems programming memory safety and performance.")

        _client, collection, config, bm25 = _setup_project(tmp_path)
        index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        original_len = len(bm25)
        assert original_len > 0

        # Create fresh BM25 index (simulating startup)
        fresh_bm25 = BM25Index()
        assert len(fresh_bm25) == 0

        # Rebuild from ChromaDB metadata
        stored = collection.get(include=["metadatas"])
        doc_ids = stored["ids"]
        tokenized_corpus = []
        for meta in stored["metadatas"]:
            bm25_tokens_str = meta.get("bm25_tokens", "")
            tokens = bm25_tokens_str.split() if bm25_tokens_str else ["_empty_"]
            tokenized_corpus.append(tokens)

        fresh_bm25.rebuild(doc_ids, tokenized_corpus)
        assert len(fresh_bm25) == original_len

        # Search should return results (python appears in only some docs)
        results = fresh_bm25.search(["python"])
        assert len(results) > 0


class TestIsolation:
    """Per-project isolation."""

    def test_per_project_isolation(self, tmp_path: Path) -> None:
        """Two projects produce independent vector stores."""
        # Project A
        proj_a = tmp_path / "project_a"
        _create_md(proj_a / "alpha.md", "Alpha", "Alpha project content about cats.")
        _client_a, coll_a, config_a, bm25_a = _setup_project(proj_a)
        result_a = index_documents(proj_a, config_a, coll_a, bm25_a, _fake_embed_fn)

        # Project B
        proj_b = tmp_path / "project_b"
        _create_md(proj_b / "beta.md", "Beta", "Beta project content about dogs.")
        _create_md(proj_b / "gamma.md", "Gamma", "More beta project content.")
        _client_b, coll_b, config_b, bm25_b = _setup_project(proj_b)
        result_b = index_documents(proj_b, config_b, coll_b, bm25_b, _fake_embed_fn)

        # Verify isolation
        assert result_a.files_indexed == 1
        assert result_b.files_indexed == 2
        assert coll_a.count() != coll_b.count() or result_a.chunks_indexed != result_b.chunks_indexed

        # Check no cross-contamination
        stored_a = coll_a.get(include=["metadatas"])
        for meta in stored_a["metadatas"]:
            assert "beta" not in meta["source_file"].lower()
            assert "gamma" not in meta["source_file"].lower()


class TestChangeDetection:
    """Content hash change detection."""

    def test_skip_unchanged_files(self, tmp_path: Path) -> None:
        """Re-indexing unchanged files skips them (no re-embedding)."""
        _create_md(tmp_path / "stable.md", "Stable", "Content that does not change.")

        _client, collection, config, bm25 = _setup_project(tmp_path)

        # First run: index everything
        result1 = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        assert result1.files_indexed == 1

        # Second run: everything unchanged
        call_count = 0
        original_fn = _fake_embed_fn

        def counting_embed(texts):
            nonlocal call_count
            call_count += 1
            return original_fn(texts)

        result2 = index_documents(tmp_path, config, collection, bm25, counting_embed)
        assert result2.files_indexed == 0
        assert result2.files_skipped == 1
        assert call_count == 0  # embed_fn never called

    def test_reindex_changed_file(self, tmp_path: Path) -> None:
        """Changing a file triggers re-indexing of that file only."""
        _create_md(tmp_path / "doc1.md", "Doc1", "Original content version one.")
        _create_md(tmp_path / "doc2.md", "Doc2", "Stable document content.")

        _client, collection, config, bm25 = _setup_project(tmp_path)
        result1 = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        assert result1.files_indexed == 2

        # Modify doc1
        _create_md(tmp_path / "doc1.md", "Doc1", "UPDATED content version two with changes.")

        result2 = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        assert result2.files_indexed == 1
        assert result2.files_skipped == 1

    def test_removed_file_cleanup(self, tmp_path: Path) -> None:
        """Deleting a file removes its chunks from ChromaDB."""
        _create_md(tmp_path / "keep.md", "Keep", "This file stays.")
        _create_md(tmp_path / "delete.md", "Delete", "This file will be removed.")

        _client, collection, config, bm25 = _setup_project(tmp_path)
        result1 = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        count_before = collection.count()
        assert result1.files_indexed == 2

        # Delete the file
        (tmp_path / "delete.md").unlink()

        result2 = index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)
        count_after = collection.count()
        assert count_after < count_before


class TestMetadata:
    """Metadata format and correctness."""

    def test_metadata_stored_correctly(self, tmp_path: Path) -> None:
        """Metadata has correct keys and types (no None, no lists)."""
        _create_md(tmp_path / "meta.md", "Test Doc", "Content for metadata test.", tags=["alpha", "beta"])

        _client, collection, config, bm25 = _setup_project(tmp_path)
        index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

        stored = collection.get(include=["metadatas"])
        assert len(stored["metadatas"]) > 0

        for meta in stored["metadatas"]:
            assert "source_file" in meta
            assert "chunk_index" in meta
            assert "content_hash" in meta
            assert "bm25_tokens" in meta

            # All values must be str, int, float, or bool
            for key, val in meta.items():
                assert isinstance(val, (str, int, float, bool)), (
                    f"Metadata key '{key}' has invalid type {type(val)}: {val}"
                )

    def test_embedding_dimensions(self, tmp_path: Path) -> None:
        """Stored embeddings have exactly 768 dimensions."""
        _create_md(tmp_path / "dims.md", "Dims", "Content for embedding dimension test.")

        _client, collection, config, bm25 = _setup_project(tmp_path)
        index_documents(tmp_path, config, collection, bm25, _fake_embed_fn)

        stored = collection.get(include=["embeddings"])
        assert len(stored["embeddings"]) > 0

        for embedding in stored["embeddings"]:
            assert len(embedding) == 768
