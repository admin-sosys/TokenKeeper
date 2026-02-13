"""Unit tests for the BM25 index module.

Tests cover:
  - Empty index search returns empty
  - Rebuild with documents sets correct length
  - Rebuild empty clears index
  - Search returns relevant results ranked correctly
  - Search respects top_k limit
  - Search only returns positive scores
  - Add documents extends index
  - Remove documents shrinks index
  - Remove nonexistent ID is a no-op
  - tokenize_for_bm25 basic, with metadata, empty metadata
  - tokens_to_metadata_string normal and empty
"""

from __future__ import annotations

import pytest

from knowledge_rag.bm25_index import (
    BM25Index,
    tokenize_for_bm25,
    tokens_to_metadata_string,
)


# ---------------------------------------------------------------------------
# tokenize_for_bm25 tests
# ---------------------------------------------------------------------------


def test_tokenize_for_bm25_basic() -> None:
    """Basic tokenization splits content, includes source file + path segments."""
    tokens = tokenize_for_bm25("hello world", "docs/readme.md")
    assert tokens == ["hello", "world", "docs/readme.md", "docs", "readme", "md"]


def test_tokenize_for_bm25_with_metadata() -> None:
    """Tokenization includes title and tags when provided."""
    tokens = tokenize_for_bm25(
        "hello", "f.md", title="My Title", tags="tag1,tag2"
    )
    assert tokens == ["hello", "f.md", "md", "my", "title", "tag1,tag2"]


def test_tokenize_for_bm25_empty_metadata() -> None:
    """Empty title and tags are excluded from tokens."""
    tokens = tokenize_for_bm25("content here", "file.md", title="", tags="")
    assert tokens == ["content", "here", "file.md", "file", "md"]


# ---------------------------------------------------------------------------
# tokens_to_metadata_string tests
# ---------------------------------------------------------------------------


def test_tokens_to_metadata_string() -> None:
    """Tokens are joined with spaces."""
    result = tokens_to_metadata_string(["hello", "world", "docs/readme.md"])
    assert result == "hello world docs/readme.md"


def test_tokens_to_metadata_string_empty() -> None:
    """Empty token list produces empty string."""
    result = tokens_to_metadata_string([])
    assert result == ""


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    """Tests for the BM25Index class."""

    def test_empty_index_search_returns_empty(self) -> None:
        """Search on empty index returns empty list."""
        idx = BM25Index()
        results = idx.search(["python"])
        assert results == []

    def test_rebuild_with_documents(self) -> None:
        """Rebuild with 3 documents sets length to 3."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["doc1", "doc2", "doc3"],
            tokenized_texts=[
                ["python", "programming"],
                ["java", "programming"],
                ["rust", "systems"],
            ],
        )
        assert len(idx) == 3

    def test_rebuild_empty_clears_index(self) -> None:
        """Rebuild with empty lists clears the index."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["doc1"],
            tokenized_texts=[["hello", "world"]],
        )
        assert len(idx) == 1

        idx.rebuild(doc_ids=[], tokenized_texts=[])
        assert len(idx) == 0
        assert idx.search(["hello"]) == []

    def test_search_returns_relevant_results(self) -> None:
        """Search for 'python' ranks python doc higher than java doc."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["python_doc", "java_doc", "rust_doc"],
            tokenized_texts=[
                ["python", "programming", "language", "scripting"],
                ["java", "coffee", "beans", "enterprise"],
                ["rust", "systems", "memory", "safety"],
            ],
        )
        results = idx.search(["python"])
        assert len(results) > 0
        # Python doc should be first (highest score)
        assert results[0][0] == "python_doc"
        assert results[0][1] > 0

    def test_search_respects_top_k(self) -> None:
        """Search returns at most top_k results."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["d1", "d2", "d3", "d4", "d5"],
            tokenized_texts=[
                ["common", "word", "extra1"],
                ["common", "word", "extra2"],
                ["common", "word", "extra3"],
                ["common", "word", "extra4"],
                ["common", "word", "extra5"],
            ],
        )
        results = idx.search(["common"], top_k=2)
        assert len(results) <= 2

    def test_search_only_positive_scores(self) -> None:
        """Search excludes results with zero or negative scores."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["match", "nomatch"],
            tokenized_texts=[
                ["python", "code"],
                ["java", "coffee"],
            ],
        )
        results = idx.search(["python"])
        for doc_id, score in results:
            assert score > 0

    def test_add_documents_extends_index(self) -> None:
        """Adding documents increases the index size."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["doc1"],
            tokenized_texts=[["hello", "world"]],
        )
        assert len(idx) == 1

        idx.add_documents(
            doc_ids=["doc2", "doc3"],
            tokenized_texts=[["foo", "bar"], ["baz", "qux"]],
        )
        assert len(idx) == 3

    def test_remove_documents_shrinks_index(self) -> None:
        """Removing documents decreases the index size."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["doc1", "doc2", "doc3"],
            tokenized_texts=[
                ["hello", "world"],
                ["foo", "bar"],
                ["baz", "qux"],
            ],
        )
        assert len(idx) == 3

        idx.remove_documents({"doc2"})
        assert len(idx) == 2

    def test_remove_nonexistent_id_is_noop(self) -> None:
        """Removing an ID not in the index does nothing."""
        idx = BM25Index()
        idx.rebuild(
            doc_ids=["doc1"],
            tokenized_texts=[["hello", "world"]],
        )
        assert len(idx) == 1

        idx.remove_documents({"nonexistent"})
        assert len(idx) == 1
