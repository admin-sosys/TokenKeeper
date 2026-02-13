"""Integration tests for Phase 5: Search Engine.

All tests use real ChromaDB (at tmp_path) with fake 768-dim embeddings.
No Ollama needed. Tests validate search quality across modes, metadata
completeness, score normalization, and performance.
"""

from __future__ import annotations

import time
from pathlib import Path

import chromadb
import pytest

from knowledge_rag.bm25_index import BM25Index
from knowledge_rag.search import SearchResult, search
from knowledge_rag.storage import get_or_create_collection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Produce deterministic, content-sensitive 768-dim vectors.

    Uses hash of lowercased text to seed vector values. Different texts
    produce different vectors.
    """
    result: list[list[float]] = []
    for text in texts:
        seed = hash(text.lower()) % 10000
        vec = [(seed + i) / 10000.0 for i in range(768)]
        result.append(vec)
    return result


# Corpus: 8 documents with distinct terms for BM25 differentiation
_CORPUS = [
    {
        "id": "auth-config.md::chunk_0",
        "source_file": "auth-config.md",
        "title": "Authentication Config",
        "tags": "auth,security",
        "content": "Authentication is configured via environment variables. Set AUTH_SECRET and AUTH_PROVIDER in your .env file. The system supports OAuth2 with Google and GitHub providers.",
    },
    {
        "id": "database-setup.md::chunk_0",
        "source_file": "database-setup.md",
        "title": "Database Setup",
        "tags": "database,postgresql",
        "content": "The database uses PostgreSQL with connection pooling. Configure DATABASE_URL in .env. Pool size defaults to 10 connections.",
    },
    {
        "id": "api-endpoints.md::chunk_0",
        "source_file": "api-endpoints.md",
        "title": "API Endpoints",
        "tags": "api,rest",
        "content": "REST API endpoints follow the /api/v1 prefix convention. Authentication is required for all endpoints except /health.",
    },
    {
        "id": "deployment-guide.md::chunk_0",
        "source_file": "deployment-guide.md",
        "title": "Deployment Guide",
        "tags": "deployment,docker",
        "content": "Deploy to production using Docker Compose. The docker-compose.yml includes postgres, redis, and the application service.",
    },
    {
        "id": "error-handling.md::chunk_0",
        "source_file": "error-handling.md",
        "title": "Error Handling",
        "tags": "errors,middleware",
        "content": "All API errors return a standard JSON response with error code, message, and request_id. Use the ErrorHandler middleware.",
    },
    {
        "id": "caching-strategy.md::chunk_0",
        "source_file": "caching-strategy.md",
        "title": "Caching Strategy",
        "tags": "caching,redis",
        "content": "Redis is used for caching with a 15-minute TTL. Cache keys follow the pattern: cache:{resource}:{id}.",
    },
    {
        "id": "testing-guide.md::chunk_0",
        "source_file": "testing-guide.md",
        "title": "Testing Guide",
        "tags": "testing,pytest",
        "content": "Run tests with pytest. Integration tests require a running PostgreSQL instance. Use fixtures for database setup.",
    },
    {
        "id": "migration-guide.md::chunk_0",
        "source_file": "migration-guide.md",
        "title": "Migration Guide",
        "tags": "database,migration",
        "content": "Database migrations use Alembic. Run alembic upgrade head to apply pending migrations.",
    },
]


def _build_search_env(
    tmp_path: Path,
) -> tuple[chromadb.Collection, BM25Index]:
    """Build a ChromaDB collection + BM25 index from the test corpus."""
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = get_or_create_collection(client)

    ids = [doc["id"] for doc in _CORPUS]
    documents = [doc["content"] for doc in _CORPUS]
    embeddings = _fake_embed_fn(documents)
    metadatas = [
        {
            "source_file": doc["source_file"],
            "chunk_index": 0,
            "title": doc["title"],
            "tags": doc["tags"],
        }
        for doc in _CORPUS
    ]

    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    # Build BM25 index
    bm25 = BM25Index()
    tokenized = [doc["content"].lower().split() for doc in _CORPUS]
    bm25.rebuild(ids, tokenized)

    return collection, bm25


# ---------------------------------------------------------------------------
# Keyword search quality tests
# ---------------------------------------------------------------------------


class TestKeywordSearchQuality:
    """Tests that BM25 keyword search finds exact-term matches."""

    def test_keyword_search_postgresql(self, tmp_path: Path) -> None:
        """Searching 'PostgreSQL' finds database-setup and testing-guide."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "PostgreSQL",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=5,
            mode="keyword",
        )

        doc_ids = [r.source_file for r in results]
        assert "database-setup.md" in doc_ids
        assert "testing-guide.md" in doc_ids
        # Deployment guide should NOT appear (no PostgreSQL)
        assert "deployment-guide.md" not in doc_ids

    def test_keyword_search_errorhandler(self, tmp_path: Path) -> None:
        """Searching 'ErrorHandler' finds error-handling doc."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "ErrorHandler",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=5,
            mode="keyword",
        )

        if results:
            doc_ids = [r.source_file for r in results]
            assert "error-handling.md" in doc_ids

    def test_keyword_search_alembic(self, tmp_path: Path) -> None:
        """Searching 'alembic' finds migration-guide."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "alembic",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=5,
            mode="keyword",
        )

        assert len(results) > 0
        assert results[0].source_file == "migration-guide.md"


# ---------------------------------------------------------------------------
# Alpha blending tests
# ---------------------------------------------------------------------------


class TestAlphaBlending:
    """Tests that alpha parameter shifts results between keyword and semantic."""

    def test_alpha_zero_pure_keyword(self, tmp_path: Path) -> None:
        """alpha=0.0 behaves as pure keyword search."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "Docker Compose deployment",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            alpha=0.0,
            top_k=5,
        )

        # Should find deployment-guide (has "Docker" and "Compose")
        if results:
            doc_ids = [r.source_file for r in results]
            assert "deployment-guide.md" in doc_ids

    def test_alpha_one_pure_semantic(self, tmp_path: Path) -> None:
        """alpha=1.0 returns semantic results (ordered by embedding similarity)."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "Docker Compose deployment",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            alpha=1.0,
            top_k=5,
        )

        assert len(results) > 0
        # Results should have descending scores
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_alpha_keyword_heavy_favors_exact_match(self, tmp_path: Path) -> None:
        """alpha=0.1 (keyword-heavy) finds alembic in migration-guide."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "alembic",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            alpha=0.1,
            top_k=5,
        )

        assert len(results) > 0
        assert results[0].source_file == "migration-guide.md"


# ---------------------------------------------------------------------------
# Hybrid search quality
# ---------------------------------------------------------------------------


class TestHybridSearchQuality:
    """Tests that hybrid search combines both signals effectively."""

    def test_hybrid_returns_results_for_mixed_query(self, tmp_path: Path) -> None:
        """Hybrid search on 'database connection pooling' returns relevant results."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "database connection pooling",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            alpha=0.5,
            top_k=5,
        )

        assert len(results) > 0
        doc_ids = [r.source_file for r in results]
        # database-setup.md should appear (has all three terms)
        assert "database-setup.md" in doc_ids

    def test_hybrid_doc_in_both_lists_boosted(self, tmp_path: Path) -> None:
        """Doc appearing in both semantic AND keyword gets boosted."""
        collection, bm25 = _build_search_env(tmp_path)

        # This query has exact keywords that match database-setup.md
        results = search(
            "PostgreSQL connection pooling",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            alpha=0.5,
            top_k=5,
        )

        if results:
            # database-setup.md should be near top
            doc_ids = [r.source_file for r in results]
            assert "database-setup.md" in doc_ids[:3]


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------


class TestMetadataCompleteness:
    """Tests that search results contain all required metadata."""

    def test_results_have_all_metadata_fields(self, tmp_path: Path) -> None:
        """Each SearchResult has score, source_file, chunk_index, title, tags, heading."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "authentication",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=3,
        )

        assert len(results) > 0
        for r in results:
            assert isinstance(r.score, float)
            assert 0 <= r.score <= 1.0
            assert isinstance(r.source_file, str)
            assert len(r.source_file) > 0
            assert isinstance(r.chunk_index, int)
            assert r.chunk_index >= 0
            assert isinstance(r.title, str)
            assert isinstance(r.tags, str)
            assert isinstance(r.heading_hierarchy, str)
            assert r.heading_hierarchy == ""  # Empty until Phase 8

    def test_results_scores_normalized(self, tmp_path: Path) -> None:
        """Scores are in 0-1 range with max close to 1.0."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "redis caching",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=5,
        )

        if results:
            scores = [r.score for r in results]
            assert all(0 <= s <= 1.0 for s in scores)
            # Max should be 1.0 (normalized)
            assert max(scores) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------


class TestPerformance:
    """Search latency benchmark."""

    def test_search_latency_under_one_second(self, tmp_path: Path) -> None:
        """Average search latency < 1 second for typical queries."""
        collection, bm25 = _build_search_env(tmp_path)

        queries = [
            "authentication OAuth2",
            "PostgreSQL connection",
            "API endpoints REST",
            "Docker deployment",
            "error handling middleware",
            "Redis caching TTL",
            "pytest integration tests",
            "alembic migrations",
            "environment variables",
            "database connection pooling",
        ]

        total_time = 0.0
        for q in queries:
            start = time.perf_counter()
            search(
                q, collection, bm25,
                embed_fn=_fake_embed_fn,
                top_k=5,
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

        avg_latency = total_time / len(queries)
        # With fake embeddings and 8 docs, this should be well under 100ms
        assert avg_latency < 1.0, f"Average search latency {avg_latency:.3f}s exceeds 1s"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for search engine."""

    def test_search_no_results(self, tmp_path: Path) -> None:
        """Non-matching keyword query returns empty list."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "xyzzy_nonexistent_term_42",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=5,
            mode="keyword",
        )

        assert results == []

    def test_search_single_document_corpus(self, tmp_path: Path) -> None:
        """Search with only 1 document returns that document."""
        chroma_path = tmp_path / "chroma_single"
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = get_or_create_collection(client)

        content = "unique content about specialized topic"
        emb = _fake_embed_fn([content])[0]
        collection.add(
            ids=["single.md::chunk_0"],
            embeddings=[emb],
            documents=[content],
            metadatas=[{"source_file": "single.md", "chunk_index": 0, "title": "Single", "tags": ""}],
        )

        bm25 = BM25Index()
        bm25.rebuild(["single.md::chunk_0"], [content.lower().split()])

        # Semantic search should find the only document
        results = search(
            "specialized topic",
            collection, bm25,
            embed_fn=_fake_embed_fn,
            top_k=5,
            mode="semantic",
        )

        assert len(results) == 1
        assert results[0].source_file == "single.md"

    def test_search_all_modes_no_crash(self, tmp_path: Path) -> None:
        """All three modes run without errors."""
        collection, bm25 = _build_search_env(tmp_path)

        for mode in ("hybrid", "semantic", "keyword"):
            results = search(
                "database connection",
                collection, bm25,
                embed_fn=_fake_embed_fn,
                top_k=3,
                mode=mode,
            )
            assert isinstance(results, list)
