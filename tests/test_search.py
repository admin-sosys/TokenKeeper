"""Unit tests for the search engine module.

Tests cover:
  - SearchResult dataclass (creation, frozen, defaults)
  - semantic_search (sorted results, top_k, empty, distance conversion)
  - keyword_search (matching, lowercasing, empty, top_k)
  - reciprocal_rank_fusion (single list, two lists, k parameter, empty)

All ChromaDB tests use real PersistentClient at tmp_path with fake 768-dim embeddings.
No Ollama needed.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from knowledge_rag.bm25_index import BM25Index
from knowledge_rag.search import (
    SearchResult,
    enrich_results,
    hybrid_search,
    keyword_search,
    normalize_scores,
    reciprocal_rank_fusion,
    search,
    semantic_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Produce deterministic, unique 768-dim vectors per text."""
    result: list[list[float]] = []
    for text in texts:
        seed = hash(text) % 10000
        vec = [(seed + i) / 10000.0 for i in range(768)]
        result.append(vec)
    return result


def _make_collection(tmp_path: Path) -> chromadb.Collection:
    """Create a real ChromaDB collection at tmp_path."""
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        "test_search", metadata={"hnsw:space": "cosine"}
    )


# ---------------------------------------------------------------------------
# SearchResult dataclass tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Construct with all fields, verify each is accessible."""
        sr = SearchResult(
            chunk_id="docs/readme.md::chunk_0",
            content="Hello world",
            score=0.95,
            source_file="docs/readme.md",
            chunk_index=0,
            total_chunks=3,
            title="README",
            tags="alpha,beta",
            heading_hierarchy="# Intro > ## Setup",
        )
        assert sr.chunk_id == "docs/readme.md::chunk_0"
        assert sr.content == "Hello world"
        assert sr.score == pytest.approx(0.95)
        assert sr.source_file == "docs/readme.md"
        assert sr.chunk_index == 0
        assert sr.total_chunks == 3
        assert sr.title == "README"
        assert sr.tags == "alpha,beta"
        assert sr.heading_hierarchy == "# Intro > ## Setup"

    def test_search_result_frozen(self) -> None:
        """SearchResult is immutable."""
        sr = SearchResult(
            chunk_id="x::chunk_0",
            content="text",
            score=0.5,
            source_file="x",
            chunk_index=0,
            total_chunks=1,
            title="T",
            tags="",
        )
        with pytest.raises((AttributeError, TypeError)):
            sr.score = 0.99  # type: ignore[misc]

    def test_search_result_heading_hierarchy_default_empty(self) -> None:
        """heading_hierarchy defaults to empty string when omitted."""
        sr = SearchResult(
            chunk_id="x::chunk_0",
            content="text",
            score=0.5,
            source_file="x",
            chunk_index=0,
            total_chunks=1,
            title="T",
            tags="",
        )
        assert sr.heading_hierarchy == ""


# ---------------------------------------------------------------------------
# semantic_search tests
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    """Tests for semantic_search using real ChromaDB."""

    def test_semantic_search_returns_sorted_by_similarity(self, tmp_path: Path) -> None:
        """Results are ordered by descending similarity."""
        collection = _make_collection(tmp_path)

        # Create 3 chunks with distinct embeddings
        emb_a = [0.1] * 768
        emb_b = [0.5] * 768
        emb_c = [0.9] * 768

        collection.add(
            ids=["a::chunk_0", "b::chunk_0", "c::chunk_0"],
            embeddings=[emb_a, emb_b, emb_c],
            documents=["doc a", "doc b", "doc c"],
            metadatas=[
                {"source_file": "a", "chunk_index": 0},
                {"source_file": "b", "chunk_index": 0},
                {"source_file": "c", "chunk_index": 0},
            ],
        )

        # Query with embedding closest to emb_c
        query_emb = [0.85] * 768
        results = semantic_search(collection, query_emb, top_k=3)

        assert len(results) == 3
        # Results should be sorted by descending similarity
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # c should be closest to query
        assert results[0][0] == "c::chunk_0"

    def test_semantic_search_top_k_limits_results(self, tmp_path: Path) -> None:
        """Only top_k results returned."""
        collection = _make_collection(tmp_path)

        for i in range(5):
            emb = [i / 10.0] * 768
            collection.add(
                ids=[f"doc{i}::chunk_0"],
                embeddings=[emb],
                documents=[f"document {i}"],
                metadatas=[{"source_file": f"doc{i}", "chunk_index": 0}],
            )

        results = semantic_search(collection, [0.3] * 768, top_k=2)
        assert len(results) == 2

    def test_semantic_search_empty_collection(self, tmp_path: Path) -> None:
        """Empty collection returns empty list."""
        collection = _make_collection(tmp_path)
        results = semantic_search(collection, [0.5] * 768, top_k=10)
        assert results == []

    def test_semantic_search_distance_to_similarity_conversion(self, tmp_path: Path) -> None:
        """Same embedding query: distance≈0, similarity≈1.0."""
        collection = _make_collection(tmp_path)

        emb = [0.5] * 768
        collection.add(
            ids=["same::chunk_0"],
            embeddings=[emb],
            documents=["exact match"],
            metadatas=[{"source_file": "same", "chunk_index": 0}],
        )

        results = semantic_search(collection, emb, top_k=1)
        assert len(results) == 1
        doc_id, similarity = results[0]
        assert doc_id == "same::chunk_0"
        # Same vector: cosine distance ≈ 0, similarity ≈ 1.0
        assert similarity == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# keyword_search tests
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    """Tests for keyword_search wrapper over BM25Index."""

    def _build_index(self) -> BM25Index:
        """Build a BM25 index with 3 docs containing distinct terms."""
        bm25 = BM25Index()
        bm25.rebuild(
            ["python.md::chunk_0", "java.md::chunk_0", "rust.md::chunk_0"],
            [
                ["python", "programming", "data", "science", "machine", "learning"],
                ["java", "enterprise", "software", "development", "web", "applications"],
                ["rust", "systems", "programming", "memory", "safety", "performance"],
            ],
        )
        return bm25

    def test_keyword_search_returns_matching_docs(self) -> None:
        """Searching for 'python' returns the python doc."""
        bm25 = self._build_index()
        results = keyword_search(bm25, "python", top_k=10)
        assert len(results) > 0
        doc_ids = [doc_id for doc_id, _ in results]
        assert "python.md::chunk_0" in doc_ids

    def test_keyword_search_lowercases_query(self) -> None:
        """UPPERCASE query still matches lowercase-indexed content."""
        bm25 = self._build_index()
        results = keyword_search(bm25, "PYTHON", top_k=10)
        assert len(results) > 0
        doc_ids = [doc_id for doc_id, _ in results]
        assert "python.md::chunk_0" in doc_ids

    def test_keyword_search_empty_index(self) -> None:
        """Searching empty index returns empty list."""
        bm25 = BM25Index()
        results = keyword_search(bm25, "anything", top_k=10)
        assert results == []

    def test_keyword_search_top_k_limits(self) -> None:
        """Results limited to top_k."""
        bm25 = BM25Index()
        # Build index with 5 docs, each with unique + shared terms
        bm25.rebuild(
            [f"doc{i}::chunk_0" for i in range(5)],
            [
                ["shared", "term", f"unique{i}", "common", "word"]
                for i in range(5)
            ],
        )
        results = keyword_search(bm25, "shared unique0", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    """Tests for the RRF combiner."""

    def test_rrf_single_list(self) -> None:
        """Single ranked list: output preserves order with RRF scores."""
        ranked = [("A", 10.0), ("B", 5.0), ("C", 1.0)]
        result = reciprocal_rank_fusion([ranked], k=60)

        # Order preserved
        assert [doc_id for doc_id, _ in result] == ["A", "B", "C"]

        # Verify RRF formula: 1/(k + rank) where rank is 1-based
        assert result[0][1] == pytest.approx(1.0 / (60 + 1))
        assert result[1][1] == pytest.approx(1.0 / (60 + 2))
        assert result[2][1] == pytest.approx(1.0 / (60 + 3))

    def test_rrf_two_lists_same_doc_boosted(self) -> None:
        """Doc appearing in both lists gets boosted to top."""
        list1 = [("A", 10.0), ("B", 5.0)]
        list2 = [("A", 8.0), ("C", 3.0)]
        result = reciprocal_rank_fusion([list1, list2], k=60)

        # A should be #1 (appears at rank 1 in both)
        assert result[0][0] == "A"
        # A's score = 1/(60+1) + 1/(60+1) = 2/(61) ≈ 0.03279
        expected_a = 2.0 / 61
        assert result[0][1] == pytest.approx(expected_a)

    def test_rrf_two_lists_different_docs(self) -> None:
        """All docs from both lists appear in output."""
        list1 = [("A", 10.0), ("B", 5.0)]
        list2 = [("C", 8.0), ("D", 3.0)]
        result = reciprocal_rank_fusion([list1, list2], k=60)

        result_ids = [doc_id for doc_id, _ in result]
        assert set(result_ids) == {"A", "B", "C", "D"}

        # Rank-1 items (A, C) should score higher than rank-2 items (B, D)
        scores = {doc_id: score for doc_id, score in result}
        assert scores["A"] > scores["B"]
        assert scores["C"] > scores["D"]

    def test_rrf_k_parameter_effect(self) -> None:
        """Smaller k gives more weight spread between top and bottom ranks."""
        ranked = [("A", 10.0), ("B", 5.0)]

        result_k1 = reciprocal_rank_fusion([ranked], k=1)
        result_k100 = reciprocal_rank_fusion([ranked], k=100)

        # With k=1: A=1/2=0.5, B=1/3≈0.333, ratio=1.5
        # With k=100: A=1/101≈0.0099, B=1/102≈0.0098, ratio≈1.01
        ratio_k1 = result_k1[0][1] / result_k1[1][1]
        ratio_k100 = result_k100[0][1] / result_k100[1][1]
        assert ratio_k1 > ratio_k100  # k=1 has more spread

    def test_rrf_empty_lists(self) -> None:
        """Empty input returns empty output."""
        result = reciprocal_rank_fusion([], k=60)
        assert result == []

        result2 = reciprocal_rank_fusion([[]], k=60)
        assert result2 == []


# ---------------------------------------------------------------------------
# hybrid_search tests
# ---------------------------------------------------------------------------


def _build_search_env(
    tmp_path: Path,
) -> tuple[chromadb.Collection, BM25Index]:
    """Build a ChromaDB collection + BM25 index with 5 docs for hybrid testing.

    Documents have distinct embeddings AND distinct BM25 terms so both
    search modes can differentiate them.
    """
    collection = _make_collection(tmp_path)
    bm25 = BM25Index()

    # 5 documents: embeddings go 0.1, 0.3, 0.5, 0.7, 0.9
    # BM25 terms are distinct for each doc
    docs = [
        ("auth.md::chunk_0", "authentication oauth2 tokens security login", [0.1] * 768),
        ("db.md::chunk_0", "database postgresql connection pooling queries", [0.3] * 768),
        ("api.md::chunk_0", "rest api endpoints http json responses", [0.5] * 768),
        ("deploy.md::chunk_0", "docker deployment kubernetes containers orchestration", [0.7] * 768),
        ("cache.md::chunk_0", "redis caching ttl invalidation memory store", [0.9] * 768),
    ]

    ids = [d[0] for d in docs]
    documents = [d[1] for d in docs]
    embeddings = [d[2] for d in docs]
    metadatas = [
        {"source_file": d[0].split("::")[0], "chunk_index": 0}
        for d in docs
    ]

    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    # Build BM25 index with tokenized content
    bm25.rebuild(ids, [doc.split() for _, doc, _ in docs])

    return collection, bm25


class TestHybridSearch:
    """Tests for hybrid_search orchestrator."""

    def test_hybrid_search_pure_semantic(self, tmp_path: Path) -> None:
        """mode='semantic' returns only semantic results (alpha forced to 1.0)."""
        collection, bm25 = _build_search_env(tmp_path)

        results = hybrid_search(
            collection, bm25, "cache test",
            embed_fn=lambda texts: [[0.85] * 768 for _ in texts],
            alpha=0.3,  # should be overridden by mode
            top_k=5,
            mode="semantic",
        )

        assert len(results) > 0
        # Results should come from semantic_search (not BM25)
        # Verify ordering is by descending similarity score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_search_pure_keyword(self, tmp_path: Path) -> None:
        """mode='keyword' returns only BM25 results (alpha forced to 0.0)."""
        collection, bm25 = _build_search_env(tmp_path)

        results = hybrid_search(
            collection, bm25, "postgresql database",
            embed_fn=lambda texts: [[0.85] * 768 for _ in texts],
            alpha=0.9,  # should be overridden by mode
            top_k=5,
            mode="keyword",
        )

        assert len(results) > 0
        doc_ids = [doc_id for doc_id, _ in results]
        assert "db.md::chunk_0" in doc_ids

    def test_hybrid_search_default_returns_results(self, tmp_path: Path) -> None:
        """Default hybrid mode returns non-empty results sorted by score."""
        collection, bm25 = _build_search_env(tmp_path)

        results = hybrid_search(
            collection, bm25, "database connection",
            embed_fn=lambda texts: [[0.35] * 768 for _ in texts],
            top_k=5,
        )

        assert len(results) > 0
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_search_alpha_shifts_ranking(self, tmp_path: Path) -> None:
        """High alpha favors semantic, low alpha favors keyword."""
        collection, bm25 = _build_search_env(tmp_path)

        # Query: embedding close to cache.md (0.9) but text matches db.md
        embed_fn_cache = lambda texts: [[0.88] * 768 for _ in texts]

        # Semantic-heavy: cache.md should rank higher
        results_semantic = hybrid_search(
            collection, bm25, "postgresql",
            embed_fn=embed_fn_cache, alpha=0.9, top_k=5,
        )

        # Keyword-heavy: db.md should rank higher (has "postgresql")
        results_keyword = hybrid_search(
            collection, bm25, "postgresql",
            embed_fn=embed_fn_cache, alpha=0.1, top_k=5,
        )

        # With alpha=0.9, cache.md (best semantic) should be near top
        semantic_ids = [doc_id for doc_id, _ in results_semantic]
        # With alpha=0.1, db.md (best keyword) should be near top
        keyword_ids = [doc_id for doc_id, _ in results_keyword]

        # db.md should rank higher in keyword-heavy search
        if "db.md::chunk_0" in keyword_ids and "db.md::chunk_0" in semantic_ids:
            keyword_rank = keyword_ids.index("db.md::chunk_0")
            semantic_rank = semantic_ids.index("db.md::chunk_0")
            assert keyword_rank <= semantic_rank

    def test_hybrid_search_mode_semantic_overrides_alpha(self, tmp_path: Path) -> None:
        """mode='semantic' with alpha=0.3 behaves as pure semantic."""
        collection, bm25 = _build_search_env(tmp_path)

        # Embedding close to cache.md
        embed_fn = lambda texts: [[0.88] * 768 for _ in texts]

        results_mode = hybrid_search(
            collection, bm25, "test",
            embed_fn=embed_fn, alpha=0.3, top_k=3, mode="semantic",
        )
        results_alpha = hybrid_search(
            collection, bm25, "test",
            embed_fn=embed_fn, alpha=1.0, top_k=3, mode="hybrid",
        )

        # Same ordering (both are pure semantic)
        ids_mode = [doc_id for doc_id, _ in results_mode]
        ids_alpha = [doc_id for doc_id, _ in results_alpha]
        assert ids_mode == ids_alpha

    def test_hybrid_search_mode_keyword_overrides_alpha(self, tmp_path: Path) -> None:
        """mode='keyword' with alpha=0.7 behaves as pure keyword."""
        collection, bm25 = _build_search_env(tmp_path)

        embed_fn = lambda texts: [[0.5] * 768 for _ in texts]

        results_mode = hybrid_search(
            collection, bm25, "docker deployment",
            embed_fn=embed_fn, alpha=0.7, top_k=3, mode="keyword",
        )
        results_alpha = hybrid_search(
            collection, bm25, "docker deployment",
            embed_fn=embed_fn, alpha=0.0, top_k=3, mode="hybrid",
        )

        # Same ordering (both are pure keyword)
        ids_mode = [doc_id for doc_id, _ in results_mode]
        ids_alpha = [doc_id for doc_id, _ in results_alpha]
        assert ids_mode == ids_alpha

    def test_hybrid_search_top_k_respected(self, tmp_path: Path) -> None:
        """top_k limits result count."""
        collection, bm25 = _build_search_env(tmp_path)

        results = hybrid_search(
            collection, bm25, "api database docker",
            embed_fn=lambda texts: [[0.5] * 768 for _ in texts],
            top_k=2,
        )

        assert len(results) <= 2

    def test_hybrid_search_empty_results(self, tmp_path: Path) -> None:
        """Empty collection + empty BM25 returns empty list."""
        collection = _make_collection(tmp_path)
        bm25 = BM25Index()

        results = hybrid_search(
            collection, bm25, "anything",
            embed_fn=lambda texts: [[0.5] * 768 for _ in texts],
            top_k=5,
        )

        assert results == []


# ---------------------------------------------------------------------------
# normalize_scores tests
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    """Tests for score normalization."""

    def test_normalize_scores_max_becomes_one(self) -> None:
        """Max score normalizes to 1.0."""
        ranked = [("A", 0.5), ("B", 0.25)]
        result = normalize_scores(ranked)
        assert result[0] == ("A", pytest.approx(1.0))
        assert result[1] == ("B", pytest.approx(0.5))

    def test_normalize_scores_empty(self) -> None:
        """Empty input returns empty."""
        assert normalize_scores([]) == []

    def test_normalize_scores_single_item(self) -> None:
        """Single item normalizes to 1.0."""
        result = normalize_scores([("A", 0.3)])
        assert result[0] == ("A", pytest.approx(1.0))

    def test_normalize_scores_all_zero(self) -> None:
        """All-zero scores returned as-is (no division by zero)."""
        ranked = [("A", 0.0)]
        result = normalize_scores(ranked)
        assert result[0] == ("A", 0.0)


# ---------------------------------------------------------------------------
# enrich_results tests
# ---------------------------------------------------------------------------


class TestEnrichResults:
    """Tests for metadata enrichment."""

    def test_enrich_results_populates_all_fields(self, tmp_path: Path) -> None:
        """All SearchResult fields populated from ChromaDB metadata."""
        collection = _make_collection(tmp_path)

        collection.add(
            ids=["doc.md::chunk_0", "doc.md::chunk_1"],
            embeddings=[[0.5] * 768, [0.6] * 768],
            documents=["first chunk content", "second chunk content"],
            metadatas=[
                {"source_file": "doc.md", "chunk_index": 0, "title": "My Doc", "tags": "alpha,beta"},
                {"source_file": "doc.md", "chunk_index": 1, "title": "My Doc", "tags": "alpha,beta"},
            ],
        )

        ranked = [("doc.md::chunk_0", 1.0), ("doc.md::chunk_1", 0.5)]
        results = enrich_results(collection, ranked)

        assert len(results) == 2
        r = results[0]
        assert r.chunk_id == "doc.md::chunk_0"
        assert r.content == "first chunk content"
        assert r.score == pytest.approx(1.0)
        assert r.source_file == "doc.md"
        assert r.chunk_index == 0
        assert r.title == "My Doc"
        assert r.tags == "alpha,beta"
        assert r.heading_hierarchy == ""

    def test_enrich_results_preserves_order(self, tmp_path: Path) -> None:
        """Output order matches ranked input order."""
        collection = _make_collection(tmp_path)

        collection.add(
            ids=["a::chunk_0", "b::chunk_0", "c::chunk_0"],
            embeddings=[[0.1] * 768, [0.5] * 768, [0.9] * 768],
            documents=["doc a", "doc b", "doc c"],
            metadatas=[
                {"source_file": "a", "chunk_index": 0},
                {"source_file": "b", "chunk_index": 0},
                {"source_file": "c", "chunk_index": 0},
            ],
        )

        ranked = [("c::chunk_0", 0.9), ("a::chunk_0", 0.5), ("b::chunk_0", 0.3)]
        results = enrich_results(collection, ranked)

        assert [r.chunk_id for r in results] == ["c::chunk_0", "a::chunk_0", "b::chunk_0"]

    def test_enrich_results_empty_input(self, tmp_path: Path) -> None:
        """Empty ranked list returns empty results."""
        collection = _make_collection(tmp_path)
        results = enrich_results(collection, [])
        assert results == []


# ---------------------------------------------------------------------------
# Full search pipeline tests
# ---------------------------------------------------------------------------


class TestSearchPipeline:
    """Tests for the top-level search() function."""

    def test_search_returns_search_results(self, tmp_path: Path) -> None:
        """search() returns list[SearchResult] with scores in 0-1 range."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "database connection",
            collection, bm25,
            embed_fn=lambda texts: [[0.35] * 768 for _ in texts],
            top_k=5,
        )

        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert 0 <= r.score <= 1.0

    def test_search_scores_normalized(self, tmp_path: Path) -> None:
        """Max score should be 1.0 (normalized)."""
        collection, bm25 = _build_search_env(tmp_path)

        results = search(
            "docker deployment",
            collection, bm25,
            embed_fn=lambda texts: [[0.5] * 768 for _ in texts],
            top_k=5,
        )

        if results:
            assert results[0].score == pytest.approx(1.0)

    def test_search_empty_query_no_crash(self, tmp_path: Path) -> None:
        """Search with non-matching query returns empty list or results, no crash."""
        collection = _make_collection(tmp_path)
        bm25 = BM25Index()

        results = search(
            "xyzzy_nonexistent",
            collection, bm25,
            embed_fn=lambda texts: [[0.5] * 768 for _ in texts],
            top_k=5,
        )

        assert isinstance(results, list)
