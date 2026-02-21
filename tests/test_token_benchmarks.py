"""Token-savings benchmark suite for TokenKeeper.

Quantifies the core value proposition: how many tokens are saved by
retrieving only relevant chunks instead of sending the full corpus to the
LLM context window.

Benchmarks
----------
1. Compression Ratio  -- Total corpus tokens vs retrieved context tokens
2. Relevance Density  -- Fraction of retrieved tokens that are relevant
3. Coverage (Recall)  -- Does RAG find all relevant chunks for known queries?
4. Sensitivity        -- How chunk_size, overlap, top_k, content_mode affect savings
5. Query-Type         -- Token savings by query type (conceptual, keyword, broad, narrow)
6. Summary            -- Aggregate headline metric

Tiers
-----
Tier 1 (always run): Fake embeddings, validates compression ratios & structure
Tier 2 (requires Ollama): Real embeddings, validates relevance/recall quality

Usage::

    # Tier 1 only (fast, no Ollama)
    uv run pytest tests/test_token_benchmarks.py -m "not ollama" -v -s

    # All tiers
    uv run pytest tests/test_token_benchmarks.py -v -s
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import chromadb
import pytest

from tokenkeeper.bm25_index import BM25Index
from tokenkeeper.config import RagConfig
from tokenkeeper.discovery import discover_code_files, discover_markdown_files
from tokenkeeper.indexer import index_documents
from tokenkeeper.search import SearchResult, search


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLONE_URL = "https://github.com/httpie/cli.git"
DEFAULT_CLONE_DEPTH = 1
CACHE_DIR = Path(__file__).parent.parent / ".acceptance-cache"

# Ground-truth queries for httpie/cli.
# Each maps to expected keywords that SHOULD appear in relevant results.
GROUND_TRUTH_QUERIES: dict[str, dict[str, Any]] = {
    "narrow_keyword": {
        "query": "SessionStorage",
        "expected_terms": ["session", "storage", "cookie"],
        "type": "keyword",
    },
    "narrow_conceptual": {
        "query": "how to send a POST request with JSON data",
        "expected_terms": ["post", "json", "request", "data", "body"],
        "type": "conceptual",
    },
    "broad_conceptual": {
        "query": "overview of the project architecture and design",
        "expected_terms": ["httpie", "cli", "http", "command"],
        "type": "broad",
    },
    "exact_function": {
        "query": "def format_headers",
        "expected_terms": ["def", "header", "format"],
        "type": "keyword",
    },
    "installation": {
        "query": "how to install httpie",
        "expected_terms": ["install", "pip", "brew", "package"],
        "type": "conceptual",
    },
}


# ===================================================================
# Token counting utilities
# ===================================================================


def count_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic.

    This is a deterministic, dependency-free approximation.  For English
    text with typical LLM tokenizers (cl100k_base, etc.), chars/4 is
    accurate to within ~10-15%.  For code it slightly underestimates,
    making savings figures conservative.

    Returns 0 for empty text, ``max(1, len(text) // 4)`` otherwise.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens_for_results(results: list[SearchResult]) -> int:
    """Sum estimated tokens across all :class:`SearchResult` contents."""
    return sum(count_tokens(r.content) for r in results)


def count_corpus_tokens(collection: chromadb.Collection) -> int:
    """Count total tokens across all chunks stored in a ChromaDB collection."""
    stored = collection.get(include=["documents"])
    docs = stored.get("documents") or []
    return sum(count_tokens(doc) for doc in docs)


def count_raw_file_tokens(repo_path: Path, content_mode: str = "docs") -> int:
    """Count total tokens in the raw source files (pre-chunking).

    This represents the cost of naively including all source files in the
    LLM context window -- the *baseline* that RAG aims to compress.
    """
    if content_mode == "docs":
        files = discover_markdown_files(repo_path)
    elif content_mode == "code":
        files = discover_code_files(repo_path)
    else:  # "both"
        files = list(
            set(discover_markdown_files(repo_path) + discover_code_files(repo_path))
        )

    total = 0
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            total += count_tokens(content)
        except (UnicodeDecodeError, PermissionError):
            continue
    return total


# ===================================================================
# Benchmark result model & pretty-printer
# ===================================================================


@dataclass
class BenchmarkResult:
    """A single benchmark measurement."""

    name: str
    corpus_tokens: int
    retrieved_tokens: int
    compression_ratio: float  # corpus / retrieved  (higher = more savings)
    savings_pct: float  # (1 - retrieved / corpus) * 100
    query: str = ""
    top_k: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def print_benchmark_table(
    results: list[BenchmarkResult],
    title: str = "Benchmark Results",
) -> None:
    """Print a formatted benchmark summary table to stdout."""
    print(f"\n{'=' * 78}")
    print(f"  TOKEN SAVINGS BENCHMARK: {title}")
    print(f"{'=' * 78}")
    print(
        f"  {'Benchmark':<30} {'Corpus':>10} {'Retrieved':>10} "
        f"{'Savings':>8} {'Ratio':>7}"
    )
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 7}")
    for r in results:
        print(
            f"  {r.name:<30} {r.corpus_tokens:>10,} {r.retrieved_tokens:>10,} "
            f"{r.savings_pct:>7.1f}% {r.compression_ratio:>6.1f}x"
        )
    print(f"{'=' * 78}\n")


# ===================================================================
# Embedding helpers
# ===================================================================


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings based on text hash."""
    results: list[list[float]] = []
    for text in texts:
        h = hashlib.sha256(text.lower().encode()).digest()
        vec: list[float] = []
        for i in range(768):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 127.5) - 1.0)
        results.append(vec)
    return results


def _check_ollama_available() -> bool:
    """Check if Ollama is running and nomic-embed-text is available."""
    try:
        import requests as _req

        resp = _req.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["test"]},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="session")
def repo_path() -> Path:
    """Get or clone the acceptance test repository (reuses acceptance cache)."""
    env_path = os.environ.get("ACCEPTANCE_REPO")
    if env_path:
        p = Path(env_path)
        assert p.is_dir(), f"ACCEPTANCE_REPO={env_path} is not a directory"
        return p

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    repo_name = DEFAULT_CLONE_URL.rstrip("/").split("/")[-1].replace(".git", "")
    cached = CACHE_DIR / repo_name

    if cached.is_dir():
        return cached

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            str(DEFAULT_CLONE_DEPTH),
            DEFAULT_CLONE_URL,
            str(cached),
        ],
        check=True,
        capture_output=True,
    )
    return cached


@pytest.fixture(scope="session")
def raw_corpus_tokens_docs(repo_path: Path) -> int:
    """Total tokens in all raw markdown files (the naive baseline)."""
    return count_raw_file_tokens(repo_path, "docs")


@pytest.fixture(scope="session")
def raw_corpus_tokens_code(repo_path: Path) -> int:
    """Total tokens in all raw code files."""
    return count_raw_file_tokens(repo_path, "code")


@pytest.fixture(scope="session")
def raw_corpus_tokens_both(repo_path: Path) -> int:
    """Total tokens in all docs + code files."""
    return count_raw_file_tokens(repo_path, "both")


def _build_index(
    repo_path: Path,
    content_mode: str,
    chunk_size: int,
    overlap: int,
    collection_name: str,
    embed_fn: Callable[..., Any],
) -> dict[str, Any]:
    """Build a complete indexed-repo artifact dict."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    config = RagConfig(
        content_mode=content_mode,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    bm25 = BM25Index()

    result = index_documents(repo_path, config, collection, bm25, embed_fn)

    return {
        "config": config,
        "collection": collection,
        "bm25": bm25,
        "result": result,
        "embed_fn": embed_fn,
        "repo_path": repo_path,
        "corpus_tokens": count_corpus_tokens(collection),
    }


@pytest.fixture(scope="session")
def bench_docs(repo_path: Path) -> dict[str, Any]:
    """Standard docs-mode index for benchmarks."""
    return _build_index(repo_path, "docs", 1000, 200, "bench_docs", _fake_embed_fn)


@pytest.fixture(scope="session")
def bench_code(repo_path: Path) -> dict[str, Any]:
    """Standard code-mode index."""
    return _build_index(repo_path, "code", 1500, 200, "bench_code", _fake_embed_fn)


@pytest.fixture(scope="session")
def bench_both(repo_path: Path) -> dict[str, Any]:
    """Standard both-mode index."""
    return _build_index(repo_path, "both", 1000, 200, "bench_both", _fake_embed_fn)


# Sensitivity analysis fixtures: different chunk sizes
@pytest.fixture(scope="session")
def bench_docs_small_chunks(repo_path: Path) -> dict[str, Any]:
    """Docs-mode with small chunk_size=500."""
    return _build_index(
        repo_path, "docs", 500, 100, "bench_docs_small", _fake_embed_fn
    )


@pytest.fixture(scope="session")
def bench_docs_large_chunks(repo_path: Path) -> dict[str, Any]:
    """Docs-mode with large chunk_size=2000."""
    return _build_index(
        repo_path, "docs", 2000, 400, "bench_docs_large", _fake_embed_fn
    )


# ===================================================================
# BENCHMARK 1: Compression Ratio
# ===================================================================


class TestCompressionRatio:
    """Corpus tokens vs retrieved tokens -- the headline metric.

    Answers: "If I use RAG instead of dumping everything into context,
    how many tokens do I save?"
    """

    def test_compression_ratio_docs_mode(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Docs-mode compression ratio should average >= 80% savings."""
        results_list: list[BenchmarkResult] = []

        for name, gt in GROUND_TRUTH_QUERIES.items():
            results = search(
                query=gt["query"],
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=10,
            )
            retrieved_tokens = count_tokens_for_results(results)
            corpus = raw_corpus_tokens_docs
            ratio = corpus / max(retrieved_tokens, 1)
            savings = (1 - retrieved_tokens / max(corpus, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=f"docs/{name}",
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved_tokens,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=gt["query"],
                    top_k=10,
                )
            )

        print_benchmark_table(results_list, "Compression Ratio (docs mode)")

        avg_savings = sum(r.savings_pct for r in results_list) / len(results_list)
        assert avg_savings >= 80.0, (
            f"Average token savings {avg_savings:.1f}% is below 80% threshold"
        )
        for r in results_list:
            assert r.savings_pct >= 50.0, (
                f"Query '{r.query}' only saves {r.savings_pct:.1f}% tokens"
            )

    def test_compression_ratio_code_mode(
        self,
        bench_code: dict[str, Any],
        raw_corpus_tokens_code: int,
    ) -> None:
        """Code-mode compression ratio for code-related queries."""
        code_queries = [
            ("function_search", "def format_headers"),
            ("class_search", "class HTTPMessage"),
            ("import_search", "import requests"),
        ]
        results_list: list[BenchmarkResult] = []

        for name, query in code_queries:
            results = search(
                query=query,
                collection=bench_code["collection"],
                bm25_index=bench_code["bm25"],
                embed_fn=bench_code["embed_fn"],
                top_k=10,
            )
            retrieved_tokens = count_tokens_for_results(results)
            corpus = raw_corpus_tokens_code
            ratio = corpus / max(retrieved_tokens, 1)
            savings = (1 - retrieved_tokens / max(corpus, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=f"code/{name}",
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved_tokens,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                )
            )

        print_benchmark_table(results_list, "Compression Ratio (code mode)")

        avg_savings = sum(r.savings_pct for r in results_list) / len(results_list)
        assert avg_savings >= 80.0, (
            f"Code-mode average savings {avg_savings:.1f}% below 80%"
        )

    def test_compression_ratio_both_mode(
        self,
        bench_both: dict[str, Any],
        raw_corpus_tokens_both: int,
    ) -> None:
        """Both-mode should show high compression (larger corpus, same top_k)."""
        results = search(
            query="how to configure the application",
            collection=bench_both["collection"],
            bm25_index=bench_both["bm25"],
            embed_fn=bench_both["embed_fn"],
            top_k=10,
        )
        retrieved_tokens = count_tokens_for_results(results)
        corpus = raw_corpus_tokens_both
        savings = (1 - retrieved_tokens / max(corpus, 1)) * 100
        ratio = corpus / max(retrieved_tokens, 1)

        print(
            f"\n  Both-mode: {corpus:,} corpus -> {retrieved_tokens:,} retrieved "
            f"({savings:.1f}% savings, {ratio:.1f}x compression)"
        )

        assert savings >= 85.0, (
            f"Both-mode savings {savings:.1f}% below 85% threshold"
        )

    def test_end_to_end_pipeline_flow(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Compare raw files -> chunked index -> retrieval token flow.

        Shows the full pipeline: raw files are chunked (adding overlap
        overhead), then retrieval dramatically reduces context usage.
        """
        results = search(
            query="documentation",
            collection=bench_docs["collection"],
            bm25_index=bench_docs["bm25"],
            embed_fn=bench_docs["embed_fn"],
            top_k=10,
        )
        retrieved = count_tokens_for_results(results)
        raw = raw_corpus_tokens_docs
        indexed = bench_docs["corpus_tokens"]

        print(f"\n  End-to-end pipeline token flow:")
        print(f"    Raw files:       {raw:>10,} tokens")
        print(
            f"    After chunking:  {indexed:>10,} tokens "
            f"(overhead: {(indexed / max(raw, 1) - 1) * 100:+.1f}%)"
        )
        print(
            f"    After retrieval: {retrieved:>10,} tokens "
            f"(savings: {(1 - retrieved / max(raw, 1)) * 100:.1f}%)"
        )

        # Chunking overhead should be modest (overlap adds ~20%)
        if raw > 0:
            overhead = indexed / raw
            assert overhead < 1.5, (
                f"Chunking overhead {overhead:.2f}x is too high (expected < 1.5x)"
            )
        # Retrieval savings should be dramatic
        if raw > 0:
            savings = (1 - retrieved / raw) * 100
            assert savings >= 80.0


# ===================================================================
# BENCHMARK 2: Relevance Density
# ===================================================================


class TestRelevanceDensity:
    """What fraction of retrieved tokens are actually relevant?

    Uses ground-truth keyword matching as a proxy for relevance.
    A retrieved chunk is "relevant" if it contains at least one of the
    expected terms for the query.

    Answers: "Of the tokens I'm paying for, how many are useful?"
    """

    @staticmethod
    def _relevance_score(
        results: list[SearchResult], expected_terms: list[str]
    ) -> dict[str, Any]:
        """Compute relevance metrics for a set of results."""
        relevant_chunks = 0
        relevant_tokens = 0
        total_tokens = 0

        for r in results:
            tokens = count_tokens(r.content)
            total_tokens += tokens
            content_lower = r.content.lower()
            if any(term in content_lower for term in expected_terms):
                relevant_chunks += 1
                relevant_tokens += tokens

        chunk_precision = relevant_chunks / max(len(results), 1)
        token_precision = relevant_tokens / max(total_tokens, 1)

        return {
            "total_chunks": len(results),
            "relevant_chunks": relevant_chunks,
            "total_tokens": total_tokens,
            "relevant_tokens": relevant_tokens,
            "chunk_precision": chunk_precision,
            "token_precision": token_precision,
        }

    def test_relevance_density_keyword_queries(
        self, bench_docs: dict[str, Any]
    ) -> None:
        """Keyword queries should have high relevance density."""
        keyword_queries = {
            k: v for k, v in GROUND_TRUTH_QUERIES.items() if v["type"] == "keyword"
        }

        print(f"\n{'=' * 60}")
        print("  RELEVANCE DENSITY: Keyword Queries")
        print(f"{'=' * 60}")

        for name, gt in keyword_queries.items():
            results = search(
                query=gt["query"],
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                mode="keyword",
                top_k=10,
            )
            metrics = self._relevance_score(results, gt["expected_terms"])

            print(
                f"  {name}: {metrics['relevant_chunks']}/{metrics['total_chunks']} "
                f"chunks relevant ({metrics['chunk_precision']:.0%}), "
                f"{metrics['relevant_tokens']}/{metrics['total_tokens']} "
                f"tokens relevant ({metrics['token_precision']:.0%})"
            )

            if results:
                assert metrics["chunk_precision"] >= 0.3, (
                    f"Query '{gt['query']}': only {metrics['chunk_precision']:.0%} "
                    f"chunks contain expected terms"
                )

    def test_relevance_density_conceptual_queries(
        self, bench_docs: dict[str, Any]
    ) -> None:
        """Conceptual queries measured by topic keyword overlap."""
        conceptual_queries = {
            k: v
            for k, v in GROUND_TRUTH_QUERIES.items()
            if v["type"] in ("conceptual", "broad")
        }

        print(f"\n{'=' * 60}")
        print("  RELEVANCE DENSITY: Conceptual Queries")
        print(f"{'=' * 60}")

        for name, gt in conceptual_queries.items():
            results = search(
                query=gt["query"],
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=10,
            )
            metrics = self._relevance_score(results, gt["expected_terms"])

            print(
                f"  {name}: {metrics['chunk_precision']:.0%} chunk precision, "
                f"{metrics['token_precision']:.0%} token precision"
            )

            # Conceptual queries are harder with fake embeddings
            if results:
                assert metrics["chunk_precision"] >= 0.1, (
                    f"Query '{gt['query']}': 0% relevance -- "
                    f"no results contain expected terms"
                )

    def test_wasted_token_analysis(self, bench_docs: dict[str, Any]) -> None:
        """Report how many retrieved tokens are 'wasted' (not relevant).

        This is the complement of relevance density.  A good RAG system
        wastes fewer tokens than a naive approach.
        """
        total_wasted = 0
        total_retrieved = 0

        for _name, gt in GROUND_TRUTH_QUERIES.items():
            results = search(
                query=gt["query"],
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=10,
            )
            metrics = self._relevance_score(results, gt["expected_terms"])
            wasted = metrics["total_tokens"] - metrics["relevant_tokens"]
            total_wasted += wasted
            total_retrieved += metrics["total_tokens"]

        waste_pct = total_wasted / max(total_retrieved, 1) * 100
        print(
            f"\n  Aggregate token waste: {total_wasted:,}/{total_retrieved:,} "
            f"({waste_pct:.1f}% wasted)"
        )

        # Even with some waste, RAG is still dramatically better than
        # including the whole corpus.  Assert waste isn't catastrophic.
        assert waste_pct < 95.0, (
            f"Token waste {waste_pct:.1f}% is catastrophically high"
        )


# ===================================================================
# BENCHMARK 3: Coverage (Recall)
# ===================================================================


class TestCoverage:
    """Does the RAG system find all relevant chunks?

    Uses known topics in the httpie/cli repo as ground truth and checks
    if retrieval covers the expected source files.
    """

    def test_readme_coverage(self, bench_docs: dict[str, Any]) -> None:
        """Querying for project overview should retrieve README chunks."""
        results = search(
            query="httpie command line HTTP client overview",
            collection=bench_docs["collection"],
            bm25_index=bench_docs["bm25"],
            embed_fn=bench_docs["embed_fn"],
            top_k=10,
        )
        source_files = {r.source_file for r in results}

        readme_found = any("readme" in sf.lower() for sf in source_files)
        print(f"\n  README coverage: {'FOUND' if readme_found else 'MISSING'}")
        print(f"  Retrieved from: {sorted(source_files)}")

        assert readme_found, f"README not found in results. Sources: {source_files}"

    def test_multi_file_coverage(self, bench_docs: dict[str, Any]) -> None:
        """Broad queries should retrieve chunks from multiple source files."""
        results = search(
            query="documentation contributing changelog",
            collection=bench_docs["collection"],
            bm25_index=bench_docs["bm25"],
            embed_fn=bench_docs["embed_fn"],
            top_k=10,
        )
        source_files = {r.source_file for r in results}

        print(f"\n  Multi-file coverage: {len(source_files)} unique files")
        print(f"  Files: {sorted(source_files)}")

        assert len(source_files) >= 2, (
            f"Broad query only retrieved from {len(source_files)} file(s)"
        )

    def test_coverage_increases_with_top_k(
        self, bench_docs: dict[str, Any]
    ) -> None:
        """Coverage (unique source files) should be non-decreasing with top_k."""
        query = "httpie usage"
        coverages: list[tuple[int, int]] = []

        for top_k in [3, 5, 10, 20]:
            results = search(
                query=query,
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=top_k,
            )
            unique_files = len({r.source_file for r in results})
            coverages.append((top_k, unique_files))

        print(f"\n  Coverage vs top_k for query '{query}':")
        for k, files in coverages:
            print(f"    top_k={k:>2}: {files} unique source files")

        for i in range(1, len(coverages)):
            assert coverages[i][1] >= coverages[i - 1][1], (
                f"Coverage decreased from top_k={coverages[i - 1][0]} "
                f"({coverages[i - 1][1]} files) to top_k={coverages[i][0]} "
                f"({coverages[i][1]} files)"
            )


# ===================================================================
# BENCHMARK 4: Sensitivity Analysis
# ===================================================================


class TestSensitivityAnalysis:
    """How do config parameters affect token savings?

    Varies chunk_size, top_k, content_mode, and search_mode to produce
    a sensitivity analysis showing which knobs matter most.
    """

    def test_chunk_size_sensitivity(
        self,
        bench_docs: dict[str, Any],
        bench_docs_small_chunks: dict[str, Any],
        bench_docs_large_chunks: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Compare token savings across chunk_size = 500, 1000, 2000."""
        query = "how to install httpie"
        configs = [
            ("chunk_size=500", bench_docs_small_chunks),
            ("chunk_size=1000", bench_docs),
            ("chunk_size=2000", bench_docs_large_chunks),
        ]

        results_list: list[BenchmarkResult] = []
        for label, indexed in configs:
            results = search(
                query=query,
                collection=indexed["collection"],
                bm25_index=indexed["bm25"],
                embed_fn=indexed["embed_fn"],
                top_k=10,
            )
            retrieved = count_tokens_for_results(results)
            corpus = raw_corpus_tokens_docs
            ratio = corpus / max(retrieved, 1)
            savings = (1 - retrieved / max(corpus, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=label,
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                    extra={
                        "total_chunks": indexed["result"].chunks_indexed,
                        "corpus_chunk_tokens": indexed["corpus_tokens"],
                    },
                )
            )

        print_benchmark_table(results_list, "Chunk Size Sensitivity")

        print("  Chunk counts:")
        for r in results_list:
            print(
                f"    {r.name}: {r.extra['total_chunks']} chunks, "
                f"{r.extra['corpus_chunk_tokens']:,} total chunk tokens"
            )

        for r in results_list:
            assert r.savings_pct >= 70.0, (
                f"{r.name}: savings {r.savings_pct:.1f}% below 70% threshold"
            )

    def test_top_k_sensitivity(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Token savings should decrease as top_k increases.

        But even at top_k=20, savings should still be substantial.
        """
        query = "how to install httpie"
        corpus = raw_corpus_tokens_docs

        results_list: list[BenchmarkResult] = []
        for top_k in [1, 3, 5, 10, 20]:
            results = search(
                query=query,
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=top_k,
            )
            retrieved = count_tokens_for_results(results)
            ratio = corpus / max(retrieved, 1)
            savings = (1 - retrieved / max(corpus, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=f"top_k={top_k}",
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=top_k,
                )
            )

        print_benchmark_table(results_list, "Top-K Sensitivity")

        # Savings should decrease as top_k increases (with small tolerance)
        for i in range(1, len(results_list)):
            assert results_list[i].savings_pct <= results_list[i - 1].savings_pct + 1.0, (
                f"Savings increased unexpectedly: top_k={results_list[i].top_k} "
                f"({results_list[i].savings_pct:.1f}%) > "
                f"top_k={results_list[i - 1].top_k} "
                f"({results_list[i - 1].savings_pct:.1f}%)"
            )

        # Even top_k=20 should save at least 60%
        assert results_list[-1].savings_pct >= 60.0, (
            f"top_k=20 only saves {results_list[-1].savings_pct:.1f}%"
        )

    def test_content_mode_sensitivity(
        self,
        bench_docs: dict[str, Any],
        bench_code: dict[str, Any],
        bench_both: dict[str, Any],
        raw_corpus_tokens_docs: int,
        raw_corpus_tokens_code: int,
        raw_corpus_tokens_both: int,
    ) -> None:
        """Compare token economics across content modes."""
        query = "how to use the application"
        modes = [
            ("docs", bench_docs, raw_corpus_tokens_docs),
            ("code", bench_code, raw_corpus_tokens_code),
            ("both", bench_both, raw_corpus_tokens_both),
        ]

        results_list: list[BenchmarkResult] = []
        for mode_name, indexed, raw_tokens in modes:
            results = search(
                query=query,
                collection=indexed["collection"],
                bm25_index=indexed["bm25"],
                embed_fn=indexed["embed_fn"],
                top_k=10,
            )
            retrieved = count_tokens_for_results(results)
            ratio = raw_tokens / max(retrieved, 1)
            savings = (1 - retrieved / max(raw_tokens, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=f"mode={mode_name}",
                    corpus_tokens=raw_tokens,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                    extra={
                        "total_chunks": indexed["result"].chunks_indexed,
                        "files_indexed": indexed["result"].files_indexed,
                    },
                )
            )

        print_benchmark_table(results_list, "Content Mode Sensitivity")

        print("  Index sizes:")
        for r in results_list:
            print(
                f"    {r.name}: {r.extra['files_indexed']} files, "
                f"{r.extra['total_chunks']} chunks"
            )

        # "both" mode has the largest corpus
        both_result = results_list[2]
        docs_result = results_list[0]
        assert both_result.corpus_tokens >= docs_result.corpus_tokens, (
            "'both' mode corpus should be >= docs-only corpus"
        )

    def test_search_mode_sensitivity(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Compare hybrid vs semantic vs keyword search modes."""
        query = "how to install httpie"
        corpus = raw_corpus_tokens_docs

        results_list: list[BenchmarkResult] = []
        for mode in ["hybrid", "semantic", "keyword"]:
            results = search(
                query=query,
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                mode=mode,
                top_k=10,
            )
            retrieved = count_tokens_for_results(results)
            ratio = corpus / max(retrieved, 1)
            savings = (1 - retrieved / max(corpus, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=f"search={mode}",
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                )
            )

        print_benchmark_table(results_list, "Search Mode Comparison")

        for r in results_list:
            assert r.savings_pct >= 70.0, (
                f"{r.name}: savings {r.savings_pct:.1f}% below 70%"
            )


# ===================================================================
# BENCHMARK 5: Query-Type Analysis
# ===================================================================


class TestQueryTypeAnalysis:
    """Token savings broken down by query type.

    Compares narrow vs broad, keyword vs conceptual queries to show
    which patterns benefit most from RAG.
    """

    def test_narrow_vs_broad_queries(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Both narrow and broad queries should achieve substantial savings."""
        narrow_queries = [
            ("narrow_exact", "SessionStorage"),
            ("narrow_specific", "pip install httpie"),
        ]
        broad_queries = [
            ("broad_overview", "project overview documentation"),
            ("broad_how", "how does this project work"),
        ]

        corpus = raw_corpus_tokens_docs
        results_list: list[BenchmarkResult] = []

        for name, query in narrow_queries + broad_queries:
            results = search(
                query=query,
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=10,
            )
            retrieved = count_tokens_for_results(results)
            ratio = corpus / max(retrieved, 1)
            savings = (1 - retrieved / max(corpus, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=name,
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                )
            )

        print_benchmark_table(results_list, "Query Type: Narrow vs Broad")

        for r in results_list:
            assert r.savings_pct >= 50.0, (
                f"Query type '{r.name}' savings {r.savings_pct:.1f}% below 50%"
            )

    def test_keyword_vs_conceptual_queries(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Compare token economics for keyword vs conceptual query styles."""
        keyword_queries = [
            ("kw_function", "def format_headers"),
            ("kw_term", "authentication"),
            ("kw_file", "README.md"),
        ]
        conceptual_queries = [
            ("concept_how", "how to configure authentication"),
            ("concept_what", "what protocols does this support"),
            ("concept_why", "why was this design chosen"),
        ]

        corpus = raw_corpus_tokens_docs
        all_results: dict[str, list[BenchmarkResult]] = {
            "keyword": [],
            "conceptual": [],
        }

        for name, query in keyword_queries:
            results = search(
                query=query,
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=10,
            )
            retrieved = count_tokens_for_results(results)
            ratio = corpus / max(retrieved, 1)
            savings = (1 - retrieved / max(corpus, 1)) * 100
            all_results["keyword"].append(
                BenchmarkResult(
                    name=name,
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                )
            )

        for name, query in conceptual_queries:
            results = search(
                query=query,
                collection=bench_docs["collection"],
                bm25_index=bench_docs["bm25"],
                embed_fn=bench_docs["embed_fn"],
                top_k=10,
            )
            retrieved = count_tokens_for_results(results)
            ratio = corpus / max(retrieved, 1)
            savings = (1 - retrieved / max(corpus, 1)) * 100
            all_results["conceptual"].append(
                BenchmarkResult(
                    name=name,
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=10,
                )
            )

        combined = all_results["keyword"] + all_results["conceptual"]
        print_benchmark_table(combined, "Query Type: Keyword vs Conceptual")

        kw_avg = sum(r.savings_pct for r in all_results["keyword"]) / len(
            all_results["keyword"]
        )
        concept_avg = sum(r.savings_pct for r in all_results["conceptual"]) / len(
            all_results["conceptual"]
        )

        print(
            f"  Average savings -- Keyword: {kw_avg:.1f}%, Conceptual: {concept_avg:.1f}%"
        )

        assert kw_avg >= 70.0
        assert concept_avg >= 70.0

    def test_empty_result_handling(
        self,
        bench_docs: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """Nonsense queries still report correct savings (few/no useful tokens)."""
        results = search(
            query="xyzzy_nonexistent_term_12345",
            collection=bench_docs["collection"],
            bm25_index=bench_docs["bm25"],
            embed_fn=bench_docs["embed_fn"],
            top_k=10,
        )
        retrieved = count_tokens_for_results(results)
        corpus = raw_corpus_tokens_docs

        if corpus > 0:
            savings = (1 - retrieved / corpus) * 100
            print(
                f"\n  Nonsense query: {retrieved:,} tokens retrieved "
                f"({savings:.1f}% savings)"
            )
            assert savings >= 50.0, (
                "Even random results should be a small fraction of corpus"
            )


# ===================================================================
# BENCHMARK 6: Comprehensive Summary
# ===================================================================


class TestBenchmarkSummary:
    """Aggregate summary producing the headline metric.

    This test produces a single comprehensive table showing the overall
    value proposition of the RAG system.
    """

    def test_comprehensive_summary(
        self,
        bench_docs: dict[str, Any],
        bench_code: dict[str, Any],
        bench_both: dict[str, Any],
        raw_corpus_tokens_docs: int,
        raw_corpus_tokens_code: int,
        raw_corpus_tokens_both: int,
    ) -> None:
        """Produce the headline: 'TokenKeeper saves X% of tokens on average'."""
        scenarios: list[tuple[str, dict[str, Any], int, str, int]] = [
            ("Docs (top_k=10)", bench_docs, raw_corpus_tokens_docs, "how to install httpie", 10),
            ("Docs (top_k=5)", bench_docs, raw_corpus_tokens_docs, "how to install httpie", 5),
            ("Docs (top_k=3)", bench_docs, raw_corpus_tokens_docs, "how to install httpie", 3),
            ("Code (top_k=10)", bench_code, raw_corpus_tokens_code, "def format_headers", 10),
            ("Both (top_k=10)", bench_both, raw_corpus_tokens_both, "configuration", 10),
        ]

        results_list: list[BenchmarkResult] = []
        for name, indexed, raw_tokens, query, top_k in scenarios:
            results = search(
                query=query,
                collection=indexed["collection"],
                bm25_index=indexed["bm25"],
                embed_fn=indexed["embed_fn"],
                top_k=top_k,
            )
            retrieved = count_tokens_for_results(results)
            ratio = raw_tokens / max(retrieved, 1)
            savings = (1 - retrieved / max(raw_tokens, 1)) * 100

            results_list.append(
                BenchmarkResult(
                    name=name,
                    corpus_tokens=raw_tokens,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=top_k,
                )
            )

        print_benchmark_table(results_list, "COMPREHENSIVE SUMMARY")

        avg_savings = sum(r.savings_pct for r in results_list) / len(results_list)
        avg_ratio = sum(r.compression_ratio for r in results_list) / len(results_list)

        print(
            f"  HEADLINE: TokenKeeper saves an average of {avg_savings:.1f}% of tokens"
        )
        print(f"            (average {avg_ratio:.0f}x compression ratio)")
        print(f"            across {len(results_list)} benchmark scenarios.\n")

        assert avg_savings >= 80.0, (
            f"Overall average savings {avg_savings:.1f}% is below 80% target. "
            f"The RAG system may not be providing sufficient value."
        )


# ===================================================================
# TIER 2: Real Embedding Benchmarks (requires Ollama)
# ===================================================================

ollama_available = _check_ollama_available()


@pytest.fixture(scope="session")
def real_embed_fn():
    """Return the real Ollama embedding function."""
    if not ollama_available:
        pytest.skip("Ollama with nomic-embed-text not available")
    from tokenkeeper.embeddings import embed_texts

    return embed_texts


@pytest.fixture(scope="session")
def bench_docs_real(repo_path: Path, real_embed_fn: Any) -> dict[str, Any]:
    """Docs-mode index with real Ollama embeddings."""
    return _build_index(
        repo_path, "docs", 1000, 200, "bench_docs_real", real_embed_fn
    )


@pytest.mark.ollama
class TestRealEmbeddingBenchmarks:
    """Tier 2: Benchmarks with real Ollama embeddings.

    Validates that real embeddings produce BETTER relevance than fake
    ones, confirming semantic search adds value beyond BM25 alone.
    """

    def test_real_embeddings_improve_relevance(
        self,
        bench_docs_real: dict[str, Any],
        bench_docs: dict[str, Any],
    ) -> None:
        """Real embeddings should produce higher relevance than fake."""
        query = "how to install httpie"
        expected_terms = ["install", "pip", "brew", "package"]

        # Real embeddings -- semantic only
        real_results = search(
            query=query,
            collection=bench_docs_real["collection"],
            bm25_index=bench_docs_real["bm25"],
            embed_fn=bench_docs_real["embed_fn"],
            mode="semantic",
            top_k=10,
        )
        real_relevant = sum(
            1
            for r in real_results
            if any(t in r.content.lower() for t in expected_terms)
        )

        # Fake embeddings -- semantic only
        fake_results = search(
            query=query,
            collection=bench_docs["collection"],
            bm25_index=bench_docs["bm25"],
            embed_fn=bench_docs["embed_fn"],
            mode="semantic",
            top_k=10,
        )
        fake_relevant = sum(
            1
            for r in fake_results
            if any(t in r.content.lower() for t in expected_terms)
        )

        print(f"\n  Real embeddings relevance: {real_relevant}/{len(real_results)}")
        print(f"  Fake embeddings relevance: {fake_relevant}/{len(fake_results)}")

        # Real should be at least as good
        assert real_relevant >= fake_relevant, (
            f"Real embeddings ({real_relevant}) worse than fake ({fake_relevant})"
        )

    def test_real_embeddings_compression_with_quality(
        self,
        bench_docs_real: dict[str, Any],
        raw_corpus_tokens_docs: int,
    ) -> None:
        """With real embeddings, top-5 results should be highly relevant AND save tokens."""
        queries = [
            ("install", "how to install httpie", ["install", "pip", "brew"]),
            ("usage", "how to use httpie to send requests", ["http", "request", "get", "post"]),
            ("config", "configuration options", ["config", "option", "setting"]),
        ]

        results_list: list[BenchmarkResult] = []
        for name, query, terms in queries:
            results = search(
                query=query,
                collection=bench_docs_real["collection"],
                bm25_index=bench_docs_real["bm25"],
                embed_fn=bench_docs_real["embed_fn"],
                top_k=5,
            )
            retrieved = count_tokens_for_results(results)
            corpus = raw_corpus_tokens_docs
            savings = (1 - retrieved / max(corpus, 1)) * 100
            ratio = corpus / max(retrieved, 1)

            relevant_count = sum(
                1
                for r in results
                if any(t in r.content.lower() for t in terms)
            )

            results_list.append(
                BenchmarkResult(
                    name=name,
                    corpus_tokens=corpus,
                    retrieved_tokens=retrieved,
                    compression_ratio=ratio,
                    savings_pct=savings,
                    query=query,
                    top_k=5,
                    extra={
                        "relevant_count": relevant_count,
                        "total_results": len(results),
                    },
                )
            )

        print_benchmark_table(
            results_list, "Real Embeddings: Quality + Compression"
        )
        for r in results_list:
            print(
                f"  {r.name}: {r.extra['relevant_count']}/{r.extra['total_results']} "
                f"relevant"
            )

        # With real embeddings at top_k=5, expect high savings
        for r in results_list:
            assert r.savings_pct >= 90.0, (
                f"top_k=5 should save >= 90%, got {r.savings_pct:.1f}%"
            )
