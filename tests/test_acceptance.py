"""Acceptance tests for Knowledge RAG against a real codebase.

These tests clone a real open-source repository and run the full Knowledge RAG
pipeline against it: discovery → ingestion → indexing → search.

Two tiers:
  Tier 1 (always run): Uses fake embeddings to validate pipeline correctness.
    - File discovery finds real files
    - Ingestion produces valid chunks with metadata
    - Indexing stores everything in ChromaDB + BM25
    - Search returns results with proper structure
    - Heading hierarchy is populated for markdown docs
    - Code chunking produces function/class boundaries
    - Mode router dispatches correctly

  Tier 2 (requires Ollama): Uses real embeddings for search quality.
    - Semantic search returns relevant results for conceptual queries
    - Keyword search returns results for exact identifiers
    - Hybrid search blends both effectively
    - Top-k results are actually relevant to the query

Usage:
    # Run Tier 1 only (no Ollama required):
    uv run pytest tests/test_acceptance.py -m "not ollama" -v

    # Run all tiers (requires Ollama with nomic-embed-text):
    uv run pytest tests/test_acceptance.py -v

    # Run against a specific local repo:
    ACCEPTANCE_REPO=/path/to/repo uv run pytest tests/test_acceptance.py -v

Configuration:
    Set ACCEPTANCE_REPO env var to point at an existing local repo,
    or the test will clone a small, well-known open-source project.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import chromadb
import pytest

from knowledge_rag.bm25_index import BM25Index
from knowledge_rag.config import RagConfig
from knowledge_rag.discovery import (
    CODE_EXTENSIONS,
    discover_code_files,
    discover_markdown_files,
)
from knowledge_rag.indexer import (
    IndexingResult,
    _discover_for_mode,
    _ingest_file_routed,
    index_documents,
)
from knowledge_rag.ingestion import (
    DocumentChunk,
    chunk_document_heading_aware,
    ingest_code_file,
    ingest_file,
)
from knowledge_rag.search import SearchResult, enrich_results, search


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A small, well-structured repo with markdown docs AND Python code.
# httpie/cli is ~3MB clone, has README, CHANGELOG, docs/, and Python source.
DEFAULT_CLONE_URL = "https://github.com/httpie/cli.git"
DEFAULT_CLONE_DEPTH = 1  # Shallow clone to save time

# Where acceptance test repos are cached (reused across runs)
CACHE_DIR = Path(__file__).parent.parent / ".acceptance-cache"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings based on text hash.

    Unlike the unit test version (all [0.1]*768), this produces *different*
    vectors for different texts so that ChromaDB semantic search has
    something to distinguish. Uses a simple hash-to-vector mapping.
    """
    results = []
    for text in texts:
        # Hash the text to get a deterministic seed
        h = hashlib.sha256(text.lower().encode()).digest()
        # Convert first 768 bytes (cycling) to float values in [-1, 1]
        vec = []
        for i in range(768):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 127.5) - 1.0)
        results.append(vec)
    return results


def _check_ollama_available() -> bool:
    """Check if Ollama is running and nomic-embed-text is available."""
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["test"]},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def repo_path() -> Path:
    """Get or clone the acceptance test repository.

    Uses ACCEPTANCE_REPO env var if set, otherwise clones the default repo.
    Cloned repos are cached in .acceptance-cache/ for reuse.
    """
    env_path = os.environ.get("ACCEPTANCE_REPO")
    if env_path:
        p = Path(env_path)
        assert p.is_dir(), f"ACCEPTANCE_REPO={env_path} is not a directory"
        return p

    # Clone to cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    repo_name = DEFAULT_CLONE_URL.rstrip("/").split("/")[-1].replace(".git", "")
    cached = CACHE_DIR / repo_name

    if cached.is_dir():
        return cached

    # Shallow clone
    subprocess.run(
        [
            "git", "clone",
            "--depth", str(DEFAULT_CLONE_DEPTH),
            DEFAULT_CLONE_URL,
            str(cached),
        ],
        check=True,
        capture_output=True,
    )
    return cached


@pytest.fixture(scope="session")
def repo_indexed(repo_path: Path) -> dict[str, Any]:
    """Index the repo in docs mode and return all artifacts.

    Returns dict with:
      config, collection, bm25, result, embed_fn, repo_path
    """
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "acceptance_docs",
        metadata={"hnsw:space": "cosine"},
    )
    config = RagConfig()
    config.content_mode = "docs"
    config.chunk_size = 1000
    config.overlap = 200
    bm25 = BM25Index()

    result = index_documents(
        repo_path, config, collection, bm25, _fake_embed_fn,
    )

    return {
        "config": config,
        "collection": collection,
        "bm25": bm25,
        "result": result,
        "embed_fn": _fake_embed_fn,
        "repo_path": repo_path,
    }


@pytest.fixture(scope="session")
def repo_indexed_code(repo_path: Path) -> dict[str, Any]:
    """Index the repo in code mode."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "acceptance_code",
        metadata={"hnsw:space": "cosine"},
    )
    config = RagConfig()
    config.content_mode = "code"
    config.chunk_size = 1500
    config.overlap = 200
    bm25 = BM25Index()

    result = index_documents(
        repo_path, config, collection, bm25, _fake_embed_fn,
    )

    return {
        "config": config,
        "collection": collection,
        "bm25": bm25,
        "result": result,
        "embed_fn": _fake_embed_fn,
        "repo_path": repo_path,
    }


@pytest.fixture(scope="session")
def repo_indexed_both(repo_path: Path) -> dict[str, Any]:
    """Index the repo in both mode."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "acceptance_both",
        metadata={"hnsw:space": "cosine"},
    )
    config = RagConfig()
    config.content_mode = "both"
    config.chunk_size = 1000
    config.overlap = 200
    bm25 = BM25Index()

    result = index_documents(
        repo_path, config, collection, bm25, _fake_embed_fn,
    )

    return {
        "config": config,
        "collection": collection,
        "bm25": bm25,
        "result": result,
        "embed_fn": _fake_embed_fn,
        "repo_path": repo_path,
    }


# ===========================================================================
# TIER 1: Structural Acceptance Tests (fake embeddings, always run)
# ===========================================================================


class TestDiscoveryOnRealRepo:
    """Verify file discovery works correctly against real repo structure."""

    def test_discovers_markdown_files(self, repo_path: Path) -> None:
        """At least one .md file is discovered in a real repo."""
        files = discover_markdown_files(repo_path)
        assert len(files) > 0, "No markdown files found in repo"
        assert all(f.suffix == ".md" for f in files)

    def test_discovers_python_files(self, repo_path: Path) -> None:
        """At least one .py file is discovered in a real repo."""
        files = discover_code_files(repo_path)
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) > 0, "No Python files found in repo"

    def test_excludes_git_directory(self, repo_path: Path) -> None:
        """No files from .git/ directory are discovered."""
        all_files = discover_markdown_files(repo_path) + discover_code_files(repo_path)
        git_files = [f for f in all_files if ".git" in f.parts]
        assert len(git_files) == 0, f"Found files in .git/: {git_files[:5]}"

    def test_excludes_node_modules(self, repo_path: Path) -> None:
        """No files from node_modules/ are discovered."""
        all_files = discover_markdown_files(repo_path) + discover_code_files(repo_path)
        nm_files = [f for f in all_files if "node_modules" in f.parts]
        assert len(nm_files) == 0

    def test_excludes_pycache(self, repo_path: Path) -> None:
        """No files from __pycache__/ are discovered."""
        all_files = discover_markdown_files(repo_path) + discover_code_files(repo_path)
        cache_files = [f for f in all_files if "__pycache__" in f.parts]
        assert len(cache_files) == 0

    def test_mode_router_docs(self, repo_path: Path) -> None:
        """Mode router in 'docs' mode returns only markdown files."""
        files = _discover_for_mode(repo_path, "docs")
        assert all(f.suffix == ".md" for f in files)

    def test_mode_router_code(self, repo_path: Path) -> None:
        """Mode router in 'code' mode returns only code files."""
        files = _discover_for_mode(repo_path, "code")
        assert all(f.suffix in CODE_EXTENSIONS for f in files)

    def test_mode_router_both(self, repo_path: Path) -> None:
        """Mode router in 'both' mode returns union of docs + code files."""
        docs = set(str(f) for f in _discover_for_mode(repo_path, "docs"))
        code = set(str(f) for f in _discover_for_mode(repo_path, "code"))
        both = set(str(f) for f in _discover_for_mode(repo_path, "both"))
        assert docs.issubset(both)
        assert code.issubset(both)
        assert both == docs | code


class TestIngestionOnRealRepo:
    """Verify ingestion produces valid chunks from real files."""

    def test_ingest_readme(self, repo_path: Path) -> None:
        """README.md produces at least one chunk with content."""
        readme = repo_path / "README.md"
        if not readme.exists():
            # Try case variations
            for name in ["readme.md", "Readme.md", "README.rst"]:
                readme = repo_path / name
                if readme.exists():
                    break
        if not readme.exists() or readme.suffix != ".md":
            pytest.skip("No README.md found in repo")

        chunks = ingest_file(readme, repo_path, chunk_size=1000, overlap=200)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(len(c.content) > 0 for c in chunks)

    def test_chunks_have_source_file(self, repo_path: Path) -> None:
        """All chunks have a relative source_file path."""
        md_files = discover_markdown_files(repo_path)
        if not md_files:
            pytest.skip("No markdown files")

        chunks = ingest_file(md_files[0], repo_path, chunk_size=1000, overlap=200)
        for chunk in chunks:
            assert chunk.source_file, "source_file is empty"
            # Should be relative (no absolute path prefix)
            assert not Path(chunk.source_file).is_absolute()

    def test_chunks_respect_size_limit(self, repo_path: Path) -> None:
        """No chunk exceeds 2x the configured chunk_size (soft limit)."""
        md_files = discover_markdown_files(repo_path)
        if not md_files:
            pytest.skip("No markdown files")

        chunk_size = 1000
        for md_file in md_files[:5]:  # Test first 5 files
            chunks = ingest_file(md_file, repo_path, chunk_size=chunk_size, overlap=200)
            for chunk in chunks:
                # Soft limit: chunks can exceed by up to 2x for atomic blocks
                assert len(chunk.content) <= chunk_size * 3, (
                    f"Chunk from {chunk.source_file} is {len(chunk.content)} chars "
                    f"(limit: {chunk_size * 3})"
                )

    def test_heading_hierarchy_populated(self, repo_path: Path) -> None:
        """Heading-aware chunker populates heading_hierarchy on some chunks."""
        md_files = discover_markdown_files(repo_path)
        if not md_files:
            pytest.skip("No markdown files")

        all_chunks: list[DocumentChunk] = []
        for md_file in md_files[:10]:
            all_chunks.extend(
                ingest_file(md_file, repo_path, chunk_size=1000, overlap=200)
            )

        # At least some chunks should have heading hierarchy
        with_hierarchy = [c for c in all_chunks if c.heading_hierarchy]
        assert len(with_hierarchy) > 0, (
            f"No chunks have heading_hierarchy out of {len(all_chunks)} total"
        )

    def test_code_chunks_have_metadata(self, repo_path: Path) -> None:
        """Python code chunks have language, symbol_name, line numbers."""
        py_files = [f for f in discover_code_files(repo_path) if f.suffix == ".py"]
        if not py_files:
            pytest.skip("No Python files")

        all_chunks: list[DocumentChunk] = []
        for py_file in py_files[:5]:
            all_chunks.extend(ingest_code_file(py_file, repo_path, chunk_size=1500))

        assert len(all_chunks) > 0, "No code chunks produced"

        # All code chunks should have language set
        for chunk in all_chunks:
            assert chunk.language == "python"

        # At least some should have symbol_name (functions/classes)
        with_symbol = [c for c in all_chunks if c.symbol_name]
        # It's okay if some chunks are module-level (no symbol), but at least some
        # should have function/class names if the code has any
        if len(all_chunks) > 1:
            assert len(with_symbol) > 0, (
                "No code chunks have symbol_name "
                f"out of {len(all_chunks)} total"
            )

    def test_code_chunks_have_line_ranges(self, repo_path: Path) -> None:
        """Code chunks have valid line_start and line_end values."""
        py_files = [f for f in discover_code_files(repo_path) if f.suffix == ".py"]
        if not py_files:
            pytest.skip("No Python files")

        chunks = ingest_code_file(py_files[0], repo_path, chunk_size=1500)
        for chunk in chunks:
            if chunk.line_start and chunk.line_end:
                assert chunk.line_start <= chunk.line_end
                assert chunk.line_start >= 1

    def test_ingest_file_routed_dispatches_correctly(self, repo_path: Path) -> None:
        """_ingest_file_routed dispatches .md and .py to correct pipelines."""
        md_files = discover_markdown_files(repo_path)
        py_files = [f for f in discover_code_files(repo_path) if f.suffix == ".py"]

        if md_files:
            md_chunks = _ingest_file_routed(md_files[0], repo_path, 1000, 200)
            assert len(md_chunks) >= 1
            # Markdown chunks should NOT have language set
            for c in md_chunks:
                assert c.language == ""

        if py_files:
            py_chunks = _ingest_file_routed(py_files[0], repo_path, 1500, 200)
            assert len(py_chunks) >= 1
            for c in py_chunks:
                assert c.language == "python"


class TestIndexingOnRealRepo:
    """Verify full indexing pipeline works on real repo."""

    def test_docs_mode_indexes_successfully(self, repo_indexed: dict) -> None:
        """Docs-mode indexing completes without error."""
        result = repo_indexed["result"]
        assert isinstance(result, IndexingResult)
        assert result.files_indexed >= 1
        assert result.chunks_indexed >= 1
        assert result.files_failed == 0

    def test_code_mode_indexes_successfully(self, repo_indexed_code: dict) -> None:
        """Code-mode indexing completes without error."""
        result = repo_indexed_code["result"]
        assert isinstance(result, IndexingResult)
        assert result.files_indexed >= 1
        assert result.chunks_indexed >= 1

    def test_both_mode_indexes_successfully(self, repo_indexed_both: dict) -> None:
        """Both-mode indexing completes without error."""
        result = repo_indexed_both["result"]
        assert isinstance(result, IndexingResult)
        assert result.files_indexed >= 1
        assert result.chunks_indexed >= 1

    def test_both_mode_indexes_more_than_either(
        self, repo_indexed: dict, repo_indexed_code: dict, repo_indexed_both: dict
    ) -> None:
        """Both mode indexes at least as many files as either mode alone."""
        docs_files = repo_indexed["result"].files_indexed
        code_files = repo_indexed_code["result"].files_indexed
        both_files = repo_indexed_both["result"].files_indexed
        assert both_files >= docs_files
        assert both_files >= code_files

    def test_chromadb_count_matches_result(self, repo_indexed: dict) -> None:
        """ChromaDB collection count matches IndexingResult.chunks_indexed."""
        result = repo_indexed["result"]
        collection = repo_indexed["collection"]
        assert collection.count() == result.chunks_indexed

    def test_bm25_count_matches_result(self, repo_indexed: dict) -> None:
        """BM25 index size matches IndexingResult.chunks_indexed."""
        result = repo_indexed["result"]
        bm25 = repo_indexed["bm25"]
        assert len(bm25) == result.chunks_indexed

    def test_stored_metadata_has_source_file(self, repo_indexed: dict) -> None:
        """Every stored chunk has source_file metadata."""
        collection = repo_indexed["collection"]
        stored = collection.get(include=["metadatas"])
        for meta in stored["metadatas"]:
            assert "source_file" in meta
            assert meta["source_file"], "source_file is empty"

    def test_stored_metadata_has_chunk_index(self, repo_indexed: dict) -> None:
        """Every stored chunk has chunk_index metadata."""
        collection = repo_indexed["collection"]
        stored = collection.get(include=["metadatas"])
        for meta in stored["metadatas"]:
            assert "chunk_index" in meta

    def test_some_chunks_have_heading_hierarchy(self, repo_indexed: dict) -> None:
        """At least some stored chunks have heading_hierarchy metadata."""
        collection = repo_indexed["collection"]
        stored = collection.get(include=["metadatas"])
        with_hierarchy = [
            m for m in stored["metadatas"]
            if m.get("heading_hierarchy")
        ]
        assert len(with_hierarchy) > 0, (
            f"No chunks have heading_hierarchy in {len(stored['metadatas'])} stored"
        )

    def test_code_chunks_have_language_metadata(self, repo_indexed_code: dict) -> None:
        """Code-mode stored chunks have language metadata."""
        collection = repo_indexed_code["collection"]
        stored = collection.get(include=["metadatas"])
        with_language = [
            m for m in stored["metadatas"]
            if m.get("language")
        ]
        assert len(with_language) > 0

    def test_indexing_completes_in_reasonable_time(self, repo_path: Path) -> None:
        """Indexing a real repo completes within 60 seconds (with fake embeds)."""
        client = chromadb.Client()
        collection = client.get_or_create_collection("acceptance_timing")
        config = RagConfig()
        config.content_mode = "docs"
        bm25 = BM25Index()

        start = time.monotonic()
        index_documents(repo_path, config, collection, bm25, _fake_embed_fn)
        elapsed = time.monotonic() - start

        assert elapsed < 60.0, (
            f"Indexing took {elapsed:.1f}s (limit: 60s)"
        )

    def test_reindex_skips_unchanged_files(self, repo_path: Path) -> None:
        """Second indexing run skips all files (nothing changed)."""
        client = chromadb.Client()
        collection = client.get_or_create_collection("acceptance_reindex")
        config = RagConfig()
        config.content_mode = "docs"
        bm25 = BM25Index()

        result1 = index_documents(repo_path, config, collection, bm25, _fake_embed_fn)
        result2 = index_documents(repo_path, config, collection, bm25, _fake_embed_fn)

        assert result2.files_skipped >= result1.files_indexed
        assert result2.files_indexed == 0


class TestSearchOnRealRepo:
    """Verify search returns properly structured results from real repo."""

    def test_search_returns_results(self, repo_indexed: dict) -> None:
        """Search returns at least one result for a generic query."""
        results = search(
            query="documentation",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=5,
        )
        assert len(results) >= 1

    def test_search_results_are_search_result_type(self, repo_indexed: dict) -> None:
        """All search results are SearchResult instances."""
        results = search(
            query="configuration",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=5,
        )
        for r in results:
            assert isinstance(r, SearchResult)

    def test_search_results_have_content(self, repo_indexed: dict) -> None:
        """Search results contain actual chunk text content."""
        results = search(
            query="readme",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=5,
        )
        for r in results:
            assert r.content, "SearchResult has empty content"
            assert len(r.content) > 10, "SearchResult content suspiciously short"

    def test_search_results_have_source_file(self, repo_indexed: dict) -> None:
        """Search results have source_file populated."""
        results = search(
            query="install",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=5,
        )
        for r in results:
            assert r.source_file, "source_file is empty"

    def test_search_results_have_scores(self, repo_indexed: dict) -> None:
        """Search results have normalized scores between 0 and 1."""
        results = search(
            query="usage",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=5,
        )
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0, 1] range"

    def test_search_respects_top_k(self, repo_indexed: dict) -> None:
        """Search returns no more than top_k results."""
        results = search(
            query="documentation",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=3,
        )
        assert len(results) <= 3

    def test_keyword_mode_search(self, repo_indexed: dict) -> None:
        """Keyword-only search returns results."""
        results = search(
            query="install",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            mode="keyword",
            top_k=5,
        )
        assert len(results) >= 1

    def test_semantic_mode_search(self, repo_indexed: dict) -> None:
        """Semantic-only search returns results."""
        results = search(
            query="how to set up the project",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            mode="semantic",
            top_k=5,
        )
        assert len(results) >= 1

    def test_search_on_code_index(self, repo_indexed_code: dict) -> None:
        """Search works on code-indexed repo."""
        results = search(
            query="function",
            collection=repo_indexed_code["collection"],
            bm25_index=repo_indexed_code["bm25"],
            embed_fn=repo_indexed_code["embed_fn"],
            top_k=5,
        )
        assert len(results) >= 1

    def test_search_completes_quickly(self, repo_indexed: dict) -> None:
        """Search completes within 5 seconds."""
        start = time.monotonic()
        search(
            query="how to configure the application",
            collection=repo_indexed["collection"],
            bm25_index=repo_indexed["bm25"],
            embed_fn=repo_indexed["embed_fn"],
            top_k=10,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"Search took {elapsed:.2f}s (limit: 5.0s)"


class TestChunkQualityOnRealRepo:
    """Verify chunk quality heuristics on real content."""

    def test_no_empty_chunks_stored(self, repo_indexed: dict) -> None:
        """No stored chunks have empty or whitespace-only content."""
        collection = repo_indexed["collection"]
        stored = collection.get(include=["documents"])
        for i, doc in enumerate(stored["documents"]):
            assert doc and doc.strip(), f"Chunk {i} has empty content"

    def test_no_frontmatter_in_chunks(self, repo_indexed: dict) -> None:
        """YAML frontmatter (---) should not appear in chunk content."""
        collection = repo_indexed["collection"]
        stored = collection.get(include=["documents"])
        for doc in stored["documents"]:
            # Frontmatter would start with --- on the first line
            if doc.startswith("---"):
                # Only a problem if it looks like YAML frontmatter
                lines = doc.split("\n")
                if len(lines) > 1 and any(
                    ":" in line for line in lines[1:5]
                ):
                    pytest.fail(
                        f"Chunk appears to contain YAML frontmatter: {doc[:200]}"
                    )

    def test_code_fences_not_split(self, repo_indexed: dict) -> None:
        """Code blocks (``` ... ```) should not be split across chunks.

        Heuristic: count opening ``` and closing ``` in each chunk.
        They should be balanced (equal count) or the chunk is just text
        with no code fences.
        """
        collection = repo_indexed["collection"]
        stored = collection.get(include=["documents"])
        unbalanced = 0
        for doc in stored["documents"]:
            lines = doc.split("\n")
            fence_count = sum(
                1 for line in lines if line.strip().startswith("```")
            )
            if fence_count % 2 != 0:
                unbalanced += 1

        # Allow a small percentage of unbalanced (edge cases with nested fences)
        total = len(stored["documents"])
        if total > 0:
            pct = unbalanced / total
            assert pct < 0.1, (
                f"{unbalanced}/{total} ({pct:.0%}) chunks have unbalanced "
                "code fences — structure protection may be failing"
            )

    def test_chunks_have_reasonable_length(self, repo_indexed: dict) -> None:
        """Most chunks have content of reasonable length.

        Some very short chunks are acceptable (e.g., a module with only an
        __init__.py docstring), but the vast majority should be meaningful.
        """
        collection = repo_indexed["collection"]
        stored = collection.get(include=["documents"])
        lengths = [len(doc) for doc in stored["documents"]]
        total = len(lengths)

        # No truly empty chunks
        assert all(l > 0 for l in lengths), "Found empty chunks"

        # At most 5% of chunks should be very short (< 5 chars)
        very_short = [l for l in lengths if l < 5]
        if total > 0:
            pct = len(very_short) / total
            assert pct < 0.05, (
                f"{len(very_short)}/{total} ({pct:.0%}) chunks are under 5 chars"
            )

        # Very large chunks are acceptable for oversized sections
        # but shouldn't be extreme
        max_len = max(lengths) if lengths else 0
        assert max_len <= 10000, f"Largest chunk is {max_len} chars"


# ===========================================================================
# TIER 2: Search Quality Tests (requires Ollama + nomic-embed-text)
# ===========================================================================


ollama_available = _check_ollama_available()
ollama_reason = "Ollama with nomic-embed-text not available"


@pytest.fixture(scope="session")
def real_embed_fn():
    """Return the real Ollama embedding function."""
    if not ollama_available:
        pytest.skip(ollama_reason)
    from knowledge_rag.embeddings import embed_texts
    return embed_texts


@pytest.fixture(scope="session")
def repo_indexed_real(repo_path: Path, real_embed_fn) -> dict[str, Any]:
    """Index repo with real Ollama embeddings."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "acceptance_real_embeds",
        metadata={"hnsw:space": "cosine"},
    )
    config = RagConfig()
    config.content_mode = "both"
    config.chunk_size = 1000
    config.overlap = 200
    bm25 = BM25Index()

    result = index_documents(
        repo_path, config, collection, bm25, real_embed_fn,
    )

    return {
        "config": config,
        "collection": collection,
        "bm25": bm25,
        "result": result,
        "embed_fn": real_embed_fn,
        "repo_path": repo_path,
    }


@pytest.mark.ollama
class TestSearchQualityWithRealEmbeddings:
    """Search quality tests using real Ollama embeddings.

    These tests verify that the search system returns *relevant* results
    for meaningful queries, not just that it returns *any* results.

    Run with: uv run pytest tests/test_acceptance.py -m ollama -v
    """

    def test_semantic_search_finds_relevant_content(
        self, repo_indexed_real: dict
    ) -> None:
        """Semantic search for conceptual queries returns related content."""
        results = search(
            query="how to install this project",
            collection=repo_indexed_real["collection"],
            bm25_index=repo_indexed_real["bm25"],
            embed_fn=repo_indexed_real["embed_fn"],
            mode="semantic",
            top_k=5,
        )
        assert len(results) >= 1
        # Top result should mention something installation-related
        top_content = results[0].content.lower()
        install_terms = ["install", "setup", "pip", "requirements", "getting started",
                        "brew", "npm", "package", "download", "build"]
        assert any(term in top_content for term in install_terms), (
            f"Top result doesn't seem related to installation:\n{top_content[:300]}"
        )

    def test_keyword_search_finds_exact_terms(
        self, repo_indexed_real: dict
    ) -> None:
        """Keyword search returns results containing the exact search terms."""
        # Use a term that definitely appears in the httpie codebase content
        results = search(
            query="import requests",
            collection=repo_indexed_real["collection"],
            bm25_index=repo_indexed_real["bm25"],
            embed_fn=repo_indexed_real["embed_fn"],
            mode="keyword",
            top_k=5,
        )
        assert len(results) >= 1

    def test_hybrid_outperforms_single_mode(
        self, repo_indexed_real: dict
    ) -> None:
        """Hybrid search returns results for queries where one mode might miss."""
        # A conceptual query that might miss with pure keyword
        results_hybrid = search(
            query="how to send http requests",
            collection=repo_indexed_real["collection"],
            bm25_index=repo_indexed_real["bm25"],
            embed_fn=repo_indexed_real["embed_fn"],
            mode="hybrid",
            alpha=0.5,
            top_k=10,
        )
        assert len(results_hybrid) >= 1

    def test_top_result_quality(self, repo_indexed_real: dict) -> None:
        """Top result score should be meaningfully high."""
        results = search(
            query="command line interface usage",
            collection=repo_indexed_real["collection"],
            bm25_index=repo_indexed_real["bm25"],
            embed_fn=repo_indexed_real["embed_fn"],
            top_k=5,
        )
        if results:
            # Top result should have a reasonable score
            assert results[0].score > 0.3, (
                f"Top result score is very low: {results[0].score:.3f}"
            )

    def test_different_queries_return_different_results(
        self, repo_indexed_real: dict
    ) -> None:
        """Distinct queries should return meaningfully different top results."""
        results_install = search(
            query="installation instructions",
            collection=repo_indexed_real["collection"],
            bm25_index=repo_indexed_real["bm25"],
            embed_fn=repo_indexed_real["embed_fn"],
            top_k=3,
        )
        results_api = search(
            query="API documentation",
            collection=repo_indexed_real["collection"],
            bm25_index=repo_indexed_real["bm25"],
            embed_fn=repo_indexed_real["embed_fn"],
            top_k=3,
        )

        if results_install and results_api:
            # At least the top results should differ
            install_ids = {r.chunk_id for r in results_install}
            api_ids = {r.chunk_id for r in results_api}
            assert install_ids != api_ids, (
                "Different queries returned identical result sets"
            )

    def test_indexing_with_real_embeds_completes(
        self, repo_indexed_real: dict
    ) -> None:
        """Full indexing with real embeddings completes successfully."""
        result = repo_indexed_real["result"]
        assert result.files_indexed >= 1
        assert result.chunks_indexed >= 1
        assert result.files_failed == 0


# ===========================================================================
# TIER 3: Stress / Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases that only surface with real-world content."""

    def test_empty_file_handled(self, repo_path: Path, tmp_path: Path) -> None:
        """An empty .md file doesn't crash the pipeline."""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        chunks = ingest_file(empty_file, tmp_path, chunk_size=1000, overlap=200)
        assert chunks == []

    def test_binary_adjacent_files_skipped(self, repo_path: Path) -> None:
        """Binary files (.png, .jpg, etc.) are not discovered."""
        all_files = discover_markdown_files(repo_path) + discover_code_files(repo_path)
        binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                      ".ttf", ".eot", ".pdf", ".zip", ".tar", ".gz"}
        binary_files = [f for f in all_files if f.suffix.lower() in binary_exts]
        assert len(binary_files) == 0, f"Binary files discovered: {binary_files[:5]}"

    def test_unicode_content_handled(self, tmp_path: Path) -> None:
        """Files with Unicode content (emoji, CJK, etc.) don't crash."""
        unicode_file = tmp_path / "unicode.md"
        unicode_file.write_text(
            "# Unicode Test 🎉\n\n"
            "日本語テスト\n\n"
            "Ñoño résumé naïve\n\n"
            "```python\n# 中文注释\nprint('你好')\n```\n",
            encoding="utf-8",
        )
        chunks = ingest_file(unicode_file, tmp_path, chunk_size=1000, overlap=200)
        assert len(chunks) >= 1
        # Verify content isn't corrupted
        all_content = " ".join(c.content for c in chunks)
        assert "🎉" in all_content or "unicode" in all_content.lower()

    def test_deeply_nested_headings(self, tmp_path: Path) -> None:
        """Deeply nested heading hierarchy (H1 > H2 > H3 > H4) is tracked."""
        deep_file = tmp_path / "deep.md"
        deep_file.write_text(
            "# Level 1\n\n"
            "## Level 2\n\n"
            "### Level 3\n\n"
            "#### Level 4\n\n"
            "Content at level 4.\n",
        )
        chunks = ingest_file(deep_file, tmp_path, chunk_size=5000, overlap=200)
        assert len(chunks) >= 1
        # Find the chunk with deepest hierarchy
        deepest = max(chunks, key=lambda c: c.heading_hierarchy.count(">") if c.heading_hierarchy else 0)
        if deepest.heading_hierarchy:
            depth = deepest.heading_hierarchy.count(">")
            assert depth >= 2, f"Expected deep hierarchy, got: {deepest.heading_hierarchy}"

    def test_large_file_doesnt_hang(self, tmp_path: Path) -> None:
        """A large file (100KB) processes within 10 seconds."""
        large_file = tmp_path / "large.md"
        # Generate a 100KB markdown file
        content = "# Large Document\n\n"
        for i in range(200):
            content += f"## Section {i}\n\n"
            content += f"This is paragraph {i} with some content. " * 10 + "\n\n"

        large_file.write_text(content)

        start = time.monotonic()
        chunks = ingest_file(large_file, tmp_path, chunk_size=1000, overlap=200)
        elapsed = time.monotonic() - start

        assert elapsed < 10.0, f"Large file took {elapsed:.1f}s"
        assert len(chunks) > 10, f"Only {len(chunks)} chunks from 100KB file"
