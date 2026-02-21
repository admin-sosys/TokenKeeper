"""Integration tests for Phase 6 (MCP Server Core) and Phase 7 (MCP Management Tools).

Tests the full pipeline: markdown files on disk -> lifespan initialization
-> indexing -> search -> formatted results.

Uses real ChromaDB at tmp_path with mocked health checks and smoke test
(no Ollama needed). Embedding is faked with deterministic hash-based vectors.

NOTE: FastMCP's @mcp.tool decorator wraps functions in FunctionTool objects,
so we test the underlying search pipeline directly using the lifespan data
rather than calling the tool wrappers.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tokenkeeper.server import _get_lifespan_data, _rebuild_bm25_from_metadata, app_lifespan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings for testing (no Ollama needed)."""
    result: list[list[float]] = []
    for text in texts:
        seed = hash(text.lower()) % 10000
        vec = [(seed + i) / 10000.0 for i in range(768)]
        result.append(vec)
    return result


def _create_test_project(tmp_path: Path) -> Path:
    """Create a test project with markdown files for indexing."""
    project = tmp_path / "project"
    project.mkdir()

    # File 1: Authentication docs
    (project / "auth.md").write_text(
        "---\ntitle: Authentication\ntags:\n  - auth\n  - security\n---\n\n"
        "Authentication is configured via environment variables. "
        "Set AUTH_SECRET and AUTH_PROVIDER in your .env file. "
        "The system supports OAuth2 with Google and GitHub providers. "
        "JWT tokens expire after 24 hours by default.",
        encoding="utf-8",
    )

    # File 2: Database setup
    (project / "database.md").write_text(
        "---\ntitle: Database Setup\ntags:\n  - database\n  - postgresql\n---\n\n"
        "The database uses PostgreSQL with connection pooling. "
        "Configure DATABASE_URL in .env. Pool size defaults to 10 connections. "
        "Use Alembic for database migrations.",
        encoding="utf-8",
    )

    # File 3: API endpoints
    (project / "api.md").write_text(
        "---\ntitle: API Endpoints\ntags:\n  - api\n  - rest\n---\n\n"
        "REST API endpoints follow the /api/v1 prefix convention. "
        "Authentication is required for all endpoints except /health. "
        "Rate limiting is configured at 100 requests per minute.",
        encoding="utf-8",
    )

    return project


def _run_lifespan(project: Path) -> dict:
    """Run the server lifespan with mocked health/smoke checks and fake embeddings."""

    async def _go():
        mock_server = MagicMock()
        with (
            patch.dict("os.environ", {"TOKENKEEPER_PROJECT": str(project)}),
            patch("tokenkeeper.health.run_startup_checks"),
            patch("tokenkeeper.embeddings.run_smoke_test"),
            patch("tokenkeeper.embeddings.embed_texts", side_effect=_fake_embed_texts),
        ):
            async with app_lifespan(mock_server) as ctx:
                # Wait for background indexing to complete before exiting
                for _ in range(200):  # 10s max
                    if ctx["indexing_state"]["status"] != "indexing":
                        break
                    await asyncio.sleep(0.05)
                return ctx

    return asyncio.run(_go())


def _search(lifespan_data: dict, query: str, top_k: int = 10, mode: str | None = None, alpha: float | None = None):
    """Run the search pipeline using lifespan data (same logic as the MCP tool)."""
    from tokenkeeper.search import search

    collection = lifespan_data["collection"]
    bm25_index = lifespan_data["bm25_index"]
    embed_fn = lifespan_data["embed_fn"]
    config = lifespan_data["config"]

    search_alpha = alpha if alpha is not None else config.alpha
    search_mode = mode if mode is not None else config.mode

    return search(
        query, collection, bm25_index, embed_fn,
        alpha=search_alpha, top_k=top_k, mode=search_mode,
    )


def _format_results(results) -> str:
    """Format results like the search_knowledge tool does."""
    if not results:
        return "No results found"

    lines = [f"Found {len(results)} results\n"]
    for i, r in enumerate(results, start=1):
        lines.append(f"--- Result {i} (score: {r.score:.3f}) ---")
        lines.append(f"Source: {r.source_file} (chunk {r.chunk_index})")
        if r.title:
            lines.append(f"Title: {r.title}")
        if r.tags:
            lines.append(f"Tags: {r.tags}")
        lines.append("")
        lines.append(r.content)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lifespan initialization tests
# ---------------------------------------------------------------------------


class TestLifespanInitialization:
    """Tests that lifespan correctly initializes all server components."""

    def test_lifespan_indexes_markdown_files(self, tmp_path: Path) -> None:
        """Lifespan indexes .md files from project root."""
        project = _create_test_project(tmp_path)
        ctx = _run_lifespan(project)

        state = ctx["indexing_state"]
        assert state["status"] == "ready"
        result = state["result"]
        assert result.files_indexed == 3
        assert result.chunks_indexed > 0

    def test_lifespan_creates_collection(self, tmp_path: Path) -> None:
        """Lifespan creates ChromaDB collection with indexed documents."""
        project = _create_test_project(tmp_path)
        ctx = _run_lifespan(project)

        collection = ctx["collection"]
        assert collection.count() > 0

    def test_lifespan_builds_bm25_index(self, tmp_path: Path) -> None:
        """Lifespan builds BM25 index from indexed documents."""
        project = _create_test_project(tmp_path)
        ctx = _run_lifespan(project)

        bm25 = ctx["bm25_index"]
        assert len(bm25) > 0

    def test_lifespan_empty_project(self, tmp_path: Path) -> None:
        """Lifespan handles empty project gracefully."""
        project = tmp_path / "empty"
        project.mkdir()

        ctx = _run_lifespan(project)

        state = ctx["indexing_state"]
        assert state["status"] == "ready"
        assert state["result"].files_indexed == 0
        assert ctx["collection"].count() == 0

    def test_lifespan_yields_all_keys(self, tmp_path: Path) -> None:
        """Lifespan context has all required keys for tools."""
        project = _create_test_project(tmp_path)
        ctx = _run_lifespan(project)

        required_keys = [
            "config", "project_root", "rag_dir",
            "collection", "bm25_index", "embed_fn", "indexing_state",
        ]
        for key in required_keys:
            assert key in ctx, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Search pipeline tests (same logic as search_knowledge tool)
# ---------------------------------------------------------------------------


class TestSearchPipeline:
    """Tests search via the pipeline function (mirrors MCP tool behavior)."""

    def test_search_returns_results(self, tmp_path: Path) -> None:
        """Search finds matching documents."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "PostgreSQL database")

        assert len(results) > 0
        source_files = [r.source_file for r in results]
        assert "database.md" in source_files

    def test_search_returns_scores(self, tmp_path: Path) -> None:
        """Search results have scores in 0-1 range."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "authentication")

        assert len(results) > 0
        for r in results:
            assert 0 <= r.score <= 1.0

    def test_search_respects_top_k(self, tmp_path: Path) -> None:
        """Search limits results to top_k."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "api", top_k=1)

        assert len(results) == 1

    def test_search_keyword_mode(self, tmp_path: Path) -> None:
        """Keyword mode finds exact term matches."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "alembic", mode="keyword")

        assert len(results) > 0
        assert results[0].source_file == "database.md"

    def test_search_no_results(self, tmp_path: Path) -> None:
        """Non-matching keyword query returns empty list."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "xyzzy_nonexistent_42", mode="keyword")

        assert results == []

    def test_search_includes_metadata(self, tmp_path: Path) -> None:
        """Results include title and tags from frontmatter."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "authentication OAuth2")

        assert len(results) > 0
        # Find the auth.md result
        auth_results = [r for r in results if r.source_file == "auth.md"]
        if auth_results:
            r = auth_results[0]
            assert r.title == "Authentication"
            assert "auth" in r.tags

    def test_search_semantic_mode(self, tmp_path: Path) -> None:
        """Semantic mode returns results sorted by score."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "security tokens", mode="semantic")

        assert len(results) > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_alpha_override(self, tmp_path: Path) -> None:
        """Per-query alpha override works."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        # alpha=0 (pure keyword) should find alembic only in database.md
        results = _search(data, "alembic", alpha=0.0)

        assert len(results) > 0
        assert results[0].source_file == "database.md"


# ---------------------------------------------------------------------------
# Result formatting tests
# ---------------------------------------------------------------------------


class TestResultFormatting:
    """Tests that results format correctly for Claude Code display."""

    def test_formatted_output_structure(self, tmp_path: Path) -> None:
        """Formatted output has result blocks with score, source, content."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "PostgreSQL database")
        output = _format_results(results)

        assert "Found" in output
        assert "Result 1" in output
        assert "score:" in output
        assert "Source:" in output
        assert "database.md" in output

    def test_formatted_empty_results(self) -> None:
        """Empty results produce friendly message."""
        output = _format_results([])
        assert "No results found" in output


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_index_then_search_finds_content(self, tmp_path: Path) -> None:
        """Documents indexed at startup are searchable."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        results = _search(data, "JWT tokens expire")

        source_files = [r.source_file for r in results]
        assert "auth.md" in source_files
        # Verify the content contains JWT
        auth_result = [r for r in results if r.source_file == "auth.md"][0]
        assert "jwt" in auth_result.content.lower()

    def test_all_files_searchable(self, tmp_path: Path) -> None:
        """All indexed files are reachable via search."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        queries_and_files = [
            ("OAuth2 authentication", "auth.md"),
            ("PostgreSQL connection pooling", "database.md"),
            ("REST API endpoints", "api.md"),
        ]

        for query, expected_file in queries_and_files:
            results = _search(data, query, top_k=5)
            source_files = [r.source_file for r in results]
            assert expected_file in source_files, (
                f"Expected {expected_file} in results for '{query}', "
                f"got {source_files}"
            )

    def test_search_completes_under_timeout(self, tmp_path: Path) -> None:
        """Search completes within 15 seconds (MCP-05 requirement)."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        start = time.perf_counter()
        _search(data, "database authentication API")
        elapsed = time.perf_counter() - start

        assert elapsed < 15.0, f"Search took {elapsed:.2f}s (limit: 15s)"

    def test_indexing_state_matches_collection(self, tmp_path: Path) -> None:
        """Indexing state chunk count matches ChromaDB document count."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        state = data["indexing_state"]
        collection = data["collection"]

        assert state["result"].chunks_indexed == collection.count()

    def test_bm25_matches_collection(self, tmp_path: Path) -> None:
        """BM25 index document count matches ChromaDB document count."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        bm25 = data["bm25_index"]
        collection = data["collection"]

        assert len(bm25) == collection.count()


# ===========================================================================
# Phase 7: MCP Management Tools — Integration Tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper for reindex tests
# ---------------------------------------------------------------------------


def _run_reindex(
    lifespan_data: dict,
    target_files: list[Path] | None = None,
    force: bool = False,
):
    """Run the indexing pipeline directly (mirrors reindex_documents tool)."""
    from tokenkeeper.indexer import index_documents

    collection = lifespan_data["collection"]
    bm25_index = lifespan_data["bm25_index"]
    embed_fn = lifespan_data["embed_fn"]
    config = lifespan_data["config"]
    project_root = Path(lifespan_data["project_root"])

    return index_documents(
        project_root, config, collection, bm25_index, embed_fn,
        target_files=target_files,
        force=force,
    )


def _get_stats(lifespan_data: dict) -> dict:
    """Query stats from ChromaDB (mirrors get_index_stats tool)."""
    collection = lifespan_data["collection"]
    bm25_index = lifespan_data["bm25_index"]

    total_chunks = collection.count()

    source_files: set[str] = set()
    max_indexed_at = 0.0

    if total_chunks > 0:
        stored = collection.get(include=["metadatas"])
        for meta in (stored["metadatas"] or []):
            source_files.add(meta.get("source_file", "<unknown>"))
            indexed_at = meta.get("indexed_at", 0)
            if isinstance(indexed_at, (int, float)) and indexed_at > max_indexed_at:
                max_indexed_at = indexed_at

    return {
        "total_chunks": total_chunks,
        "unique_files": len(source_files),
        "source_files": source_files,
        "max_indexed_at": max_indexed_at,
        "bm25_count": len(bm25_index),
    }


# ---------------------------------------------------------------------------
# Reindex documents tests
# ---------------------------------------------------------------------------


class TestReindexDocuments:
    """Integration tests for reindex_documents tool logic."""

    def test_full_reindex_updates_collection(self, tmp_path: Path) -> None:
        """After modifying a file, full reindex updates ChromaDB."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        initial_count = data["collection"].count()
        assert initial_count > 0

        # Modify auth.md to add more content
        (project / "auth.md").write_text(
            "---\ntitle: Authentication v2\ntags:\n  - auth\n---\n\n"
            "Completely rewritten authentication docs with SSO support. "
            "SAML and OIDC protocols are now available. "
            "Session management uses Redis for token storage.",
            encoding="utf-8",
        )

        result = _run_reindex(data)

        assert result.files_indexed >= 1
        # auth.md was changed, so it should be reindexed

    def test_partial_reindex_specific_file(self, tmp_path: Path) -> None:
        """Pass one file path, only that file reindexed."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        # Modify database.md
        (project / "database.md").write_text(
            "---\ntitle: Database v2\ntags:\n  - database\n---\n\n"
            "Switched to MySQL with connection pooling via HikariCP.",
            encoding="utf-8",
        )

        result = _run_reindex(data, target_files=[project / "database.md"])

        assert result.files_indexed == 1
        # Check the new content is searchable
        results = _search(data, "MySQL HikariCP", mode="keyword")
        assert len(results) > 0

    def test_partial_reindex_unchanged_skipped(self, tmp_path: Path) -> None:
        """Partial reindex with unchanged file is skipped."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        # Don't modify auth.md — reindex should skip it
        result = _run_reindex(data, target_files=[project / "auth.md"])

        assert result.files_skipped == 1
        assert result.files_indexed == 0

    def test_force_reindex_unchanged_file(self, tmp_path: Path) -> None:
        """force=True reindexes even unchanged files."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        result = _run_reindex(data, force=True)

        # All 3 files should be reindexed even though unchanged
        assert result.files_indexed == 3
        assert result.files_skipped == 0

    def test_reindex_adds_new_file(self, tmp_path: Path) -> None:
        """Add a new .md file, reindex finds it."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        initial_count = data["collection"].count()

        # Add a new file
        (project / "deployment.md").write_text(
            "---\ntitle: Deployment\ntags:\n  - deploy\n---\n\n"
            "Deploy to Kubernetes using Helm charts. "
            "CI/CD pipeline uses GitHub Actions.",
            encoding="utf-8",
        )

        result = _run_reindex(data)

        assert result.files_indexed >= 1
        assert data["collection"].count() > initial_count

    def test_reindex_removes_deleted_file(self, tmp_path: Path) -> None:
        """Delete a file, full reindex cleans up chunks."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        initial_count = data["collection"].count()

        # Delete api.md
        (project / "api.md").unlink()

        result = _run_reindex(data)

        # Collection should have fewer chunks now
        assert data["collection"].count() < initial_count

    def test_reindex_stores_indexed_at(self, tmp_path: Path) -> None:
        """After reindex, chunks have indexed_at in metadata."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        stored = data["collection"].get(include=["metadatas"])
        for meta in stored["metadatas"]:
            assert "indexed_at" in meta
            assert isinstance(meta["indexed_at"], float)
            assert meta["indexed_at"] > 0

    def test_partial_reindex_preserves_other_files(self, tmp_path: Path) -> None:
        """Partial reindex doesn't affect other files' chunks."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        # Count chunks for api.md before
        stored_before = data["collection"].get(
            where={"source_file": "api.md"}, include=["metadatas"]
        )
        api_count_before = len(stored_before["ids"])

        # Modify and partially reindex only auth.md
        (project / "auth.md").write_text(
            "---\ntitle: Auth Updated\ntags:\n  - auth\n---\n\n"
            "Updated authentication with passkeys and WebAuthn.",
            encoding="utf-8",
        )
        _run_reindex(data, target_files=[project / "auth.md"])

        # api.md chunks should be unchanged
        stored_after = data["collection"].get(
            where={"source_file": "api.md"}, include=["metadatas"]
        )
        assert len(stored_after["ids"]) == api_count_before


# ---------------------------------------------------------------------------
# Index stats tests
# ---------------------------------------------------------------------------


class TestGetIndexStats:
    """Integration tests for get_index_stats tool logic."""

    def test_stats_shows_total_chunks(self, tmp_path: Path) -> None:
        """Chunk count matches collection.count()."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        stats = _get_stats(data)
        assert stats["total_chunks"] == data["collection"].count()
        assert stats["total_chunks"] > 0

    def test_stats_shows_unique_files(self, tmp_path: Path) -> None:
        """Lists correct number of source files."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        stats = _get_stats(data)
        assert stats["unique_files"] == 3
        assert "auth.md" in stats["source_files"]
        assert "database.md" in stats["source_files"]
        assert "api.md" in stats["source_files"]

    def test_stats_shows_bm25_count(self, tmp_path: Path) -> None:
        """BM25 doc count matches chunk count."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        stats = _get_stats(data)
        assert stats["bm25_count"] == stats["total_chunks"]

    def test_stats_includes_last_indexed(self, tmp_path: Path) -> None:
        """Has a last-indexed timestamp."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        stats = _get_stats(data)
        assert stats["max_indexed_at"] > 0

    def test_stats_empty_index(self, tmp_path: Path) -> None:
        """Stats work with empty collection."""
        project = tmp_path / "empty"
        project.mkdir()
        data = _run_lifespan(project)

        stats = _get_stats(data)
        assert stats["total_chunks"] == 0
        assert stats["unique_files"] == 0
        assert stats["bm25_count"] == 0

    def test_stats_after_reindex(self, tmp_path: Path) -> None:
        """Stats update after reindex adds a new file."""
        project = _create_test_project(tmp_path)
        data = _run_lifespan(project)

        stats_before = _get_stats(data)

        # Add a new file and reindex
        (project / "monitoring.md").write_text(
            "---\ntitle: Monitoring\ntags:\n  - observability\n---\n\n"
            "Prometheus metrics and Grafana dashboards for monitoring.",
            encoding="utf-8",
        )
        _run_reindex(data)

        stats_after = _get_stats(data)
        assert stats_after["total_chunks"] > stats_before["total_chunks"]
        assert stats_after["unique_files"] == stats_before["unique_files"] + 1
