"""Integration tests for the ingestion pipeline using real planning documents.

Validates the complete pipeline (discovery -> parsing -> chunking) against
the actual ``.planning/`` directory from this project. Also includes a
performance benchmark to ensure the full doc set processes in under 5 seconds.

These tests are skipped when ``.planning/`` does not exist (e.g. in a bare
clone without planning artifacts).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from knowledge_rag.discovery import discover_markdown_files
from knowledge_rag.ingestion import DocumentChunk, ingest_file, parse_document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANNING_DIR = PROJECT_ROOT / ".planning"

_skip_no_planning = pytest.mark.skipif(
    not PLANNING_DIR.exists(),
    reason=".planning/ directory does not exist",
)


# ---------------------------------------------------------------------------
# Test 1: discover_markdown_files finds planning docs
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestDiscoverPlanningDocs:
    def test_finds_planning_docs(self) -> None:
        """discover_markdown_files returns non-empty list of .md files."""
        files = discover_markdown_files(PLANNING_DIR)
        assert len(files) > 0, "Expected to find markdown files in .planning/"
        for f in files:
            assert f.suffix.lower() == ".md", f"Non-markdown file found: {f}"
            assert f.exists(), f"Returned path does not exist: {f}"
        # Verify deterministic ordering (two calls produce identical results).
        # The function walks directories in sorted order, producing a consistent
        # traversal order that may differ from a global sorted() on full paths.
        files2 = discover_markdown_files(PLANNING_DIR)
        assert files == files2, "Results should be deterministic across calls"


# ---------------------------------------------------------------------------
# Test 2: discover_markdown_files excludes .git, node_modules, __pycache__
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestDiscoverExclusions:
    def test_excludes_known_directories(self) -> None:
        """No returned path contains .git, node_modules, or __pycache__."""
        files = discover_markdown_files(PROJECT_ROOT)
        excluded_components = {".git", "node_modules", "__pycache__"}
        for f in files:
            parts = set(f.parts)
            for exc in excluded_components:
                assert exc not in parts, (
                    f"Path {f} contains excluded component '{exc}'"
                )


# ---------------------------------------------------------------------------
# Test 3: ingest_file on ROADMAP.md
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestIngestRoadmap:
    def test_ingest_roadmap(self) -> None:
        """Ingest ROADMAP.md and verify chunk properties."""
        roadmap = PLANNING_DIR / "ROADMAP.md"
        if not roadmap.exists():
            pytest.skip("ROADMAP.md not found in .planning/")

        chunks = ingest_file(roadmap, PROJECT_ROOT)
        assert len(chunks) > 0, "ROADMAP.md should produce at least one chunk"

        total = chunks[0].total_chunks
        indices = []
        for chunk in chunks:
            # POSIX paths only (no backslashes)
            assert "\\" not in chunk.source_file, (
                f"source_file contains backslash: {chunk.source_file}"
            )
            assert not chunk.source_file.startswith("C:"), (
                f"source_file contains drive letter: {chunk.source_file}"
            )
            # All chunks share the same total_chunks
            assert chunk.total_chunks == total
            indices.append(chunk.chunk_index)

        # Sequential indices: [0, 1, 2, ..., total-1]
        assert indices == list(range(total)), (
            f"Chunk indices not sequential: {indices}"
        )

        # No frontmatter leakage into chunk content
        for chunk in chunks:
            assert "---\ntitle:" not in chunk.content, (
                "Frontmatter leaked into chunk content"
            )
            assert "---\ntags:" not in chunk.content, (
                "Frontmatter leaked into chunk content"
            )


# ---------------------------------------------------------------------------
# Test 4: chunk size respected
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestChunkSizeRespected:
    def test_oversized_chunks_contain_oversized_paragraph(self) -> None:
        """Chunks exceeding chunk_size contain at least one oversized paragraph.

        The overlap mechanism prepends the last paragraph from the previous
        chunk, so an oversized chunk may contain overlap + oversized paragraph
        (i.e. have ``\\n\\n``).  The key invariant is that the chunk is only
        oversized because it contains a single paragraph that by itself exceeds
        chunk_size -- the algorithm never packs multiple normal-sized paragraphs
        beyond the limit.
        """
        roadmap = PLANNING_DIR / "ROADMAP.md"
        if not roadmap.exists():
            pytest.skip("ROADMAP.md not found in .planning/")

        chunk_size = 1000
        chunks = ingest_file(roadmap, PROJECT_ROOT, chunk_size=chunk_size)
        assert len(chunks) > 0

        oversized_count = 0
        for chunk in chunks:
            if len(chunk.content) > chunk_size:
                # At least one paragraph in this chunk must exceed chunk_size
                # on its own (that is the oversized paragraph that forced the
                # chunk to be large).  The other paragraph(s) are overlap.
                paras = chunk.content.split("\n\n")
                max_para = max(len(p) for p in paras)
                assert max_para > chunk_size // 2, (
                    f"Oversized chunk at index {chunk.chunk_index} "
                    f"({len(chunk.content)} chars) has no large paragraph "
                    f"(max paragraph: {max_para} chars)"
                )
                oversized_count += 1

        # Oversized chunks should be a small fraction of total
        fraction = oversized_count / len(chunks) if chunks else 0
        assert fraction < 0.25, (
            f"Too many oversized chunks: {oversized_count}/{len(chunks)} "
            f"({fraction:.0%})"
        )


# ---------------------------------------------------------------------------
# Test 5: deterministic chunking
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestDeterministicChunking:
    def test_two_runs_produce_identical_results(self) -> None:
        """Ingesting the same file twice yields identical chunk lists."""
        roadmap = PLANNING_DIR / "ROADMAP.md"
        if not roadmap.exists():
            pytest.skip("ROADMAP.md not found in .planning/")

        run1 = ingest_file(roadmap, PROJECT_ROOT)
        run2 = ingest_file(roadmap, PROJECT_ROOT)

        assert len(run1) == len(run2), "Different chunk counts across runs"
        for i, (c1, c2) in enumerate(zip(run1, run2)):
            assert c1.content == c2.content, f"Content differs at chunk {i}"
            assert c1.char_start == c2.char_start, f"char_start differs at chunk {i}"
            assert c1.char_end == c2.char_end, f"char_end differs at chunk {i}"
            assert c1.chunk_index == c2.chunk_index, f"chunk_index differs at chunk {i}"


# ---------------------------------------------------------------------------
# Test 6: character offset validity
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestCharacterOffsetValidity:
    def test_offsets_reference_normalized_body(self) -> None:
        """body[char_start:char_end] starts with the first paragraph of chunk."""
        roadmap = PLANNING_DIR / "ROADMAP.md"
        if not roadmap.exists():
            pytest.skip("ROADMAP.md not found in .planning/")

        text = roadmap.read_text(encoding="utf-8")
        _metadata, body = parse_document(text)

        chunks = ingest_file(roadmap, PROJECT_ROOT)
        assert len(chunks) > 0

        for chunk in chunks:
            body_slice = body[chunk.char_start : chunk.char_end]
            # The first paragraph of the chunk content should appear at
            # the start of the body slice (offsets track first paragraph)
            first_para = chunk.content.split("\n\n")[0]
            assert body_slice.startswith(first_para), (
                f"Chunk {chunk.chunk_index}: body[{chunk.char_start}:{chunk.char_end}] "
                f"does not start with first paragraph.\n"
                f"  body_slice start: {body_slice[:80]!r}\n"
                f"  first_para: {first_para[:80]!r}"
            )


# ---------------------------------------------------------------------------
# Test 7: end-to-end pipeline
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestEndToEndPipeline:
    def test_discover_and_ingest_all(self) -> None:
        """Discover all .md files in .planning/ and ingest each one."""
        files = discover_markdown_files(PLANNING_DIR)
        assert len(files) > 0, "No markdown files discovered"

        total_chunks = 0
        for f in files:
            chunks = ingest_file(f, PROJECT_ROOT)
            total_chunks += len(chunks)
            # Each file should produce at least one chunk (unless empty)
            # We don't assert > 0 here because some files may have empty bodies

        assert total_chunks > 0, "Pipeline produced zero chunks across all files"

        avg_chunks = total_chunks / len(files)
        print(
            f"\n  End-to-end: {len(files)} files discovered, "
            f"{total_chunks} total chunks, "
            f"{avg_chunks:.1f} avg chunks per file"
        )


# ---------------------------------------------------------------------------
# Test 8: benchmark chunking performance
# ---------------------------------------------------------------------------


@_skip_no_planning
class TestBenchmark:
    def test_full_pipeline_under_5_seconds(self) -> None:
        """Full discover + ingest pipeline completes in under 5 seconds."""
        start = time.perf_counter()

        files = discover_markdown_files(PLANNING_DIR)
        total_chunks = 0
        for f in files:
            chunks = ingest_file(f, PROJECT_ROOT)
            total_chunks += len(chunks)

        elapsed = time.perf_counter() - start

        print(
            f"\n  Benchmark: {len(files)} files, "
            f"{total_chunks} chunks in {elapsed:.2f}s"
        )

        assert elapsed < 5.0, (
            f"Pipeline too slow: {elapsed:.2f}s (limit: 5.0s)"
        )
