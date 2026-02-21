"""Comprehensive tests for tokenkeeper.ingestion module.

Covers: DocumentChunk dataclass construction and immutability,
normalize_whitespace edge cases (CRLF, newline collapsing, space
collapsing, strip), parse_document frontmatter extraction with
body normalisation, chunk_document paragraph-aware chunking, and
ingest_file pipeline integration.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pytest

from tokenkeeper.ingestion import (
    DocumentChunk,
    HeadingStack,
    chunk_document,
    chunk_document_heading_aware,
    chunk_typescript_code,
    ingest_code_file,
    ingest_file,
    normalize_whitespace,
    parse_document,
)


# ---------------------------------------------------------------------------
# DocumentChunk dataclass
# ---------------------------------------------------------------------------


class TestDocumentChunk:
    def test_construction_with_required_fields(self) -> None:
        chunk = DocumentChunk(
            content="hello world",
            source_file="docs/readme.md",
            chunk_index=0,
            char_start=0,
            char_end=11,
            total_chunks=1,
        )
        assert chunk.content == "hello world"
        assert chunk.source_file == "docs/readme.md"
        assert chunk.chunk_index == 0
        assert chunk.char_start == 0
        assert chunk.char_end == 11
        assert chunk.total_chunks == 1

    def test_default_frontmatter_is_empty_dict(self) -> None:
        chunk = DocumentChunk(
            content="text",
            source_file="a.md",
            chunk_index=0,
            char_start=0,
            char_end=4,
            total_chunks=1,
        )
        assert chunk.frontmatter == {}

    def test_frontmatter_provided(self) -> None:
        meta = {"title": "Test", "tags": ["a", "b"]}
        chunk = DocumentChunk(
            content="text",
            source_file="a.md",
            chunk_index=0,
            char_start=0,
            char_end=4,
            total_chunks=1,
            frontmatter=meta,
        )
        assert chunk.frontmatter == {"title": "Test", "tags": ["a", "b"]}

    def test_frozen_raises_on_attribute_assignment(self) -> None:
        chunk = DocumentChunk(
            content="text",
            source_file="a.md",
            chunk_index=0,
            char_start=0,
            char_end=4,
            total_chunks=1,
        )
        with pytest.raises(AttributeError):
            chunk.content = "new"  # type: ignore[misc]

    def test_equality_for_identical_chunks(self) -> None:
        kwargs = {
            "content": "text",
            "source_file": "a.md",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "total_chunks": 1,
        }
        assert DocumentChunk(**kwargs) == DocumentChunk(**kwargs)

    def test_inequality_for_different_chunks(self) -> None:
        base = {
            "content": "text",
            "source_file": "a.md",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 4,
            "total_chunks": 1,
        }
        chunk_a = DocumentChunk(**base)
        chunk_b = DocumentChunk(**{**base, "chunk_index": 1})
        assert chunk_a != chunk_b

    def test_source_file_stores_posix_path(self) -> None:
        chunk = DocumentChunk(
            content="text",
            source_file="docs/guides/intro.md",
            chunk_index=0,
            char_start=0,
            char_end=4,
            total_chunks=1,
        )
        assert "/" in chunk.source_file
        assert "\\" not in chunk.source_file


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_crlf_to_lf(self) -> None:
        assert normalize_whitespace("a\r\nb") == "a\nb"

    def test_collapse_three_plus_newlines(self) -> None:
        assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_preserve_double_newlines(self) -> None:
        assert normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_collapse_multiple_spaces(self) -> None:
        assert normalize_whitespace("a   b") == "a b"

    def test_collapse_tabs(self) -> None:
        assert normalize_whitespace("a\t\tb") == "a b"

    def test_collapse_mixed_spaces_and_tabs(self) -> None:
        assert normalize_whitespace("a \t \t b") == "a b"

    def test_strip_leading_and_trailing_whitespace(self) -> None:
        assert normalize_whitespace("  a\n\nb  ") == "a\n\nb"

    def test_mixed_crlf_and_excess_newlines(self) -> None:
        result = normalize_whitespace("a\r\n\r\n\r\n\r\nb")
        assert result == "a\n\nb"

    def test_mixed_everything(self) -> None:
        text = "  hello\r\n  world  \r\n\r\n\r\n\r\nend  "
        result = normalize_whitespace(text)
        # "  world  " collapses to " world " (trailing space preserved mid-text)
        assert result == "hello\n world \n\nend"

    def test_empty_string(self) -> None:
        assert normalize_whitespace("") == ""

    def test_whitespace_only_string(self) -> None:
        assert normalize_whitespace("   \n\n\n  ") == ""

    def test_single_newline_preserved(self) -> None:
        assert normalize_whitespace("a\nb") == "a\nb"

    def test_no_change_for_clean_text(self) -> None:
        text = "Hello world.\n\nSecond paragraph."
        assert normalize_whitespace(text) == text


# ---------------------------------------------------------------------------
# parse_document
# ---------------------------------------------------------------------------


class TestParseDocument:
    def test_standard_yaml_frontmatter(self) -> None:
        text = "---\ntitle: Test\ntags: [a, b]\n---\nBody text"
        metadata, body = parse_document(text)
        assert metadata["title"] == "Test"
        assert metadata["tags"] == ["a", "b"]
        assert body == "Body text"

    def test_no_frontmatter_returns_empty_dict(self) -> None:
        text = "Just body text"
        metadata, body = parse_document(text)
        assert metadata == {}
        assert body == "Just body text"

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\nBody text"
        metadata, body = parse_document(text)
        assert metadata == {}
        assert body == "Body text"

    def test_frontmatter_with_date_field(self) -> None:
        text = "---\ndate: 2025-01-15\n---\nBody"
        metadata, body = parse_document(text)
        # PyYAML auto-parses dates into datetime.date objects
        assert isinstance(metadata["date"], datetime.date)

    def test_horizontal_rule_in_body_preserved(self) -> None:
        text = "---\ntitle: Test\n---\nBody\n\n---\n\nMore body"
        metadata, body = parse_document(text)
        assert metadata["title"] == "Test"
        assert "---" in body
        assert "More body" in body

    def test_body_whitespace_is_normalized(self) -> None:
        text = "---\ntitle: Test\n---\nBody   with   spaces\r\n\r\n\r\n\r\nEnd"
        metadata, body = parse_document(text)
        assert metadata["title"] == "Test"
        assert body == "Body with spaces\n\nEnd"

    def test_nested_yaml_frontmatter(self) -> None:
        text = "---\nmeta:\n  key: value\n---\nBody"
        metadata, body = parse_document(text)
        assert metadata["meta"]["key"] == "value"
        assert body == "Body"

    def test_frontmatter_with_multiple_fields(self) -> None:
        text = (
            "---\n"
            "title: My Doc\n"
            "author: Alice\n"
            "version: 2\n"
            "draft: true\n"
            "---\n"
            "Content here"
        )
        metadata, body = parse_document(text)
        assert metadata["title"] == "My Doc"
        assert metadata["author"] == "Alice"
        assert metadata["version"] == 2
        assert metadata["draft"] is True
        assert body == "Content here"

    def test_empty_string_input(self) -> None:
        metadata, body = parse_document("")
        assert metadata == {}
        assert body == ""

    def test_malformed_yaml_frontmatter_falls_back_to_plain_text(self) -> None:
        """Malformed YAML in frontmatter should not crash; treat as plain text."""
        # This YAML mixes list items with mapping-style keys, triggering a
        # ParserError in PyYAML (real-world case from 02-01-SUMMARY.md).
        text = (
            "---\n"
            "provides:\n"
            "  - item one\n"
            "  - [build-system] with hatchling\n"
            "affects: [02-02]\n"
            "---\n"
            "Body content here."
        )
        metadata, body = parse_document(text)
        assert metadata == {}
        # The entire text (including malformed frontmatter) becomes the body
        assert "Body content here." in body

    def test_malformed_yaml_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Malformed YAML frontmatter logs a warning."""
        text = (
            "---\n"
            "bad:\n"
            "  - item\n"
            "    key: value\n"
            "---\n"
            "Body."
        )
        with caplog.at_level(logging.WARNING, logger="tokenkeeper.ingestion"):
            parse_document(text)
        assert any("frontmatter" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocument:
    """Tests for the paragraph-aware greedy accumulator chunking algorithm."""

    def test_empty_body_returns_empty_list(self) -> None:
        assert chunk_document("") == []

    def test_whitespace_only_body_returns_empty_list(self) -> None:
        assert chunk_document("   \n\n\n   ") == []

    def test_single_short_paragraph_returns_one_chunk(self) -> None:
        body = "Hello world."
        chunks = chunk_document(body, chunk_size=1000)
        assert len(chunks) == 1
        content, char_start, char_end = chunks[0]
        assert content == "Hello world."
        assert char_start == 0
        assert char_end == len("Hello world.")

    def test_two_paragraphs_fitting_in_one_chunk(self) -> None:
        body = "Paragraph one.\n\nParagraph two."
        chunks = chunk_document(body, chunk_size=1000)
        assert len(chunks) == 1
        content, char_start, char_end = chunks[0]
        assert content == "Paragraph one.\n\nParagraph two."
        assert char_start == 0
        assert char_end == len(body)

    def test_multiple_paragraphs_split_into_chunks(self) -> None:
        # Each paragraph is ~10 chars. chunk_size=25 means ~2 paragraphs per chunk.
        body = "AAAAAAAAAA\n\nBBBBBBBBBB\n\nCCCCCCCCCC\n\nDDDDDDDDDD"
        chunks = chunk_document(body, chunk_size=25, overlap=200)
        # With chunk_size=25, joining two 10-char paras = 22 chars (fits).
        # Adding third (10+2=12 more chars = 34) exceeds 25, so finalize.
        assert len(chunks) >= 2
        # All content accounted for: first chunk has at least 2 paragraphs
        assert "AAAAAAAAAA" in chunks[0][0]
        assert "BBBBBBBBBB" in chunks[0][0]

    def test_overlap_repeats_last_paragraph(self) -> None:
        body = "Alpha\n\nBravo\n\nCharlie\n\nDelta"
        # chunk_size=15: "Alpha" (5) + "\n\n" + "Bravo" (5) = 12 fits.
        # + "\n\nCharlie" (9) = 21 > 15 -> finalize ["Alpha", "Bravo"]
        # Next chunk starts with overlap = "Bravo"
        chunks = chunk_document(body, chunk_size=15, overlap=200)
        assert len(chunks) >= 2
        # Overlap: last paragraph of chunk 0 appears at start of chunk 1
        first_chunk_paras = chunks[0][0].split("\n\n")
        second_chunk_paras = chunks[1][0].split("\n\n")
        assert first_chunk_paras[-1] == second_chunk_paras[0]

    def test_oversized_paragraph_kept_intact(self) -> None:
        big_para = "x" * 2000
        body = big_para
        chunks = chunk_document(body, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0][0] == big_para
        assert chunks[0][1] == 0
        assert chunks[0][2] == 2000

    def test_oversized_paragraph_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # Oversized paragraph preceded by a normal one, so the "current is not empty" branch triggers
        body = "Short.\n\n" + "x" * 2000
        with caplog.at_level(logging.WARNING, logger="tokenkeeper.ingestion"):
            chunk_document(body, chunk_size=1000)
        assert any("Oversized paragraph" in msg for msg in caplog.messages)

    def test_character_offsets_reference_body(self) -> None:
        body = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_document(body, chunk_size=1000)
        # Single chunk: offsets span the full body
        assert len(chunks) == 1
        content, char_start, char_end = chunks[0]
        assert body[char_start:char_end] == content

    def test_character_offsets_correct_for_multiple_chunks(self) -> None:
        body = "Alpha\n\nBravo\n\nCharlie\n\nDelta\n\nEcho"
        chunks = chunk_document(body, chunk_size=15, overlap=200)
        # Verify each chunk's offsets can extract content from body
        for content, char_start, char_end in chunks:
            # The content is a join of paragraphs, so the slice from body
            # should contain all paragraphs in the content
            body_slice = body[char_start:char_end]
            # Every paragraph in the chunk content should appear in the body slice
            for para in content.split("\n\n"):
                assert para in body_slice

    def test_deterministic_output(self) -> None:
        body = "One\n\nTwo\n\nThree\n\nFour\n\nFive"
        result1 = chunk_document(body, chunk_size=10, overlap=200)
        result2 = chunk_document(body, chunk_size=10, overlap=200)
        assert result1 == result2

    def test_returns_tuples_of_three(self) -> None:
        chunks = chunk_document("Hello world.", chunk_size=1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3
        assert isinstance(chunks[0][0], str)
        assert isinstance(chunks[0][1], int)
        assert isinstance(chunks[0][2], int)

    def test_no_empty_chunks_produced(self) -> None:
        body = "A\n\nB\n\nC\n\nD\n\nE"
        chunks = chunk_document(body, chunk_size=5, overlap=200)
        for content, _, _ in chunks:
            assert content.strip() != ""

    def test_all_paragraphs_present_in_at_least_one_chunk(self) -> None:
        body = "Alpha\n\nBravo\n\nCharlie\n\nDelta"
        chunks = chunk_document(body, chunk_size=15, overlap=200)
        all_content = " ".join(c[0] for c in chunks)
        for para in ["Alpha", "Bravo", "Charlie", "Delta"]:
            assert para in all_content


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------


class TestIngestFile:
    """Tests for the ingest_file pipeline function."""

    def test_basic_markdown_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("---\ntitle: Test\n---\nHello world.", encoding="utf-8")
        chunks = ingest_file(md, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].source_file == "doc.md"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert chunks[0].frontmatter == {"title": "Test"}

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        md = tmp_path / "plain.md"
        md.write_text("Just some text.", encoding="utf-8")
        chunks = ingest_file(md, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].frontmatter == {}
        assert chunks[0].content == "Just some text."

    def test_empty_body_returns_empty_list(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("---\ntitle: Empty\n---\n", encoding="utf-8")
        chunks = ingest_file(md, tmp_path)
        assert chunks == []

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        md = tmp_path / "blank.md"
        md.write_text("", encoding="utf-8")
        chunks = ingest_file(md, tmp_path)
        assert chunks == []

    def test_non_utf8_file_returns_empty_list(self, tmp_path: Path) -> None:
        md = tmp_path / "binary.md"
        md.write_bytes(b"\xff\xfe\x00\x01invalid utf-8 \x80\x81")
        chunks = ingest_file(md, tmp_path)
        assert chunks == []

    def test_non_utf8_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        md = tmp_path / "bad.md"
        md.write_bytes(b"\x80\x81\x82\x83")
        with caplog.at_level(logging.WARNING, logger="tokenkeeper.ingestion"):
            ingest_file(md, tmp_path)
        assert any("non-UTF-8" in msg or "Skipping" in msg for msg in caplog.messages)

    def test_source_file_uses_posix_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "docs" / "guides"
        sub.mkdir(parents=True)
        md = sub / "intro.md"
        md.write_text("Content here.", encoding="utf-8")
        chunks = ingest_file(md, tmp_path)
        assert chunks[0].source_file == "docs/guides/intro.md"
        assert "\\" not in chunks[0].source_file

    def test_multiple_chunks_have_correct_indices(self, tmp_path: Path) -> None:
        # Create a file with enough content to produce multiple chunks
        paras = [f"Paragraph {i} with some extra text." for i in range(20)]
        body = "\n\n".join(paras)
        md = tmp_path / "long.md"
        md.write_text(body, encoding="utf-8")
        chunks = ingest_file(md, tmp_path, chunk_size=100, overlap=50)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)
            assert chunk.source_file == "long.md"

    def test_all_chunks_share_same_frontmatter(self, tmp_path: Path) -> None:
        paras = [f"Paragraph {i} content." for i in range(20)]
        body = "\n\n".join(paras)
        text = f"---\ntitle: Shared\ntags: [a, b]\n---\n{body}"
        md = tmp_path / "shared.md"
        md.write_text(text, encoding="utf-8")
        chunks = ingest_file(md, tmp_path, chunk_size=80, overlap=30)
        assert len(chunks) > 1
        expected_meta = {"title": "Shared", "tags": ["a", "b"]}
        for chunk in chunks:
            assert chunk.frontmatter == expected_meta

    def test_chunk_size_and_overlap_params_forwarded(self, tmp_path: Path) -> None:
        paras = ["A" * 50, "B" * 50, "C" * 50, "D" * 50]
        body = "\n\n".join(paras)
        md = tmp_path / "sized.md"
        md.write_text(body, encoding="utf-8")
        # Small chunk_size forces multiple chunks
        chunks_small = ingest_file(md, tmp_path, chunk_size=60, overlap=30)
        # Large chunk_size keeps everything in one chunk
        chunks_large = ingest_file(md, tmp_path, chunk_size=5000, overlap=200)
        assert len(chunks_small) > len(chunks_large)

    def test_char_offsets_populated(self, tmp_path: Path) -> None:
        md = tmp_path / "offsets.md"
        md.write_text("Some content here.", encoding="utf-8")
        chunks = ingest_file(md, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len("Some content here.")

    def test_empty_body_after_frontmatter_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        md = tmp_path / "meta_only.md"
        md.write_text("---\ntitle: Only Meta\n---\n   \n\n  ", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="tokenkeeper.ingestion"):
            result = ingest_file(md, tmp_path)
        assert result == []
        assert any("Empty" in msg or "empty" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# HeadingStack
# ---------------------------------------------------------------------------


class TestHeadingStack:
    """Tests for the HeadingStack heading hierarchy tracker."""

    def test_empty_path(self) -> None:
        """New stack returns empty string for path()."""
        stack = HeadingStack()
        assert stack.path() == ""

    def test_single_heading(self) -> None:
        """Single push produces a single-element path."""
        stack = HeadingStack()
        stack.push(1, "Title")
        assert stack.path() == "# Title"

    def test_nested_headings(self) -> None:
        """Pushing h1, h2, h3 in order produces a three-part path."""
        stack = HeadingStack()
        stack.push(1, "Config")
        stack.push(2, "Database")
        stack.push(3, "Pool")
        assert stack.path() == "# Config > ## Database > ### Pool"

    def test_sibling_heading_pops(self) -> None:
        """Pushing a heading at the same level replaces its predecessor."""
        stack = HeadingStack()
        stack.push(2, "A")
        stack.push(2, "B")
        assert stack.path() == "## B"

    def test_parent_heading_pops_children(self) -> None:
        """Pushing a higher-level heading pops all deeper headings."""
        stack = HeadingStack()
        stack.push(1, "Old")
        stack.push(2, "Child")
        stack.push(3, "Grandchild")
        stack.push(1, "New")
        assert stack.path() == "# New"

    def test_jump_up_levels(self) -> None:
        """Jumping from h3 back to h2 pops h3 and replaces h2."""
        stack = HeadingStack()
        stack.push(1, "Root")
        stack.push(2, "Section")
        stack.push(3, "Subsection")
        stack.push(2, "Sibling")
        assert stack.path() == "# Root > ## Sibling"

    def test_copy_independent(self) -> None:
        """A copy is independent -- mutating original does not affect copy."""
        stack = HeadingStack()
        stack.push(1, "Root")
        stack.push(2, "Child")
        snapshot = stack.copy()
        # Mutate original
        stack.push(3, "Grandchild")
        # Copy should be unchanged
        assert snapshot.path() == "# Root > ## Child"
        assert stack.path() == "# Root > ## Child > ### Grandchild"

    def test_len(self) -> None:
        """__len__ returns the number of headings on the stack."""
        stack = HeadingStack()
        assert len(stack) == 0
        stack.push(1, "A")
        assert len(stack) == 1
        stack.push(2, "B")
        assert len(stack) == 2
        stack.push(3, "C")
        assert len(stack) == 3
        # Sibling pop: replaces h3
        stack.push(3, "D")
        assert len(stack) == 3
        # Parent pop: pops h3 and h2
        stack.push(1, "E")
        assert len(stack) == 1

    def test_push_deeper_without_parent(self) -> None:
        """Pushing a deep heading directly (no parent) works fine."""
        stack = HeadingStack()
        stack.push(3, "Deep")
        assert stack.path() == "### Deep"


# ---------------------------------------------------------------------------
# chunk_document_heading_aware
# ---------------------------------------------------------------------------


class TestChunkDocumentHeadingAware:
    """Tests for the heading-aware chunking algorithm."""

    def test_single_section_atomic(self) -> None:
        """One heading + short content -> 1 chunk with heading path."""
        body = "## Setup\nInstall the package."
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 1
        content, char_start, char_end, hierarchy = chunks[0]
        assert "## Setup" in content
        assert "Install the package." in content
        assert hierarchy == "## Setup"

    def test_two_sections(self) -> None:
        """Two H2 headings -> 2 chunks with correct paths."""
        body = "## Alpha\nFirst section.\n\n## Beta\nSecond section."
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 2
        assert chunks[0][3] == "## Alpha"
        assert "First section." in chunks[0][0]
        assert chunks[1][3] == "## Beta"
        assert "Second section." in chunks[1][0]

    def test_nested_headings(self) -> None:
        """H1 > H2 > H3 -> chunks carry full hierarchy path."""
        body = "# Root\nIntro.\n\n## Child\nDetails.\n\n### Grandchild\nDeep content."
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 3
        assert chunks[0][3] == "# Root"
        assert chunks[1][3] == "# Root > ## Child"
        assert chunks[2][3] == "# Root > ## Child > ### Grandchild"

    def test_heading_hierarchy_updated_on_sibling(self) -> None:
        """H1 > H2 'A' > text > H2 'B' > text -> second chunk has '# X > ## B'."""
        body = "# X\nIntro.\n\n## A\nContent A.\n\n## B\nContent B."
        chunks = chunk_document_heading_aware(body)
        # Should have 3 chunks: # X, ## A, ## B
        assert len(chunks) == 3
        assert chunks[1][3] == "# X > ## A"
        assert chunks[2][3] == "# X > ## B"

    def test_no_headings_fallback(self) -> None:
        """Plain text with no headings -> falls back to paragraph chunking with empty hierarchy."""
        body = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_document_heading_aware(body, chunk_size=1000)
        assert len(chunks) >= 1
        # All chunks should have empty hierarchy
        for chunk in chunks:
            assert chunk[3] == ""
        # Content should be present
        all_content = " ".join(c[0] for c in chunks)
        assert "Paragraph one." in all_content
        assert "Paragraph two." in all_content

    def test_code_block_not_split(self) -> None:
        """Section with code block stays in one chunk."""
        body = "## Code\nHere is code:\n\n```python\ndef foo():\n    pass\n```\n\nEnd."
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 1
        assert "```python" in chunks[0][0]
        assert "def foo():" in chunks[0][0]
        assert "```" in chunks[0][0]

    def test_heading_inside_code_block_ignored(self) -> None:
        """# inside ``` block is not treated as heading."""
        body = "## Real\nBefore code.\n\n```\n# Not a heading\nsome code\n```\n\nAfter code."
        chunks = chunk_document_heading_aware(body)
        # Should be 1 chunk -- the # inside the code block is not a heading
        assert len(chunks) == 1
        assert "# Not a heading" in chunks[0][0]
        assert chunks[0][3] == "## Real"

    def test_tilde_fence_protection(self) -> None:
        """~~~ fences also protected."""
        body = "## Section\nText.\n\n~~~\n# Fake heading\ncode\n~~~\n\nMore text."
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 1
        assert "# Fake heading" in chunks[0][0]
        assert chunks[0][3] == "## Section"

    def test_table_not_split(self) -> None:
        """Section with pipe table stays in one chunk."""
        body = "## Data\nResults:\n\n| Name | Value |\n|------|-------|\n| A    | 1     |\n| B    | 2     |"
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 1
        assert "| Name | Value |" in chunks[0][0]
        assert "| A    | 1     |" in chunks[0][0]

    def test_empty_body(self) -> None:
        """Empty body returns empty list."""
        assert chunk_document_heading_aware("") == []
        assert chunk_document_heading_aware("   ") == []

    def test_content_before_first_heading(self) -> None:
        """Text before any heading -> chunk with empty hierarchy."""
        body = "Preamble text.\n\n## First\nContent."
        chunks = chunk_document_heading_aware(body)
        assert len(chunks) == 2
        assert chunks[0][3] == ""
        assert "Preamble text." in chunks[0][0]
        assert chunks[1][3] == "## First"
        assert "Content." in chunks[1][0]


# ---------------------------------------------------------------------------
# Python code chunking (09-04)
# ---------------------------------------------------------------------------


class TestPythonCodeChunking:
    """Tests for chunk_python_code() and related helpers."""

    def test_single_function(self) -> None:
        code = "def foo():\n    return 42\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c[4] == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0][3] == "foo"

    def test_two_functions(self) -> None:
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        func_chunks = [c for c in chunks if c[4] == "function"]
        assert len(func_chunks) == 2
        names = {c[3] for c in func_chunks}
        assert names == {"foo", "bar"}

    def test_class_small(self) -> None:
        code = "class MyClass:\n    def method(self):\n        pass\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code, chunk_size=5000)
        class_chunks = [c for c in chunks if c[4] == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0][3] == "MyClass"

    def test_module_level_code(self) -> None:
        code = "import os\nimport sys\n\nX = 42\n\ndef foo():\n    pass\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        module_chunks = [c for c in chunks if c[4] == "module"]
        assert len(module_chunks) >= 1
        assert "import os" in module_chunks[0][0]

    def test_async_function(self) -> None:
        code = "async def fetch():\n    return await something()\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        func_chunks = [c for c in chunks if c[4] == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0][3] == "fetch"

    def test_decorated_function(self) -> None:
        code = "@decorator\ndef foo():\n    pass\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        func_chunks = [c for c in chunks if c[4] == "function"]
        assert len(func_chunks) == 1
        assert "@decorator" in func_chunks[0][0]

    def test_line_numbers(self) -> None:
        code = "import os\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        func_chunks = sorted(
            [c for c in chunks if c[4] == "function"], key=lambda c: c[5]
        )
        assert func_chunks[0][3] == "foo"
        assert func_chunks[0][5] == 3  # line_start
        assert func_chunks[1][3] == "bar"
        assert func_chunks[1][5] == 6

    def test_syntax_error_fallback(self) -> None:
        code = "def broken(:\n    pass\n"
        from tokenkeeper.ingestion import chunk_python_code

        chunks = chunk_python_code(code)
        assert len(chunks) >= 1
        # All chunks should be "module" type (fallback)
        assert all(c[4] == "module" for c in chunks)

    def test_ingest_code_file_python(self, tmp_path) -> None:
        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    return 'world'\n")
        from tokenkeeper.ingestion import ingest_code_file

        chunks = ingest_code_file(py_file, tmp_path)
        assert len(chunks) >= 1
        assert chunks[0].language == "python"

    def test_ingest_code_file_language_field(self, tmp_path) -> None:
        py_file = tmp_path / "module.py"
        py_file.write_text("X = 1\n\ndef foo():\n    pass\n")
        from tokenkeeper.ingestion import ingest_code_file

        chunks = ingest_code_file(py_file, tmp_path)
        for chunk in chunks:
            assert chunk.language == "python"


# ---------------------------------------------------------------------------
# Oversized section splitting (08-05)
# ---------------------------------------------------------------------------


class TestOversizedSectionSplitting:
    """Tests for oversized section handling in chunk_document_heading_aware."""

    def test_oversized_section_split_at_paragraphs(self) -> None:
        """Section with multiple paragraphs exceeding chunk_size is split."""
        # 5 paragraphs of ~300 chars each -> total ~1500 chars
        paras = [f"Paragraph {i}. " + "x" * 280 for i in range(5)]
        body = "## Big Section\n\n" + "\n\n".join(paras)
        chunks = chunk_document_heading_aware(body, chunk_size=700)
        # Should produce more than 1 chunk
        assert len(chunks) > 1
        # All sub-chunks carry the heading hierarchy
        for chunk in chunks:
            assert chunk[3] == "## Big Section"

    def test_oversized_split_preserves_heading_hierarchy(self) -> None:
        """All sub-chunks of an oversized section share the parent heading path."""
        paras = ["Long paragraph. " + "w" * 400 for _ in range(4)]
        body = "# Root\n\n## Child\n\n" + "\n\n".join(paras)
        chunks = chunk_document_heading_aware(body, chunk_size=600)
        # Find chunks from the ## Child section
        child_chunks = [c for c in chunks if "## Child" in c[3]]
        assert len(child_chunks) > 1
        for c in child_chunks:
            assert c[3] == "# Root > ## Child"

    def test_oversized_section_protects_code_block(self) -> None:
        """Code block within oversized section is never split."""
        code_block = "```python\n" + "\n".join(
            f"def func_{i}(): pass" for i in range(20)
        ) + "\n```"
        body = f"## Code Heavy\nIntro paragraph.\n\n{code_block}\n\nEnd paragraph."
        chunks = chunk_document_heading_aware(body, chunk_size=200)
        # The code block should appear in a single chunk
        code_chunk = [c for c in chunks if "```python" in c[0]]
        assert len(code_chunk) >= 1
        # Verify the code block is intact (not split)
        for c in code_chunk:
            assert "def func_0(): pass" in c[0]
            assert "def func_19(): pass" in c[0]

    def test_oversized_section_protects_table(self) -> None:
        """Table within oversized section is not split across chunks."""
        rows = "\n".join(f"| item_{i} | value_{i} |" for i in range(15))
        table = f"| Name | Value |\n|------|-------|\n{rows}"
        body = f"## Data\nHere is data:\n\n{table}\n\nConclusion text."
        chunks = chunk_document_heading_aware(body, chunk_size=200)
        # Find the chunk containing the table
        table_chunks = [c for c in chunks if "| Name | Value |" in c[0]]
        assert len(table_chunks) >= 1
        # Table should be intact
        for c in table_chunks:
            assert "| item_0 | value_0 |" in c[0]
            assert "| item_14 | value_14 |" in c[0]

    def test_small_section_not_split(self) -> None:
        """Section under chunk_size remains atomic (no splitting)."""
        body = "## Small\nJust a little content."
        chunks = chunk_document_heading_aware(body, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0][3] == "## Small"


# ---------------------------------------------------------------------------
# ingest_file heading hierarchy (08-06)
# ---------------------------------------------------------------------------


class TestIngestFileHeadingHierarchy:
    """Tests that ingest_file returns DocumentChunks with heading_hierarchy."""

    def test_ingest_file_heading_hierarchy(self, tmp_path: Path) -> None:
        """File with headings -> DocumentChunks have heading_hierarchy populated."""
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Title\n\nIntro.\n\n## Section\n\nContent here.")

        chunks = ingest_file(md_file, tmp_path, chunk_size=2000)
        # Should have chunks with heading_hierarchy
        hierarchies = [c.heading_hierarchy for c in chunks]
        assert any("# Title" in h for h in hierarchies)

    def test_ingest_file_no_headings_empty_hierarchy(self, tmp_path: Path) -> None:
        """File with no headings -> heading_hierarchy is empty string."""
        md_file = tmp_path / "plain.md"
        md_file.write_text("Just some text.\n\nAnother paragraph.")

        chunks = ingest_file(md_file, tmp_path, chunk_size=2000)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.heading_hierarchy == ""

    def test_ingest_file_backward_compatible(self, tmp_path: Path) -> None:
        """Existing fields are still populated correctly."""
        md_file = tmp_path / "compat.md"
        md_file.write_text(
            "---\ntitle: Test\n---\n\n## Section A\n\nParagraph one."
        )

        chunks = ingest_file(md_file, tmp_path, chunk_size=2000)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source_file == "compat.md"
            assert chunk.frontmatter.get("title") == "Test"
            assert chunk.content
            assert chunk.total_chunks >= 1
