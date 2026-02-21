"""Document ingestion pipeline for TokenKeeper.

Provides the ``DocumentChunk`` data model, ``HeadingStack`` heading hierarchy
tracker, YAML frontmatter extraction via ``python-frontmatter``, whitespace
normalisation, paragraph-aware chunking, markdown structure detection, and
file-level ingestion pipeline.

Exports:
    DocumentChunk                 -- Immutable chunk of a parsed document
    HeadingStack                  -- Tracks the current heading hierarchy as a stack
    parse_document                -- Extract YAML frontmatter and normalised body
    normalize_whitespace          -- Collapse CRLF, excess newlines, and runs of spaces
    chunk_document                -- Split normalised body into overlapping paragraph-bounded chunks
    chunk_document_heading_aware  -- Split body into heading-bounded chunks with structure protection
    ingest_file                   -- Full pipeline: read file -> parse -> chunk -> DocumentChunk list
    detect_heading                -- Identify ATX headings (H1-H6) with level and text
    is_code_fence                 -- Detect code fence boundaries (``` or ~~~)
    detect_table_lines            -- Find contiguous pipe-delimited table blocks
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

logger = logging.getLogger("tokenkeeper.ingestion")

# ---------------------------------------------------------------------------
# Compiled regexes (module-level for performance)
# ---------------------------------------------------------------------------

_COLLAPSE_NEWLINES = re.compile(r"\n{3,}")
_COLLAPSE_SPACES = re.compile(r"[ \t]{2,}")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_TABLE_LINE_RE = re.compile(r"^\|.*\|$")

# ---------------------------------------------------------------------------
# TypeScript/JavaScript boundary patterns (column 0 only)
# ---------------------------------------------------------------------------

_TS_BOUNDARY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, symbol_type, name_group_description)
    # function declarations: function foo(, export function foo(, async function foo(
    (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "function", "group1"),
    # class declarations: class Foo, export class Foo
    (re.compile(r"^(?:export\s+)?class\s+(\w+)"), "class", "group1"),
    # arrow functions assigned to const/let/var: const foo = (...) => {
    (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("), "function", "group1"),
    # arrow functions with single param: const foo = async x => {
    (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?[a-zA-Z_]\w*\s*=>"), "function", "group1"),
    # export default function/class
    (re.compile(r"^export\s+default\s+(?:async\s+)?function(?:\s+(\w+))?"), "export", "group1_or_default"),
    (re.compile(r"^export\s+default\s+class(?:\s+(\w+))?"), "export", "group1_or_default"),
    # interface declarations
    (re.compile(r"^(?:export\s+)?interface\s+(\w+)"), "interface", "group1"),
    # type alias declarations
    (re.compile(r"^(?:export\s+)?type\s+(\w+)"), "type", "group1"),
    # enum declarations
    (re.compile(r"^(?:export\s+)?enum\s+(\w+)"), "enum", "group1"),
]

# ---------------------------------------------------------------------------
# Language detection map for code files
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
}

# ---------------------------------------------------------------------------
# Markdown structure detection
# ---------------------------------------------------------------------------


def detect_heading(line: str) -> tuple[int, str] | None:
    """Detect ATX-style heading (# through ######).

    Args:
        line: A single line of text (no trailing newline).

    Returns:
        (level, heading_text) tuple if heading detected, None otherwise.
        Level is 1-6 corresponding to # through ######.
    """
    m = _HEADING_RE.match(line.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def is_code_fence(line: str) -> bool:
    """Check if line starts/ends a fenced code block.

    Matches lines starting with 3+ backticks or 3+ tildes.
    The opening fence may have a language identifier after it.

    Args:
        line: A single line of text.

    Returns:
        True if this is a code fence boundary line.
    """
    return bool(_CODE_FENCE_RE.match(line.strip()))


def detect_table_lines(lines: list[str], start: int) -> int:
    """Find the end of a contiguous table block starting at ``start``.

    A table block is a sequence of pipe-delimited lines (|...|).
    Includes header, separator (|---|---|), and data rows.

    Args:
        lines: All lines of the document.
        start: Index of the first pipe-delimited line.

    Returns:
        End index (exclusive) of the table block.
    """
    end = start
    while end < len(lines) and _TABLE_LINE_RE.match(lines[end].strip()):
        end += 1
    return end


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """An immutable chunk of a parsed document.

    Each chunk carries its own text content, positional metadata within the
    source document, and any YAML frontmatter extracted from the file header.

    Attributes:
        content:            Raw markdown text (original case preserved).
        source_file:        Relative path from project root (POSIX format).
        chunk_index:        0-based position within the document.
        char_start:         Character offset start in post-normalisation body.
        char_end:           Character offset end in post-normalisation body.
        total_chunks:       Total number of chunks in the source document.
        frontmatter:        Extracted YAML fields (empty dict when absent).
        heading_hierarchy:  Heading context path (e.g. "# Config > ## Database > ### Pool").
        language:           Programming language (e.g. "python", "typescript").
        symbol_name:        Function or class name for code chunks.
        symbol_type:        Symbol kind: "function", "class", "method", "module".
        line_start:         1-based line number start in source file.
        line_end:           1-based line number end in source file.
    """

    content: str
    source_file: str
    chunk_index: int
    char_start: int
    char_end: int
    total_chunks: int
    frontmatter: dict = field(default_factory=dict)
    heading_hierarchy: str = ""
    language: str = ""
    symbol_name: str = ""
    symbol_type: str = ""
    line_start: int = 0
    line_end: int = 0


# ---------------------------------------------------------------------------
# Heading hierarchy tracker
# ---------------------------------------------------------------------------


class HeadingStack:
    """Tracks the current heading hierarchy as a stack.

    When a heading of level N is pushed, all headings with level >= N
    are popped first. This correctly handles jumping from ### to ## (pops ###)
    and from # to ### (keeps #, pushes ###).

    Usage:
        stack = HeadingStack()
        stack.push(1, "Config")       # stack: ["# Config"]
        stack.push(2, "Database")     # stack: ["# Config", "## Database"]
        stack.push(3, "Pool")         # stack: ["# Config", "## Database", "### Pool"]
        stack.push(2, "Cache")        # stack: ["# Config", "## Cache"]
        stack.path()                  # "# Config > ## Cache"
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        """Push a heading, popping any headings at same or deeper level."""
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))

    def path(self) -> str:
        """Return the heading hierarchy as a formatted path string.

        Returns:
            E.g. "# Config > ## Database > ### Pool", or "" if empty.
        """
        if not self._stack:
            return ""
        parts = []
        for level, text in self._stack:
            prefix = "#" * level
            parts.append(f"{prefix} {text}")
        return " > ".join(parts)

    def copy(self) -> HeadingStack:
        """Return a shallow copy of this stack."""
        new = HeadingStack()
        new._stack = list(self._stack)
        return new

    def __len__(self) -> int:
        return len(self._stack)


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Normalise whitespace for consistent downstream processing.

    Transformations applied in order:

    1. Convert CRLF (``\\r\\n``) to LF (``\\n``).
    2. Collapse runs of three or more newlines down to exactly two
       (preserving paragraph breaks).
    3. Collapse runs of multiple spaces/tabs within each line to a
       single space.
    4. Strip leading and trailing whitespace from the entire result.

    Args:
        text: Raw document text (may contain mixed line endings).

    Returns:
        Cleaned text with consistent whitespace.
    """
    text = text.replace("\r\n", "\n")
    text = _COLLAPSE_NEWLINES.sub("\n\n", text)
    lines = text.split("\n")
    lines = [_COLLAPSE_SPACES.sub(" ", line) for line in lines]
    text = "\n".join(lines)
    return text.strip()


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def parse_document(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Uses ``python-frontmatter`` which splits on the *first* pair of ``---``
    delimiters (``FM_BOUNDARY.split(text, 2)``), so horizontal rules
    (``---``) appearing later in the body are preserved as content.

    The body text is passed through :func:`normalize_whitespace` before
    being returned.

    Args:
        text: Full document text, optionally starting with ``---`` YAML
              frontmatter block.

    Returns:
        A ``(metadata, body)`` tuple where *metadata* is a plain ``dict``
        of YAML fields (empty when no frontmatter is present) and *body*
        is the normalised document content.
    """
    try:
        post = frontmatter.loads(text)
        metadata = dict(post.metadata)
        body = normalize_whitespace(post.content)
    except Exception:
        # Malformed YAML frontmatter -- treat entire text as body with no metadata.
        logger.warning("Failed to parse YAML frontmatter; treating as plain text")
        metadata = {}
        body = normalize_whitespace(text)
    return metadata, body


# ---------------------------------------------------------------------------
# Paragraph-aware chunking
# ---------------------------------------------------------------------------


def chunk_document(
    body: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[tuple[str, int, int]]:
    """Split normalised body text into overlapping, paragraph-bounded chunks.

    Uses the *Paragraph-Aware Greedy Accumulator* algorithm: paragraphs are
    accumulated until adding the next paragraph would exceed *chunk_size*,
    then the current chunk is finalised and the next chunk begins with the
    last paragraph from the previous chunk (overlap).

    A single paragraph that exceeds *chunk_size* is kept intact (never broken
    mid-thought) and a warning is logged.

    Args:
        body:       Post-normalisation document body (output of
                    :func:`parse_document`).
        chunk_size: Soft character limit per chunk (default 1000).
        overlap:    Soft guideline for overlap size (default 200).  Actual
                    overlap equals the length of the last paragraph from the
                    previous chunk.

    Returns:
        A list of ``(content, char_start, char_end)`` tuples where
        *char_start* and *char_end* are character offsets into *body*.
    """
    paragraphs = body.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # Pre-compute character offsets of each paragraph in the original body.
    para_offsets: list[int] = []
    offset = 0
    for para in paragraphs:
        idx = body.find(para, offset)
        para_offsets.append(idx)
        offset = idx + len(para)

    chunks: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_size = 0

    for i, para in enumerate(paragraphs):
        para_size = len(para)
        join_cost = 2 if current else 0  # "\n\n" separator

        if current and current_size + join_cost + para_size > chunk_size:
            # Finalize current chunk
            chunk_text = "\n\n".join(current)
            char_start = para_offsets[i - len(current)]
            char_end = para_offsets[i - 1] + len(current[-1])
            chunks.append((chunk_text, char_start, char_end))

            if para_size > chunk_size:
                logger.warning(
                    "Oversized paragraph (%d chars > chunk_size %d) at offset %d",
                    para_size,
                    chunk_size,
                    para_offsets[i],
                )

            # Overlap: start new chunk with last paragraph from previous chunk
            last = current[-1]
            current = [last]
            current_size = len(last)

        current.append(para)
        current_size += join_cost + para_size

    # Finalize last chunk
    if current:
        chunk_text = "\n\n".join(current)
        start_idx = len(paragraphs) - len(current)
        char_start = para_offsets[start_idx]
        char_end = para_offsets[-1] + len(paragraphs[-1])
        chunks.append((chunk_text, char_start, char_end))

    return chunks


# ---------------------------------------------------------------------------
# Heading-aware chunking — helper utilities
# ---------------------------------------------------------------------------


def _identify_blocks(content: str) -> list[str]:
    """Split section content into blocks that must not be split.

    Returns a list of text blocks where each block is either:
    - A normal paragraph (text between blank lines)
    - A complete fenced code block (``` ... ```)
    - A complete table (contiguous |...| lines)

    Code blocks and tables are treated as atomic — they are never broken
    across chunk boundaries.
    """
    lines = content.split("\n")
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code fence toggle
        if is_code_fence(line):
            if not in_fence:
                # Starting a code block.  Flush any accumulated paragraph first.
                if current:
                    _flush_paragraphs(current, blocks)
                    current = []
                # Accumulate the entire code block as one atomic block.
                fence_lines: list[str] = [line]
                i += 1
                while i < len(lines):
                    fence_lines.append(lines[i])
                    if is_code_fence(lines[i]):
                        break
                    i += 1
                blocks.append("\n".join(fence_lines))
                i += 1
                continue
            else:
                # Shouldn't normally reach here but handle gracefully
                current.append(line)
                in_fence = False
                i += 1
                continue

        # Table detection (contiguous pipe-delimited lines)
        if _TABLE_LINE_RE.match(line.strip()):
            # Flush any accumulated paragraph first.
            if current:
                _flush_paragraphs(current, blocks)
                current = []
            table_lines: list[str] = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(table_lines))
            continue

        # Regular line
        current.append(line)
        i += 1

    # Flush remaining
    if current:
        _flush_paragraphs(current, blocks)

    return blocks


def _flush_paragraphs(lines: list[str], blocks: list[str]) -> None:
    """Split accumulated lines at paragraph boundaries and add to blocks."""
    text = "\n".join(lines)
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        stripped = para.strip()
        if stripped:
            blocks.append(stripped)


def _split_oversized_section(
    content: str,
    heading_hierarchy: str,
    char_start: int,
    chunk_size: int,
) -> list[tuple[str, int, int, str]]:
    """Split an oversized section into sub-chunks at block boundaries.

    Uses :func:`_identify_blocks` to find atomic units (code blocks, tables,
    paragraphs), then applies the greedy accumulator to build sub-chunks
    that each fit within *chunk_size*.  Every sub-chunk inherits the parent
    section's *heading_hierarchy*.

    A single block that exceeds *chunk_size* is kept intact (never broken).

    Args:
        content:           Full text of the oversized section.
        heading_hierarchy: Heading hierarchy path for all sub-chunks.
        char_start:        Character offset of this section in the body.
        chunk_size:        Soft character limit per sub-chunk.

    Returns:
        List of ``(content, char_start, char_end, heading_hierarchy)`` tuples.
    """
    blocks = _identify_blocks(content)
    if not blocks:
        return [(content, char_start, char_start + len(content), heading_hierarchy)]

    sub_chunks: list[tuple[str, int, int, str]] = []
    current_blocks: list[str] = []
    current_size = 0

    for block in blocks:
        block_size = len(block)
        join_cost = 2 if current_blocks else 0  # "\n\n" separator

        if current_blocks and current_size + join_cost + block_size > chunk_size:
            # Finalize current sub-chunk
            sub_text = "\n\n".join(current_blocks)
            offset = content.find(current_blocks[0])
            if offset == -1:
                offset = 0
            sub_start = char_start + offset
            sub_end = sub_start + len(sub_text)
            sub_chunks.append((sub_text, sub_start, sub_end, heading_hierarchy))

            # Overlap: start new chunk with last block
            last = current_blocks[-1]
            current_blocks = [last]
            current_size = len(last)

        current_blocks.append(block)
        current_size += join_cost + block_size

    # Finalize last sub-chunk
    if current_blocks:
        sub_text = "\n\n".join(current_blocks)
        offset = content.find(current_blocks[0])
        if offset == -1:
            offset = 0
        sub_start = char_start + offset
        sub_end = sub_start + len(sub_text)
        sub_chunks.append((sub_text, sub_start, sub_end, heading_hierarchy))

    return sub_chunks


# ---------------------------------------------------------------------------
# Heading-aware chunking
# ---------------------------------------------------------------------------


def chunk_document_heading_aware(
    body: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[tuple[str, int, int, str]]:
    """Split normalised body into heading-bounded chunks with structure protection.

    Uses headings as primary split boundaries. Within each section:
    - Code blocks (fenced with ``` or ~~~) are never split
    - Tables (contiguous pipe-delimited lines) are never split
    - Sections under chunk_size are kept as atomic chunks

    If no headings are found, falls back to paragraph-aware chunking
    (via :func:`chunk_document`) with empty heading hierarchy.

    Args:
        body:       Post-normalisation document body.
        chunk_size: Soft character limit per chunk.
        overlap:    Soft overlap guideline (reserved for future use;
                    heading context provides natural overlap).

    Returns:
        List of ``(content, char_start, char_end, heading_hierarchy)`` tuples.
    """
    if not body or not body.strip():
        return []

    lines = body.split("\n")

    # Quick scan: check if any headings exist outside code fences.
    has_headings = False
    fence_check = False
    for line in lines:
        if is_code_fence(line):
            fence_check = not fence_check
            continue
        if not fence_check and detect_heading(line) is not None:
            has_headings = True
            break

    # Fallback: no headings at all -> delegate to paragraph-aware chunker.
    if not has_headings:
        raw = chunk_document(body, chunk_size, overlap)
        return [(content, start, end, "") for content, start, end in raw]

    # --- Main heading-aware walk ---
    heading_stack = HeadingStack()
    section_lines: list[str] = []
    section_start: int = 0
    current_heading_path: str = ""
    in_code_fence: bool = False
    char_offset: int = 0

    chunks: list[tuple[str, int, int, str]] = []

    def _finalize_section() -> None:
        """Finalize the current section as a chunk if it has non-empty content.

        If the section content exceeds *chunk_size*, splits it into
        sub-chunks at paragraph boundaries while treating code blocks
        and tables as atomic units.
        """
        if not section_lines:
            return
        content = "\n".join(section_lines).strip()
        if not content:
            return
        # Find char_start/char_end in original body.
        start = body.find(content, section_start)
        if start == -1:
            start = section_start
        end = start + len(content)

        if len(content) <= chunk_size:
            chunks.append((content, start, end, current_heading_path))
        else:
            # Oversized section: split at paragraph boundaries.
            sub_chunks = _split_oversized_section(
                content, current_heading_path, start, chunk_size,
            )
            chunks.extend(sub_chunks)

    for line in lines:
        line_len = len(line)

        # Code fence toggle (always append to current section).
        if is_code_fence(line):
            in_code_fence = not in_code_fence
            section_lines.append(line)
            char_offset += line_len + 1  # +1 for the \n
            continue

        # Inside code fence: always accumulate, never split.
        if in_code_fence:
            section_lines.append(line)
            char_offset += line_len + 1
            continue

        # Check for heading outside code fence.
        heading = detect_heading(line)
        if heading is not None:
            level, text = heading
            # Finalize the previous section.
            _finalize_section()
            # Update heading stack.
            heading_stack.push(level, text)
            current_heading_path = heading_stack.path()
            # Start new section with this heading line.
            section_lines = [line]
            section_start = char_offset
            char_offset += line_len + 1
            continue

        # Regular line: accumulate.
        section_lines.append(line)
        char_offset += line_len + 1

    # Finalize the last section.
    _finalize_section()

    return chunks


# ---------------------------------------------------------------------------
# Python code chunking
# ---------------------------------------------------------------------------


def chunk_python_code(
    content: str,
    chunk_size: int = 1000,
) -> list[tuple[str, int, int, str, str, int, int]]:
    """Split Python source code at function/class boundaries using ``ast``.

    Uses Python's ``ast`` module for reliable boundary detection.  Each
    top-level function, async function, or class becomes its own chunk.
    Module-level code (imports, constants, assignments outside any
    function/class) is collected into a "module" chunk.

    If a class exceeds *chunk_size*, its methods are extracted as
    individual chunks with ``symbol_type="method"`` and
    ``symbol_name="ClassName.method_name"``.

    Falls back to simple line-based splitting when ``ast.parse`` raises
    ``SyntaxError``.

    Args:
        content:    Python source code as a string.
        chunk_size: Soft character limit per chunk.

    Returns:
        List of 7-tuples:
        ``(content, char_start, char_end, symbol_name, symbol_type, line_start, line_end)``
    """
    lines = content.splitlines()

    # --- AST fallback ---------------------------------------------------
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning("ast.parse failed, falling back to line-based splitting")
        return _python_line_fallback(content, lines, chunk_size)

    # Collect top-level node spans
    nodes: list[tuple[int, int, str, str]] = []  # (start, end, name, type)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append((node.lineno, node.end_lineno or node.lineno, node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            nodes.append((node.lineno, node.end_lineno or node.lineno, node.name, "class"))

    # Sort by start line
    nodes.sort(key=lambda n: n[0])

    chunks: list[tuple[str, int, int, str, str, int, int]] = []

    # Collect module-level code (lines not inside any node)
    covered = set()
    for start_line, end_line, _name, _stype in nodes:
        for ln in range(start_line, end_line + 1):
            covered.add(ln)

    # Module-level lines
    module_lines: list[str] = []
    module_line_nums: list[int] = []
    for i, line in enumerate(lines, start=1):
        if i not in covered:
            module_lines.append(line)
            module_line_nums.append(i)

    if module_lines:
        mod_text = "\n".join(module_lines).strip()
        if mod_text:
            mod_start = _char_offset_for_line(lines, module_line_nums[0])
            mod_end = _char_offset_for_line(lines, module_line_nums[-1]) + len(lines[module_line_nums[-1] - 1])
            chunks.append((mod_text, mod_start, mod_end, "", "module", module_line_nums[0], module_line_nums[-1]))

    # Process each node
    for start_line, end_line, name, stype in nodes:
        node_text = "\n".join(lines[start_line - 1 : end_line]).rstrip()
        char_start = _char_offset_for_line(lines, start_line)
        char_end = char_start + len(node_text)

        if stype == "class" and len(node_text) > chunk_size:
            # Split oversized class into methods
            class_chunks = _split_class_into_methods(lines, start_line, end_line, name)
            chunks.extend(class_chunks)
        else:
            # Include decorator lines if any
            actual_start = _find_decorator_start(lines, start_line)
            if actual_start < start_line:
                node_text = "\n".join(lines[actual_start - 1 : end_line]).rstrip()
                char_start = _char_offset_for_line(lines, actual_start)
                char_end = char_start + len(node_text)

            chunks.append((node_text, char_start, char_end, name, stype, actual_start, end_line))

    # Sort by char_start
    chunks.sort(key=lambda c: c[1])
    return chunks


def _char_offset_for_line(lines: list[str], line_num: int) -> int:
    """Compute the character offset of the start of a 1-based line number."""
    offset = 0
    for i in range(line_num - 1):
        offset += len(lines[i]) + 1  # +1 for newline
    return offset


def _find_decorator_start(lines: list[str], func_line: int) -> int:
    """Walk backwards to find decorators above a function/class definition.

    Args:
        lines:     All source lines.
        func_line: 1-based line number of the def/class line.

    Returns:
        1-based line number of the first decorator, or func_line if none.
    """
    start = func_line
    i = func_line - 2  # 0-based index of line above
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("@"):
            start = i + 1  # Convert back to 1-based
            i -= 1
        else:
            break
    return start


def _split_class_into_methods(
    lines: list[str],
    class_start: int,
    class_end: int,
    class_name: str,
) -> list[tuple[str, int, int, str, str, int, int]]:
    """Split an oversized class into method-level chunks.

    Parses the class body to find method definitions and extracts each as
    a separate chunk with ``symbol_type="method"`` and
    ``symbol_name="ClassName.method_name"``.

    Args:
        lines:       All source lines.
        class_start: 1-based start line of the class.
        class_end:   1-based end line of the class.
        class_name:  The class name.

    Returns:
        List of 7-tuples for each method (and optionally the class header).
    """
    chunks: list[tuple[str, int, int, str, str, int, int]] = []
    class_text = "\n".join(lines[class_start - 1 : class_end])

    try:
        tree = ast.parse(class_text)
    except SyntaxError:
        # Can't parse the class — return as single chunk
        char_start = _char_offset_for_line(lines, class_start)
        return [(class_text.rstrip(), char_start, char_start + len(class_text.rstrip()),
                 class_name, "class", class_start, class_end)]

    cls_nodes = list(ast.iter_child_nodes(tree))
    if not cls_nodes:
        char_start = _char_offset_for_line(lines, class_start)
        return [(class_text.rstrip(), char_start, char_start + len(class_text.rstrip()),
                 class_name, "class", class_start, class_end)]

    # Find the class definition node
    cls_def = None
    for node in cls_nodes:
        if isinstance(node, ast.ClassDef):
            cls_def = node
            break

    if cls_def is None:
        char_start = _char_offset_for_line(lines, class_start)
        return [(class_text.rstrip(), char_start, char_start + len(class_text.rstrip()),
                 class_name, "class", class_start, class_end)]

    methods = [
        n for n in ast.iter_child_nodes(cls_def)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not methods:
        char_start = _char_offset_for_line(lines, class_start)
        return [(class_text.rstrip(), char_start, char_start + len(class_text.rstrip()),
                 class_name, "class", class_start, class_end)]

    # Class header = everything before first method (adjusted to original lines)
    first_method_line = class_start + methods[0].lineno - 1
    header_text = "\n".join(lines[class_start - 1 : first_method_line - 1]).rstrip()
    if header_text:
        char_start = _char_offset_for_line(lines, class_start)
        chunks.append((header_text, char_start, char_start + len(header_text),
                       class_name, "class", class_start, first_method_line - 1))

    # Each method as a separate chunk
    for method in methods:
        meth_start = class_start + method.lineno - 1
        meth_end = class_start + (method.end_lineno or method.lineno) - 1
        meth_text = "\n".join(lines[meth_start - 1 : meth_end]).rstrip()
        char_start = _char_offset_for_line(lines, meth_start)
        symbol = f"{class_name}.{method.name}"
        chunks.append((meth_text, char_start, char_start + len(meth_text),
                       symbol, "method", meth_start, meth_end))

    return chunks


def _python_line_fallback(
    content: str,
    lines: list[str],
    chunk_size: int,
) -> list[tuple[str, int, int, str, str, int, int]]:
    """Line-based fallback for Python files that fail AST parsing."""
    chunks: list[tuple[str, int, int, str, str, int, int]] = []
    current_lines: list[str] = []
    current_start = 1
    current_chars = 0

    for i, line in enumerate(lines):
        line_len = len(line) + 1
        if current_chars + line_len > chunk_size and current_lines:
            text = "\n".join(current_lines).rstrip()
            char_start = _char_offset_for_line(lines, current_start)
            chunks.append((text, char_start, char_start + len(text),
                           "", "module", current_start, current_start + len(current_lines) - 1))
            current_lines = [line]
            current_start = i + 1
            current_chars = line_len
        else:
            current_lines.append(line)
            current_chars += line_len

    if current_lines:
        text = "\n".join(current_lines).rstrip()
        char_start = _char_offset_for_line(lines, current_start)
        chunks.append((text, char_start, char_start + len(text),
                       "", "module", current_start, current_start + len(current_lines) - 1))

    return chunks


# ---------------------------------------------------------------------------
# TypeScript / JavaScript code chunking
# ---------------------------------------------------------------------------


def _find_block_end(lines: list[str], start_line: int) -> int:
    """Find end of a brace-delimited block using brace counting.

    Scans from *start_line* forward, counting ``{`` and ``}`` characters.
    Returns the line index (exclusive) where the brace count returns to
    zero (or fewer), meaning the top-level block has closed.

    Args:
        lines:      All source lines of the file.
        start_line: 0-based index of the line where the block opens.

    Returns:
        Exclusive end line index.  Falls back to ``len(lines)`` if braces
        never balance (e.g. syntax error in source).
    """
    brace_count = 0
    for i in range(start_line, len(lines)):
        brace_count += lines[i].count("{") - lines[i].count("}")
        if brace_count <= 0 and i > start_line:
            return i + 1
    return len(lines)


def _match_ts_boundary(line: str) -> tuple[str, str] | None:
    """Try to match a TypeScript/JavaScript boundary pattern on a line.

    Only matches lines at column 0 (no leading whitespace).

    Args:
        line: A single source line.

    Returns:
        ``(symbol_name, symbol_type)`` if matched, ``None`` otherwise.
    """
    if not line or line[0].isspace():
        return None

    for pattern, symbol_type, name_kind in _TS_BOUNDARY_PATTERNS:
        m = pattern.match(line)
        if m:
            if name_kind == "group1_or_default":
                name = m.group(1) if m.group(1) else "default"
            else:
                name = m.group(1)
            return name, symbol_type
    return None


def _attach_leading_comments(lines: list[str], boundary_line: int) -> int:
    """Walk backwards from *boundary_line* to include JSDoc or line comments.

    Attaches contiguous ``/** ... */`` blocks and ``//`` comment lines that
    immediately precede the boundary.

    Args:
        lines:         All source lines.
        boundary_line: 0-based index of the detected boundary line.

    Returns:
        Adjusted start line (may equal *boundary_line* if no comments found).
    """
    start = boundary_line
    i = boundary_line - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped == "*/":
            start = i
            i -= 1
        elif stripped.startswith("/**"):
            start = i
            break
        else:
            break
    return start


def chunk_typescript_code(
    content: str,
    chunk_size: int = 1000,
) -> list[tuple[str, int, int, str, str, int, int]]:
    """Split TypeScript/JavaScript source at function/class/export boundaries.

    Uses regex-based detection for common TS/JS patterns at column 0.
    Falls back to line-based splitting for files with no detected boundaries.

    Algorithm:
        1. Scan each line for boundary patterns (function, class, interface,
           type, enum, arrow function const, export default).
        2. For each boundary, attach any preceding JSDoc/comment lines.
        3. Use brace counting ``{``/``}`` to find the block end.
        4. Collect module-level code between boundaries as "module" chunks.
        5. If no boundaries found, fall back to line-based splitting.

    Args:
        content:    Raw TypeScript or JavaScript source code.
        chunk_size: Soft character limit per chunk (used for line-based
                    fallback when no boundaries are detected).

    Returns:
        A list of 7-tuples:
        ``(content, char_start, char_end, symbol_name, symbol_type, line_start, line_end)``
        where *line_start* and *line_end* are 1-based line numbers.
    """
    if not content.strip():
        return []

    lines = content.split("\n")

    # Phase 1: Detect all boundary positions
    boundaries: list[tuple[int, str, str]] = []  # (line_idx, name, type)
    for idx, line in enumerate(lines):
        result = _match_ts_boundary(line)
        if result:
            boundaries.append((idx, result[0], result[1]))

    # Phase 2: Fallback to line-based splitting if no boundaries found
    if not boundaries:
        return _line_based_fallback(content, lines, chunk_size)

    # Phase 3: Build chunks from boundaries
    chunks: list[tuple[str, int, int, str, str, int, int]] = []

    # Compute character offset for each line start
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1  # +1 for newline

    # Track the end of the last boundary block for module-level code detection
    last_block_end = 0

    for b_idx, (line_idx, sym_name, sym_type) in enumerate(boundaries):
        # Attach leading comments (JSDoc, //)
        chunk_start_line = _attach_leading_comments(lines, line_idx)

        # Module-level code between previous block end and this boundary
        if chunk_start_line > last_block_end:
            module_lines = lines[last_block_end:chunk_start_line]
            module_text = "\n".join(module_lines).strip()
            if module_text:
                char_start = line_offsets[last_block_end]
                char_end = line_offsets[chunk_start_line - 1] + len(lines[chunk_start_line - 1])
                chunks.append((
                    module_text,
                    char_start,
                    char_end,
                    "module",
                    "module",
                    last_block_end + 1,  # 1-based
                    chunk_start_line,    # 1-based (inclusive)
                ))

        # Find block end using brace counting
        block_end_line = _find_block_end(lines, line_idx)

        # Build the chunk content
        chunk_lines = lines[chunk_start_line:block_end_line]
        chunk_text = "\n".join(chunk_lines)
        char_start = line_offsets[chunk_start_line]
        # char_end: end of last line in the block
        end_line_idx = block_end_line - 1
        if end_line_idx < len(lines):
            char_end = line_offsets[end_line_idx] + len(lines[end_line_idx])
        else:
            char_end = len(content)

        chunks.append((
            chunk_text,
            char_start,
            char_end,
            sym_name,
            sym_type,
            chunk_start_line + 1,  # 1-based
            block_end_line,        # 1-based (inclusive)
        ))

        last_block_end = block_end_line

    # Trailing module-level code after the last boundary block
    if last_block_end < len(lines):
        trailing_lines = lines[last_block_end:]
        trailing_text = "\n".join(trailing_lines).strip()
        if trailing_text:
            char_start = line_offsets[last_block_end]
            char_end = len(content)
            chunks.append((
                trailing_text,
                char_start,
                char_end,
                "module",
                "module",
                last_block_end + 1,
                len(lines),
            ))

    return chunks


def _line_based_fallback(
    content: str,
    lines: list[str],
    chunk_size: int,
) -> list[tuple[str, int, int, str, str, int, int]]:
    """Split source into fixed-size line groups when no boundaries are found.

    Args:
        content:    Full source text.
        lines:      Pre-split lines of source.
        chunk_size: Target character count per chunk.

    Returns:
        List of 7-tuples with symbol_name="module" and symbol_type="module".
    """
    chunks: list[tuple[str, int, int, str, str, int, int]] = []

    # Compute character offset for each line
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1

    current_lines: list[str] = []
    current_start = 0
    current_size = 0

    for idx, line in enumerate(lines):
        line_len = len(line) + 1  # +1 for newline
        if current_lines and current_size + line_len > chunk_size:
            # Finalize current chunk
            chunk_text = "\n".join(current_lines)
            char_start = line_offsets[current_start]
            end_idx = idx - 1
            char_end = line_offsets[end_idx] + len(lines[end_idx])
            chunks.append((
                chunk_text,
                char_start,
                char_end,
                "module",
                "module",
                current_start + 1,  # 1-based
                idx,                # 1-based (inclusive)
            ))
            current_lines = []
            current_start = idx
            current_size = 0

        current_lines.append(line)
        current_size += line_len

    # Finalize last chunk
    if current_lines:
        chunk_text = "\n".join(current_lines)
        char_start = line_offsets[current_start]
        char_end = len(content)
        chunks.append((
            chunk_text,
            char_start,
            char_end,
            "module",
            "module",
            current_start + 1,
            len(lines),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Code file ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_code_file(
    file_path: Path,
    project_root: Path,
    chunk_size: int = 1000,
) -> list[DocumentChunk]:
    """Ingest a source code file into a list of :class:`DocumentChunk` objects.

    Routes to language-specific chunkers based on file extension:
    - ``.py`` -> ``chunk_python_code()`` (AST-based)
    - ``.ts``, ``.tsx``, ``.js``, ``.jsx`` -> ``chunk_typescript_code()`` (regex-based)

    Args:
        file_path:    Path to a source code file.
        project_root: Root directory for computing relative ``source_file``.
        chunk_size:   Soft character limit per chunk.

    Returns:
        A list of :class:`DocumentChunk` instances with language, symbol_name,
        symbol_type, line_start, and line_end populated.  Empty list if the
        file cannot be read or has no content.
    """
    suffix = file_path.suffix.lower()
    language = _LANG_MAP.get(suffix)
    if not language:
        logger.warning("Unsupported code file extension: %s", suffix)
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping non-UTF-8 code file: %s", file_path)
        return []

    if not content.strip():
        logger.warning("Empty code file: %s", file_path)
        return []

    source = file_path.relative_to(project_root).as_posix()

    # Route to the appropriate language-specific chunker
    if language == "python":
        raw_chunks = chunk_python_code(content, chunk_size=chunk_size)
    elif language in ("typescript", "javascript"):
        raw_chunks = chunk_typescript_code(content, chunk_size=chunk_size)
    else:
        logger.warning("No chunker for language: %s", language)
        return []

    return [
        DocumentChunk(
            content=chunk_content,
            source_file=source,
            chunk_index=i,
            char_start=char_start,
            char_end=char_end,
            total_chunks=len(raw_chunks),
            language=language,
            symbol_name=sym_name,
            symbol_type=sym_type,
            line_start=line_start,
            line_end=line_end,
        )
        for i, (chunk_content, char_start, char_end, sym_name, sym_type, line_start, line_end) in enumerate(raw_chunks)
    ]


# ---------------------------------------------------------------------------
# File-level ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_file(
    file_path: Path,
    project_root: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Ingest a single file into a list of :class:`DocumentChunk` objects.

    Pipeline:

    1. Read *file_path* as UTF-8 text (skip non-UTF-8 files with a warning).
    2. Parse YAML frontmatter and normalise the body via :func:`parse_document`.
    3. If the body is empty after normalisation, return an empty list.
    4. Chunk the body via :func:`chunk_document`.
    5. Build :class:`DocumentChunk` instances with source-relative POSIX path,
       chunk indices, character offsets, and frontmatter metadata.

    Args:
        file_path:    Absolute or relative path to a markdown file.
        project_root: Root directory for computing relative ``source_file``.
        chunk_size:   Soft character limit per chunk (forwarded to
                      :func:`chunk_document`).
        overlap:      Soft overlap guideline (forwarded to
                      :func:`chunk_document`).

    Returns:
        A list of :class:`DocumentChunk` instances.  Empty list if the file
        cannot be read, has no body content, or is not valid UTF-8.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping non-UTF-8 file: %s", file_path)
        return []

    metadata, body = parse_document(text)

    if not body.strip():
        logger.warning("Empty document body: %s", file_path)
        return []

    raw_chunks = chunk_document_heading_aware(body, chunk_size=chunk_size, overlap=overlap)
    source = file_path.relative_to(project_root).as_posix()

    return [
        DocumentChunk(
            content=content,
            source_file=source,
            chunk_index=i,
            char_start=char_start,
            char_end=char_end,
            total_chunks=len(raw_chunks),
            frontmatter=metadata,
            heading_hierarchy=heading_hierarchy,
        )
        for i, (content, char_start, char_end, heading_hierarchy) in enumerate(raw_chunks)
    ]
