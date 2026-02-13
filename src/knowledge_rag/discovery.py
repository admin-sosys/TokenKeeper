"""File discovery for Knowledge RAG.

Walks project directories to find files by extension, handles glob expansion,
exclusion pruning, and deduplication.

Exports:
    EXCLUDED_DIRS            -- Default directories to exclude from traversal
    CODE_EXTENSIONS          -- Supported source code file extensions
    discover_files           -- Generic file finder by extension set
    discover_markdown_files  -- Recursively find .md files under a root
    discover_code_files      -- Recursively find source code files under a root
    resolve_paths            -- Resolve configured paths to a deduplicated file list
"""

from __future__ import annotations

import logging
from pathlib import Path

from knowledge_rag.config import RagConfig

logger = logging.getLogger("knowledge_rag.discovery")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".rag",
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
})

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".rs", ".go", ".java", ".rb",
    ".cpp", ".c", ".h", ".hpp",
    ".cs", ".swift", ".kt",
    ".json",
})


# ---------------------------------------------------------------------------
# Directory walking
# ---------------------------------------------------------------------------


def discover_files(
    root: Path,
    extensions: frozenset[str],
    excluded: frozenset[str] = EXCLUDED_DIRS,
) -> list[Path]:
    """Recursively find files with specified extensions under *root*.

    Uses Python 3.12 ``Path.walk()`` with **slice assignment** on
    ``dirnames`` to prune excluded directories so they are never entered.

    Args:
        root: Directory to walk.
        extensions: File extensions to include (e.g., ``{".py", ".ts"}``).
        excluded: Directory names to prune from traversal.

    Returns:
        Sorted list of ``Path`` objects for matching files.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in root.walk():
        # CRITICAL: slice assignment prunes in-place so walk() skips them.
        # Plain reassignment (dirnames = ...) would NOT prune.
        dirnames[:] = sorted(d for d in dirnames if d not in excluded)
        for name in sorted(filenames):
            if any(name.lower().endswith(ext) for ext in extensions):
                files.append(dirpath / name)
    logger.info("Discovered %d files (%s) under %s", len(files), extensions, root)
    return files


def discover_markdown_files(
    root: Path,
    excluded: frozenset[str] = EXCLUDED_DIRS,
) -> list[Path]:
    """Recursively find all markdown files under *root*.

    Uses Python 3.12 ``Path.walk()`` with **slice assignment** on
    ``dirnames`` to prune excluded directories so they are never entered.

    Args:
        root: Directory to walk.
        excluded: Directory names to prune from traversal.

    Returns:
        Sorted list of ``Path`` objects for all ``.md`` files found.
    """
    return discover_files(root, frozenset({".md", ".mdx"}), excluded)


def discover_code_files(
    root: Path,
    excluded: frozenset[str] = EXCLUDED_DIRS,
) -> list[Path]:
    """Recursively find all source code files under *root*.

    Delegates to :func:`discover_files` with :data:`CODE_EXTENSIONS`.

    Args:
        root: Directory to walk.
        excluded: Directory names to prune from traversal.

    Returns:
        Sorted list of ``Path`` objects for source code files.
    """
    return discover_files(root, CODE_EXTENSIONS, excluded)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_paths(config: RagConfig) -> list[Path]:
    """Resolve configured paths to a sorted, deduplicated file list.

    Behaviour:
    - If ``config.paths`` is empty, scans the entire ``config.project_root``.
    - For each configured path override:
      - **Directory:** recursively discover markdown files within it.
      - **Glob pattern** (contains ``*`` or ``?``): expand relative to
        ``project_root`` and collect matching ``.md`` files.
      - **Specific file:** include if it exists and ends in ``.md``.
    - All paths are deduplicated by their resolved absolute path.
    - The final list is sorted for deterministic output.

    Args:
        config: A :class:`RagConfig` instance with ``project_root`` and
                ``paths`` fields.

    Returns:
        Sorted, deduplicated list of ``Path`` objects.
    """
    if not config.paths:
        return discover_markdown_files(config.project_root)

    seen: set[Path] = set()
    result: list[Path] = []

    for override in config.paths:
        target = config.project_root / override.path

        if target.is_dir():
            for f in discover_markdown_files(target):
                resolved = f.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    result.append(f)
        elif "*" in override.path or "?" in override.path:
            for f in sorted(config.project_root.glob(override.path)):
                if f.is_file() and f.name.lower().endswith(".md"):
                    resolved = f.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        result.append(f)
        elif target.is_file() and target.name.lower().endswith(".md"):
            resolved = target.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(target)
        else:
            logger.warning("Path not found or not a markdown file: %s", override.path)

    return sorted(result)
