"""Tests for tokenkeeper.discovery module.

Covers: discover_files (generic extension-based discovery), discover_markdown_files
(recursive walk, exclusion pruning, sorting, case-insensitive .md matching),
discover_code_files (source code discovery), resolve_paths (empty paths, directory
paths, glob patterns, specific files, deduplication), CODE_EXTENSIONS, and
EXCLUDED_DIRS constants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenkeeper.config import PathOverride, RagConfig
from tokenkeeper.discovery import (
    CODE_EXTENSIONS,
    EXCLUDED_DIRS,
    discover_code_files,
    discover_files,
    discover_markdown_files,
    resolve_paths,
)


# ---------------------------------------------------------------------------
# EXCLUDED_DIRS constant
# ---------------------------------------------------------------------------


class TestExcludedDirs:
    def test_excluded_dirs_is_frozenset(self) -> None:
        assert isinstance(EXCLUDED_DIRS, frozenset)

    def test_excluded_dirs_contains_expected(self) -> None:
        expected = {".rag", ".git", "node_modules", "__pycache__", ".venv", "venv", ".env"}
        assert EXCLUDED_DIRS == expected


# ---------------------------------------------------------------------------
# discover_markdown_files
# ---------------------------------------------------------------------------


class TestDiscoverMarkdownFiles:
    def test_finds_md_files_in_root(self, tmp_path: Path) -> None:
        """Finds .md files directly in root directory."""
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "code.py").write_text("print('hello')")

        result = discover_markdown_files(tmp_path)
        names = [f.name for f in result]
        assert "readme.md" in names
        assert "notes.md" in names
        assert "code.py" not in names

    def test_finds_md_files_recursively(self, tmp_path: Path) -> None:
        """Discovers .md files in nested subdirectories."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "docs" / "sub").mkdir()
        (tmp_path / "docs" / "sub" / "deep.md").write_text("# Deep")
        (tmp_path / "top.md").write_text("# Top")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 3
        names = [f.name for f in result]
        assert "guide.md" in names
        assert "deep.md" in names
        assert "top.md" in names

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        """Output list is sorted for deterministic results."""
        (tmp_path / "c.md").write_text("c")
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")

        result = discover_markdown_files(tmp_path)
        assert result == sorted(result)

    def test_case_insensitive_md_extension(self, tmp_path: Path) -> None:
        """Matches .MD, .Md, .mD variants (case-insensitive)."""
        (tmp_path / "upper.MD").write_text("upper")
        (tmp_path / "mixed.Md").write_text("mixed")
        (tmp_path / "lower.md").write_text("lower")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 3

    def test_excludes_git_directory(self, tmp_path: Path) -> None:
        """Files inside .git/ are excluded."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD.md").write_text("git internal")
        (tmp_path / "readme.md").write_text("# Readme")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "readme.md"

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        """Files inside node_modules/ are excluded."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg").mkdir()
        (tmp_path / "node_modules" / "pkg" / "README.md").write_text("pkg readme")
        (tmp_path / "docs.md").write_text("# Docs")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "docs.md"

    def test_excludes_all_default_dirs(self, tmp_path: Path) -> None:
        """All EXCLUDED_DIRS are pruned from traversal."""
        for dirname in EXCLUDED_DIRS:
            d = tmp_path / dirname
            d.mkdir()
            (d / "hidden.md").write_text("hidden")

        (tmp_path / "visible.md").write_text("visible")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "visible.md"

    def test_custom_exclusion_set(self, tmp_path: Path) -> None:
        """Accepts a custom frozenset to override default exclusions."""
        (tmp_path / "keep").mkdir()
        (tmp_path / "keep" / "file.md").write_text("kept")
        (tmp_path / "skip").mkdir()
        (tmp_path / "skip" / "file.md").write_text("skipped")

        result = discover_markdown_files(tmp_path, excluded=frozenset({"skip"}))
        assert len(result) == 1
        names = [f.name for f in result]
        assert "file.md" in names
        # Verify the kept file is from "keep" dir
        assert "keep" in str(result[0])

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Returns empty list when no .md files exist."""
        (tmp_path / "code.py").write_text("print()")

        result = discover_markdown_files(tmp_path)
        assert result == []

    def test_pruning_uses_slice_assignment(self, tmp_path: Path) -> None:
        """Excluded dirs are genuinely pruned (not just filtered post-walk).

        If slice assignment is used correctly, subdirectories of excluded
        dirs are never entered. We verify by checking that deeply nested
        files inside excluded dirs are not found.
        """
        # Create deep nesting inside excluded dir
        deep = tmp_path / ".git" / "refs" / "heads"
        deep.mkdir(parents=True)
        (deep / "main.md").write_text("branch ref")

        # Create a normal file
        (tmp_path / "readme.md").write_text("# Readme")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "readme.md"

    def test_non_md_files_ignored(self, tmp_path: Path) -> None:
        """Only .md files are collected, other extensions are skipped."""
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "script.py").write_text("print()")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "styles.css").write_text("body {}")
        (tmp_path / "notes.txt").write_text("notes")

        result = discover_markdown_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "readme.md"


# ---------------------------------------------------------------------------
# resolve_paths
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, paths: list[str] | None = None) -> RagConfig:
    """Helper to create a RagConfig with given paths."""
    overrides = [PathOverride(path=p) for p in paths] if paths else []
    return RagConfig(project_root=tmp_path, paths=overrides)


class TestResolvePaths:
    def test_empty_paths_scans_project_root(self, tmp_path: Path) -> None:
        """When config.paths is empty, scans entire project_root."""
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")

        config = _make_config(tmp_path)
        result = resolve_paths(config)
        assert len(result) == 2

    def test_directory_path_scans_recursively(self, tmp_path: Path) -> None:
        """A directory path in config.paths scans that directory."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "docs" / "sub").mkdir()
        (tmp_path / "docs" / "sub" / "deep.md").write_text("# Deep")
        (tmp_path / "other.md").write_text("# Other")

        config = _make_config(tmp_path, paths=["docs"])
        result = resolve_paths(config)
        assert len(result) == 2
        names = [f.name for f in result]
        assert "guide.md" in names
        assert "deep.md" in names
        assert "other.md" not in names

    def test_glob_pattern_expands(self, tmp_path: Path) -> None:
        """Glob patterns with * expand to matching .md files."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "api.md").write_text("# API")
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "docs" / "data.json").write_text("{}")

        config = _make_config(tmp_path, paths=["docs/*.md"])
        result = resolve_paths(config)
        assert len(result) == 2
        names = [f.name for f in result]
        assert "api.md" in names
        assert "guide.md" in names

    def test_specific_file_path(self, tmp_path: Path) -> None:
        """A specific .md file path is included directly."""
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "other.md").write_text("# Other")

        config = _make_config(tmp_path, paths=["readme.md"])
        result = resolve_paths(config)
        assert len(result) == 1
        assert result[0].name == "readme.md"

    def test_overlapping_paths_deduplicated(self, tmp_path: Path) -> None:
        """Overlapping paths produce deduplicated results."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")

        # Both "docs" dir and specific file point to same .md
        config = _make_config(tmp_path, paths=["docs", "docs/guide.md"])
        result = resolve_paths(config)
        assert len(result) == 1

    def test_nonexistent_path_logged_and_skipped(self, tmp_path: Path) -> None:
        """Non-existent paths are skipped with a warning."""
        (tmp_path / "readme.md").write_text("# Readme")

        config = _make_config(tmp_path, paths=["nonexistent.md", "readme.md"])
        result = resolve_paths(config)
        assert len(result) == 1
        assert result[0].name == "readme.md"

    def test_non_md_specific_file_skipped(self, tmp_path: Path) -> None:
        """Specific file paths that aren't .md are skipped."""
        (tmp_path / "data.json").write_text("{}")

        config = _make_config(tmp_path, paths=["data.json"])
        result = resolve_paths(config)
        assert result == []

    def test_glob_with_question_mark(self, tmp_path: Path) -> None:
        """Glob patterns with ? single-char wildcard expand correctly."""
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")
        (tmp_path / "document.md").write_text("# Document")

        config = _make_config(tmp_path, paths=["doc?.md"])
        result = resolve_paths(config)
        assert len(result) == 2
        names = [f.name for f in result]
        assert "doc1.md" in names
        assert "doc2.md" in names
        # "document.md" does NOT match doc?.md (too many chars after "doc")
        assert "document.md" not in names

    def test_results_sorted(self, tmp_path: Path) -> None:
        """Final results are sorted."""
        (tmp_path / "z.md").write_text("z")
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "m.md").write_text("m")

        config = _make_config(tmp_path)
        result = resolve_paths(config)
        assert result == sorted(result)

    def test_mixed_path_types(self, tmp_path: Path) -> None:
        """Handles a mix of directories, globs, and specific files."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "todo.md").write_text("# TODO")
        (tmp_path / "readme.md").write_text("# Readme")

        config = _make_config(tmp_path, paths=["docs", "notes/*.md", "readme.md"])
        result = resolve_paths(config)
        assert len(result) == 3
        names = [f.name for f in result]
        assert "guide.md" in names
        assert "todo.md" in names
        assert "readme.md" in names


# ---------------------------------------------------------------------------
# CODE_EXTENSIONS constant
# ---------------------------------------------------------------------------


class TestCodeExtensions:
    def test_code_extensions_is_frozenset(self) -> None:
        assert isinstance(CODE_EXTENSIONS, frozenset)

    def test_code_extensions_contains_expected(self) -> None:
        """Core language extensions are present in the constant."""
        for ext in (".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go",
                     ".java", ".rb", ".cpp", ".c", ".h", ".hpp",
                     ".cs", ".swift", ".kt"):
            assert ext in CODE_EXTENSIONS, f"{ext} missing from CODE_EXTENSIONS"

    def test_code_extensions_does_not_contain_markdown(self) -> None:
        """Markdown is not a code extension."""
        assert ".md" not in CODE_EXTENSIONS


# ---------------------------------------------------------------------------
# discover_files (generic)
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    def test_discover_files_by_extension(self, tmp_path: Path) -> None:
        """Finds only files matching the requested extension set."""
        (tmp_path / "app.py").write_text("print('hello')")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "style.css").write_text("body {}")

        result = discover_files(tmp_path, frozenset({".py"}))
        names = [f.name for f in result]
        assert "app.py" in names
        assert "readme.md" not in names
        assert "style.css" not in names

    def test_discover_files_multiple_extensions(self, tmp_path: Path) -> None:
        """Finds files matching any of several extensions."""
        (tmp_path / "app.py").write_text("print()")
        (tmp_path / "index.ts").write_text("export {}")
        (tmp_path / "readme.md").write_text("# Readme")

        result = discover_files(tmp_path, frozenset({".py", ".ts"}))
        names = [f.name for f in result]
        assert "app.py" in names
        assert "index.ts" in names
        assert "readme.md" not in names

    def test_discover_files_respects_exclusions(self, tmp_path: Path) -> None:
        """Excluded directories are pruned from traversal."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "dep.py").write_text("# dep")
        (tmp_path / "app.py").write_text("# app")

        result = discover_files(tmp_path, frozenset({".py"}))
        names = [f.name for f in result]
        assert "app.py" in names
        assert "dep.py" not in names

    def test_discover_files_empty_result(self, tmp_path: Path) -> None:
        """Returns empty list when no files match."""
        (tmp_path / "readme.md").write_text("# Readme")

        result = discover_files(tmp_path, frozenset({".py"}))
        assert result == []

    def test_discover_files_results_sorted(self, tmp_path: Path) -> None:
        """Output is sorted for deterministic results."""
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        (tmp_path / "m.py").write_text("")

        result = discover_files(tmp_path, frozenset({".py"}))
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# discover_code_files
# ---------------------------------------------------------------------------


class TestDiscoverCodeFiles:
    def test_discover_code_files_python(self, tmp_path: Path) -> None:
        """Discovers Python source files."""
        (tmp_path / "app.py").write_text("print('hello')")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "utils.py").write_text("def helper(): pass")

        result = discover_code_files(tmp_path)
        names = [f.name for f in result]
        assert "app.py" in names
        assert "utils.py" in names

    def test_discover_code_files_typescript(self, tmp_path: Path) -> None:
        """Discovers TypeScript source files."""
        (tmp_path / "index.ts").write_text("export const x = 1;")
        (tmp_path / "component.tsx").write_text("export default function() {}")

        result = discover_code_files(tmp_path)
        names = [f.name for f in result]
        assert "index.ts" in names
        assert "component.tsx" in names

    def test_discover_code_files_excludes_dirs(self, tmp_path: Path) -> None:
        """Files in excluded directories (e.g., node_modules) are not found."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib").mkdir()
        (tmp_path / "node_modules" / "lib" / "index.js").write_text("// dep")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.cpython-312.pyc").write_text("")
        (tmp_path / "app.py").write_text("# app")

        result = discover_code_files(tmp_path)
        names = [f.name for f in result]
        assert "app.py" in names
        assert "index.js" not in names

    def test_discover_code_files_mixed(self, tmp_path: Path) -> None:
        """Discovers code files (.py, .ts, .json, .mjs) but not markdown (.md)."""
        (tmp_path / "app.py").write_text("print()")
        (tmp_path / "index.ts").write_text("export {}")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "config.mjs").write_text("export default {}")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        result = discover_code_files(tmp_path)
        names = [f.name for f in result]
        assert "app.py" in names
        assert "index.ts" in names
        assert "data.json" in names
        assert "config.mjs" in names
        assert "readme.md" not in names
        assert "image.png" not in names

    def test_discover_code_files_various_languages(self, tmp_path: Path) -> None:
        """Discovers files from various supported languages."""
        for ext in (".rs", ".go", ".java", ".rb", ".cpp", ".c", ".h",
                     ".hpp", ".cs", ".swift", ".kt", ".js", ".jsx"):
            (tmp_path / f"file{ext}").write_text(f"// {ext}")

        result = discover_code_files(tmp_path)
        assert len(result) == 13  # All 13 extensions above


# ---------------------------------------------------------------------------
# discover_markdown_files backward compatibility
# ---------------------------------------------------------------------------


class TestDiscoverMarkdownStillWorks:
    def test_discover_markdown_still_works(self, tmp_path: Path) -> None:
        """discover_markdown_files() still finds .md files after refactor."""
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "app.py").write_text("print()")

        result = discover_markdown_files(tmp_path)
        names = [f.name for f in result]
        assert "readme.md" in names
        assert "guide.md" in names
        assert "app.py" not in names
        assert len(result) == 2

    def test_discover_markdown_custom_exclusion(self, tmp_path: Path) -> None:
        """Custom exclusion set still works after refactor."""
        (tmp_path / "keep").mkdir()
        (tmp_path / "keep" / "file.md").write_text("kept")
        (tmp_path / "skip").mkdir()
        (tmp_path / "skip" / "file.md").write_text("skipped")

        result = discover_markdown_files(tmp_path, excluded=frozenset({"skip"}))
        assert len(result) == 1
        assert "keep" in str(result[0])
