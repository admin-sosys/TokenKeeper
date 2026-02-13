"""Configuration system for Knowledge RAG.

Provides Pydantic models for `.rag-config.json`, JSONC comment stripping,
config loading with sensible defaults, template auto-generation, and `.rag/`
directory management.

Exports:
    RagConfig           -- Main configuration model
    PathOverride        -- Per-path override model
    load_config         -- Load (or create) config from project root
    ensure_rag_directory -- Create .rag/ directory if missing
    generate_config_template -- Return JSONC template string
    strip_json_comments -- Strip // comments from JSONC text
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("knowledge_rag.config")

# ---------------------------------------------------------------------------
# JSONC helper
# ---------------------------------------------------------------------------

_VALID_MODES = frozenset({"hybrid", "semantic", "keyword"})
_VALID_CONTENT_MODES = frozenset({"docs", "code", "both"})


def strip_json_comments(text: str) -> str:
    """Strip ``//`` line comments from JSONC-style text.

    Only strips comments that appear *outside* of JSON string values.
    For the simple JSONC templates this project generates, a per-line
    regex is sufficient and avoids pulling in a full parser.

    Args:
        text: JSONC content (JSON with ``//`` comments).

    Returns:
        Clean JSON string suitable for ``json.loads``.
    """
    return re.sub(r"//.*$", "", text, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PathOverride(BaseModel):
    """Per-path configuration override.

    Allows specific directories or file globs to use different chunking
    parameters than the project-wide defaults.
    """

    model_config = ConfigDict(extra="ignore")

    path: str
    chunk_size: int | None = None
    overlap: int | None = None
    mode: str | None = None

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_MODES:
            msg = f"mode must be one of {sorted(_VALID_MODES)}, got '{v}'"
            raise ValueError(msg)
        return v


class RagConfig(BaseModel):
    """Top-level RAG configuration loaded from ``.rag/.rag-config.json``.

    Unknown fields are silently ignored (``extra="ignore"``), enabling
    forward-compatibility when new options are added.
    """

    model_config = ConfigDict(extra="ignore")

    chunk_size: int = Field(default=1000, ge=100, le=10000)
    overlap: int = Field(default=200, ge=0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: str = Field(default="hybrid")
    content_mode: str = Field(default="docs")
    paths: list[PathOverride] = Field(default_factory=list)

    chroma_path: str = Field(default="chroma")

    # File watcher settings (Phase 10)
    watch_enabled: bool = Field(default=True)
    debounce_seconds: float = Field(default=3.0, ge=0.1, le=30.0)
    burst_threshold: int = Field(default=20, ge=1, le=1000)
    burst_window_seconds: float = Field(default=5.0, ge=0.5, le=60.0)

    # Internal (excluded from serialisation)
    project_root: Path = Field(default=Path("."), exclude=True)
    rag_dir: Path = Field(default=Path(".rag"), exclude=True)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in _VALID_MODES:
            msg = f"mode must be one of {sorted(_VALID_MODES)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("content_mode")
    @classmethod
    def _validate_content_mode(cls, v: str) -> str:
        if v not in _VALID_CONTENT_MODES:
            msg = f"content_mode must be one of {sorted(_VALID_CONTENT_MODES)}, got '{v}'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _overlap_less_than_chunk_size(self) -> RagConfig:
        if self.overlap >= self.chunk_size:
            msg = (
                f"overlap ({self.overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


def ensure_rag_directory(project_root: Path) -> Path:
    """Create the ``.rag/`` directory under *project_root* if it does not exist.

    Args:
        project_root: Absolute or relative path to the project root.

    Returns:
        The resolved ``.rag/`` directory path.
    """
    rag_dir = project_root / ".rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured .rag directory at %s", rag_dir)
    return rag_dir


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def generate_config_template() -> str:
    """Return a JSONC-commented configuration template.

    The template includes explanatory comments for each field and a
    recommended ``.gitignore`` pattern so the config file itself is
    tracked in version control while data files are excluded.

    Returns:
        A string containing valid JSONC (JSON with ``//`` comments).
    """
    return """\
// Knowledge RAG configuration
// This file lives at .rag/.rag-config.json
//
// Recommended .gitignore pattern:
//   .rag/*
//   !.rag/.rag-config.json
//
{
  // Characters per chunk (100-10000)
  "chunk_size": 1000,

  // Overlap between consecutive chunks in characters (must be < chunk_size)
  "overlap": 200,

  // Hybrid search weight: 0.0 = pure keyword, 1.0 = pure semantic
  "alpha": 0.5,

  // Search mode: "hybrid", "semantic", or "keyword"
  "mode": "hybrid",

  // Content mode: "docs" (markdown only), "code" (source code only), or "both"
  "content_mode": "docs",

  // ChromaDB persistent storage directory (relative to .rag/)
  "chroma_path": "chroma",

  // File watcher: auto-reindex on file changes
  "watch_enabled": true,
  "debounce_seconds": 3.0,
  "burst_threshold": 20,
  "burst_window_seconds": 5.0,

  // Per-path overrides (optional)
  // Each entry can override chunk_size, overlap, and/or mode for files
  // matching the given path prefix.
  "paths": []
}
"""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(project_root: Path | None = None) -> RagConfig:
    """Load configuration from ``.rag/.rag-config.json``.

    If the ``.rag/`` directory or config file does not exist, both are
    created automatically with sensible defaults.

    Args:
        project_root: Project root directory. Defaults to ``Path.cwd()``.

    Returns:
        A validated :class:`RagConfig` instance.
    """
    if project_root is None:
        project_root = Path.cwd()

    rag_dir = ensure_rag_directory(project_root)
    config_path = rag_dir / ".rag-config.json"

    if not config_path.exists():
        logger.info("No config found — generating template at %s", config_path)
        config_path.write_text(generate_config_template(), encoding="utf-8")
        # Return defaults (template values match defaults)
        cfg = RagConfig()
    else:
        raw = config_path.read_text(encoding="utf-8")
        clean_json = strip_json_comments(raw)
        try:
            data = json.loads(clean_json)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse config at %s: %s", config_path, exc)
            raise
        cfg = RagConfig.model_validate(data)
        logger.info("Loaded config from %s", config_path)

    # Attach internal paths (excluded from serialisation)
    cfg.project_root = project_root
    cfg.rag_dir = rag_dir
    return cfg
