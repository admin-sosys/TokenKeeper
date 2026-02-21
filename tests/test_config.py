"""Comprehensive tests for tokenkeeper.config module.

Covers: default values, lenient validation, constraint enforcement,
per-path overrides, JSONC comment stripping, template generation,
and filesystem-based load_config behaviour.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tokenkeeper.config import (
    PathOverride,
    RagConfig,
    generate_config_template,
    load_config,
    strip_json_comments,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_chunk_size(self) -> None:
        cfg = RagConfig()
        assert cfg.chunk_size == 1000

    def test_default_overlap(self) -> None:
        cfg = RagConfig()
        assert cfg.overlap == 200

    def test_default_alpha(self) -> None:
        cfg = RagConfig()
        assert cfg.alpha == 0.5

    def test_default_mode(self) -> None:
        cfg = RagConfig()
        assert cfg.mode == "hybrid"

    def test_default_paths_empty(self) -> None:
        cfg = RagConfig()
        assert cfg.paths == []

    def test_chroma_path_default(self) -> None:
        cfg = RagConfig()
        assert cfg.chroma_path == "chroma"

    def test_content_mode_default_is_docs(self) -> None:
        cfg = RagConfig()
        assert cfg.content_mode == "docs"


# ---------------------------------------------------------------------------
# Lenient validation (CFG-03)
# ---------------------------------------------------------------------------


class TestLenientValidation:
    def test_unknown_fields_ignored(self) -> None:
        cfg = RagConfig.model_validate(
            {"chunk_size": 500, "unknown_field": "ignored"}
        )
        assert cfg.chunk_size == 500
        assert not hasattr(cfg, "unknown_field")

    def test_partial_config_uses_defaults(self) -> None:
        cfg = RagConfig.model_validate({"chunk_size": 800})
        assert cfg.chunk_size == 800
        assert cfg.overlap == 200
        assert cfg.alpha == 0.5
        assert cfg.mode == "hybrid"


# ---------------------------------------------------------------------------
# Validation constraints
# ---------------------------------------------------------------------------


class TestValidationConstraints:
    def test_chunk_size_too_small(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(chunk_size=50)

    def test_chunk_size_too_large(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(chunk_size=20000)

    def test_overlap_negative(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(overlap=-1)

    def test_alpha_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(alpha=-0.1)

    def test_alpha_above_one(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(alpha=1.5)

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(mode="invalid")

    def test_overlap_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(chunk_size=500, overlap=500)

    def test_overlap_greater_than_chunk_size(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(chunk_size=500, overlap=600)

    def test_valid_boundary_chunk_size_min(self) -> None:
        cfg = RagConfig(chunk_size=100, overlap=0)
        assert cfg.chunk_size == 100

    def test_valid_boundary_chunk_size_max(self) -> None:
        cfg = RagConfig(chunk_size=10000)
        assert cfg.chunk_size == 10000

    def test_valid_boundary_alpha_zero(self) -> None:
        cfg = RagConfig(alpha=0.0)
        assert cfg.alpha == 0.0

    def test_valid_boundary_alpha_one(self) -> None:
        cfg = RagConfig(alpha=1.0)
        assert cfg.alpha == 1.0

    def test_valid_modes(self) -> None:
        for mode in ("hybrid", "semantic", "keyword"):
            cfg = RagConfig(mode=mode)
            assert cfg.mode == mode

    def test_content_mode_valid_code(self) -> None:
        cfg = RagConfig(content_mode="code")
        assert cfg.content_mode == "code"

    def test_content_mode_valid_both(self) -> None:
        cfg = RagConfig(content_mode="both")
        assert cfg.content_mode == "both"

    def test_content_mode_invalid_raises(self) -> None:
        with pytest.raises(ValidationError):
            RagConfig(content_mode="invalid")


# ---------------------------------------------------------------------------
# Per-path overrides
# ---------------------------------------------------------------------------


class TestPathOverrides:
    def test_path_override_parses(self) -> None:
        cfg = RagConfig.model_validate(
            {"paths": [{"path": "docs/", "chunk_size": 500}]}
        )
        assert len(cfg.paths) == 1
        assert cfg.paths[0].path == "docs/"
        assert cfg.paths[0].chunk_size == 500
        assert cfg.paths[0].overlap is None
        assert cfg.paths[0].mode is None

    def test_path_override_unknown_fields_ignored(self) -> None:
        po = PathOverride.model_validate(
            {"path": "src/", "chunk_size": 300, "unknown_key": "ignored"}
        )
        assert po.path == "src/"
        assert po.chunk_size == 300
        assert not hasattr(po, "unknown_key")

    def test_path_override_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            PathOverride(path="docs/", mode="invalid")


# ---------------------------------------------------------------------------
# JSONC comment stripping
# ---------------------------------------------------------------------------


class TestJSONCStripping:
    def test_inline_comment_stripped(self) -> None:
        raw = '{"key": "value"} // comment'
        result = strip_json_comments(raw)
        data = json.loads(result)
        assert data == {"key": "value"}

    def test_multiline_comments_stripped(self) -> None:
        raw = (
            "// Top-level comment\n"
            "{\n"
            '  "chunk_size": 500, // inline\n'
            '  // full line comment\n'
            '  "mode": "semantic"\n'
            "}\n"
        )
        result = strip_json_comments(raw)
        data = json.loads(result)
        assert data["chunk_size"] == 500
        assert data["mode"] == "semantic"

    def test_no_comments_unchanged(self) -> None:
        raw = '{"key": "value"}'
        result = strip_json_comments(raw)
        assert json.loads(result) == {"key": "value"}


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


class TestTemplateGeneration:
    def test_template_non_empty(self) -> None:
        tmpl = generate_config_template()
        assert len(tmpl) > 0

    def test_template_parses_after_stripping(self) -> None:
        tmpl = generate_config_template()
        clean = strip_json_comments(tmpl)
        data = json.loads(clean)
        assert isinstance(data, dict)

    def test_template_produces_valid_config(self) -> None:
        tmpl = generate_config_template()
        clean = strip_json_comments(tmpl)
        data = json.loads(clean)
        cfg = RagConfig.model_validate(data)
        assert cfg.chunk_size == 1000
        assert cfg.overlap == 200
        assert cfg.alpha == 0.5
        assert cfg.mode == "hybrid"

    def test_chroma_path_in_template(self) -> None:
        tmpl = generate_config_template()
        clean = strip_json_comments(tmpl)
        data = json.loads(clean)
        assert data["chroma_path"] == "chroma"

    def test_content_mode_in_template(self) -> None:
        tmpl = generate_config_template()
        clean = strip_json_comments(tmpl)
        data = json.loads(clean)
        assert data["content_mode"] == "docs"


# ---------------------------------------------------------------------------
# load_config with filesystem (tmp_path)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_creates_rag_directory(self, tmp_path: object) -> None:
        """Loading from empty directory creates .rag/ and config file."""
        from pathlib import Path

        root = Path(str(tmp_path))
        cfg = load_config(root)
        rag_dir = root / ".rag"
        config_file = rag_dir / ".rag-config.json"
        assert rag_dir.is_dir()
        assert config_file.is_file()
        assert cfg.chunk_size == 1000

    def test_returns_defaults_from_fresh_directory(self, tmp_path: object) -> None:
        from pathlib import Path

        root = Path(str(tmp_path))
        cfg = load_config(root)
        assert cfg.chunk_size == 1000
        assert cfg.overlap == 200
        assert cfg.alpha == 0.5
        assert cfg.mode == "hybrid"
        assert cfg.paths == []

    def test_loads_custom_config(self, tmp_path: object) -> None:
        from pathlib import Path

        root = Path(str(tmp_path))
        rag_dir = root / ".rag"
        rag_dir.mkdir()
        config_path = rag_dir / ".rag-config.json"
        config_path.write_text(
            json.dumps({"chunk_size": 800, "alpha": 0.7, "mode": "semantic"}),
            encoding="utf-8",
        )
        cfg = load_config(root)
        assert cfg.chunk_size == 800
        assert cfg.alpha == 0.7
        assert cfg.mode == "semantic"
        # Defaults for unspecified
        assert cfg.overlap == 200

    def test_partial_config_merges_with_defaults(self, tmp_path: object) -> None:
        from pathlib import Path

        root = Path(str(tmp_path))
        rag_dir = root / ".rag"
        rag_dir.mkdir()
        config_path = rag_dir / ".rag-config.json"
        config_path.write_text(
            json.dumps({"chunk_size": 1500}),
            encoding="utf-8",
        )
        cfg = load_config(root)
        assert cfg.chunk_size == 1500
        assert cfg.overlap == 200
        assert cfg.alpha == 0.5
        assert cfg.mode == "hybrid"

    def test_chroma_path_round_trips(self, tmp_path: object) -> None:
        """chroma_path persists through save/load cycle."""
        from pathlib import Path

        root = Path(str(tmp_path))
        rag_dir = root / ".rag"
        rag_dir.mkdir()
        config_path = rag_dir / ".rag-config.json"
        config_path.write_text(
            json.dumps({"chroma_path": "my_vectors"}),
            encoding="utf-8",
        )
        cfg = load_config(root)
        assert cfg.chroma_path == "my_vectors"

    def test_project_root_and_rag_dir_set(self, tmp_path: object) -> None:
        from pathlib import Path

        root = Path(str(tmp_path))
        cfg = load_config(root)
        assert cfg.project_root == root
        assert cfg.rag_dir == root / ".rag"
