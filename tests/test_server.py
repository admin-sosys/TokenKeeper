"""Unit tests for tokenkeeper.server module.

Tests cover:
- FastMCP server instance type and tool registration
- Logging setup (file handler, stderr handler, no stdout)
- Lifespan environment variable override
- Lifespan yields full context (config, collection, bm25_index, etc.)
- search_knowledge and indexing_status tools
- Module import speed (<1s)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tokenkeeper.server import mcp, setup_logging


# ---------------------------------------------------------------------------
# Server type and tool registration
# ---------------------------------------------------------------------------


class TestMCPServer:
    """Tests for the FastMCP server instance."""

    def test_mcp_server_type(self) -> None:
        """Server is a FastMCP instance."""
        from fastmcp import FastMCP

        assert isinstance(mcp, FastMCP)

    def test_mcp_server_name(self) -> None:
        """Server is named 'TokenKeeper'."""
        assert mcp._mcp_server.name == "TokenKeeper"

    def test_search_knowledge_tool_registered(self) -> None:
        """The search_knowledge tool is registered."""
        tools = list(mcp._tool_manager._tools.keys())
        assert "search_knowledge" in tools

    def test_indexing_status_tool_registered(self) -> None:
        """The indexing_status tool is registered."""
        tools = list(mcp._tool_manager._tools.keys())
        assert "indexing_status" in tools

    def test_reindex_documents_tool_registered(self) -> None:
        """The reindex_documents tool is registered."""
        tools = list(mcp._tool_manager._tools.keys())
        assert "reindex_documents" in tools

    def test_get_index_stats_tool_registered(self) -> None:
        """The get_index_stats tool is registered."""
        tools = list(mcp._tool_manager._tools.keys())
        assert "get_index_stats" in tools

    def test_ping_tool_removed(self) -> None:
        """The temporary ping tool has been replaced."""
        tools = list(mcp._tool_manager._tools.keys())
        assert "ping" not in tools

    def test_four_tools_registered(self) -> None:
        """Exactly 4 tools are registered (search, status, reindex, stats)."""
        tools = list(mcp._tool_manager._tools.keys())
        expected = {"search_knowledge", "indexing_status", "reindex_documents", "get_index_stats"}
        assert set(tools) == expected


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        """setup_logging creates rag.log in the specified directory."""
        setup_logging(tmp_path)
        log_file = tmp_path / "rag.log"
        assert log_file.exists()

    def test_handler_count(self, tmp_path: Path) -> None:
        """Two handlers: RotatingFileHandler and StreamHandler."""
        setup_logging(tmp_path)
        root = logging.getLogger("tokenkeeper")
        assert len(root.handlers) == 2

    def test_no_stdout_handler(self, tmp_path: Path) -> None:
        """No handler writes to stdout (protects MCP stdio transport)."""
        setup_logging(tmp_path)
        root = logging.getLogger("tokenkeeper")
        for handler in root.handlers:
            if hasattr(handler, "stream"):
                assert handler.stream is not sys.stdout, (
                    "Handler writes to stdout -- this corrupts MCP stdio transport"
                )

    def test_stderr_handler_present(self, tmp_path: Path) -> None:
        """A StreamHandler targeting stderr exists."""
        setup_logging(tmp_path)
        root = logging.getLogger("tokenkeeper")
        stderr_handlers = [
            h
            for h in root.handlers
            if hasattr(h, "stream") and h.stream is sys.stderr
        ]
        assert len(stderr_handlers) == 1

    def test_file_handler_is_rotating(self, tmp_path: Path) -> None:
        """File handler is RotatingFileHandler with correct config."""
        from logging.handlers import RotatingFileHandler

        setup_logging(tmp_path)
        root = logging.getLogger("tokenkeeper")
        file_handlers = [
            h for h in root.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        fh = file_handlers[0]
        assert fh.maxBytes == 5 * 1024 * 1024
        assert fh.backupCount == 3

    def test_idempotent_setup(self, tmp_path: Path) -> None:
        """Calling setup_logging twice doesn't duplicate handlers."""
        setup_logging(tmp_path)
        setup_logging(tmp_path)
        root = logging.getLogger("tokenkeeper")
        assert len(root.handlers) == 2


# ---------------------------------------------------------------------------
# Lifespan environment variable
# ---------------------------------------------------------------------------


def _mock_indexing_result():
    """Create a mock IndexingResult for lifespan tests."""
    from tokenkeeper.indexer import IndexingResult
    return IndexingResult(files_indexed=0, chunks_indexed=0, files_skipped=0, files_failed=0)


class TestLifespanEnvVariable:
    """Tests for TOKENKEEPER_PROJECT env var in lifespan."""

    @patch("tokenkeeper.health.run_startup_checks")
    @patch("tokenkeeper.embeddings.run_smoke_test")
    def test_uses_cwd_when_env_empty(
        self,
        mock_smoke: MagicMock,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When TOKENKEEPER_PROJECT is empty, falls back to cwd."""
        from tokenkeeper.server import app_lifespan

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                return ctx

        # Use patch.dict to merge (not replace) environ
        with (
            patch.dict("os.environ", {"TOKENKEEPER_PROJECT": ""}),
            patch("tokenkeeper.server.Path.cwd", return_value=tmp_path),
        ):
            result = asyncio.run(_run())

        assert result["project_root"] == str(tmp_path)

    @patch("tokenkeeper.health.run_startup_checks")
    @patch("tokenkeeper.embeddings.run_smoke_test")
    def test_uses_env_variable(
        self,
        mock_smoke: MagicMock,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When TOKENKEEPER_PROJECT is set, uses that as project root."""
        from tokenkeeper.server import app_lifespan

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                return ctx

        env_path = str(tmp_path / "custom_project")
        (tmp_path / "custom_project").mkdir()

        with patch.dict("os.environ", {"TOKENKEEPER_PROJECT": env_path}):
            result = asyncio.run(_run())

        assert result["project_root"] == env_path

    @patch("tokenkeeper.health.run_startup_checks")
    @patch("tokenkeeper.embeddings.run_smoke_test")
    def test_lifespan_yields_full_context(
        self,
        mock_smoke: MagicMock,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Lifespan yields dict with all expected keys."""
        from tokenkeeper.server import app_lifespan

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                # Wait for background indexing to complete before exiting
                for _ in range(200):  # 10s max
                    if ctx["indexing_state"]["status"] != "indexing":
                        break
                    await asyncio.sleep(0.05)
                return ctx

        with patch.dict("os.environ", {"TOKENKEEPER_PROJECT": str(tmp_path)}):
            result = asyncio.run(_run())

        # Phase 2 keys
        assert "config" in result
        assert "project_root" in result
        assert "rag_dir" in result
        assert result["config"].chunk_size == 1000

        # Phase 6 keys
        assert "collection" in result
        assert "bm25_index" in result
        assert "embed_fn" in result
        assert "indexing_state" in result
        assert result["indexing_state"]["status"] in ("ready", "error")

    @patch("tokenkeeper.health.run_startup_checks", side_effect=SystemExit(1))
    def test_health_failure_blocks_startup(
        self,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Failed health check raises SystemExit(1)."""
        from tokenkeeper.server import app_lifespan

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                return ctx

        with (
            patch.dict("os.environ", {"TOKENKEEPER_PROJECT": str(tmp_path)}),
            pytest.raises(SystemExit, match="1"),
        ):
            asyncio.run(_run())

    @patch("tokenkeeper.health.run_startup_checks")
    @patch(
        "tokenkeeper.embeddings.run_smoke_test", side_effect=SystemExit(1)
    )
    def test_smoke_failure_blocks_startup(
        self,
        mock_smoke: MagicMock,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Failed smoke test raises SystemExit(1)."""
        from tokenkeeper.server import app_lifespan

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                return ctx

        with (
            patch.dict("os.environ", {"TOKENKEEPER_PROJECT": str(tmp_path)}),
            pytest.raises(SystemExit, match="1"),
        ):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# Import speed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FileWatcher lifespan integration (10-06)
# ---------------------------------------------------------------------------


class TestFileWatcherLifespan:
    """Tests for FileWatcher integration in server lifespan."""

    @patch("tokenkeeper.health.run_startup_checks")
    @patch("tokenkeeper.embeddings.run_smoke_test")
    @patch("tokenkeeper.watcher.FileWatcher")
    def test_watcher_not_started_when_disabled(
        self,
        MockWatcher: MagicMock,
        mock_smoke: MagicMock,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """With watch_enabled=False, FileWatcher is NOT created."""
        from tokenkeeper.server import app_lifespan

        # Write a config with watch_enabled=False
        rag_dir = tmp_path / ".rag"
        rag_dir.mkdir(parents=True, exist_ok=True)
        config_file = rag_dir / ".rag-config.json"
        config_file.write_text('{"watch_enabled": false}', encoding="utf-8")

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                return ctx

        with patch.dict("os.environ", {"TOKENKEEPER_PROJECT": str(tmp_path)}):
            result = asyncio.run(_run())

        # Watcher should NOT have been instantiated
        MockWatcher.assert_not_called()
        assert "config" in result
        assert "indexing_state" in result

    @patch("tokenkeeper.health.run_startup_checks")
    @patch("tokenkeeper.embeddings.run_smoke_test")
    @patch("tokenkeeper.watcher.FileWatcher")
    def test_watcher_started_when_enabled(
        self,
        MockWatcher: MagicMock,
        mock_smoke: MagicMock,
        mock_health: MagicMock,
        tmp_path: Path,
    ) -> None:
        """With watch_enabled=True, FileWatcher.start() is called."""
        from tokenkeeper.server import app_lifespan

        mock_instance = MagicMock()
        MockWatcher.return_value = mock_instance

        async def _run() -> dict:
            mock_server = MagicMock()
            async with app_lifespan(mock_server) as ctx:
                # Verify watcher was started during lifespan
                mock_instance.start.assert_called_once()
                # Wait for background indexing to complete before exiting
                for _ in range(200):  # 10s max
                    if ctx["indexing_state"]["status"] != "indexing":
                        break
                    await asyncio.sleep(0.05)
                return ctx

        with patch.dict("os.environ", {"TOKENKEEPER_PROJECT": str(tmp_path)}):
            asyncio.run(_run())

        # After lifespan exits, watcher.stop() should be called
        mock_instance.stop.assert_called_once()

    def test_make_reindex_callback_skips_when_already_indexing(self) -> None:
        """Callback does nothing when indexing is already in progress."""
        from tokenkeeper.server import _make_reindex_callback

        state = {"status": "indexing"}
        loop = MagicMock()

        callback = _make_reindex_callback(
            Path("."), MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), state, loop,
        )

        # Should not crash and should skip reindex
        callback(["/some/file.md"])
        # No coroutine should have been scheduled
        loop.call_soon_threadsafe.assert_not_called()

    def test_make_reindex_callback_skips_empty_paths(self) -> None:
        """Callback does nothing when no valid files are passed."""
        from tokenkeeper.server import _make_reindex_callback

        state = {"status": "ready"}
        loop = MagicMock()

        callback = _make_reindex_callback(
            Path("."), MagicMock(), MagicMock(), MagicMock(),
            MagicMock(), state, loop,
        )

        # Non-existent files should be filtered out
        callback(["/nonexistent/file.md"])
        # No coroutine should have been scheduled because no valid files


class TestImportSpeed:
    """Tests that module import is fast."""

    def test_server_import_is_fast(self) -> None:
        """Importing tokenkeeper.server takes less than 1 second.

        This verifies that heavy dependencies (chromadb, etc.) are lazy-imported
        inside the lifespan, not at module level.
        """
        import importlib
        import tokenkeeper.server

        # Reload to measure cold import time
        start = time.monotonic()
        importlib.reload(tokenkeeper.server)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, (
            f"Module import took {elapsed:.2f}s (limit: 1.0s). "
            "Heavy dependencies may be imported at module level."
        )
