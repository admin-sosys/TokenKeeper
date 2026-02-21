"""Unit tests for tokenkeeper.health startup checks.

All HTTP calls are mocked so tests run without Ollama or ChromaDB
services. ChromaDB accessibility is tested with a skip marker when
chromadb is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests as _requests

from tokenkeeper.health import (
    check_chromadb_accessible,
    check_embedding_model,
    check_ollama_running,
    run_startup_checks,
)
from tokenkeeper.types import MODEL_NAME, OLLAMA_REQUEST_TIMEOUT, OLLAMA_TAGS_URL


# ---------------------------------------------------------------------------
# check_ollama_running
# ---------------------------------------------------------------------------


class TestCheckOllamaRunning:
    """Tests for check_ollama_running()."""

    @patch("requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        """200 response with models returns healthy status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}]}
        mock_get.return_value = mock_resp

        result = check_ollama_running()

        assert result.healthy is True
        assert "1 models available" in result.detail
        assert result.fix == ""
        mock_get.assert_called_once_with(
            OLLAMA_TAGS_URL, timeout=OLLAMA_REQUEST_TIMEOUT,
        )

    @patch("requests.get")
    def test_connection_refused(self, mock_get: MagicMock) -> None:
        """ConnectionError when Ollama is not running."""
        mock_get.side_effect = _requests.ConnectionError("Connection refused")

        result = check_ollama_running()

        assert result.healthy is False
        assert "connection refused" in result.detail.lower()
        assert "ollama serve" in result.fix.lower()

    @patch("requests.get")
    def test_timeout(self, mock_get: MagicMock) -> None:
        """Timeout when Ollama is slow to respond."""
        mock_get.side_effect = _requests.Timeout("timed out")

        result = check_ollama_running()

        assert result.healthy is False
        assert "timed out" in result.detail.lower()
        assert "ollama serve" in result.fix.lower()


# ---------------------------------------------------------------------------
# check_embedding_model
# ---------------------------------------------------------------------------


class TestCheckEmbeddingModel:
    """Tests for check_embedding_model()."""

    @patch("requests.get")
    def test_model_found(self, mock_get: MagicMock) -> None:
        """Model present in tags response returns healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "llama3:latest"},
            ],
        }
        mock_get.return_value = mock_resp

        result = check_embedding_model()

        assert result.healthy is True
        assert "nomic-embed-text:latest" in result.detail
        assert result.fix == ""

    @patch("requests.get")
    def test_model_not_found(self, mock_get: MagicMock) -> None:
        """Model missing from tags response returns unhealthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:latest"}],
        }
        mock_get.return_value = mock_resp

        result = check_embedding_model()

        assert result.healthy is False
        assert MODEL_NAME in result.detail
        assert f"ollama pull {MODEL_NAME}" in result.fix


# ---------------------------------------------------------------------------
# check_chromadb_accessible
# ---------------------------------------------------------------------------


def _can_import_chromadb() -> bool:
    """Check whether chromadb can be imported."""
    try:
        import chromadb  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


_has_chromadb = _can_import_chromadb()


class TestCheckChromadbAccessible:
    """Tests for check_chromadb_accessible()."""

    @pytest.mark.skipif(not _has_chromadb, reason="chromadb not installed")
    def test_success(self, tmp_path: object) -> None:
        """ChromaDB PersistentClient creates and heartbeats successfully."""
        result = check_chromadb_accessible()

        assert result.healthy is True
        assert "persistent" in result.detail.lower()
        assert result.fix == ""


# ---------------------------------------------------------------------------
# run_startup_checks
# ---------------------------------------------------------------------------


class TestRunStartupChecks:
    """Tests for run_startup_checks()."""

    @patch("tokenkeeper.health.check_chromadb_accessible")
    @patch("tokenkeeper.health.check_embedding_model")
    @patch("tokenkeeper.health.check_ollama_running")
    def test_all_pass(
        self,
        mock_ollama: MagicMock,
        mock_model: MagicMock,
        mock_chroma: MagicMock,
    ) -> None:
        """All checks passing does not raise SystemExit."""
        from tokenkeeper.types import HealthStatus

        mock_ollama.return_value = HealthStatus("Ollama", True, "ok", "")
        mock_model.return_value = HealthStatus("Model", True, "ok", "")
        mock_chroma.return_value = HealthStatus("ChromaDB", True, "ok", "")

        # Should not raise
        run_startup_checks()

    @patch("tokenkeeper.health.check_chromadb_accessible")
    @patch("tokenkeeper.health.check_embedding_model")
    @patch("tokenkeeper.health.check_ollama_running")
    def test_one_fails_raises_systemexit(
        self,
        mock_ollama: MagicMock,
        mock_model: MagicMock,
        mock_chroma: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """One failing check raises SystemExit(1) with stderr output."""
        from tokenkeeper.types import HealthStatus

        mock_ollama.return_value = HealthStatus(
            "Ollama", False, "Not running", "run ollama serve",
        )
        mock_model.return_value = HealthStatus("Model", True, "ok", "")
        mock_chroma.return_value = HealthStatus("ChromaDB", True, "ok", "")

        with pytest.raises(SystemExit) as exc_info:
            run_startup_checks()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        # Diagnostic output must go to stderr, not stdout
        assert captured.out == ""
        assert "FAIL" in captured.err
        assert "Ollama" in captured.err
        assert "ollama serve" in captured.err
