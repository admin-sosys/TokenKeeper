"""Tests for the startup health check script.

Validates that each health check function returns a well-formed HealthStatus,
and that the orchestration functions (run_health_checks, print_health_report)
work correctly.

Tests that depend on Ollama skip gracefully when it is not running.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import requests

from scripts.health_check import (
    HealthStatus,
    check_chromadb_accessible,
    check_embedding_model,
    check_ollama_running,
    check_python_version,
    check_windows_platform,
    run_health_checks,
)


# ---------------------------------------------------------------------------
# Helper: detect if Ollama is running (reuse pattern from test_ollama.py)
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"


def ollama_is_running() -> bool:
    """Check if Ollama server is accessible."""
    try:
        resp = requests.get(TAGS_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Tests: Python version check
# ---------------------------------------------------------------------------


class TestPythonVersionCheck:
    """Tests for check_python_version()."""

    def test_python_version_check_returns_healthy(self) -> None:
        """check_python_version() returns healthy=True on Python 3.12."""
        result = check_python_version()
        assert isinstance(result, HealthStatus)
        assert result.healthy is True
        assert result.name == "Python Version"
        assert "3.12" in result.detail

    def test_python_version_check_has_empty_fix_when_healthy(self) -> None:
        """Healthy status has empty fix string."""
        result = check_python_version()
        assert result.fix == ""


# ---------------------------------------------------------------------------
# Tests: Platform check
# ---------------------------------------------------------------------------


class TestPlatformCheck:
    """Tests for check_windows_platform()."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_platform_check_returns_healthy_on_windows(self) -> None:
        """check_windows_platform() returns healthy=True on Windows."""
        result = check_windows_platform()
        assert isinstance(result, HealthStatus)
        assert result.healthy is True
        assert result.name == "Platform"
        assert "Windows" in result.detail

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_platform_check_detail_says_native(self) -> None:
        """On native Windows the detail includes '(native)'."""
        result = check_windows_platform()
        assert "native" in result.detail

    @pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows test")
    def test_platform_check_returns_unhealthy_on_non_windows(self) -> None:
        """check_windows_platform() returns healthy=False on non-Windows."""
        result = check_windows_platform()
        assert isinstance(result, HealthStatus)
        assert result.healthy is False


# ---------------------------------------------------------------------------
# Tests: Ollama check (skip if not running)
# ---------------------------------------------------------------------------


class TestOllamaCheck:
    """Tests for check_ollama_running()."""

    @pytest.mark.skipif(
        not ollama_is_running(),
        reason="Ollama not running on localhost:11434",
    )
    def test_ollama_check_when_running(self) -> None:
        """check_ollama_running() returns healthy=True if Ollama is accessible."""
        result = check_ollama_running()
        assert isinstance(result, HealthStatus)
        assert result.healthy is True
        assert result.name == "Ollama"
        assert "Running" in result.detail
        assert result.fix == ""

    def test_ollama_check_returns_health_status(self) -> None:
        """check_ollama_running() always returns a HealthStatus regardless of Ollama state."""
        result = check_ollama_running()
        assert isinstance(result, HealthStatus)
        assert result.name == "Ollama"
        assert isinstance(result.healthy, bool)
        assert isinstance(result.detail, str)
        assert isinstance(result.fix, str)


# ---------------------------------------------------------------------------
# Tests: Embedding model check (skip if Ollama not running)
# ---------------------------------------------------------------------------


class TestEmbeddingModelCheck:
    """Tests for check_embedding_model()."""

    @pytest.mark.skipif(
        not ollama_is_running(),
        reason="Ollama not running on localhost:11434",
    )
    def test_embedding_model_check(self) -> None:
        """check_embedding_model() returns healthy=True if nomic-embed-text is available."""
        result = check_embedding_model()
        assert isinstance(result, HealthStatus)
        assert result.name == "Embedding Model"
        # Model may or may not be available depending on whether it was pulled
        if result.healthy:
            assert "nomic-embed-text" in result.detail
            assert result.fix == ""

    def test_embedding_model_check_returns_health_status(self) -> None:
        """check_embedding_model() always returns a HealthStatus."""
        result = check_embedding_model()
        assert isinstance(result, HealthStatus)
        assert result.name == "Embedding Model"


# ---------------------------------------------------------------------------
# Tests: ChromaDB check
# ---------------------------------------------------------------------------


class TestChromaDBCheck:
    """Tests for check_chromadb_accessible()."""

    def test_chromadb_check_returns_healthy(self) -> None:
        """check_chromadb_accessible() returns healthy=True."""
        result = check_chromadb_accessible()
        assert isinstance(result, HealthStatus)
        assert result.healthy is True
        assert result.name == "ChromaDB"
        assert "1.5.0" in result.detail

    def test_chromadb_check_has_empty_fix_when_healthy(self) -> None:
        """Healthy ChromaDB check has empty fix string."""
        result = check_chromadb_accessible()
        assert result.fix == ""


# ---------------------------------------------------------------------------
# Tests: HealthStatus structure
# ---------------------------------------------------------------------------


class TestHealthStatus:
    """Tests for the HealthStatus named tuple."""

    def test_health_status_has_fix_when_unhealthy(self) -> None:
        """An unhealthy HealthStatus should have a non-empty fix field."""
        status = HealthStatus(
            name="Test Check",
            healthy=False,
            detail="Something went wrong",
            fix="Do this to fix it",
        )
        assert status.healthy is False
        assert status.fix != ""

    def test_health_status_fields(self) -> None:
        """HealthStatus has name, healthy, detail, and fix fields."""
        status = HealthStatus(
            name="Test",
            healthy=True,
            detail="All good",
            fix="",
        )
        assert status.name == "Test"
        assert status.healthy is True
        assert status.detail == "All good"
        assert status.fix == ""


# ---------------------------------------------------------------------------
# Tests: run_health_checks orchestration
# ---------------------------------------------------------------------------


class TestRunHealthChecks:
    """Tests for run_health_checks()."""

    def test_run_health_checks_returns_all_five(self) -> None:
        """run_health_checks() returns a list of exactly 5 HealthStatus objects."""
        results = run_health_checks()
        assert len(results) == 5
        for result in results:
            assert isinstance(result, HealthStatus)
            assert result.name != ""
            assert result.detail != ""

    def test_run_health_checks_names_are_unique(self) -> None:
        """Each check has a unique name."""
        results = run_health_checks()
        names = [r.name for r in results]
        assert len(names) == len(set(names)), f"Duplicate names found: {names}"

    def test_run_health_checks_expected_names(self) -> None:
        """The five checks cover the expected prerequisites."""
        results = run_health_checks()
        names = [r.name for r in results]
        assert "Python Version" in names
        assert "Platform" in names
        assert "Ollama" in names
        assert "Embedding Model" in names
        assert "ChromaDB" in names


# ---------------------------------------------------------------------------
# Tests: Exit code (subprocess)
# ---------------------------------------------------------------------------


class TestHealthCheckExitCode:
    """Tests for the health check script exit code behavior."""

    def test_health_check_exit_code(self) -> None:
        """Running scripts/health_check.py produces exit code 0 if all pass, 1 otherwise.

        Since Ollama may not be running, we accept either exit code but verify
        the script runs without crashing (no exit code 2 or other errors).
        """
        result = subprocess.run(
            ["uv", "run", "python", "scripts/health_check.py"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # Script should exit with 0 (all pass) or 1 (some fail), never 2+ (crash)
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # Output should contain the health check header
        assert "TokenKeeper - Health Check" in result.stdout
        # Output should contain PASS/FAIL indicators
        assert "[PASS]" in result.stdout or "[FAIL]" in result.stdout
