"""Startup health checks for Knowledge RAG MCP server.

Verifies all runtime prerequisites before the MCP server begins accepting
requests:
  1. Ollama server is running and accessible
  2. nomic-embed-text embedding model is available
  3. ChromaDB PersistentClient initializes correctly

Failed checks block startup with actionable fix suggestions printed to
stderr. All output uses logging or ``file=sys.stderr`` -- never bare
``print()`` to stdout -- to avoid corrupting the MCP stdio transport.

Adapted from scripts/health_check.py (Phase 1) to use shared constants
from knowledge_rag.types instead of hardcoded values.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile

from knowledge_rag.types import (
    HealthStatus,
    MODEL_NAME,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_TAGS_URL,
)

logger = logging.getLogger("knowledge_rag.health")


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_ollama_running() -> HealthStatus:
    """Verify Ollama server is running and accessible.

    Sends GET to the tags endpoint (from types.py) to check server status
    and count available models.
    """
    try:
        import requests  # noqa: PLC0415 -- lazy import
    except ImportError:
        return HealthStatus(
            "Ollama",
            False,
            "requests library not installed",
            "Install requests: uv add requests",
        )

    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            model_count = len(data.get("models", []))
            return HealthStatus(
                "Ollama",
                True,
                f"Running, {model_count} models available",
                "",
            )
        return HealthStatus(
            "Ollama",
            False,
            f"HTTP {resp.status_code}",
            "Start Ollama: run 'ollama serve' or launch the Ollama desktop app",
        )
    except requests.ConnectionError:
        return HealthStatus(
            "Ollama",
            False,
            "Not running (connection refused)",
            "Start Ollama: run 'ollama serve' or launch the Ollama desktop app",
        )
    except requests.Timeout:
        return HealthStatus(
            "Ollama",
            False,
            "Request timed out",
            "Start Ollama: run 'ollama serve' or launch the Ollama desktop app",
        )
    except requests.RequestException as exc:
        return HealthStatus(
            "Ollama",
            False,
            f"Error: {exc}",
            "Start Ollama: run 'ollama serve' or launch the Ollama desktop app",
        )


def check_embedding_model() -> HealthStatus:
    """Verify the configured embedding model is available in Ollama.

    Parses /api/tags response and checks for MODEL_NAME (from types.py)
    using substring match to handle tags like 'nomic-embed-text:latest'.
    """
    try:
        import requests  # noqa: PLC0415 -- lazy import
    except ImportError:
        return HealthStatus(
            "Embedding Model",
            False,
            "requests library not installed",
            "Install requests: uv add requests",
        )

    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return HealthStatus(
                "Embedding Model",
                False,
                "Cannot check (Ollama returned non-200)",
                f"Pull the model: ollama pull {MODEL_NAME}",
            )
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]
        matched = [name for name in model_names if MODEL_NAME in name]
        if matched:
            return HealthStatus(
                "Embedding Model",
                True,
                f"{matched[0]} available",
                "",
            )
        return HealthStatus(
            "Embedding Model",
            False,
            f"{MODEL_NAME} not found (available: {model_names})",
            f"Pull the model: ollama pull {MODEL_NAME}",
        )
    except requests.ConnectionError:
        return HealthStatus(
            "Embedding Model",
            False,
            "Cannot check (Ollama not running)",
            f"Pull the model: ollama pull {MODEL_NAME}",
        )
    except requests.RequestException as exc:
        return HealthStatus(
            "Embedding Model",
            False,
            f"Cannot check ({exc})",
            f"Pull the model: ollama pull {MODEL_NAME}",
        )


def check_chromadb_accessible() -> HealthStatus:
    """Verify ChromaDB PersistentClient initializes correctly.

    Creates a PersistentClient with default settings (RustBindingsAPI) in a
    temp directory, calls heartbeat() and get_version(), then cleans up.
    """
    try:
        import chromadb  # noqa: PLC0415 -- lazy import
    except ImportError:
        return HealthStatus(
            "ChromaDB",
            False,
            "chromadb library not installed",
            "Install ChromaDB: uv add chromadb==1.5.0",
        )

    tmp_dir = tempfile.mkdtemp(prefix="chroma_health_")
    try:
        client = chromadb.PersistentClient(path=tmp_dir)
        heartbeat = client.heartbeat()
        version = client.get_version()
        return HealthStatus(
            "ChromaDB",
            True,
            f"v{version}, persistent, heartbeat={heartbeat}ns",
            "",
        )
    except Exception as exc:
        return HealthStatus(
            "ChromaDB",
            False,
            f"{type(exc).__name__}: {exc}",
            "Reinstall ChromaDB: uv add chromadb==1.5.0 --reinstall",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_startup_checks() -> None:
    """Run all startup health checks and block on failure.

    Executes each check in order. If any check fails, prints a diagnostic
    report to stderr (with actionable fix suggestions) and raises
    SystemExit(1) to block MCP server startup.

    All output goes to stderr and the module logger -- never stdout --
    to avoid corrupting the MCP stdio transport.
    """
    checks = [
        check_ollama_running(),
        check_embedding_model(),
        check_chromadb_accessible(),
    ]

    failed = [c for c in checks if not c.healthy]

    if not failed:
        logger.info("All %d startup checks passed", len(checks))
        return

    # Report failures to stderr with actionable fixes
    print(
        "\n=== Knowledge RAG - Startup Health Check FAILED ===\n",
        file=sys.stderr,
    )
    for check in checks:
        status = "PASS" if check.healthy else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}", file=sys.stderr)
        if not check.healthy and check.fix:
            print(f"         Fix: {check.fix}", file=sys.stderr)

    print(
        f"\n=== {len(failed)} of {len(checks)} checks FAILED ===\n",
        file=sys.stderr,
    )

    logger.error(
        "Startup blocked: %d health check(s) failed: %s",
        len(failed),
        ", ".join(c.name for c in failed),
    )

    raise SystemExit(1)
