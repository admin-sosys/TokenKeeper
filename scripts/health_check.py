"""Startup health check script for TokenKeeper.

Verifies all platform prerequisites before RAG operations:
  1. Python version is 3.12.x
  2. Platform is Windows 11 (native, not WSL)
  3. Ollama server is running and accessible
  4. nomic-embed-text embedding model is available
  5. ChromaDB PersistentClient initializes correctly

Satisfies PLT-03: Startup health checks verify Ollama running, ChromaDB
accessible, embedding model available, Python 3.12.

Usage:
    uv run python scripts/health_check.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
from typing import NamedTuple


class HealthStatus(NamedTuple):
    """Result of a single health check."""

    name: str
    healthy: bool
    detail: str
    fix: str  # Actionable fix instruction (empty string if healthy)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_python_version() -> HealthStatus:
    """Verify Python 3.12.x is running.

    ChromaDB's onnxruntime dependency lacks Python 3.13 wheels,
    so 3.12.x is required.
    """
    ver = sys.version_info
    is_312 = ver.major == 3 and ver.minor == 12
    detail = f"Python {ver.major}.{ver.minor}.{ver.micro}"
    if not is_312:
        detail += f" (required: 3.12.x)"
    fix = "" if is_312 else "Install Python 3.12 via uv: uv python install 3.12"
    return HealthStatus("Python Version", is_312, detail, fix)


def check_windows_platform() -> HealthStatus:
    """Verify running on native Windows 11 (not WSL).

    PLT-01 requires native Windows 11. WSL is detected by checking
    for /proc/version containing 'microsoft'.
    """
    system = platform.system()
    is_windows = system == "Windows"

    if is_windows:
        # Check for WSL indicators
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    return HealthStatus(
                        "Platform",
                        False,
                        "WSL detected (not native Windows)",
                        "This system must run on native Windows 11 (not WSL)",
                    )
        except (FileNotFoundError, OSError):
            pass  # No /proc/version means native Windows -- good
        detail = f"Windows {platform.release()} (native)"
        return HealthStatus("Platform", True, detail, "")

    detail = f"{system} {platform.release()}"
    return HealthStatus(
        "Platform",
        False,
        detail,
        "This system must run on native Windows 11 (not WSL)",
    )


def check_ollama_running() -> HealthStatus:
    """Verify Ollama server is running and accessible on localhost:11434.

    Sends GET /api/tags to check server status and count available models.
    """
    try:
        import requests
    except ImportError:
        return HealthStatus(
            "Ollama",
            False,
            "requests library not installed",
            "Install requests: uv add requests",
        )

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
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
            "Not running (connection refused on port 11434)",
            "Start Ollama: run 'ollama serve' or launch the Ollama desktop app",
        )
    except requests.Timeout:
        return HealthStatus(
            "Ollama",
            False,
            "Request timed out (port 11434)",
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
    """Verify nomic-embed-text model is available in Ollama.

    Parses /api/tags response and checks for 'nomic-embed-text' in model
    names using substring match (handles tags like 'nomic-embed-text:latest').
    """
    try:
        import requests
    except ImportError:
        return HealthStatus(
            "Embedding Model",
            False,
            "requests library not installed",
            "Install requests: uv add requests",
        )

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return HealthStatus(
                "Embedding Model",
                False,
                "Cannot check (Ollama returned non-200)",
                "Pull the model: ollama pull nomic-embed-text",
            )
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]
        matched = [name for name in model_names if "nomic-embed-text" in name]
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
            f"nomic-embed-text not found (available: {model_names})",
            "Pull the model: ollama pull nomic-embed-text",
        )
    except requests.ConnectionError:
        return HealthStatus(
            "Embedding Model",
            False,
            "Cannot check (Ollama not running)",
            "Pull the model: ollama pull nomic-embed-text",
        )
    except requests.RequestException as exc:
        return HealthStatus(
            "Embedding Model",
            False,
            f"Cannot check ({exc})",
            "Pull the model: ollama pull nomic-embed-text",
        )


def check_chromadb_accessible() -> HealthStatus:
    """Verify ChromaDB PersistentClient initializes correctly.

    Creates a PersistentClient with default settings (RustBindingsAPI) in a
    temp directory, calls heartbeat() and get_version(), then cleans up.

    Note: Plan 01-02 confirmed RustBindingsAPI works on this Windows system.
    The SegmentAPI bypass from research was unnecessary.
    """
    try:
        import chromadb
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


def run_health_checks() -> list[HealthStatus]:
    """Run all health checks in order and return results.

    Checks are run sequentially. Even if an earlier check fails (e.g.,
    Ollama not running), later checks still execute to give a complete
    picture of the system state.
    """
    return [
        check_python_version(),
        check_windows_platform(),
        check_ollama_running(),
        check_embedding_model(),
        check_chromadb_accessible(),
    ]


def print_health_report(checks: list[HealthStatus]) -> None:
    """Print a formatted health report to stdout.

    Shows [PASS] or [FAIL] for each check with details, and actionable
    fix instructions for any failures.
    """
    print()
    print("=== TokenKeeper - Health Check ===")
    print()

    for check in checks:
        status = "PASS" if check.healthy else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")
        if not check.healthy and check.fix:
            print(f"         Fix: {check.fix}")

    print()
    passed = sum(1 for c in checks if c.healthy)
    total = len(checks)
    failed = total - passed

    if failed == 0:
        print(f"=== All {total} checks passed ===")
    else:
        print(f"=== {failed} of {total} checks FAILED ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_health_checks()
    print_health_report(results)
    all_healthy = all(c.healthy for c in results)
    sys.exit(0 if all_healthy else 1)
