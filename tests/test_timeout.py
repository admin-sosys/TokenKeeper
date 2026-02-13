"""Tests for PLT-02: ChromaDB timeout wrapper prevents Windows hang.

Validates that the process-based timeout wrapper:
- Returns results from fast operations
- Terminates slow operations after the configured timeout
- Propagates exceptions from subprocesses
- Supports configurable timeout durations
- Completes operations that finish before timeout

All test target functions are defined at module level (not inside
test functions) because Windows multiprocessing uses 'spawn' mode,
which requires all Process targets to be importable at module level.
"""

import time

import pytest

from scripts.validate_timeout import run_with_timeout


# ---------------------------------------------------------------------------
# Module-level test target functions
# (MUST be at module level for Windows multiprocessing 'spawn')
# ---------------------------------------------------------------------------

def _return_value():
    """Return a simple string value for fast-operation test."""
    return "hello"


def _sleep_forever():
    """Sleep for 300 seconds. Used to test timeout termination."""
    time.sleep(300)


def _raise_error():
    """Raise a ValueError. Used to test exception propagation."""
    raise ValueError("test error")


def _sleep_briefly():
    """Sleep briefly and return. Tests operation completing before timeout."""
    time.sleep(0.5)
    return "completed"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTimeoutWrapper:
    """Tests for the run_with_timeout process-based timeout wrapper."""

    def test_fast_operation_returns_result(self):
        """Verify that a fast operation returns its result through the wrapper.

        The timeout wrapper should be transparent for operations that
        complete well within the timeout limit. The return value must
        be passed back from the subprocess to the caller.
        """
        result = run_with_timeout(_return_value, 5)
        assert result == "hello"

    def test_slow_operation_raises_timeout(self):
        """Verify that a slow operation raises TimeoutError after timeout.

        This is the core safety mechanism. When ChromaDB's Rust FFI
        deadlocks, the subprocess must be terminated after the timeout
        period rather than hanging forever. Wall time should be close
        to the timeout value, not the operation's natural duration.
        """
        start = time.perf_counter()
        with pytest.raises(TimeoutError):
            run_with_timeout(_sleep_forever, 2)
        elapsed = time.perf_counter() - start

        # Should complete in roughly 2 seconds, with generous buffer
        # for process spawn overhead on Windows
        assert 1.5 <= elapsed <= 8.0, (
            f"Expected wall time between 1.5s and 8s, got {elapsed:.2f}s"
        )

    def test_exception_propagates_from_subprocess(self):
        """Verify that exceptions from the subprocess propagate to the caller.

        When the wrapped function raises an exception, it should be
        re-raised in the calling process with the same type and message.
        This ensures error handling works correctly through the wrapper.
        """
        with pytest.raises(ValueError, match="test error"):
            run_with_timeout(_raise_error, 5)

    def test_timeout_is_configurable(self):
        """Verify that different timeout values produce different wait times.

        The timeout parameter must actually control how long the wrapper
        waits before terminating. A longer timeout should result in a
        longer wall time when the operation exceeds both timeouts.
        """
        start1 = time.perf_counter()
        with pytest.raises(TimeoutError):
            run_with_timeout(_sleep_forever, 1)
        elapsed1 = time.perf_counter() - start1

        start2 = time.perf_counter()
        with pytest.raises(TimeoutError):
            run_with_timeout(_sleep_forever, 3)
        elapsed2 = time.perf_counter() - start2

        assert elapsed2 > elapsed1, (
            f"timeout=3 ({elapsed2:.2f}s) should take longer "
            f"than timeout=1 ({elapsed1:.2f}s)"
        )

    def test_operation_completing_before_timeout(self):
        """Verify that operations completing before timeout return normally.

        A 0.5-second sleep with a 10-second timeout should complete
        successfully and return the result, not raise TimeoutError.
        This ensures the wrapper does not interfere with normal
        operations.
        """
        result = run_with_timeout(_sleep_briefly, 10)
        assert result == "completed"
