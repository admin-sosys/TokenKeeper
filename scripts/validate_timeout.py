"""
Process-Based Timeout Wrapper for ChromaDB Hang Prevention

Implements a reusable `run_with_timeout()` function that uses
`multiprocessing.Process` to run operations in a subprocess with a
configurable timeout. If the operation exceeds the timeout, the
subprocess is forcefully terminated.

Why process-based (not thread-based): ChromaDB's Rust FFI can deadlock
on Windows (issue #5937). Thread-based timeouts CANNOT kill deadlocked
Rust FFI calls because Python threads are cooperative and Rust FFI
holds the GIL. A subprocess can be forcefully terminated by the OS.

Usage:
    uv run python scripts/validate_timeout.py

Exit codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback


# ---------------------------------------------------------------------------
# Module-level worker function (REQUIRED for Windows multiprocessing 'spawn')
# ---------------------------------------------------------------------------

def _worker(func, args, kwargs, result_queue):
    """Execute func in a subprocess and put the result on the queue.

    This function MUST be at module level for Windows multiprocessing
    compatibility. Windows uses 'spawn' (not 'fork'), so the target
    function must be importable at module level.
    """
    try:
        result = func(*args, **kwargs)
        result_queue.put(("success", result))
    except Exception as exc:
        # Send the exception info as a string since not all exceptions
        # can be pickled across process boundaries
        result_queue.put(("error", type(exc).__name__, str(exc)))


# ---------------------------------------------------------------------------
# Public API: run_with_timeout
# ---------------------------------------------------------------------------

def run_with_timeout(func, timeout=30.0, *args, **kwargs):
    """Run func in a subprocess with a timeout.

    Args:
        func: Callable to execute. MUST be defined at module level
              (not a lambda, closure, or local function) for Windows
              multiprocessing compatibility.
        timeout: Maximum seconds to wait. Default 30.0.
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        The return value of func(*args, **kwargs).

    Raises:
        TimeoutError: If func does not complete within timeout seconds.
        RuntimeError: If the worker process exits without returning a result.
        Exception: Re-raises whatever exception func raised (by type name).
    """
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_worker,
        args=(func, args, kwargs, result_queue),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        # Timeout exceeded -- terminate the subprocess
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        raise TimeoutError(
            f"Operation timed out after {timeout} seconds"
        )

    # Process exited -- check for result
    if result_queue.empty():
        raise RuntimeError(
            "Worker process exited without returning a result."
        )

    item = result_queue.get_nowait()
    if item[0] == "success":
        return item[1]
    elif item[0] == "error":
        # Reconstruct the exception by name
        exc_type_name = item[1]
        exc_message = item[2]
        # Map common built-in exception types back to their classes
        _builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        builtin_exceptions = {
            name: obj for name, obj in _builtins.items()
            if isinstance(obj, type) and issubclass(obj, BaseException)
        }
        exc_class = builtin_exceptions.get(exc_type_name, RuntimeError)
        raise exc_class(exc_message)
    else:
        raise RuntimeError(f"Unexpected result format from worker: {item}")


# ---------------------------------------------------------------------------
# Module-level test target functions
# (MUST be at module level for Windows multiprocessing 'spawn')
# ---------------------------------------------------------------------------

def _return_42():
    """Return the integer 42. Used to test fast operation completion."""
    return 42


def _sleep_long():
    """Sleep for 60 seconds. Used to test timeout termination."""
    time.sleep(60)
    return "should not reach here"


def _raise_zero_div():
    """Raise a ZeroDivisionError. Used to test exception propagation."""
    return 1 / 0


def _chromadb_crud_in_subprocess(persist_dir):
    """Run ChromaDB CRUD operations in a subprocess.

    Creates a ChromaDB PersistentClient, creates a collection, adds
    2 documents, verifies the count, and cleans up. Uses default
    ChromaDB settings (RustBindingsAPI).

    Args:
        persist_dir: Directory for ChromaDB persistence.

    Returns:
        dict with 'count' key indicating number of documents added.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("timeout_test")
    collection.add(
        ids=["t1", "t2"],
        documents=[
            "Timeout wrapper test document one",
            "Timeout wrapper test document two",
        ],
        metadatas=[
            {"source": "timeout_test"},
            {"source": "timeout_test"},
        ],
    )
    count = collection.count()
    client.delete_collection("timeout_test")
    return {"count": count}


# ---------------------------------------------------------------------------
# Validation script (runs when executed directly)
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all timeout wrapper validation tests."""
    print("=" * 55)
    print("=== Process-Based Timeout Wrapper Validation ===")
    print("=" * 55)
    print(f"  Platform: Windows 11")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Multiprocessing start method: {multiprocessing.get_start_method()}")
    print()

    passed = 0
    failed = 0
    total = 6

    # Test 1: Fast operation completes
    print("--- Test 1: Fast operation completes ---")
    try:
        result = run_with_timeout(_return_42, timeout=5)
        if result == 42:
            print(f"  [PASS] run_with_timeout(_return_42, timeout=5) returned {result}")
            passed += 1
        else:
            print(f"  [FAIL] Expected 42, got {result}")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] Unexpected exception: {type(e).__name__}: {e}")
        failed += 1
    print()

    # Test 2: Slow operation times out
    print("--- Test 2: Slow operation times out ---")
    start_time = time.perf_counter()
    try:
        run_with_timeout(_sleep_long, timeout=2)
        print("  [FAIL] Expected TimeoutError, but no exception raised")
        failed += 1
    except TimeoutError as e:
        elapsed = time.perf_counter() - start_time
        if elapsed < 8:
            print(f"  [PASS] TimeoutError raised after {elapsed:.2f}s (timeout=2): {e}")
            passed += 1
        else:
            print(f"  [FAIL] TimeoutError raised but took {elapsed:.2f}s (too slow)")
            failed += 1
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"  [FAIL] Expected TimeoutError, got {type(e).__name__}: {e} ({elapsed:.2f}s)")
        failed += 1
    print()

    # Test 3: Exception propagates
    print("--- Test 3: Exception propagates from subprocess ---")
    try:
        run_with_timeout(_raise_zero_div, timeout=5)
        print("  [FAIL] Expected ZeroDivisionError, but no exception raised")
        failed += 1
    except ZeroDivisionError as e:
        print(f"  [PASS] ZeroDivisionError propagated: {e}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Expected ZeroDivisionError, got {type(e).__name__}: {e}")
        failed += 1
    print()

    # Test 4: ChromaDB operation within timeout
    print("--- Test 4: ChromaDB operation within timeout ---")
    chroma_dir = tempfile.mkdtemp(prefix="timeout_chroma_")
    try:
        result = run_with_timeout(
            _chromadb_crud_in_subprocess, 30, chroma_dir
        )
        if isinstance(result, dict) and result.get("count") == 2:
            print(f"  [PASS] ChromaDB subprocess returned count={result['count']}")
            passed += 1
        else:
            print(f"  [FAIL] Unexpected result: {result}")
            failed += 1
    except Exception as e:
        print(f"  [FAIL] ChromaDB subprocess failed: {type(e).__name__}: {e}")
        failed += 1
    finally:
        shutil.rmtree(chroma_dir, ignore_errors=True)
    print()

    # Test 5: No zombie processes after timeout
    print("--- Test 5: No zombie processes after timeout ---")
    result_queue = multiprocessing.Queue()
    zombie_process = multiprocessing.Process(
        target=_worker,
        args=(_sleep_long, (), {}, result_queue),
    )
    zombie_process.start()
    zombie_process.join(timeout=1)
    if zombie_process.is_alive():
        zombie_process.terminate()
        zombie_process.join(timeout=5)
        if zombie_process.is_alive():
            zombie_process.kill()
            zombie_process.join(timeout=5)

    if not zombie_process.is_alive():
        print(f"  [PASS] Process terminated cleanly (is_alive={zombie_process.is_alive()})")
        passed += 1
    else:
        print(f"  [FAIL] Zombie process still alive after kill")
        failed += 1
    print()

    # Test 6: Configurable timeout
    print("--- Test 6: Configurable timeout ---")
    start1 = time.perf_counter()
    try:
        run_with_timeout(_sleep_long, timeout=1)
    except TimeoutError:
        pass
    elapsed1 = time.perf_counter() - start1

    start2 = time.perf_counter()
    try:
        run_with_timeout(_sleep_long, timeout=3)
    except TimeoutError:
        pass
    elapsed2 = time.perf_counter() - start2

    if elapsed2 > elapsed1:
        print(
            f"  [PASS] timeout=1 took {elapsed1:.2f}s, "
            f"timeout=3 took {elapsed2:.2f}s (longer as expected)"
        )
        passed += 1
    else:
        print(
            f"  [FAIL] timeout=1 took {elapsed1:.2f}s, "
            f"timeout=3 took {elapsed2:.2f}s (expected second to be longer)"
        )
        failed += 1
    print()

    # Summary
    print("=" * 55)
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    if passed == total:
        print("  RESULT: ALL TIMEOUT WRAPPER VALIDATIONS PASSED")
    else:
        print("  RESULT: SOME VALIDATIONS FAILED")
    print("=" * 55)

    return 0 if passed == total else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
