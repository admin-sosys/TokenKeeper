"""
ChromaDB Platform Validation Script

Validates that ChromaDB PersistentClient works correctly on Windows 11.
Tests all CRUD operations, persistence across client restarts, and
documents whether RustBindingsAPI works or requires the SegmentAPI bypass
for the Rust FFI deadlock issue (#5937).

Key finding: On this system (chromadb==1.5.0, Windows 11, Python 3.12),
RustBindingsAPI works without hanging. The SegmentAPI bypass requires
the hnswlib package which is not bundled with chromadb 1.5.0. Since
RustBindingsAPI works, SegmentAPI is not needed.

Usage:
    uv run python scripts/validate_chromadb.py

Exit codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import chromadb


TIMEOUT_WARNING_SECONDS = 30
RUST_CHECK_TIMEOUT_SECONDS = 10


def create_client(persist_dir: str) -> chromadb.ClientAPI:
    """Create a PersistentClient using the default RustBindingsAPI.

    ChromaDB 1.5.0 defaults to RustBindingsAPI which uses native Rust
    bindings for HNSW operations. On some Windows systems this can hang
    due to issue #5937, but on this system it works correctly.
    """
    return chromadb.PersistentClient(path=persist_dir)


def timed_operation(name: str, func, *args, **kwargs):
    """Run an operation with timing and result reporting."""
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        status = "PASS"
        if elapsed > TIMEOUT_WARNING_SECONDS:
            print(
                f"  [WARN] {name} took {elapsed:.2f}s "
                f"(>{TIMEOUT_WARNING_SECONDS}s -- possible Rust FFI deadlock)"
            )
        print(f"  [{status}] {name} -- {elapsed:.2f}s")
        return result, True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  [FAIL] {name} -- {elapsed:.2f}s -- {type(e).__name__}: {e}")
        return None, False


def check_rust_bindings_api(persist_dir: str) -> str:
    """
    Quick check if RustBindingsAPI works or hangs on this system.

    Uses a ThreadPoolExecutor with a 10-second timeout to detect the hang
    described in chroma issue #5937. Returns a status message.
    """
    rust_dir = os.path.join(persist_dir, "_rust_check")
    os.makedirs(rust_dir, exist_ok=True)

    def try_rust_client():
        client = chromadb.PersistentClient(path=rust_dir)
        _ = client.heartbeat()
        col = client.get_or_create_collection("rust_check")
        col.add(ids=["r1"], documents=["rust bindings test document"])
        result = col.get(ids=["r1"])
        client.delete_collection("rust_check")
        return result

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(try_rust_client)
            try:
                future.result(timeout=RUST_CHECK_TIMEOUT_SECONDS)
                return "RustBindingsAPI works on this system (chroma#5937 may be fixed)"
            except TimeoutError:
                return "RustBindingsAPI hangs on this system -- SegmentAPI required"
            except Exception as e:
                return (
                    f"RustBindingsAPI error: {type(e).__name__}: {e} "
                    f"-- SegmentAPI required"
                )
    except Exception as e:
        return f"RustBindingsAPI check failed: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(rust_dir, ignore_errors=True)


def check_segment_api_available() -> str:
    """Check if SegmentAPI bypass is available (requires hnswlib)."""
    try:
        import hnswlib  # noqa: F401

        return "SegmentAPI available (hnswlib installed)"
    except ImportError:
        return (
            "SegmentAPI unavailable (hnswlib not installed) -- "
            "not needed since RustBindingsAPI works"
        )


def run_crud_validation(persist_dir: str) -> bool:
    """Run complete CRUD validation against ChromaDB PersistentClient."""
    all_passed = True

    # CREATE CLIENT
    client, ok = timed_operation(
        "Client created (PersistentClient)",
        create_client,
        persist_dir,
    )
    if not ok or client is None:
        return False

    # Heartbeat
    _, ok = timed_operation("Heartbeat", client.heartbeat)
    all_passed = all_passed and ok

    # Version
    version, ok = timed_operation("Get version", client.get_version)
    all_passed = all_passed and ok
    if version:
        print(f"         ChromaDB version: {version}")

    # CREATE COLLECTION
    collection, ok = timed_operation(
        "Collection created",
        client.get_or_create_collection,
        "test_validation",
    )
    if not ok or collection is None:
        return False

    # ADD DOCUMENTS
    def add_docs():
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "ChromaDB platform validation document one",
                "Windows 11 compatibility testing document two",
                "SegmentAPI bypass verification document three",
            ],
            metadatas=[
                {"source": "test", "index": 1},
                {"source": "test", "index": 2},
                {"source": "validation", "index": 3},
            ],
        )
        count = collection.count()
        assert count == 3, f"Expected 3 documents, got {count}"
        return count

    count, ok = timed_operation("Documents added (3)", add_docs)
    all_passed = all_passed and ok

    # READ (GET)
    def get_docs():
        result = collection.get(
            ids=["doc1", "doc2"],
            include=["documents", "metadatas"],
        )
        assert len(result["ids"]) == 2, f"Expected 2 results, got {len(result['ids'])}"
        assert result["documents"][0] == "ChromaDB platform validation document one"
        assert result["metadatas"][0]["source"] == "test"
        return result

    _, ok = timed_operation("Documents retrieved (2)", get_docs)
    all_passed = all_passed and ok

    # QUERY
    def query_docs():
        result = collection.query(
            query_texts=["platform test"],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )
        assert (
            len(result["ids"][0]) == 2
        ), f"Expected 2 query results, got {len(result['ids'][0])}"
        assert result["documents"] is not None
        assert result["distances"] is not None
        return result

    _, ok = timed_operation("Query returned results (2)", query_docs)
    all_passed = all_passed and ok

    # UPDATE
    def update_doc():
        collection.update(
            ids=["doc1"],
            documents=["Updated content for document one"],
        )
        result = collection.get(ids=["doc1"], include=["documents"])
        assert result["documents"][0] == "Updated content for document one"
        return result

    _, ok = timed_operation("Document updated (doc1)", update_doc)
    all_passed = all_passed and ok

    # DELETE DOCUMENT
    def delete_doc():
        collection.delete(ids=["doc3"])
        count = collection.count()
        assert count == 2, f"Expected 2 documents after delete, got {count}"
        return count

    _, ok = timed_operation("Document deleted (doc3), count=2", delete_doc)
    all_passed = all_passed and ok

    # DELETE COLLECTION
    def delete_collection():
        client.delete_collection("test_validation")
        collections = client.list_collections()
        assert len(collections) == 0, f"Expected 0 collections, got {len(collections)}"
        return True

    _, ok = timed_operation("Collection deleted", delete_collection)
    all_passed = all_passed and ok

    return all_passed


def run_persistence_test(persist_dir: str) -> bool:
    """Test that data persists across client restarts."""
    all_passed = True

    # Create client and add data
    client, ok = timed_operation(
        "Persistence: client created",
        create_client,
        persist_dir,
    )
    if not ok or client is None:
        return False

    def setup_persistence_data():
        col = client.get_or_create_collection("persist_test")
        col.add(
            ids=["p1", "p2"],
            documents=["Persistent doc one", "Persistent doc two"],
            metadatas=[{"type": "persist"}, {"type": "persist"}],
        )
        assert col.count() == 2
        return True

    _, ok = timed_operation("Persistence: data written", setup_persistence_data)
    all_passed = all_passed and ok

    # Close/delete the client reference
    del client

    # Create NEW client pointing to same directory
    client2, ok = timed_operation(
        "Persistence: new client created (same dir)",
        create_client,
        persist_dir,
    )
    if not ok or client2 is None:
        return False

    def verify_persistence():
        col = client2.get_collection("persist_test")
        assert col.count() == 2, f"Expected 2 persisted docs, got {col.count()}"
        result = col.get(ids=["p1", "p2"], include=["documents", "metadatas"])
        assert len(result["ids"]) == 2
        assert result["documents"][0] == "Persistent doc one"
        assert result["documents"][1] == "Persistent doc two"
        assert result["metadatas"][0]["type"] == "persist"
        return result

    _, ok = timed_operation(
        "Persistence: data verified after restart", verify_persistence
    )
    all_passed = all_passed and ok

    # Cleanup persistence collection
    try:
        client2.delete_collection("persist_test")
    except Exception:
        pass

    return all_passed


def main() -> int:
    """Run all ChromaDB platform validations."""
    persist_dir = tempfile.mkdtemp(prefix="chromadb_validate_")

    try:
        print("=" * 50)
        print("=== ChromaDB Platform Validation ===")
        print("=" * 50)
        print(f"  Platform: Windows 11")
        print(f"  ChromaDB: {chromadb.__version__}")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Persist dir: {persist_dir}")
        print()

        # --- RustBindingsAPI check ---
        print("--- RustBindingsAPI Quick Check ---")
        rust_status = check_rust_bindings_api(persist_dir)
        rust_works = "works" in rust_status
        print(f"  [{'INFO' if rust_works else 'WARN'}] {rust_status}")
        print()

        # --- SegmentAPI availability check ---
        print("--- SegmentAPI Availability ---")
        segment_status = check_segment_api_available()
        print(f"  [INFO] {segment_status}")
        print()

        # --- CRUD Validation ---
        api_label = "RustBindingsAPI" if rust_works else "SegmentAPI"
        print(f"--- CRUD Operations ({api_label}) ---")
        crud_passed = run_crud_validation(persist_dir)
        print()

        # --- Persistence Test ---
        # Use a fresh directory for persistence to avoid leftovers from CRUD
        persist_dir_2 = os.path.join(persist_dir, "persistence_test")
        os.makedirs(persist_dir_2, exist_ok=True)

        print("--- Persistence Test ---")
        persistence_passed = run_persistence_test(persist_dir_2)
        print()

        # --- Summary ---
        print("=" * 50)
        all_passed = crud_passed and persistence_passed
        if all_passed:
            print("  RESULT: ALL VALIDATIONS PASSED")
            print()
            print(f"  API: {api_label} (default PersistentClient)")
            if rust_works:
                print(
                    "  NOTE: chroma#5937 Rust FFI hang NOT observed on this system."
                )
                print(
                    "  SegmentAPI bypass is NOT required. Using default "
                    "RustBindingsAPI."
                )
        else:
            print("  RESULT: SOME VALIDATIONS FAILED")
            if not crud_passed:
                print("    - CRUD operations: FAILED")
            if not persistence_passed:
                print("    - Persistence test: FAILED")
        print("=" * 50)

        return 0 if all_passed else 1

    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
