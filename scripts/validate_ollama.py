"""Validate Ollama embedding generation with nomic-embed-text on Windows 11.

PLT-01/PLT-03: Confirms that Ollama HTTP server is accessible, nomic-embed-text
model is available, and embedding generation produces correct 768-dimensional
vectors for single, batch, and determinism scenarios.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_ENDPOINT = f"{OLLAMA_URL}/api/embed"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"
MODEL_NAME = "nomic-embed-text"
EXPECTED_DIMS = 768
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vectors_nearly_equal(a: list[float], b: list[float], tol: float = 1e-6) -> bool:
    """Check if two vectors are nearly equal within tolerance."""
    if len(a) != len(b):
        return False
    return all(abs(x - y) < tol for x, y in zip(a, b))


def embed(input_data: str | list[str]) -> dict[str, Any]:
    """Call Ollama embed API and return the JSON response."""
    payload: dict[str, Any] = {"model": MODEL_NAME, "input": input_data}
    resp = requests.post(EMBED_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def print_result(label: str, passed: bool, detail: str = "") -> None:
    """Print a [PASS] or [FAIL] result line."""
    tag = "[PASS]" if passed else "[FAIL]"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag} {label}{suffix}")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def check_server_running() -> bool:
    """1. Verify Ollama server is running and accessible."""
    print("\n--- Check 1: Ollama server running ---")
    try:
        resp = requests.get(TAGS_ENDPOINT, timeout=REQUEST_TIMEOUT)
        passed = resp.status_code == 200
        print_result("Ollama server accessible", passed, f"status={resp.status_code}")
        return passed
    except requests.ConnectionError:
        print_result("Ollama server accessible", False, "connection refused")
        print(
            "\n  Ollama is not running. Start it with: ollama serve\n"
            "  (or ensure the Ollama desktop app is running)"
        )
        return False
    except requests.Timeout:
        print_result("Ollama server accessible", False, "request timed out")
        print("\n  Ollama did not respond within the timeout period.")
        return False
    except requests.RequestException as exc:
        print_result("Ollama server accessible", False, str(exc))
        return False


def check_model_available() -> bool:
    """2. Verify nomic-embed-text model is pulled."""
    print("\n--- Check 2: nomic-embed-text model available ---")
    try:
        resp = requests.get(TAGS_ENDPOINT, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]

        found = any(MODEL_NAME in name for name in model_names)
        print_result(
            "nomic-embed-text model found",
            found,
            f"models: {model_names}" if not found else "",
        )
        if not found:
            print(f"\n  Model not found. Run: ollama pull {MODEL_NAME}")
        return found
    except requests.RequestException as exc:
        print_result("nomic-embed-text model found", False, str(exc))
        return False


def check_single_embedding() -> bool:
    """3. Generate a single embedding and validate dimensions."""
    print("\n--- Check 3: Single embedding generation ---")
    try:
        t0 = time.perf_counter()
        data = embed("test embedding generation")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        embeddings = data.get("embeddings", [])
        count_ok = len(embeddings) == 1
        print_result("Returned 1 embedding", count_ok, f"got {len(embeddings)}")

        if not count_ok:
            return False

        dims = len(embeddings[0])
        dims_ok = dims == EXPECTED_DIMS
        print_result(f"Embedding has {EXPECTED_DIMS} dimensions", dims_ok, f"got {dims}")

        # Check all values are finite floats
        all_finite = all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in embeddings[0]
        )
        print_result("All values are finite floats", all_finite)

        # Report timing from response or wall-clock
        total_ns = data.get("total_duration")
        if total_ns is not None:
            api_ms = total_ns / 1_000_000
            print_result("Timing", True, f"API={api_ms:.1f}ms, wall={elapsed_ms:.1f}ms")
        else:
            print_result("Timing", True, f"wall={elapsed_ms:.1f}ms")

        return count_ok and dims_ok and all_finite
    except requests.RequestException as exc:
        print_result("Single embedding generation", False, str(exc))
        return False


def check_batch_embedding() -> bool:
    """4. Generate batch embeddings and validate count + dimensions."""
    print("\n--- Check 4: Batch embedding generation ---")
    try:
        inputs = ["first document", "second document", "third document"]
        data = embed(inputs)
        embeddings = data.get("embeddings", [])

        count_ok = len(embeddings) == 3
        print_result(f"Returned {len(inputs)} embeddings", count_ok, f"got {len(embeddings)}")

        if not count_ok:
            return False

        all_dims_ok = True
        for i, emb in enumerate(embeddings):
            d = len(emb)
            ok = d == EXPECTED_DIMS
            if not ok:
                print_result(f"  Embedding[{i}] has {EXPECTED_DIMS} dims", False, f"got {d}")
                all_dims_ok = False

        if all_dims_ok:
            print_result(f"All embeddings have {EXPECTED_DIMS} dimensions", True)

        return count_ok and all_dims_ok
    except requests.RequestException as exc:
        print_result("Batch embedding generation", False, str(exc))
        return False


def check_distinct_embeddings() -> bool:
    """5. Distinct inputs produce distinct embeddings."""
    print("\n--- Check 5: Distinct inputs produce distinct embeddings ---")
    try:
        text_a = "the cat sat on the mat"
        text_b = "quantum chromodynamics explains the strong nuclear force"

        data = embed([text_a, text_b])
        embeddings = data.get("embeddings", [])

        if len(embeddings) != 2:
            print_result("Got 2 embeddings", False, f"got {len(embeddings)}")
            return False

        vec_a, vec_b = embeddings[0], embeddings[1]

        # Check first 10 values differ
        first_10_differ = any(
            abs(vec_a[i] - vec_b[i]) > 1e-9 for i in range(min(10, len(vec_a)))
        )
        print_result("First 10 values differ", first_10_differ)

        sim = cosine_similarity(vec_a, vec_b)
        sim_ok = sim < 0.99
        print_result(
            "Cosine similarity < 0.99",
            sim_ok,
            f"similarity={sim:.6f}",
        )

        return first_10_differ and sim_ok
    except requests.RequestException as exc:
        print_result("Distinct embedding comparison", False, str(exc))
        return False


def check_determinism() -> bool:
    """6. Same input produces identical (or near-identical) embeddings."""
    print("\n--- Check 6: Embedding determinism ---")
    try:
        text = "determinism check for embedding stability"

        data1 = embed(text)
        data2 = embed(text)

        emb1 = data1.get("embeddings", [[]])[0]
        emb2 = data2.get("embeddings", [[]])[0]

        if not emb1 or not emb2:
            print_result("Got embeddings for both calls", False)
            return False

        equal = vectors_nearly_equal(emb1, emb2, tol=1e-6)
        print_result("Identical input produces identical output", equal)

        if not equal:
            # Show how different they are
            sim = cosine_similarity(emb1, emb2)
            print_result("Cosine similarity of repeated inputs", sim > 0.9999, f"sim={sim:.8f}")

        return equal
    except requests.RequestException as exc:
        print_result("Determinism check", False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all validation checks and return exit code."""
    print("=" * 60)
    print("Ollama Embedding Validation (nomic-embed-text)")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    # Check 1: Server running (gate -- abort if down)
    server_ok = check_server_running()
    results.append(("Ollama server running", server_ok))
    if not server_ok:
        print("\n" + "=" * 60)
        print("ABORTED: Ollama server is not accessible.")
        print("Cannot proceed without the Ollama server running.")
        print("=" * 60)
        return 1

    # Check 2: Model available (gate -- abort if missing)
    model_ok = check_model_available()
    results.append(("nomic-embed-text available", model_ok))
    if not model_ok:
        print("\n" + "=" * 60)
        print("ABORTED: nomic-embed-text model is not available.")
        print(f"Run: ollama pull {MODEL_NAME}")
        print("=" * 60)
        return 1

    # Check 3-6: Embedding validations
    results.append(("Single embedding (768-dim)", check_single_embedding()))
    results.append(("Batch embedding (3 inputs)", check_batch_embedding()))
    results.append(("Distinct embeddings", check_distinct_embeddings()))
    results.append(("Embedding determinism", check_determinism()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for label, passed in results:
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} {label}")
        if not passed:
            all_passed = False

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\n  {passed_count}/{total_count} checks passed.")

    if all_passed:
        print("\n  Ollama embedding generation is fully operational.")
        return 0
    else:
        print("\n  Some checks failed. Review output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
