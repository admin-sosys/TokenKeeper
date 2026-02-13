"""Tests for PLT-01/PLT-03: Ollama embedding generation on Windows 11.

Validates that Ollama HTTP server is accessible, nomic-embed-text model is
available, and embedding generation produces correct 768-dimensional vectors.

Tests are skipped with a clear message if Ollama is not running.
"""

from __future__ import annotations

import math

import pytest
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_ENDPOINT = f"{OLLAMA_URL}/api/embed"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"
MODEL_NAME = "nomic-embed-text"
EXPECTED_DIMS = 768
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Skip condition: skip all tests if Ollama is not running
# ---------------------------------------------------------------------------

def ollama_is_running() -> bool:
    """Check if Ollama server is accessible."""
    try:
        resp = requests.get(TAGS_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_is_running(),
    reason="Ollama is not running on localhost:11434 -- start Ollama to run these tests",
)


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


def get_embedding(input_data: str | list[str]) -> dict:
    """Call Ollama embed API."""
    payload = {"model": MODEL_NAME, "input": input_data}
    resp = requests.post(EMBED_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOllamaServer:
    """Tests for Ollama server accessibility."""

    def test_ollama_server_accessible(self) -> None:
        """GET /api/tags returns 200 with JSON body containing 'models' key."""
        resp = requests.get(TAGS_ENDPOINT, timeout=REQUEST_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data, "Response JSON must contain 'models' key"

    def test_nomic_embed_text_available(self) -> None:
        """nomic-embed-text model is listed in /api/tags response."""
        resp = requests.get(TAGS_ENDPOINT, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]
        assert any(
            MODEL_NAME in name for name in model_names
        ), f"nomic-embed-text not found in models: {model_names}"


class TestSingleEmbedding:
    """Tests for single embedding generation."""

    def test_single_embedding_768_dimensions(self) -> None:
        """POST /api/embed with single string returns 1 embedding of 768 dims."""
        data = get_embedding("test embedding generation")
        embeddings = data.get("embeddings", [])

        assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
        assert (
            len(embeddings[0]) == EXPECTED_DIMS
        ), f"Expected {EXPECTED_DIMS} dimensions, got {len(embeddings[0])}"

    def test_embedding_values_are_floats(self) -> None:
        """All values in an embedding are finite floats (not NaN, not Inf)."""
        data = get_embedding("check float values")
        embeddings = data.get("embeddings", [])
        assert len(embeddings) >= 1, "No embeddings returned"

        vec = embeddings[0]
        for i, val in enumerate(vec):
            assert isinstance(val, (int, float)), (
                f"Value at index {i} is {type(val).__name__}, expected float"
            )
            assert math.isfinite(val), (
                f"Value at index {i} is not finite: {val}"
            )


class TestBatchEmbedding:
    """Tests for batch embedding generation."""

    def test_batch_embedding_returns_correct_count(self) -> None:
        """POST /api/embed with list of 3 strings returns 3 embeddings, each 768 dims."""
        inputs = ["first document", "second document", "third document"]
        data = get_embedding(inputs)
        embeddings = data.get("embeddings", [])

        assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
        for i, emb in enumerate(embeddings):
            assert len(emb) == EXPECTED_DIMS, (
                f"Embedding[{i}]: expected {EXPECTED_DIMS} dims, got {len(emb)}"
            )


class TestEmbeddingQuality:
    """Tests for embedding distinctness and semantic properties."""

    def test_distinct_inputs_produce_distinct_embeddings(self) -> None:
        """Semantically different texts produce embeddings with cosine similarity < 0.99."""
        text_a = "the cat sat on the mat"
        text_b = "quantum chromodynamics explains the strong nuclear force"

        data = get_embedding([text_a, text_b])
        embeddings = data.get("embeddings", [])
        assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"

        vec_a, vec_b = embeddings[0], embeddings[1]

        # Vectors should not be identical
        first_10_differ = any(
            abs(vec_a[i] - vec_b[i]) > 1e-9 for i in range(min(10, len(vec_a)))
        )
        assert first_10_differ, "First 10 values of distinct inputs should differ"

        sim = cosine_similarity(vec_a, vec_b)
        assert sim < 0.99, (
            f"Cosine similarity {sim:.6f} >= 0.99 -- distinct texts should not be near-identical"
        )
