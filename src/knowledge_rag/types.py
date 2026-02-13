"""Shared types, constants, and utilities for Knowledge RAG.

Provides the canonical HealthStatus type, Ollama connection constants,
smoke test parameters, and a pure-Python cosine similarity function.
All production modules import from here instead of scripts/.
"""

from __future__ import annotations

import math
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class HealthStatus(NamedTuple):
    """Result of a single health check."""

    name: str
    healthy: bool
    detail: str
    fix: str  # Actionable fix instruction (empty string if healthy)


# ---------------------------------------------------------------------------
# Ollama connection constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_EMBED_URL: str = "http://localhost:11434/api/embed"
OLLAMA_TAGS_URL: str = "http://localhost:11434/api/tags"
MODEL_NAME: str = "nomic-embed-text"
EXPECTED_EMBEDDING_DIMS: int = 768
OLLAMA_REQUEST_TIMEOUT: int = 30  # seconds
EMBED_MAX_CHARS: int = 2048  # nomic-embed-text context is 8192 tokens; code tokenizes ~2-3 chars/token

# ---------------------------------------------------------------------------
# Google Generative AI embedding constants
# ---------------------------------------------------------------------------

GOOGLE_EMBED_MODEL: str = "models/gemini-embedding-001"
GOOGLE_EMBED_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"{GOOGLE_EMBED_MODEL}:batchEmbedContents"
)
GOOGLE_EXPECTED_DIMS: int = 3072
GOOGLE_EMBED_MAX_CHARS: int = 8000  # gemini-embedding-001 supports much larger input
GOOGLE_BATCH_SIZE: int = 100  # max texts per batchEmbedContents call


# ---------------------------------------------------------------------------
# Smoke test constants (reuse Phase 1 validation strings per CONTEXT.md)
# ---------------------------------------------------------------------------

SMOKE_TEST_A: str = "the cat sat on the mat"
SMOKE_TEST_B: str = "quantum chromodynamics explains the strong nuclear force"
SMOKE_TEST_MAX_SIMILARITY: float = 0.99


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Pure Python implementation (no numpy dependency). Returns a float
    in the range [-1.0, 1.0]. Returns 0.0 if either vector has zero
    magnitude.

    Args:
        a: First vector.
        b: Second vector (must be same length as *a*).

    Returns:
        Cosine similarity score.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
