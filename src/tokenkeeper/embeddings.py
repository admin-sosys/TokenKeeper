"""Embedding providers and batch embedder.

Supports multiple embedding backends:
- **Ollama** (local): ``embed_texts`` — uses nomic-embed-text (768 dims)
- **Google Gemini** (cloud): ``embed_texts_google`` — uses gemini-embedding-001 (3072 dims)

Also provides a startup smoke test and ``embed_chunks_batched`` for batch
embedding with retry logic.

Exports:
    run_smoke_test        -- Block startup if embeddings are broken
    embed_texts           -- Embed a list of texts via local Ollama
    embed_texts_google    -- Embed a list of texts via Google Generative AI API
    embed_chunks_batched  -- Batch embed with retry and exponential backoff
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable

import requests

from tokenkeeper.types import (
    EMBED_MAX_CHARS,
    EXPECTED_EMBEDDING_DIMS,
    GOOGLE_EMBED_MAX_CHARS,
    GOOGLE_EMBED_MODEL,
    GOOGLE_EMBED_URL,
    GOOGLE_EXPECTED_DIMS,
    MODEL_NAME,
    OLLAMA_EMBED_URL,
    OLLAMA_REQUEST_TIMEOUT,
    SMOKE_TEST_A,
    SMOKE_TEST_B,
    SMOKE_TEST_MAX_SIMILARITY,
    cosine_similarity,
)

logger = logging.getLogger("tokenkeeper.embeddings")

import os as _os

BATCH_SIZE = int(_os.environ.get("RAG_BATCH_SIZE", "15"))
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(message: str) -> None:
    """Log an error, print to stderr, and exit the process.

    This is called when the embedding subsystem is in a broken state that
    cannot be recovered from at runtime (e.g. Ollama is down, model missing,
    or the uppercase regression is present).

    Args:
        message: Human-readable failure description.

    Raises:
        SystemExit: Always raised with exit code 1.
    """
    logger.error(message)
    print(message, file=sys.stderr)  # noqa: T201
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    *,
    timeout: int | None = None,
) -> list[list[float]]:
    """Embed a list of texts using the local Ollama instance.

    All input strings are **lowercased** before being sent to Ollama to
    work around the known uppercase embedding regression where certain
    Ollama versions return identical vectors regardless of input casing.

    Args:
        texts: Strings to embed.
        timeout: HTTP request timeout in seconds.  Defaults to
            :data:`tokenkeeper.types.OLLAMA_REQUEST_TIMEOUT`.

    Returns:
        A list of embedding vectors (each a list of floats with length
        :data:`tokenkeeper.types.EXPECTED_EMBEDDING_DIMS`).

    Raises:
        ConnectionError: If the Ollama server is unreachable.
        ValueError: If the returned vectors have unexpected dimensions.
    """
    if timeout is None:
        timeout = OLLAMA_REQUEST_TIMEOUT

    # Truncate oversized texts to stay within model context window.
    # nomic-embed-text supports ~8192 tokens; code can tokenize at
    # ~1.3 tokens/char so EMBED_MAX_CHARS is set conservatively.
    lowered: list[str] = []
    for t in texts:
        lc = t.lower()
        if len(lc) > EMBED_MAX_CHARS:
            logger.debug(
                "Truncating text from %d to %d chars for embedding",
                len(lc), EMBED_MAX_CHARS,
            )
            lc = lc[:EMBED_MAX_CHARS]
        lowered.append(lc)

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": MODEL_NAME, "input": lowered},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_EMBED_URL}. "
            "Is Ollama running?  Try: ollama serve"
        ) from exc
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Ollama request failed: {exc}"
        ) from exc

    data = response.json()
    embeddings: list[list[float]] = data.get("embeddings", [])

    # Validate dimensions
    for idx, vec in enumerate(embeddings):
        if len(vec) != EXPECTED_EMBEDDING_DIMS:
            msg = (
                f"Embedding {idx} has {len(vec)} dimensions, "
                f"expected {EXPECTED_EMBEDDING_DIMS}. "
                f"Model '{MODEL_NAME}' may be incorrect or corrupted. "
                f"Try: ollama pull {MODEL_NAME}"
            )
            raise ValueError(msg)

    return embeddings


def _get_google_api_key() -> str:
    """Get Google API key from environment."""
    import os

    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Set it to your Google Generative AI API key."
        )
    return key


def embed_texts_google(
    texts: list[str],
    *,
    timeout: int = 60,
) -> list[list[float]]:
    """Embed texts using Google Generative AI (gemini-embedding-001).

    Uses the batchEmbedContents endpoint for efficient batch processing.
    Supports up to 100 texts per call with 3072-dimensional output.

    Args:
        texts: Strings to embed.
        timeout: HTTP request timeout in seconds.

    Returns:
        A list of embedding vectors (each a list of 3072 floats).

    Raises:
        ConnectionError: If the Google API is unreachable.
        ValueError: If the API key is missing or response is malformed.
    """
    if not texts:
        return []

    api_key = _get_google_api_key()
    url = f"{GOOGLE_EMBED_URL}?key={api_key}"

    # Truncate oversized texts
    processed: list[str] = []
    for t in texts:
        if len(t) > GOOGLE_EMBED_MAX_CHARS:
            logger.debug(
                "Truncating text from %d to %d chars for Google embedding",
                len(t), GOOGLE_EMBED_MAX_CHARS,
            )
            t = t[:GOOGLE_EMBED_MAX_CHARS]
        processed.append(t)

    try:
        response = requests.post(
            url,
            json={
                "requests": [
                    {
                        "model": GOOGLE_EMBED_MODEL,
                        "content": {"parts": [{"text": t}]},
                    }
                    for t in processed
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            "Cannot reach Google Generative AI API. Check your internet connection."
        ) from exc
    except requests.RequestException as exc:
        raise ConnectionError(f"Google embedding request failed: {exc}") from exc

    data = response.json()

    if "error" in data:
        raise ValueError(f"Google API error: {data['error'].get('message', data['error'])}")

    embeddings_raw = data.get("embeddings", [])
    embeddings: list[list[float]] = [e["values"] for e in embeddings_raw]

    # Validate dimensions
    for idx, vec in enumerate(embeddings):
        if len(vec) != GOOGLE_EXPECTED_DIMS:
            raise ValueError(
                f"Google embedding {idx} has {len(vec)} dims, "
                f"expected {GOOGLE_EXPECTED_DIMS}."
            )

    return embeddings


def run_smoke_test() -> None:
    """Run the embedding smoke test.

    Embeds two semantically distinct strings and verifies they produce
    **different** vectors (cosine similarity < ``SMOKE_TEST_MAX_SIMILARITY``).
    This catches the Ollama uppercase embedding regression where all inputs
    map to the same vector.

    On failure, prints actionable fix instructions to stderr and exits the
    process with code 1.  This is designed to be called once at server
    startup (~2-3 seconds).

    Raises:
        SystemExit: If the smoke test fails for any reason.
    """
    logger.info("Running embedding smoke test...")

    try:
        vectors = embed_texts([SMOKE_TEST_A, SMOKE_TEST_B])
    except (ConnectionError, ValueError) as exc:
        _fail(
            f"Embedding smoke test failed: {exc}\n"
            f"Fix: Ensure Ollama is running and the model is pulled:\n"
            f"  ollama serve\n"
            f"  ollama pull {MODEL_NAME}"
        )
        return  # unreachable, but satisfies type checkers

    if len(vectors) != 2:
        _fail(
            f"Embedding smoke test failed: expected 2 vectors, got {len(vectors)}.\n"
            f"Fix: ollama pull {MODEL_NAME}"
        )
        return

    sim = cosine_similarity(vectors[0], vectors[1])
    logger.info(
        "Smoke test similarity: %.6f (threshold: %.2f)",
        sim,
        SMOKE_TEST_MAX_SIMILARITY,
    )

    if sim >= SMOKE_TEST_MAX_SIMILARITY:
        _fail(
            f"Embedding smoke test FAILED: cosine similarity {sim:.6f} >= "
            f"{SMOKE_TEST_MAX_SIMILARITY}.\n"
            f"This indicates the Ollama uppercase embedding regression "
            f"(identical vectors for distinct strings).\n"
            f"Fix: ollama pull {MODEL_NAME}"
        )
        return

    logger.info("Embedding smoke test passed (similarity=%.6f).", sim)


def embed_chunks_batched(
    texts: list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embed texts in batches with retry and exponential backoff.

    Splits *texts* into batches of *batch_size*, calls *embed_fn* on each
    batch, and collects the results into a flat list preserving input order.

    On failure, each batch is retried up to *max_retries* times (so
    ``max_retries + 1`` total attempts) with exponential backoff between
    retries.  If a batch still fails after exhausting retries, the entire
    operation is aborted with a :class:`RuntimeError`.

    Args:
        texts: Strings to embed.
        embed_fn: Callable that accepts a list of strings and returns a
            list of embedding vectors (each a list of floats).
        batch_size: Maximum number of texts per batch.
        max_retries: Number of retries after the initial attempt.
        backoff_base: Base for the exponential backoff formula
            ``backoff_base ** (attempt + 1)``.
        progress_callback: Optional callback invoked after each successful
            batch with ``(done, total)`` counts.

    Returns:
        Flat list of embedding vectors in the same order as *texts*.

    Raises:
        RuntimeError: If any batch fails after all retry attempts.
    """
    if not texts:
        return []

    total = len(texts)
    all_embeddings: list[list[float]] = []

    # Split into batches
    batches: list[list[str]] = []
    for i in range(0, total, batch_size):
        batches.append(texts[i : i + batch_size])

    for batch_num, batch_texts in enumerate(batches, start=1):
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                embeddings = embed_fn(batch_texts)
                all_embeddings.extend(embeddings)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max_retries:
                    delay = backoff_base ** (attempt + 1)
                    time.sleep(delay)

        if last_error is not None:
            msg = (
                f"Batch {batch_num} failed after {max_retries} retries: "
                f"{last_error}"
            )
            raise RuntimeError(msg)

        done = len(all_embeddings)
        pct = done / total * 100
        logger.info("Embedded %d/%d chunks (%.0f%%)", done, total, pct)

        if progress_callback is not None:
            progress_callback(done, total)

    return all_embeddings
