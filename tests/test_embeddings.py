"""Unit tests for the embedding smoke test, Ollama embedding helper, and batch embedder.

All tests mock ``requests.post`` so they run without a live Ollama instance.
Tests verify:
  - embed_texts success path (correct dimensions, count)
  - embed_texts lowercases input
  - embed_texts connection error handling
  - embed_texts wrong dimensions raises ValueError
  - run_smoke_test pass (distinct vectors)
  - run_smoke_test fail (identical vectors -> SystemExit)
  - run_smoke_test fail (Ollama down -> SystemExit)
  - embed_chunks_batched: single batch, multiple batches, order, empty input
  - embed_chunks_batched: retry success, retry exhausted, abort on failure
  - embed_chunks_batched: progress callback, custom batch size, backoff timing
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from tokenkeeper.types import (
    EXPECTED_EMBEDDING_DIMS,
    MODEL_NAME,
    SMOKE_TEST_A,
    SMOKE_TEST_B,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vector(seed: int = 42) -> list[float]:
    """Return a deterministic random vector of EXPECTED_EMBEDDING_DIMS."""
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(EXPECTED_EMBEDDING_DIMS)]


def _make_response(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock ``requests.Response`` returning *embeddings*."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embeddings": embeddings}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# embed_texts tests
# ---------------------------------------------------------------------------


@patch("tokenkeeper.embeddings.requests.post")
def test_embed_texts_success(mock_post: MagicMock) -> None:
    """embed_texts returns 2 vectors with correct dimensions."""
    from tokenkeeper.embeddings import embed_texts

    vec_a = _random_vector(1)
    vec_b = _random_vector(2)
    mock_post.return_value = _make_response([vec_a, vec_b])

    result = embed_texts(["hello", "world"])

    assert len(result) == 2
    assert len(result[0]) == EXPECTED_EMBEDDING_DIMS
    assert len(result[1]) == EXPECTED_EMBEDDING_DIMS
    assert result[0] == vec_a
    assert result[1] == vec_b


@patch("tokenkeeper.embeddings.requests.post")
def test_embed_texts_lowercases_input(mock_post: MagicMock) -> None:
    """embed_texts lowercases all input strings before sending to Ollama."""
    from tokenkeeper.embeddings import embed_texts

    vec = _random_vector(1)
    mock_post.return_value = _make_response([vec])

    embed_texts(["HELLO World"])

    # Inspect what was POSTed
    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["input"] == ["hello world"]
    assert payload["model"] == MODEL_NAME


@patch("tokenkeeper.embeddings.requests.post")
def test_embed_texts_connection_error(mock_post: MagicMock) -> None:
    """embed_texts raises ConnectionError when Ollama is unreachable."""
    import requests as real_requests

    from tokenkeeper.embeddings import embed_texts

    mock_post.side_effect = real_requests.ConnectionError("refused")

    with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
        embed_texts(["test"])


@patch("tokenkeeper.embeddings.requests.post")
def test_embed_texts_wrong_dimensions(mock_post: MagicMock) -> None:
    """embed_texts raises ValueError when vector dims != EXPECTED_EMBEDDING_DIMS."""
    from tokenkeeper.embeddings import embed_texts

    bad_vec = [0.1] * 512  # Wrong: 512 instead of 768
    mock_post.return_value = _make_response([bad_vec])

    with pytest.raises(ValueError, match="512 dimensions.*expected 768"):
        embed_texts(["test"])


# ---------------------------------------------------------------------------
# run_smoke_test tests
# ---------------------------------------------------------------------------


@patch("tokenkeeper.embeddings.requests.post")
def test_smoke_test_pass(mock_post: MagicMock) -> None:
    """run_smoke_test succeeds when vectors are distinct (low similarity)."""
    from tokenkeeper.embeddings import run_smoke_test

    # Two clearly different vectors -> cosine similarity well below 0.99
    vec_a = _random_vector(100)
    vec_b = _random_vector(200)
    mock_post.return_value = _make_response([vec_a, vec_b])

    # Should not raise
    run_smoke_test()

    # Verify the correct strings were sent (lowercased)
    call_args = mock_post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["input"] == [SMOKE_TEST_A.lower(), SMOKE_TEST_B.lower()]


@patch("tokenkeeper.embeddings.requests.post")
def test_smoke_test_fail_identical_vectors(
    mock_post: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_smoke_test exits with code 1 when vectors are identical (regression)."""
    from tokenkeeper.embeddings import run_smoke_test

    # Same vector for both -> cosine similarity = 1.0
    identical = _random_vector(42)
    mock_post.return_value = _make_response([identical, identical])

    with pytest.raises(SystemExit) as exc_info:
        run_smoke_test()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "ollama pull" in captured.err
    assert MODEL_NAME in captured.err
    assert "uppercase embedding regression" in captured.err.lower() or "FAILED" in captured.err


@patch("tokenkeeper.embeddings.requests.post")
def test_smoke_test_fail_ollama_down(
    mock_post: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_smoke_test exits with code 1 when Ollama is unreachable."""
    import requests as real_requests

    from tokenkeeper.embeddings import run_smoke_test

    mock_post.side_effect = real_requests.ConnectionError("Connection refused")

    with pytest.raises(SystemExit) as exc_info:
        run_smoke_test()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "ollama pull" in captured.err or "ollama serve" in captured.err


# ---------------------------------------------------------------------------
# embed_chunks_batched tests
# ---------------------------------------------------------------------------


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake embeddings (768-dim) for each input text."""
    return [[0.1] * 768 for _ in texts]


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_single_batch(mock_sleep: MagicMock) -> None:
    """10 texts with batch_size=15 -> single call to embed_fn."""
    from tokenkeeper.embeddings import embed_chunks_batched

    calls: list[list[str]] = []

    def tracking_fn(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return _fake_embed_fn(texts)

    texts = [f"text_{i}" for i in range(10)]
    result = embed_chunks_batched(texts, tracking_fn, batch_size=15)

    assert len(calls) == 1
    assert len(calls[0]) == 10
    assert len(result) == 10
    assert all(len(v) == 768 for v in result)


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_multiple_batches(mock_sleep: MagicMock) -> None:
    """35 texts with batch_size=15 -> 3 calls (15, 15, 5)."""
    from tokenkeeper.embeddings import embed_chunks_batched

    calls: list[list[str]] = []

    def tracking_fn(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return _fake_embed_fn(texts)

    texts = [f"text_{i}" for i in range(35)]
    result = embed_chunks_batched(texts, tracking_fn, batch_size=15)

    assert len(calls) == 3
    assert len(calls[0]) == 15
    assert len(calls[1]) == 15
    assert len(calls[2]) == 5
    assert len(result) == 35


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_preserves_order(mock_sleep: MagicMock) -> None:
    """Embeddings returned in same order as input texts."""
    from tokenkeeper.embeddings import embed_chunks_batched

    def order_fn(texts: list[str]) -> list[list[float]]:
        # Return unique vectors based on text index
        return [[float(t.split("_")[1])] * 768 for t in texts]

    texts = [f"text_{i}" for i in range(25)]
    result = embed_chunks_batched(texts, order_fn, batch_size=10)

    assert len(result) == 25
    for i in range(25):
        assert result[i][0] == float(i), f"Embedding {i} out of order"


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_empty_input(mock_sleep: MagicMock) -> None:
    """0 texts -> 0 embeddings, no calls to embed_fn."""
    from tokenkeeper.embeddings import embed_chunks_batched

    calls: list[list[str]] = []

    def tracking_fn(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return _fake_embed_fn(texts)

    result = embed_chunks_batched([], tracking_fn)

    assert result == []
    assert len(calls) == 0


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_retry_success(mock_sleep: MagicMock) -> None:
    """embed_fn fails once, succeeds on retry -> returns embeddings."""
    from tokenkeeper.embeddings import embed_chunks_batched

    call_count = 0

    def flaky_fn(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Ollama timeout")
        return _fake_embed_fn(texts)

    texts = [f"text_{i}" for i in range(5)]
    result = embed_chunks_batched(texts, flaky_fn, batch_size=15)

    assert len(result) == 5
    assert call_count == 2  # 1 fail + 1 success


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_retry_exhausted_raises(mock_sleep: MagicMock) -> None:
    """embed_fn always fails -> RuntimeError after all attempts."""
    from tokenkeeper.embeddings import embed_chunks_batched

    call_count = 0

    def always_fail(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Ollama down")

    texts = [f"text_{i}" for i in range(5)]
    with pytest.raises(RuntimeError, match="(?i)batch"):
        embed_chunks_batched(texts, always_fail, batch_size=15, max_retries=3)

    # max_retries=3 means 4 total attempts (1 initial + 3 retries)
    assert call_count == 4


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_abort_on_failure(mock_sleep: MagicMock) -> None:
    """embed_fn fails on 2nd batch after retries -> RuntimeError."""
    from tokenkeeper.embeddings import embed_chunks_batched

    batch_num = 0

    def fail_second_batch(texts: list[str]) -> list[list[float]]:
        nonlocal batch_num
        # First batch succeeds, all calls for second batch fail
        if batch_num == 0 and len(texts) == 5:
            # This is first batch (batch_size=5), succeed
            batch_num = 1
            return _fake_embed_fn(texts)
        elif batch_num == 0:
            batch_num = 1
            return _fake_embed_fn(texts)
        raise ConnectionError("Ollama down on batch 2")

    texts = [f"text_{i}" for i in range(10)]
    with pytest.raises(RuntimeError, match="(?i)batch"):
        embed_chunks_batched(texts, fail_second_batch, batch_size=5, max_retries=3)


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_progress_callback(mock_sleep: MagicMock) -> None:
    """Callback called with (done, total) after each batch."""
    from tokenkeeper.embeddings import embed_chunks_batched

    progress: list[tuple[int, int]] = []

    def record_progress(done: int, total: int) -> None:
        progress.append((done, total))

    texts = [f"text_{i}" for i in range(35)]
    embed_chunks_batched(
        texts, _fake_embed_fn, batch_size=15, progress_callback=record_progress
    )

    assert progress == [(15, 35), (30, 35), (35, 35)]


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_progress_callback_none(mock_sleep: MagicMock) -> None:
    """No callback provided -> no error."""
    from tokenkeeper.embeddings import embed_chunks_batched

    texts = [f"text_{i}" for i in range(5)]
    # Should not raise even without callback
    result = embed_chunks_batched(texts, _fake_embed_fn, batch_size=15)

    assert len(result) == 5


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_custom_batch_size(mock_sleep: MagicMock) -> None:
    """batch_size=5 with 12 texts -> 3 calls (5, 5, 2)."""
    from tokenkeeper.embeddings import embed_chunks_batched

    calls: list[list[str]] = []

    def tracking_fn(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return _fake_embed_fn(texts)

    texts = [f"text_{i}" for i in range(12)]
    result = embed_chunks_batched(texts, tracking_fn, batch_size=5)

    assert len(calls) == 3
    assert len(calls[0]) == 5
    assert len(calls[1]) == 5
    assert len(calls[2]) == 2
    assert len(result) == 12


@patch("tokenkeeper.embeddings.time.sleep")
def test_batch_backoff_timing(mock_sleep: MagicMock) -> None:
    """Mock time.sleep, verify exponential delays on retries."""
    from tokenkeeper.embeddings import embed_chunks_batched

    call_count = 0

    def fail_twice(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("Ollama timeout")
        return _fake_embed_fn(texts)

    texts = [f"text_{i}" for i in range(5)]
    result = embed_chunks_batched(
        texts, fail_twice, batch_size=15, backoff_base=2.0
    )

    assert len(result) == 5
    assert call_count == 3  # 2 fails + 1 success

    # Verify exponential backoff: sleep(2^1=2.0), sleep(2^2=4.0)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(2.0)  # backoff_base^1
    mock_sleep.assert_any_call(4.0)  # backoff_base^2
