"""BM25 keyword index for hybrid search.

Provides an in-memory BM25 index backed by :pypi:`rank-bm25` that supports
incremental add/remove operations via full corpus rebuild (rank-bm25 has
no incremental API).

Exports:
    BM25Index               -- In-memory BM25 index with rebuild/add/remove/search
    tokenize_for_bm25       -- Tokenize a chunk + metadata for BM25 indexing
    tokens_to_metadata_string -- Convert token list to storable metadata string
"""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

logger = logging.getLogger("tokenkeeper.bm25_index")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_for_bm25(
    chunk_content: str,
    source_file: str,
    title: str = "",
    tags: str = "",
) -> list[str]:
    """Tokenize chunk content and metadata for BM25 indexing.

    Concatenates non-empty arguments with spaces and splits on whitespace.
    Title is lowercased; other fields are kept as-is.

    Args:
        chunk_content: The text content of the chunk.
        source_file: The source file path (included as a single token).
        title: Optional document title (lowercased before splitting).
        tags: Optional comma-separated tags (included as a single token).

    Returns:
        List of tokens for BM25 indexing.
    """
    parts: list[str] = []

    # Content tokens (split on whitespace)
    if chunk_content:
        parts.extend(chunk_content.split())

    # Source file as a single token + path segments as individual tokens
    if source_file:
        parts.append(source_file)
        # Split path into searchable segments: "data/tax-kb/2026/federal.json"
        # -> ["data", "tax", "kb", "2026", "federal", "json"]
        for segment in source_file.replace("/", " ").replace("-", " ").replace("_", " ").replace(".", " ").split():
            if len(segment) > 1:  # skip single-char fragments
                parts.append(segment.lower())

    # Title tokens (lowercased, split on whitespace)
    if title:
        parts.extend(title.lower().split())

    # Tags as a single token
    if tags:
        parts.append(tags)

    return parts


def tokens_to_metadata_string(tokens: list[str]) -> str:
    """Convert a token list to a space-separated string for metadata storage.

    Args:
        tokens: List of string tokens.

    Returns:
        Space-joined string, or empty string for empty input.
    """
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class BM25Index:
    """In-memory BM25 keyword index.

    Wraps :class:`rank_bm25.BM25Okapi` with document ID tracking and
    incremental add/remove support (via full rebuild, since rank-bm25
    has no incremental API).

    Typical usage::

        idx = BM25Index()
        idx.rebuild(doc_ids, tokenized_texts)
        results = idx.search(["query", "tokens"], top_k=10)
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []

    def rebuild(
        self,
        doc_ids: list[str],
        tokenized_texts: list[list[str]],
    ) -> None:
        """Replace all index state with the given documents.

        Args:
            doc_ids: Document identifiers (parallel with tokenized_texts).
            tokenized_texts: Pre-tokenized document texts.
        """
        self._doc_ids = list(doc_ids)
        self._corpus = list(tokenized_texts)
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
        logger.debug("BM25 index rebuilt with %d documents", len(self._doc_ids))

    def add_documents(
        self,
        doc_ids: list[str],
        tokenized_texts: list[list[str]],
    ) -> None:
        """Add documents to the index (triggers full rebuild).

        Args:
            doc_ids: Document identifiers to add.
            tokenized_texts: Pre-tokenized texts for the new documents.
        """
        self._doc_ids.extend(doc_ids)
        self._corpus.extend(tokenized_texts)
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
        logger.debug("Added %d documents, total now %d", len(doc_ids), len(self._doc_ids))

    def remove_documents(self, doc_ids_to_remove: set[str]) -> None:
        """Remove documents by ID (triggers full rebuild).

        IDs not present in the index are silently ignored.

        Args:
            doc_ids_to_remove: Set of document IDs to remove.
        """
        if not doc_ids_to_remove:
            return

        pairs = [
            (did, corpus)
            for did, corpus in zip(self._doc_ids, self._corpus)
            if did not in doc_ids_to_remove
        ]

        if pairs:
            self._doc_ids, self._corpus = map(list, zip(*pairs))
        else:
            self._doc_ids = []
            self._corpus = []

        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
        logger.debug(
            "Removed %d IDs, %d documents remain",
            len(doc_ids_to_remove),
            len(self._doc_ids),
        )

    def search(
        self,
        query_tokens: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search the index for the given query tokens.

        Args:
            query_tokens: Tokenized query.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score,
            containing only results with positive scores.
        """
        if self._bm25 is None or not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair with doc IDs, filter positive scores, sort descending
        scored = [
            (did, float(score))
            for did, score in zip(self._doc_ids, scores)
            if score > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._doc_ids)
