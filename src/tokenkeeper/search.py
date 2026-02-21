"""Search engine for TokenKeeper.

Provides the hybrid search pipeline combining ChromaDB semantic search with
BM25 keyword search via Reciprocal Rank Fusion (RRF).

Exports:
    SearchResult             -- Frozen dataclass for search results
    semantic_search          -- Query ChromaDB by vector similarity
    keyword_search           -- Query BM25 index by keyword tokens
    reciprocal_rank_fusion   -- Combine ranked lists using RRF
    hybrid_search            -- Orchestrate semantic + keyword + RRF
    normalize_scores         -- Normalize scores to 0-1 range
    enrich_results           -- Convert (doc_id, score) to SearchResult
    search                   -- Top-level search pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import chromadb

from tokenkeeper.bm25_index import BM25Index

logger = logging.getLogger("tokenkeeper.search")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with metadata.

    Attributes:
        chunk_id:           Unique chunk identifier (e.g. "docs/readme.md::chunk_0")
        content:            Chunk text content
        score:              Relevance score (0-1 normalized)
        source_file:        Source document path (e.g. "docs/readme.md")
        chunk_index:        Position within source document
        total_chunks:       Total chunks from the source document
        title:              Document title from frontmatter
        tags:               Comma-separated tags from frontmatter
        heading_hierarchy:  Heading context path (e.g. "# Config > ## Database")
        language:           Programming language for code chunks
        symbol_name:        Function or class name for code chunks
        symbol_type:        Symbol kind: "function", "class", "method", "module"
        line_start:         1-based start line in source file
        line_end:           1-based end line in source file
    """

    chunk_id: str
    content: str
    score: float
    source_file: str
    chunk_index: int
    total_chunks: int
    title: str
    tags: str
    heading_hierarchy: str = ""
    language: str = ""
    symbol_name: str = ""
    symbol_type: str = ""
    line_start: int = 0
    line_end: int = 0


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


def semantic_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Query ChromaDB by vector similarity.

    Args:
        collection:      ChromaDB collection with cosine space.
        query_embedding: Pre-computed query embedding vector.
        top_k:           Maximum number of results.

    Returns:
        List of (doc_id, similarity_score) sorted by descending similarity.
        Similarity = 1.0 - cosine_distance.
    """
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    ids = results["ids"][0]
    distances = results["distances"][0]

    if not ids:
        return []

    # ChromaDB cosine space: distance = 1 - similarity
    pairs: list[tuple[str, float]] = []
    for doc_id, distance in zip(ids, distances):
        similarity = 1.0 - distance
        pairs.append((doc_id, similarity))

    logger.debug("Semantic search returned %d results", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def keyword_search(
    bm25_index: BM25Index,
    query_text: str,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Tokenize query and search BM25 index.

    Args:
        bm25_index: In-memory BM25 keyword index.
        query_text: Raw query string (will be lowercased).
        top_k:      Maximum number of results.

    Returns:
        List of (doc_id, bm25_score) sorted by descending score.
    """
    tokens = query_text.lower().split()
    return bm25_index.search(tokens, top_k=top_k)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    For each document, RRF score = sum(1 / (k + rank)) across all lists
    where rank is 1-based position in each list.

    Args:
        ranked_lists: List of ranked result lists, each containing
                      (doc_id, score) tuples.
        k:            RRF parameter (default 60, from original paper).

    Returns:
        Combined ranked list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return combined


# ---------------------------------------------------------------------------
# Hybrid search orchestrator
# ---------------------------------------------------------------------------


def hybrid_search(
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    alpha: float = 0.5,
    top_k: int = 10,
    mode: str = "hybrid",
) -> list[tuple[str, float]]:
    """Orchestrate semantic + keyword search with alpha-weighted RRF.

    Supports three modes:
      - ``"semantic"`` — pure semantic search (forces alpha=1.0)
      - ``"keyword"``  — pure keyword search (forces alpha=0.0)
      - ``"hybrid"``   — blend both via alpha-weighted RRF

    Args:
        collection:  ChromaDB collection with cosine space.
        bm25_index:  In-memory BM25 keyword index.
        query:       Raw query string.
        embed_fn:    Callable that embeds a list of strings to vectors.
        alpha:       Blend weight (0.0=keyword, 1.0=semantic). Overridden by mode.
        top_k:       Maximum number of results to return.
        mode:        Search mode: "hybrid", "semantic", or "keyword".

    Returns:
        List of (doc_id, weighted_rrf_score) sorted by descending score.
    """
    # Mode shortcuts override alpha
    if mode == "semantic":
        alpha = 1.0
    elif mode == "keyword":
        alpha = 0.0

    # Embed query
    query_embedding = embed_fn([query])[0]

    # Over-fetch for better fusion (cap at 200 to support higher top_k)
    fetch_k = min(top_k * 3, 200)

    # Pure semantic shortcut
    if alpha >= 1.0:
        return semantic_search(collection, query_embedding, fetch_k)[:top_k]

    # Pure keyword shortcut
    if alpha <= 0.0:
        return keyword_search(bm25_index, query, fetch_k)[:top_k]

    # Hybrid: run both searches
    semantic_results = semantic_search(collection, query_embedding, fetch_k)
    keyword_results = keyword_search(bm25_index, query, fetch_k)

    # Alpha-weighted RRF scoring
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(semantic_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha / (60 + rank)
    for rank, (doc_id, _) in enumerate(keyword_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) / (60 + rank)

    combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.debug(
        "Hybrid search: query=%r, mode=%s, alpha=%.2f, results=%d",
        query[:50],
        mode,
        alpha,
        len(combined[:top_k]),
    )
    return combined[:top_k]


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def normalize_scores(
    ranked: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize scores to 0-1 range by dividing by max score.

    Args:
        ranked: List of (doc_id, score) tuples.

    Returns:
        Same list with scores divided by max score (max becomes 1.0).
        Empty input returns empty list. All-zero scores returned as-is.
    """
    if not ranked:
        return []

    max_score = max(score for _, score in ranked)
    if max_score <= 0:
        return ranked

    return [(doc_id, score / max_score) for doc_id, score in ranked]


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------


def enrich_results(
    collection: chromadb.Collection,
    ranked: list[tuple[str, float]],
) -> list[SearchResult]:
    """Convert (doc_id, score) tuples into fully-enriched SearchResult objects.

    Fetches metadata and documents from ChromaDB in a single batch query.

    Args:
        collection: ChromaDB collection containing the chunks.
        ranked:     List of (doc_id, score) tuples in ranked order.

    Returns:
        List of SearchResult objects preserving the ranked order.
    """
    if not ranked:
        return []

    ids = [doc_id for doc_id, _ in ranked]

    # Batch-fetch from ChromaDB
    fetched = collection.get(ids=ids, include=["metadatas", "documents"])

    # Build lookup dict: {id: (metadata, document)}
    lookup: dict[str, tuple[dict, str]] = {}
    for i, doc_id in enumerate(fetched["ids"]):
        meta = fetched["metadatas"][i] if fetched["metadatas"] else {}
        doc = fetched["documents"][i] if fetched["documents"] else ""
        lookup[doc_id] = (meta, doc)

    # Pre-compute total_chunks per source_file from all result chunks
    source_counts: dict[str, int] = {}
    for doc_id in ids:
        if doc_id in lookup:
            sf = lookup[doc_id][0].get("source_file", "")
            source_counts[sf] = source_counts.get(sf, 0) + 1

    # Build SearchResult list preserving order
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        if doc_id not in lookup:
            continue

        meta, content = lookup[doc_id]
        source_file = meta.get("source_file", "")

        results.append(
            SearchResult(
                chunk_id=doc_id,
                content=content,
                score=score,
                source_file=source_file,
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=source_counts.get(source_file, 0),
                title=str(meta.get("title", "")),
                tags=str(meta.get("tags", "")),
                heading_hierarchy=str(meta.get("heading_hierarchy", "")),
                language=str(meta.get("language", "")),
                symbol_name=str(meta.get("symbol_name", "")),
                symbol_type=str(meta.get("symbol_type", "")),
                line_start=int(meta.get("line_start", 0)),
                line_end=int(meta.get("line_end", 0)),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Top-level search pipeline
# ---------------------------------------------------------------------------


def search(
    query: str,
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    embed_fn: Callable[[list[str]], list[list[float]]],
    alpha: float = 0.5,
    top_k: int = 10,
    mode: str = "hybrid",
) -> list[SearchResult]:
    """Full search pipeline: embed → search → fuse → normalize → enrich.

    This is the top-level public API for Phase 6 (MCP Server) to call.

    Args:
        query:       Raw query string.
        collection:  ChromaDB collection with cosine space.
        bm25_index:  In-memory BM25 keyword index.
        embed_fn:    Callable that embeds a list of strings to vectors.
        alpha:       Blend weight (0.0=keyword, 1.0=semantic).
        top_k:       Maximum number of results.
        mode:        Search mode: "hybrid", "semantic", or "keyword".

    Returns:
        List of SearchResult objects with normalized scores and metadata.
    """
    # Step 1: Run hybrid search pipeline
    ranked = hybrid_search(collection, bm25_index, query, embed_fn, alpha, top_k, mode)

    # Step 2: Normalize scores to 0-1 range
    normalized = normalize_scores(ranked)

    # Step 3: Enrich with metadata from ChromaDB
    results = enrich_results(collection, normalized)

    logger.info(
        "Search: query=%r, mode=%s, alpha=%.2f, results=%d",
        query[:50],
        mode,
        alpha,
        len(results),
    )
    return results
