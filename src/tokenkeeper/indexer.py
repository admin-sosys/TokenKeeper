"""Indexing orchestrator for TokenKeeper.

Ties together file discovery, document ingestion, batch embedding, ChromaDB
storage, and BM25 index updates into a single ``index_documents()`` pipeline.

Exports:
    IndexingResult        -- Frozen dataclass summarising an indexing run
    index_documents       -- Full indexing pipeline: discover -> hash -> ingest -> embed -> store
    _read_file_with_retry -- Read file with exponential backoff on PermissionError (PLT-05)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chromadb

from tokenkeeper.bm25_index import BM25Index, tokenize_for_bm25, tokens_to_metadata_string
from tokenkeeper.config import RagConfig
from tokenkeeper.discovery import CODE_EXTENSIONS, discover_code_files, discover_markdown_files
from tokenkeeper.embeddings import embed_chunks_batched
from tokenkeeper.ingestion import (
    DocumentChunk,
    chunk_document_heading_aware,
    ingest_code_file,
    ingest_file,
    normalize_whitespace,
)
from tokenkeeper.storage import (
    compute_file_hash,
    file_needs_reindex,
    get_chunk_ids_for_file,
    make_chunk_id,
    replace_file_chunks,
)

logger = logging.getLogger("tokenkeeper.indexer")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexingResult:
    """Summary of an indexing run.

    Attributes:
        files_indexed:  Number of files that were (re-)indexed.
        chunks_indexed: Total number of chunks stored across all indexed files.
        files_skipped:  Number of files skipped because content was unchanged.
        files_failed:   Number of files that could not be read (PermissionError, etc.).
    """

    files_indexed: int
    chunks_indexed: int
    files_skipped: int
    files_failed: int


# ---------------------------------------------------------------------------
# File reading with retry (Windows file locking -- PLT-05)
# ---------------------------------------------------------------------------


def _read_file_with_retry(
    path: Path,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> str:
    """Read file with exponential backoff retry on PermissionError.

    Windows file locking (antivirus, editor save-in-progress) can cause
    transient PermissionError. Retries with delays: 0.5s, 1.0s, 2.0s.

    Args:
        path: File to read.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).

    Returns:
        File content as string.

    Raises:
        PermissionError: If all retries exhausted.
        UnicodeDecodeError: If file is not valid UTF-8 (not retried).
    """
    for attempt in range(max_retries + 1):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "PermissionError reading %s (attempt %d/%d), retrying in %.1fs",
                    path, attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "PermissionError reading %s after %d retries",
                    path, max_retries + 1,
                )
                raise


# ---------------------------------------------------------------------------
# Mode routing helpers (09-06)
# ---------------------------------------------------------------------------


def _discover_for_mode(
    project_root: Path,
    content_mode: str,
) -> list[Path]:
    """Discover files based on content mode.

    Args:
        project_root: Root directory.
        content_mode: ``"docs"``, ``"code"``, or ``"both"``.

    Returns:
        Sorted list of files to index.
    """
    if content_mode == "docs":
        return discover_markdown_files(project_root)
    elif content_mode == "code":
        return discover_code_files(project_root)
    elif content_mode == "both":
        md_files = discover_markdown_files(project_root)
        code_files = discover_code_files(project_root)
        all_files = list(set(md_files + code_files))
        return sorted(all_files)
    else:
        logger.warning("Unknown content_mode '%s', defaulting to docs", content_mode)
        return discover_markdown_files(project_root)


def _json_value_to_prose(key: str, value: object) -> str:
    """Convert a JSON key-value pair into a natural language sentence."""
    if isinstance(value, str):
        return f"{key}: {value}"
    elif isinstance(value, (int, float)):
        return f"{key}: {value}"
    elif isinstance(value, bool):
        return f"{key}: {'yes' if value else 'no'}"
    elif isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return f"{key}: {', '.join(str(v) for v in value)}"
        return f"{key}: [{len(value)} items]"
    elif isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items() if isinstance(v, (str, int, float, bool))]
        return f"{key} - {'; '.join(parts)}" if parts else f"{key}: [object]"
    return f"{key}: {value}"


def _ingest_json_file(
    file_path: Path,
    project_root: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Ingest a JSON file with structure-aware chunking.

    Parses JSON and creates semantically meaningful chunks:
    - **Arrays**: Each item becomes its own chunk with a natural language preamble.
    - **Objects**: Each top-level key becomes a chunk; nested objects are flattened
      into prose descriptions.

    A file-context preamble (derived from the file path) is prepended to every
    chunk so the embedding model knows what domain the data belongs to.

    Falls back to raw-text chunking if the file isn't valid JSON.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping non-UTF-8 JSON file: %s", file_path)
        return []

    source = file_path.relative_to(project_root).as_posix()

    # Build a context preamble from the file path
    # "data/tax-kb/2026/deadlines/federal-deadlines.json"
    # -> "Context: tax-kb 2026 deadlines federal-deadlines"
    path_parts = source.replace(".json", "").replace("/", " ").replace("-", " ").replace("_", " ")
    preamble = f"[File: {source}]"

    # Try to parse as JSON
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fall back to raw-text chunking
        body = normalize_whitespace(text)
        if not body.strip():
            return []
        raw_chunks = chunk_document_heading_aware(body, chunk_size=chunk_size, overlap=overlap)
        return [
            DocumentChunk(
                content=f"{preamble}\n{content}",
                source_file=source,
                chunk_index=i,
                char_start=char_start,
                char_end=char_end,
                total_chunks=len(raw_chunks),
                heading_hierarchy="",
            )
            for i, (content, char_start, char_end, _) in enumerate(raw_chunks)
        ]

    chunks: list[DocumentChunk] = []

    if isinstance(data, list):
        # Array of items — each item becomes a chunk with prose preamble
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Build natural language description from fields
                prose_parts = [preamble]
                name = item.get("name") or item.get("title") or item.get("id") or f"Item {i + 1}"
                prose_parts.append(f"{name}.")
                for k, v in item.items():
                    if k in ("name", "title", "id"):
                        continue  # already used as header
                    prose_parts.append(_json_value_to_prose(k, v))
                content = "\n".join(prose_parts)
            else:
                content = f"{preamble}\n{json.dumps(item, indent=2)}"

            chunks.append(DocumentChunk(
                content=content,
                source_file=source,
                chunk_index=i,
                char_start=0,
                char_end=len(content),
                total_chunks=len(data),
                heading_hierarchy=f"[{i}]",
            ))

    elif isinstance(data, dict):
        # Object — chunk by top-level keys
        keys = list(data.keys())
        for i, key in enumerate(keys):
            value = data[key]
            prose_parts = [preamble, f"{key}:"]

            if isinstance(value, dict):
                # Flatten nested object into prose
                for k, v in value.items():
                    if isinstance(v, dict):
                        # Two levels deep
                        sub_parts = [f"{sk}: {sv}" for sk, sv in v.items()
                                     if isinstance(sv, (str, int, float, bool))]
                        prose_parts.append(f"  {k}: {'; '.join(sub_parts)}" if sub_parts else f"  {k}: [object]")
                    elif isinstance(v, list):
                        if all(isinstance(item, dict) for item in v):
                            # Array of objects under a key (e.g., scheduleH.lines)
                            prose_parts.append(f"  {k} ({len(v)} items):")
                            for j, item in enumerate(v):
                                item_desc = "; ".join(
                                    f"{ik}: {iv}" for ik, iv in item.items()
                                    if isinstance(iv, (str, int, float, bool))
                                )
                                prose_parts.append(f"    {item_desc}")
                                # Prevent chunks from getting too large
                                if len("\n".join(prose_parts)) > chunk_size:
                                    break
                        else:
                            prose_parts.append(f"  {k}: {', '.join(str(x) for x in v[:10])}")
                    else:
                        prose_parts.append(f"  {k}: {v}")
            elif isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    for j, item in enumerate(value):
                        item_desc = "; ".join(
                            f"{ik}: {iv}" for ik, iv in item.items()
                            if isinstance(iv, (str, int, float, bool))
                        )
                        prose_parts.append(f"  {item_desc}")
                        if len("\n".join(prose_parts)) > chunk_size:
                            break
                else:
                    prose_parts.append(f"  {json.dumps(value[:10])}")
            else:
                prose_parts.append(f"  {value}")

            content = "\n".join(prose_parts)
            chunks.append(DocumentChunk(
                content=content,
                source_file=source,
                chunk_index=i,
                char_start=0,
                char_end=len(content),
                total_chunks=len(keys),
                heading_hierarchy=key,
            ))

    if not chunks:
        # Empty JSON or unexpected structure — fall back to raw text
        body = normalize_whitespace(text)
        if body.strip():
            chunks.append(DocumentChunk(
                content=f"{preamble}\n{body}",
                source_file=source,
                chunk_index=0,
                char_start=0,
                char_end=len(body),
                total_chunks=1,
                heading_hierarchy="",
            ))

    return chunks


def _ingest_file_routed(
    file_path: Path,
    project_root: Path,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Route file to correct ingestion pipeline based on extension.

    - ``.md`` files go through :func:`ingest_file` (heading-aware markdown chunker).
    - Code files (``.py``, ``.ts``, etc.) go through :func:`ingest_code_file`.
    - Unsupported extensions return an empty list.

    Args:
        file_path:    Path to the file.
        project_root: Root directory for relative path computation.
        chunk_size:   Soft character limit per chunk.
        overlap:      Overlap size (only used for markdown).

    Returns:
        List of :class:`DocumentChunk` instances.
    """
    ext = file_path.suffix.lower()
    if ext in (".md", ".mdx"):
        return ingest_file(file_path, project_root, chunk_size, overlap)
    elif ext == ".json":
        return _ingest_json_file(file_path, project_root, chunk_size, overlap)
    elif ext in CODE_EXTENSIONS:
        return ingest_code_file(file_path, project_root, chunk_size)
    else:
        logger.warning("Unsupported file extension for ingestion: %s", ext)
        return []


# ---------------------------------------------------------------------------
# Indexing pipeline
# ---------------------------------------------------------------------------


def index_documents(
    project_root: Path,
    config: RagConfig,
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    embed_fn: Callable[[list[str]], list[list[float]]],
    progress_callback: Callable[[int, int], None] | None = None,
    target_files: list[Path] | None = None,
    force: bool = False,
) -> IndexingResult:
    """Run the full indexing pipeline.

    Steps:

    1. **Discover** markdown files under *project_root* (or use *target_files*).
    2. **Check hashes** to determine which files need re-indexing (skip if *force*).
    3. **Detect removed files** (skipped for partial reindex with *target_files*).
    4. **Ingest** changed files into :class:`DocumentChunk` lists.
    5. **Generate embeddings** in batches via *embed_fn*.
    6. **Store** chunks and embeddings in ChromaDB (per-file replacement).
    7. **Update BM25** index with tokenised chunk content.

    Args:
        project_root:      Root directory containing markdown files.
        config:            RAG configuration (chunk_size, overlap, etc.).
        collection:        ChromaDB collection for vector storage.
        bm25_index:        In-memory BM25 keyword index.
        embed_fn:          Callable that embeds a list of strings to vectors.
        progress_callback: Optional ``(done, total)`` callback for embedding progress.
        target_files:      Optional list of specific files to index. When set,
                           only these files are processed and removed-file cleanup
                           is skipped. If ``None``, all markdown files are discovered.
        force:             If ``True``, reindex files even if their content hash
                           has not changed. Default ``False``.

    Returns:
        An :class:`IndexingResult` with indexing statistics.

    Raises:
        RuntimeError: If any embedding batch fails after retries (abort semantics).
    """
    # ------------------------------------------------------------------
    # Step 1: Discover files (or use target_files for partial reindex)
    # ------------------------------------------------------------------
    if target_files is not None:
        discovered_files = target_files
        logger.info("Partial reindex: %d target files", len(discovered_files))
    else:
        content_mode = getattr(config, "content_mode", "docs")
        discovered_files = _discover_for_mode(project_root, content_mode)
        logger.info("Discovered %d files (content_mode=%s)", len(discovered_files), content_mode)

    # ------------------------------------------------------------------
    # Step 2: Check which files need re-indexing
    # ------------------------------------------------------------------
    files_to_index: list[Path] = []
    file_contents: dict[str, str] = {}  # source_file -> file content (for hash)
    files_skipped = 0
    files_failed = 0

    # Build a set of discovered source_file posix paths for step 3
    discovered_posix: set[str] = set()

    for file_path in discovered_files:
        source_file = file_path.relative_to(project_root).as_posix()
        discovered_posix.add(source_file)

        try:
            content = _read_file_with_retry(file_path)
        except (PermissionError, OSError) as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            files_failed += 1
            continue

        if force or file_needs_reindex(collection, source_file, content):
            files_to_index.append(file_path)
            file_contents[source_file] = content
        else:
            files_skipped += 1

    logger.info(
        "Files to index: %d, skipped (unchanged): %d, failed: %d",
        len(files_to_index),
        files_skipped,
        files_failed,
    )

    # ------------------------------------------------------------------
    # Step 3: Detect removed files and clean up (skip for partial reindex)
    # ------------------------------------------------------------------
    if target_files is None:
        stored_result = collection.get(include=["metadatas"])
        stored_source_files: set[str] = set()
        if stored_result["metadatas"]:
            for meta in stored_result["metadatas"]:
                sf = meta.get("source_file")
                if sf:
                    stored_source_files.add(sf)

        removed_files = stored_source_files - discovered_posix
        for removed_sf in removed_files:
            stale_ids = get_chunk_ids_for_file(collection, removed_sf)
            if stale_ids:
                collection.delete(ids=stale_ids)
                logger.info("Deleted %d stale chunks for removed file %s", len(stale_ids), removed_sf)
            # Remove from BM25
            bm25_index.remove_documents(set(stale_ids))
    else:
        logger.debug("Partial reindex: skipping removed-file cleanup")

    # ------------------------------------------------------------------
    # Step 4: Ingest changed files
    # ------------------------------------------------------------------
    file_chunks: list[tuple[str, list[DocumentChunk]]] = []
    for file_path in files_to_index:
        source_file = file_path.relative_to(project_root).as_posix()
        chunks = _ingest_file_routed(file_path, project_root, config.chunk_size, config.overlap)
        if chunks:
            file_chunks.append((source_file, chunks))

    # ------------------------------------------------------------------
    # Step 5: Generate embeddings
    # ------------------------------------------------------------------
    # Collect all chunk texts in order, enriched with file-path context
    # so the embedding model knows what domain each chunk belongs to.
    # The [File: ...] preamble is only added to the embedding input,
    # not to the stored chunk content (unless the chunker already added it).
    all_texts: list[str] = []
    for source_file, chunks in file_chunks:
        preamble = f"[File: {source_file}]"
        for chunk in chunks:
            text = chunk.content
            if not text.startswith("[File:"):
                text = f"{preamble}\n{text}"
            all_texts.append(text)

    all_embeddings: list[list[float]] = []
    if all_texts:
        # RuntimeError from embed_chunks_batched propagates (abort semantics)
        all_embeddings = embed_chunks_batched(
            all_texts,
            embed_fn,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Step 6: Store in ChromaDB (per-file) and Step 7: Update BM25
    # ------------------------------------------------------------------
    total_chunks_indexed = 0
    embed_offset = 0

    for source_file, chunks in file_chunks:
        n_chunks = len(chunks)
        file_embeddings = all_embeddings[embed_offset : embed_offset + n_chunks]
        embed_offset += n_chunks

        # Compute file-level hash for change detection
        file_content = file_contents.get(source_file, "")
        file_hash = compute_file_hash(file_content) if file_content else ""

        # Build BM25 token strings for metadata storage
        bm25_token_strings: list[str] = []
        new_bm25_ids: list[str] = []
        new_bm25_tokens: list[list[str]] = []
        for chunk in chunks:
            chunk_id = make_chunk_id(source_file, chunk.chunk_index)
            fm = chunk.frontmatter or {}
            title = fm.get("title", "") or ""
            tags_raw = fm.get("tags", [])
            tags_str = ",".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)

            tokens = tokenize_for_bm25(
                chunk.content.lower(),
                source_file,
                title=title,
                tags=tags_str,
            )
            new_bm25_ids.append(chunk_id)
            new_bm25_tokens.append(tokens)
            bm25_token_strings.append(tokens_to_metadata_string(tokens))

        # Step 6: Replace file chunks in ChromaDB (with file hash + BM25 tokens)
        replace_file_chunks(
            collection, source_file, chunks, file_embeddings,
            file_hash=file_hash,
            bm25_tokens_list=bm25_token_strings,
            indexed_at=time.time(),
        )
        total_chunks_indexed += n_chunks

        # Step 7: Update BM25 -- remove old entries for this file, add new
        existing_bm25_ids = {
            did for did in bm25_index._doc_ids
            if did.startswith(source_file + "::chunk_")
        }
        if existing_bm25_ids:
            bm25_index.remove_documents(existing_bm25_ids)

        if new_bm25_ids:
            bm25_index.add_documents(new_bm25_ids, new_bm25_tokens)

    # ------------------------------------------------------------------
    # Step 8: Return result
    # ------------------------------------------------------------------
    result = IndexingResult(
        files_indexed=len(file_chunks),
        chunks_indexed=total_chunks_indexed,
        files_skipped=files_skipped,
        files_failed=files_failed,
    )
    logger.info(
        "Indexing complete: %d files indexed (%d chunks), %d skipped, %d failed",
        result.files_indexed,
        result.chunks_indexed,
        result.files_skipped,
        result.files_failed,
    )
    return result
