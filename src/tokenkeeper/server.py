"""FastMCP server for TokenKeeper.

Hosts the MCP server with a lifespan that:
  1. Runs health checks and embedding smoke test
  2. Loads configuration
  3. Creates ChromaDB client and collection
  4. Runs initial document indexing
  5. Rebuilds BM25 index from stored metadata

Tools:
  - search_knowledge:   Hybrid search over indexed documents
  - indexing_status:     Report on index state and last indexing results
  - reindex_documents:   Trigger full or partial document reindexing
  - get_index_stats:     Show index statistics (files, chunks, timestamps)

All logging goes to ``.rag/rag.log`` and stderr -- **never** stdout --
to avoid corrupting the MCP stdio transport.

Exports:
    mcp          -- The FastMCP server instance
    setup_logging -- Configure file + stderr logging
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator

from fastmcp import Context, FastMCP


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(rag_dir: Path) -> None:
    """Configure logging to ``rag.log`` (file) and stderr (console).

    - **File handler:** RotatingFileHandler at DEBUG level, 5 MB max,
      3 backup files, written to ``<rag_dir>/rag.log``.
    - **Console handler:** StreamHandler on **stderr** at WARNING level.
    - **Never stdout:** All output avoids stdout to protect the MCP
      stdio transport.

    Args:
        rag_dir: The ``.rag/`` directory where ``rag.log`` will be created.
    """
    root_logger = logging.getLogger("tokenkeeper")
    root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    root_logger.handlers.clear()

    # File handler: 5 MB, 3 backups
    log_path = rag_dir / "rag.log"
    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    # Console handler: stderr only, WARNING level
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

logger = logging.getLogger("tokenkeeper.server")


def _make_reindex_callback(
    project_root: Path,
    config: "RagConfig",  # noqa: F821
    collection: "chromadb.Collection",  # noqa: F821
    bm25_index: "BM25Index",  # noqa: F821
    embed_fn,
    indexing_state: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
) -> "Callable[[list[str]], None]":  # noqa: F821
    """Create a thread-safe callback for FileWatcher to trigger reindex.

    The FileWatcher's flush-loop thread calls this callback with a list
    of changed file paths.  The callback dispatches an async coroutine
    to the server's event loop via :func:`asyncio.run_coroutine_threadsafe`
    to perform incremental reindexing without blocking the watcher thread.

    Args:
        project_root: Project root directory.
        config:       Current RAG configuration.
        collection:   ChromaDB collection.
        bm25_index:   In-memory BM25 index.
        embed_fn:     Embedding function.
        indexing_state: Mutable indexing state dict.
        loop:         The asyncio event loop to dispatch to.

    Returns:
        A callback ``(paths: list[str]) -> None`` suitable for FileWatcher.
    """

    def _on_files_changed(paths: list[str]) -> None:
        if indexing_state.get("status") == "indexing":
            logger.debug("Skipping watcher reindex: already indexing")
            return

        target_files = [Path(p) for p in paths if Path(p).is_file()]
        if not target_files:
            return

        async def _do_reindex() -> None:
            indexing_state["status"] = "indexing"
            indexing_state["started_at"] = time.time()
            try:
                from tokenkeeper.indexer import index_documents  # noqa: PLC0415

                result = await asyncio.to_thread(
                    index_documents,
                    project_root, config, collection, bm25_index, embed_fn,
                    target_files=target_files,
                )
                indexing_state["status"] = "ready"
                indexing_state["result"] = result
                indexing_state["finished_at"] = time.time()
                _rebuild_bm25_from_metadata(collection, bm25_index)
                logger.info("Watcher reindex complete: %d files", result.files_indexed)
            except Exception as exc:
                indexing_state["status"] = "error"
                indexing_state["error"] = str(exc)
                indexing_state["finished_at"] = time.time()
                logger.error("Watcher reindex failed: %s", exc)

        asyncio.run_coroutine_threadsafe(_do_reindex(), loop)

    return _on_files_changed


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Server lifespan: health checks, indexing, and context setup.

    Runs in order:
      1. Determine project root from ``TOKENKEEPER_PROJECT`` env var
         or ``Path.cwd()``.
      2. Ensure ``.rag/`` directory exists.
      3. Set up logging to ``.rag/rag.log`` + stderr.
      4. Run startup health checks (SystemExit on failure).
      5. Run embedding smoke test (SystemExit on failure).
      6. Load configuration from ``.rag/.rag-config.json``.
      7. Create ChromaDB client and collection.
      8. Build BM25 index from stored metadata.
      9. Run initial indexing in a thread (non-blocking).
     10. Yield context dict for tool access.

    Yields:
        Dict with config, project_root, rag_dir, collection, bm25_index,
        embed_fn, and indexing_result keys.
    """
    # 1. Project root
    project_root_env = os.environ.get("TOKENKEEPER_PROJECT")
    if project_root_env:
        project_root = Path(project_root_env)
    else:
        project_root = Path.cwd()

    # 2. Ensure .rag/ directory
    from tokenkeeper.config import ensure_rag_directory  # noqa: PLC0415

    rag_dir = ensure_rag_directory(project_root)

    # 3. Logging
    setup_logging(rag_dir)
    logger.info("Starting TokenKeeper server (root=%s)", project_root)

    # 4. Health checks (lazy import)
    from tokenkeeper.health import run_startup_checks  # noqa: PLC0415

    run_startup_checks()

    # 5. Smoke test (lazy import)
    from tokenkeeper.embeddings import run_smoke_test  # noqa: PLC0415

    run_smoke_test()

    # 6. Config
    from tokenkeeper.config import load_config  # noqa: PLC0415

    config = load_config(project_root)
    logger.info(
        "Config loaded: chunk_size=%d, overlap=%d, mode=%s, alpha=%.2f",
        config.chunk_size,
        config.overlap,
        config.mode,
        config.alpha,
    )

    # 7. ChromaDB client and collection
    from tokenkeeper.storage import (  # noqa: PLC0415
        create_chroma_client,
        get_or_create_collection,
    )

    chroma_path = rag_dir / config.chroma_path
    chroma_client = create_chroma_client(chroma_path)
    collection = get_or_create_collection(chroma_client)
    logger.info("ChromaDB collection ready (%d documents)", collection.count())

    # 8. Build BM25 index from stored metadata
    from tokenkeeper.bm25_index import BM25Index  # noqa: PLC0415

    bm25_index = BM25Index()
    _rebuild_bm25_from_metadata(collection, bm25_index)

    # 9. Embed function (lazy import)
    from tokenkeeper.embeddings import embed_texts  # noqa: PLC0415

    # 10. Run initial indexing as a background task (non-blocking)
    # The lifespan must yield quickly so the MCP server can respond to
    # the initialize handshake before Claude Code's startup timeout.
    from tokenkeeper.indexer import IndexingResult, index_documents  # noqa: PLC0415

    # Mutable state for indexing status
    indexing_state: dict[str, Any] = {
        "status": "indexing",
        "started_at": time.time(),
        "result": None,
        "error": None,
    }

    def _run_indexing() -> IndexingResult:
        return index_documents(
            project_root, config, collection, bm25_index, embed_texts,
        )

    async def _background_index() -> None:
        try:
            result = await asyncio.to_thread(_run_indexing)
            indexing_state["status"] = "ready"
            indexing_state["result"] = result
            indexing_state["finished_at"] = time.time()
            _rebuild_bm25_from_metadata(collection, bm25_index)
            logger.info(
                "Initial indexing complete: %d files (%d chunks), %d skipped",
                result.files_indexed,
                result.chunks_indexed,
                result.files_skipped,
            )
        except Exception as exc:
            indexing_state["status"] = "error"
            indexing_state["error"] = str(exc)
            indexing_state["finished_at"] = time.time()
            logger.error("Initial indexing failed: %s", exc)

    indexing_task = asyncio.create_task(_background_index())

    # 11. Start file watcher (if enabled)
    watcher = None
    if getattr(config, "watch_enabled", False):
        from tokenkeeper.discovery import EXCLUDED_DIRS  # noqa: PLC0415
        from tokenkeeper.watcher import FileWatcher  # noqa: PLC0415

        # Determine watched extensions based on content_mode
        content_mode = getattr(config, "content_mode", "docs")
        watch_extensions: set[str] = set()
        if content_mode in ("docs", "both"):
            watch_extensions.add(".md")
        if content_mode in ("code", "both"):
            from tokenkeeper.discovery import CODE_EXTENSIONS  # noqa: PLC0415
            watch_extensions.update(CODE_EXTENSIONS)

        loop = asyncio.get_event_loop()

        on_changed = _make_reindex_callback(
            project_root, config, collection, bm25_index,
            embed_texts, indexing_state, loop,
        )

        watcher = FileWatcher(
            watch_root=project_root,
            extensions=frozenset(watch_extensions),
            excluded_dirs=EXCLUDED_DIRS,
            debounce_seconds=getattr(config, "debounce_seconds", 3.0),
            burst_threshold=getattr(config, "burst_threshold", 20),
            burst_window_seconds=getattr(config, "burst_window_seconds", 5.0),
            on_files_changed=on_changed,
        )
        watcher.start()
        logger.info(
            "File watcher started (debounce=%.1fs, content_mode=%s)",
            getattr(config, "debounce_seconds", 3.0),
            content_mode,
        )

    # 12. Yield context for tool access
    yield {
        "config": config,
        "project_root": str(project_root),
        "rag_dir": str(rag_dir),
        "collection": collection,
        "bm25_index": bm25_index,
        "embed_fn": embed_texts,
        "indexing_state": indexing_state,
    }

    # Shutdown
    if not indexing_task.done():
        indexing_task.cancel()
        logger.info("Cancelled in-progress indexing task")
    if watcher:
        watcher.stop()
        logger.info("File watcher stopped")
    logger.info("TokenKeeper server shutting down")


def _rebuild_bm25_from_metadata(
    collection: "chromadb.Collection",  # noqa: F821
    bm25_index: "BM25Index",  # noqa: F821
) -> None:
    """Rebuild BM25 index from stored bm25_tokens metadata in ChromaDB.

    This avoids re-embedding on startup by reading pre-computed token
    strings from the collection metadata.
    """
    stored = collection.get(include=["metadatas"])
    if not stored["ids"]:
        logger.info("BM25 rebuild: collection empty, nothing to rebuild")
        return

    doc_ids: list[str] = []
    tokenized: list[list[str]] = []

    for i, doc_id in enumerate(stored["ids"]):
        meta = stored["metadatas"][i] if stored["metadatas"] else {}
        bm25_tokens_str = meta.get("bm25_tokens", "")
        if bm25_tokens_str:
            tokens = bm25_tokens_str.split()
        else:
            tokens = ["_empty_"]
        doc_ids.append(doc_id)
        tokenized.append(tokens)

    if doc_ids:
        bm25_index.rebuild(doc_ids, tokenized)
        logger.info("BM25 index rebuilt with %d documents", len(doc_ids))


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("TokenKeeper", lifespan=app_lifespan)


# ---------------------------------------------------------------------------
# Helper to access lifespan state
# ---------------------------------------------------------------------------


def _get_lifespan_data(ctx: Context) -> dict[str, Any]:
    """Extract lifespan data dict from Context.

    Returns:
        The lifespan context dict, or empty dict if unavailable.
    """
    return ctx.fastmcp._lifespan_result or {}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
async def search_knowledge(
    query: str,
    ctx: Context,
    top_k: int = 10,
    alpha: float | None = None,
    mode: str | None = None,
) -> str:
    """Search indexed project documents using hybrid semantic + keyword search.

    Returns the most relevant document chunks for the given query. Results
    include the chunk text, source file path, relevance score, and metadata.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return (1-50, default 10).
        alpha: Override the project's hybrid search weight.
               0.0 = pure keyword, 1.0 = pure semantic.
               If not set, uses the project's configured alpha.
        mode:  Override search mode: "hybrid", "semantic", or "keyword".
               If not set, uses the project's configured mode.

    Returns:
        Formatted search results with source, score, and content.
    """
    data = _get_lifespan_data(ctx)
    if not data:
        return "Error: Server not initialized. Check logs for startup errors."

    collection = data.get("collection")
    bm25_index = data.get("bm25_index")
    embed_fn = data.get("embed_fn")
    config = data.get("config")

    if not all([collection, bm25_index, embed_fn, config]):
        return "Error: Server components not ready. Check startup logs."

    # Check indexing state
    indexing_state = data.get("indexing_state", {})
    if indexing_state.get("status") == "indexing":
        return "Indexing in progress. Please wait and try again."

    if indexing_state.get("status") == "error":
        return f"Indexing failed: {indexing_state.get('error', 'unknown error')}"

    # Validate inputs
    top_k = max(1, min(top_k, 50))
    search_alpha = alpha if alpha is not None else config.alpha
    search_mode = mode if mode is not None else config.mode

    # Run search in thread to avoid blocking event loop
    from tokenkeeper.search import search  # noqa: PLC0415

    def _do_search():
        return search(
            query,
            collection,
            bm25_index,
            embed_fn,
            alpha=search_alpha,
            top_k=top_k,
            mode=search_mode,
        )

    results = await asyncio.to_thread(_do_search)

    if not results:
        return f"No results found for: {query}"

    # Format results for Claude Code
    lines: list[str] = []
    lines.append(f"Found {len(results)} results for: {query}\n")

    for i, r in enumerate(results, start=1):
        lines.append(f"--- Result {i} (score: {r.score:.3f}) ---")
        lines.append(f"Source: {r.source_file} (chunk {r.chunk_index})")
        if r.title:
            lines.append(f"Title: {r.title}")
        if r.tags:
            lines.append(f"Tags: {r.tags}")
        if r.heading_hierarchy:
            lines.append(f"Section: {r.heading_hierarchy}")
        if r.language:
            lines.append(f"Language: {r.language}")
        if r.symbol_name:
            lines.append(f"Symbol: {r.symbol_name} ({r.symbol_type})")
        if r.line_start and r.line_end:
            lines.append(f"Lines: {r.line_start}-{r.line_end}")
        lines.append("")
        lines.append(r.content)
        lines.append("")

    return "\n".join(lines)


@mcp.tool
async def indexing_status(ctx: Context) -> str:
    """Check the current state of document indexing.

    Returns information about:
    - Whether indexing is complete, in progress, or failed
    - Number of files indexed, skipped, and failed
    - Number of chunks in the index
    - Total documents in ChromaDB

    Returns:
        Formatted status report.
    """
    data = _get_lifespan_data(ctx)
    if not data:
        return "Error: Server not initialized."

    indexing_state = data.get("indexing_state", {})
    collection = data.get("collection")
    config = data.get("config")

    lines: list[str] = []
    lines.append("=== TokenKeeper Indexing Status ===\n")

    # Server info
    lines.append(f"Project root: {data.get('project_root', 'unknown')}")
    if config:
        lines.append(f"Search mode: {config.mode} (alpha={config.alpha:.2f})")
        lines.append(f"Chunk size: {config.chunk_size}, overlap: {config.overlap}")
    lines.append("")

    # Indexing state
    status = indexing_state.get("status", "unknown")
    lines.append(f"Indexing status: {status}")

    if status == "indexing":
        started = indexing_state.get("started_at", 0)
        elapsed = time.time() - started
        lines.append(f"Elapsed time: {elapsed:.1f}s")

    elif status == "ready":
        result = indexing_state.get("result")
        if result:
            lines.append(f"Files indexed: {result.files_indexed}")
            lines.append(f"Chunks indexed: {result.chunks_indexed}")
            lines.append(f"Files skipped (unchanged): {result.files_skipped}")
            lines.append(f"Files failed: {result.files_failed}")
        started = indexing_state.get("started_at", 0)
        finished = indexing_state.get("finished_at", 0)
        if started and finished:
            lines.append(f"Indexing duration: {finished - started:.1f}s")

    elif status == "error":
        lines.append(f"Error: {indexing_state.get('error', 'unknown')}")

    # Collection stats
    if collection:
        count = collection.count()
        lines.append(f"\nChromaDB documents: {count}")

    return "\n".join(lines)


@mcp.tool
async def reindex_documents(
    ctx: Context,
    paths: list[str] | None = None,
    force: bool = False,
) -> str:
    """Trigger reindexing of project documents.

    Runs document indexing to update the search index. Can reindex all
    documents or specific files. Use force=True to reindex even if
    files haven't changed.

    Args:
        paths: Optional list of file paths (relative to project root) to
               reindex. If omitted, reindexes all discoverable documents.
        force: If True, reindex files even if their content hash hasn't
               changed. Default False.

    Returns:
        Status message with indexing results.
    """
    data = _get_lifespan_data(ctx)
    if not data:
        return "Error: Server not initialized."

    indexing_state = data.get("indexing_state", {})

    # Reject concurrent reindex
    if indexing_state.get("status") == "indexing":
        return "Reindex already in progress. Use indexing_status to check progress."

    collection = data.get("collection")
    bm25_index = data.get("bm25_index")
    embed_fn = data.get("embed_fn")
    config = data.get("config")
    project_root = Path(data.get("project_root", "."))

    if not all([collection, bm25_index, embed_fn, config]):
        return "Error: Server components not ready."

    # Resolve target files if provided
    target_files = None
    if paths:
        target_files = []
        for p in paths:
            resolved = project_root / p
            if resolved.is_file():
                target_files.append(resolved)
            else:
                return f"Error: File not found: {p}"

    # Update state to indexing
    indexing_state["status"] = "indexing"
    indexing_state["started_at"] = time.time()
    indexing_state["result"] = None
    indexing_state["error"] = None

    from tokenkeeper.indexer import index_documents as _index_documents  # noqa: PLC0415

    def _run_reindex():
        return _index_documents(
            project_root, config, collection, bm25_index, embed_fn,
            target_files=target_files,
            force=force,
        )

    try:
        result = await asyncio.to_thread(_run_reindex)
        indexing_state["status"] = "ready"
        indexing_state["result"] = result
        indexing_state["finished_at"] = time.time()

        # Rebuild BM25 from metadata for consistency
        _rebuild_bm25_from_metadata(collection, bm25_index)

        duration = indexing_state["finished_at"] - indexing_state["started_at"]
        mode_str = "partial" if paths else "full"
        force_str = " (forced)" if force else ""

        lines: list[str] = [
            f"Reindex complete ({mode_str}{force_str}) in {duration:.1f}s",
            f"Files indexed: {result.files_indexed}",
            f"Chunks indexed: {result.chunks_indexed}",
            f"Files skipped: {result.files_skipped}",
            f"Files failed: {result.files_failed}",
            f"Total chunks in index: {collection.count()}",
        ]
        return "\n".join(lines)

    except Exception as exc:
        indexing_state["status"] = "error"
        indexing_state["error"] = str(exc)
        indexing_state["finished_at"] = time.time()
        logger.error("Reindex failed: %s", exc)
        return f"Reindex failed: {exc}"


@mcp.tool
async def get_index_stats(ctx: Context) -> str:
    """Get statistics about the current document index.

    Returns counts of indexed files and chunks, BM25 index size,
    last-indexed timestamp, and storage location.

    Returns:
        Formatted statistics report.
    """
    data = _get_lifespan_data(ctx)
    if not data:
        return "Error: Server not initialized."

    collection = data.get("collection")
    bm25_index = data.get("bm25_index")
    config = data.get("config")

    lines: list[str] = ["=== TokenKeeper Index Statistics ===\n"]

    # Project info
    lines.append(f"Project root: {data.get('project_root', 'unknown')}")
    lines.append(f"RAG directory: {data.get('rag_dir', 'unknown')}")
    lines.append("")

    if collection:
        total_chunks = collection.count()
        lines.append(f"Total chunks in ChromaDB: {total_chunks}")

        # Query metadata for unique files and timestamps
        if total_chunks > 0:
            stored = collection.get(include=["metadatas"])
            source_files: set[str] = set()
            max_indexed_at = 0.0

            for meta in (stored["metadatas"] or []):
                source_files.add(meta.get("source_file", "<unknown>"))
                indexed_at = meta.get("indexed_at", 0)
                if isinstance(indexed_at, (int, float)) and indexed_at > max_indexed_at:
                    max_indexed_at = indexed_at

            lines.append(f"Unique source files: {len(source_files)}")

            if max_indexed_at > 0:
                from datetime import datetime, timezone  # noqa: PLC0415

                ts = datetime.fromtimestamp(max_indexed_at, tz=timezone.utc)
                lines.append(f"Last indexed: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            else:
                lines.append("Last indexed: unknown (no timestamp in metadata)")

            # List source files
            lines.append("\nIndexed files:")
            for f in sorted(source_files):
                lines.append(f"  - {f}")
        else:
            lines.append("Unique source files: 0")
            lines.append("Last indexed: never")
    else:
        lines.append("ChromaDB collection: not available")

    # BM25 stats
    lines.append("")
    if bm25_index:
        lines.append(f"BM25 index documents: {len(bm25_index)}")
    else:
        lines.append("BM25 index: not available")

    # Config info
    if config:
        lines.append(f"\nSearch config: mode={config.mode}, alpha={config.alpha:.2f}")
        lines.append(f"Chunk config: size={config.chunk_size}, overlap={config.overlap}")

    return "\n".join(lines)
