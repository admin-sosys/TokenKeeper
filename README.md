[![PyPI](https://img.shields.io/pypi/v/tokenkeeper.svg)](https://pypi.org/project/tokenkeeper/)
[![CI](https://github.com/admin-sosys/TokenKeeper/actions/workflows/ci.yml/badge.svg)](https://github.com/admin-sosys/TokenKeeper/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/admin-sosys/TokenKeeper/branch/master/graph/badge.svg)](https://codecov.io/gh/admin-sosys/TokenKeeper)

<!-- MCP Install Badges -->
[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_MCP_Server-0078d4?style=flat-square&logo=visual-studio-code&logoColor=white)](https://insiders.vscode.dev/redirect?url=vscode%3A%2F%2Fsettings%2FTokenKeeper)
[![Install in Cursor](https://img.shields.io/badge/Cursor-Install_MCP_Server-000000?style=flat-square&logo=cursor&logoColor=white)](https://cursor.sh)

# TokenKeeper

**Local RAG memory for Claude Code.** Reduce prompt token consumption by ~80% on knowledge-heavy projects.

TokenKeeper is an MCP server that indexes your project's documents and code, then exposes semantic search tools to Claude Code. Instead of loading entire files into context, your agents query for only the relevant chunks.

## The Problem

On a project with 34 phases of planning docs, a single agent cycle loads **141K tokens (70% of context)** just for background knowledge — before it starts working. Quality degrades as context fills up.

## The Solution

TokenKeeper replaces "load everything" with "query what's relevant":

| | Traditional | With TokenKeeper |
|---|---|---|
| Prompt tokens | 141,345 | 26,959 |
| Context used | 70.7% | 13.5% |
| **Tokens saved** | — | **114,386 (80.9%)** |

Your agents stay in the high-quality zone of their context window.

## How It Works

```
Your project files
    |
    v
[Indexer] --> Chunks with embeddings --> ChromaDB (persistent vectors)
                                              |
                                              v
Claude Code agent --> search_knowledge("topic") --> Top-k relevant chunks
```

- **Hybrid search** — semantic similarity (vector) + keyword matching (BM25), merged via Reciprocal Rank Fusion
- **Local-first** — Ollama for embeddings, ChromaDB for storage. No cloud, no API keys required
- **Auto-indexing** — file watcher detects changes and re-indexes automatically
- **Per-project isolation** — each project gets its own `.rag/` directory

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Install

```bash
pip install tokenkeeper
ollama pull nomic-embed-text
```

### Add to Any Project

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "tokenkeeper": {
      "command": "tokenkeeper",
      "env": {
        "TOKENKEEPER_PROJECT": "${workspaceFolder}"
      }
    }
  }
}
```

Start (or restart) Claude Code in that project. TokenKeeper will:

1. Create a `.rag/` directory for index data
2. Index all markdown, JSON, and code files
3. Expose 4 MCP tools for search and management

Add `.rag/` to your project's `.gitignore`.

### Verify

Ask Claude Code:

```
Check the indexing status
```

Then test a search:

```
Search the knowledge base for "authentication flow and session management"
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Hybrid semantic + keyword search across indexed content |
| `indexing_status` | Check if indexing is complete, in progress, or failed |
| `reindex_documents` | Trigger manual reindexing (all or specific files) |
| `get_index_stats` | Index statistics — file count, chunk count, timestamps |

### search_knowledge Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Natural language search query |
| `top_k` | int | 10 | Results to return (1-50) |
| `alpha` | float | 0.5 | Hybrid weight: 0.0 = keyword only, 1.0 = semantic only |
| `mode` | string | "hybrid" | `"hybrid"`, `"semantic"`, or `"keyword"` |

## Configuration

TokenKeeper auto-creates `.rag/.rag-config.json` on first run:

```json
{
  "content_mode": "docs",
  "chunk_size": 1000,
  "overlap": 200,
  "alpha": 0.5,
  "mode": "hybrid",
  "watch_enabled": true,
  "debounce_seconds": 3.0
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `content_mode` | `"docs"` | `"docs"` (md/json), `"code"` (source files), or `"both"` |
| `chunk_size` | `1000` | Characters per chunk (100-10000) |
| `overlap` | `200` | Character overlap between chunks |
| `alpha` | `0.5` | Hybrid search weight |
| `mode` | `"hybrid"` | Search strategy |
| `watch_enabled` | `true` | Auto-reindex on file changes |

## Architecture

```
TokenKeeper/
  src/tokenkeeper/
    server.py          # FastMCP server + lifespan
    indexer.py         # Discovery -> ingestion -> embedding -> storage
    search.py          # Hybrid search with RRF fusion
    embeddings.py      # Ollama (local) or Google Gemini (cloud)
    storage.py         # ChromaDB persistent client
    bm25_index.py      # BM25 keyword index
    watcher.py         # File system monitoring with debounce
    config.py          # Pydantic configuration
    health.py          # Startup health checks
```

**Stack**: Python 3.10+ | FastMCP | ChromaDB 1.5.0 | Ollama | BM25

## Embedding Providers

### Ollama (Default, Local)

- Model: `nomic-embed-text` (768 dimensions)
- No API key needed
- Runs on CPU (no GPU required)

### Google Gemini (Optional, Cloud)

- Model: `gemini-embedding-001` (3072 dimensions)
- Requires `GOOGLE_API_KEY` environment variable
- Higher quality embeddings, but requires internet

## File Types Indexed

| Mode | Extensions |
|------|-----------|
| `"docs"` | `.md`, `.mdx`, `.json` |
| `"code"` | `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.mjs`, `.go`, `.rs`, `.java`, `.rb`, `.c`, `.cpp`, `.h` |
| `"both"` | All of the above |

**Always excluded**: `node_modules/`, `.git/`, `.next/`, `__pycache__/`, `.rag/`, `dist/`, `build/`

## Performance

| Metric | Value |
|--------|-------|
| First index (500 files) | ~3-5 minutes |
| Subsequent startups | ~5 seconds (cached) |
| Search latency | ~150ms per query |
| Storage | ~100-200 MB per 2000-file project |

## Testing

```bash
# All tests (skip Ollama-dependent if not running)
uv run pytest tests/ -v --tb=short

# Token savings benchmark
uv run pytest tests/test_practical_token_savings.py -v -s

# Agent comparison (RAG vs traditional)
uv run pytest tests/test_agent_comparison.py -v -s
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Ollama connection refused" | Run `ollama serve` to start the server |
| "nomic-embed-text not found" | Run `ollama pull nomic-embed-text` |
| Claude Code doesn't show RAG tools | Ensure `.mcp.json` is in project root, run `tokenkeeper --help` to verify install, restart Claude Code |
| 0 chunks indexed | Check `TOKENKEEPER_PROJECT` env var points to your project root |
| Slow first index | Normal — subsequent starts load cached ChromaDB in ~5 seconds |
| Search returns irrelevant results | Try `mode: "keyword"` or lower `alpha` to 0.3 |

## Docs

- [QUICKSTART.md](QUICKSTART.md) — Setup, toggling, A/B testing, GSD workflow integration
- [IMPLEMENTATION-GUIDE.md](IMPLEMENTATION-GUIDE.md) — Architecture deep dive, cost analysis, integration patterns

## License

MIT
