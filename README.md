<div align="center">

# TokenKeeper

**Local RAG memory for AI coding agents**

Give your AI coding agents long-term project memory without burning through context windows.

[![PyPI](https://img.shields.io/pypi/v/tokenkeeper.svg)](https://pypi.org/project/tokenkeeper/)
[![CI](https://github.com/admin-sosys/TokenKeeper/actions/workflows/ci.yml/badge.svg)](https://github.com/admin-sosys/TokenKeeper/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/admin-sosys/TokenKeeper/branch/master/graph/badge.svg)](https://codecov.io/gh/admin-sosys/TokenKeeper)

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Server-0078d4?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://insiders.vscode.dev/redirect?url=vscode%3A%2F%2Fsettings%2FTokenKeeper)
[![Install in Cursor](https://img.shields.io/badge/Cursor-Install_Server-000000?style=for-the-badge&logo=cursor&logoColor=white)](https://cursor.sh)

</div>

---

TokenKeeper is an [MCP server](https://modelcontextprotocol.io/) that indexes your project's documents and code into a local vector database, then exposes semantic search tools to your AI agents. Instead of loading entire files into context, agents query for only the relevant chunks — **reducing prompt tokens by up to 80%**.

## Table of Contents

- [Why TokenKeeper](#why-tokenkeeper)
- [Quick Start](#quick-start)
- [Features](#features)
- [How It Works](#how-it-works)
- [MCP Tools](#mcp-tools)
- [Configuration](#configuration)
- [Embedding Providers](#embedding-providers)
- [Architecture](#architecture)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Why TokenKeeper

Large projects accumulate documentation — planning docs, architecture decisions, API specs, changelogs, code comments. When AI agents load all of that into context every cycle, two things happen:

1. **Token costs explode** — you burn through context loading background knowledge before the agent starts working
2. **Quality degrades** — as context fills up, the agent loses focus on the actual task

TokenKeeper replaces "load everything" with "query what's relevant":

| | Without TokenKeeper | With TokenKeeper |
|---|---|---|
| Prompt tokens per cycle | ~141K | ~27K |
| Context window used | ~70% | ~13% |
| **Tokens saved** | — | **~80%** |

> Your agents stay in the high-quality zone of their context window, with more room for reasoning and code generation.

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Install

```bash
pip install tokenkeeper
ollama pull nomic-embed-text
```

### Add to Your Project

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

Start (or restart) your AI coding agent in that project. TokenKeeper will:

1. Create a `.rag/` directory for index data
2. Index your markdown, JSON, and code files
3. Expose search and management tools via MCP

> Add `.rag/` to your `.gitignore`.

### Verify

Ask your agent:

```
Check the indexing status
```

Then test a search:

```
Search the knowledge base for "authentication flow"
```

## Features

### Hybrid Search

Combines **semantic similarity** (vector embeddings) with **keyword matching** (BM25), merged via Reciprocal Rank Fusion. Get the best of both worlds — conceptual understanding plus exact-term precision.

### Local-First

All processing happens on your machine. [Ollama](https://ollama.com/) handles embeddings, [ChromaDB](https://www.trychroma.com/) stores vectors. No cloud services, no API keys, no data leaving your system.

### Heading-Aware Chunking

Markdown documents are chunked at section boundaries, preserving heading hierarchy. Code blocks and tables are never split. Each chunk carries its full heading path as metadata for context-rich search results.

### Code-Aware Indexing

Source code files are chunked at function and class boundaries using AST parsing (Python) and pattern detection (TypeScript/JavaScript). Supports `docs`, `code`, or `both` modes.

### Auto-Indexing

A file watcher monitors your project for changes and re-indexes automatically with debouncing. Git operation bursts (like `checkout`) are detected and batched to avoid CPU spikes.

### Per-Project Isolation

Each project gets its own `.rag/` directory with independent vector storage and configuration. No cross-contamination between projects.

## How It Works

```
Your project files (.md, .py, .ts, .json, ...)
    │
    ▼
┌─────────┐     ┌──────────────────┐     ┌───────────────────────┐
│ Indexer  │────▶│ Heading-aware    │────▶│ ChromaDB              │
│          │     │ chunking +       │     │ (persistent vectors)  │
│          │     │ embeddings       │     │ + BM25 keyword index  │
└─────────┘     └──────────────────┘     └───────────┬───────────┘
                                                      │
                                                      ▼
                                         ┌───────────────────────┐
AI Agent ──── search_knowledge("topic") ─▶│ Hybrid search (RRF)  │──▶ Top-k chunks
                                         └───────────────────────┘
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Hybrid semantic + keyword search across all indexed content |
| `indexing_status` | Check if indexing is complete, in progress, or failed |
| `reindex_documents` | Trigger manual reindexing (all files or specific paths) |
| `get_index_stats` | Index statistics — file count, chunk count, timestamps |

### search_knowledge

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language search query |
| `top_k` | int | 10 | Number of results (1-50) |
| `alpha` | float | 0.5 | Hybrid weight: `0.0` = keyword only, `1.0` = semantic only |
| `mode` | string | `"hybrid"` | `"hybrid"`, `"semantic"`, or `"keyword"` |

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
| `chunk_size` | `1000` | Characters per chunk (100–10,000) |
| `overlap` | `200` | Character overlap between chunks |
| `alpha` | `0.5` | Hybrid search weight |
| `mode` | `"hybrid"` | Search strategy |
| `watch_enabled` | `true` | Auto-reindex on file changes |
| `debounce_seconds` | `3.0` | Delay before reindexing after changes |

### File Types

| Mode | Extensions |
|------|-----------|
| `"docs"` | `.md`, `.mdx`, `.json` |
| `"code"` | `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.go`, `.rs`, `.java`, `.rb`, `.c`, `.cpp`, `.h` |
| `"both"` | All of the above |

**Always excluded:** `node_modules/`, `.git/`, `.next/`, `__pycache__/`, `.rag/`, `dist/`, `build/`

## Embedding Providers

| Provider | Model | Dimensions | API Key | Notes |
|----------|-------|-----------|---------|-------|
| **Ollama** (default) | `nomic-embed-text` | 768 | None | Local, runs on CPU, no GPU required |
| Google Gemini | `gemini-embedding-001` | 3,072 | `GOOGLE_API_KEY` | Cloud, higher quality, requires internet |

## Architecture

```
src/tokenkeeper/
  server.py          # FastMCP server + lifespan management
  indexer.py         # File discovery → ingestion → embedding → storage
  search.py          # Hybrid search with Reciprocal Rank Fusion
  embeddings.py      # Ollama (local) or Google Gemini (cloud)
  storage.py         # ChromaDB persistent vector store
  bm25_index.py      # BM25 keyword index
  watcher.py         # File system monitoring with debounce + burst detection
  config.py          # Pydantic configuration with validation
  health.py          # Startup health checks (Ollama, ChromaDB, model)
```

**Stack:** Python 3.10+ · [FastMCP](https://github.com/jlowin/fastmcp) · [ChromaDB](https://www.trychroma.com/) · [Ollama](https://ollama.com/) · BM25

## Performance

| Metric | Value |
|--------|-------|
| First index (500 files) | ~3–5 minutes |
| Subsequent startups | ~5 seconds (cached) |
| Search latency | ~150ms per query |
| Storage overhead | ~100–200 MB per 2,000-file project |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Ollama connection refused" | Run `ollama serve` to start the Ollama server |
| "nomic-embed-text not found" | Run `ollama pull nomic-embed-text` |
| Agent doesn't show search tools | Check `.mcp.json` is in project root, run `tokenkeeper --help`, restart agent |
| 0 chunks indexed | Verify `TOKENKEEPER_PROJECT` env var points to your project root |
| Slow first index | Normal — subsequent starts use cached vectors (~5 seconds) |
| Irrelevant search results | Try `mode: "keyword"` or lower `alpha` to `0.3` |

## Documentation

| | | |
|---|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Setup, configuration, and usage patterns | |
| [IMPLEMENTATION-GUIDE.md](IMPLEMENTATION-GUIDE.md) | Architecture deep dive, cost analysis, integration patterns | |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, testing, and PR process | |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing commands, and PR guidelines.

```bash
# Development setup
git clone https://github.com/admin-sosys/TokenKeeper.git
cd TokenKeeper
uv sync
ollama pull nomic-embed-text

# Run tests
uv run pytest tests/ -v --tb=short
```

## License

[MIT](LICENSE)
