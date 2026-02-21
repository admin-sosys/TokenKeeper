# TokenKeeper Quickstart

How to add TokenKeeper to any project, toggle it on/off for A/B testing, and verify it's working.

---

## New Project Setup (From Scratch)

### 1. Make sure Ollama is running

```bash
ollama serve
ollama pull nomic-embed-text
```

### 2. Drop `.mcp.json` into your project root

Create this file at the root of whatever project you want to index:

```json
{
  "mcpServers": {
    "tokenkeeper": {
      "command": "/path/to/TokenKeeper/.venv/bin/python",
      "args": ["-m", "tokenkeeper"],
      "env": {
        "TOKENKEEPER_PROJECT": "${workspaceFolder}"
      }
    }
  }
}
```

> **Windows**: Use `.venv\Scripts\python.exe` (e.g., `C:\\path\\to\\TokenKeeper\\.venv\\Scripts\\python.exe`)
> **Linux/Mac**: Use `.venv/bin/python` (e.g., `/home/user/TokenKeeper/.venv/bin/python`)

That's it. Next time you open Claude Code in that directory, it will:

1. Spin up the TokenKeeper MCP server
2. Create a `.rag/` directory in your project
3. Index all markdown, JSON, and code files
4. Expose 4 tools: `search_knowledge`, `indexing_status`, `reindex_documents`, `get_index_stats`

### 3. Verify it's alive

Open Claude Code in your project and say:

```
Check the indexing status
```

You should see something like:

```
Indexing status: ready
Files indexed: 247
Chunks indexed: 1,843
Duration: 45.2s
```

### 4. Test a search

```
Search the knowledge base for "authentication flow and session management"
```

If you get ranked results with scores and source files, it's working.

---

## Existing Project Setup

Identical to above. Just drop the `.mcp.json` file into your existing project root. TokenKeeper doesn't modify any of your files — it only creates a `.rag/` directory for its own index data.

**Add `.rag/` to your `.gitignore`:**

```bash
echo ".rag/" >> .gitignore
```

The index is local and rebuilds automatically. No need to commit it.

---

## Toggling On/Off for A/B Testing

### Method 1: Rename the config file (Recommended)

**Turn OFF:**
```bash
mv .mcp.json .mcp.json.disabled
```

**Turn ON:**
```bash
mv .mcp.json.disabled .mcp.json
```

Restart Claude Code after each toggle. When `.mcp.json` is missing or renamed, Claude Code won't start the MCP server and the 4 tools disappear from the session. Your agents run in traditional mode — loading full files.

When `.mcp.json` is present, the tools are available and agents can query RAG instead of reading full documents.

### Method 2: Comment out the server (JSON doesn't support comments, so use an empty object)

**Turn OFF** — replace `.mcp.json` contents with:
```json
{
  "mcpServers": {}
}
```

**Turn ON** — restore the original contents.

### Method 3: Keep it running but don't use it

The MCP server being active doesn't force anything. It just makes the tools *available*. If your agents/rules don't reference `search_knowledge`, they'll behave traditionally. This lets you A/B test within the same session:

- **Traditional**: "Read .planning/ROADMAP.md and tell me the Phase 14 goals"
- **RAG**: "Search the knowledge base for Phase 14 goals and deliverables"

Compare the token counts and quality side by side.

---

## Configuration

On first run, TokenKeeper creates `.rag/.rag-config.json` with defaults. You can tune it:

```json
{
  "content_mode": "both",
  "chunk_size": 1000,
  "overlap": 200,
  "alpha": 0.5,
  "mode": "hybrid",
  "watch_enabled": true,
  "debounce_seconds": 3.0
}
```

### What to change

| Setting | Default | When to change |
|---------|---------|---------------|
| `content_mode` | `"docs"` | Set to `"both"` if you want code files indexed too |
| `chunk_size` | `1000` | Lower to `500` for dense code, raise to `2000` for long docs |
| `alpha` | `0.5` | Raise to `0.7` for conceptual queries, lower to `0.3` for keyword-heavy |
| `watch_enabled` | `true` | Set `false` if file watcher causes issues |

### Forcing a reindex

If you change config or want a clean rebuild:

```
Reindex all documents with force=true
```

Or delete `.rag/chroma/` and restart Claude Code.

---

## What Gets Indexed

By content mode:

| Mode | File Types |
|------|-----------|
| `"docs"` | `.md`, `.mdx`, `.json` |
| `"code"` | `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.mjs`, `.go`, `.rs`, `.java`, `.rb`, `.c`, `.cpp`, `.h` |
| `"both"` | All of the above |

**Always excluded:** `node_modules/`, `.git/`, `.next/`, `__pycache__/`, `.rag/`, `dist/`, `build/`

---

## How Search Works

TokenKeeper runs **hybrid search** — combining semantic similarity (vector embeddings) with keyword matching (BM25). Results are merged using Reciprocal Rank Fusion (RRF).

```
Your query
    |
    v
 [Embed query] ──> Vector search (ChromaDB cosine similarity)
    |                                    |
    v                                    v
 [Tokenize query] ──> BM25 keyword search
                                         |
                                         v
                          RRF fusion (weighted by alpha)
                                         |
                                         v
                              Top-k ranked results
```

Each result includes:
- **Score** (0-1, normalized)
- **Source file** (relative path)
- **Chunk content** (the actual text)
- **Metadata** (title, tags, heading hierarchy, language, symbol info)

---

## Using TokenKeeper in GSD Workflows

### The pattern

Instead of agents reading entire planning docs:

```
# Before (141K tokens)
Read .planning/ROADMAP.md
Read .planning/STATE.md
Read .planning/PROJECT.md
Read all 128 SUMMARY.md files
Read all .claude/rules/*.md
```

Query for what's relevant:

```
# After (27K tokens)
Read .planning/STATE.md                          ← keep (small, always needed)
Read .planning/phases/23-*/23-RESEARCH.md        ← keep (phase-specific)
Search: "Phase 23 form wizard goals and scope"   ← replaces ROADMAP.md
Search: "prior work on tax form generation"      ← replaces 128 SUMMARYs
Search: "form generation patterns and templates"  ← replaces supplementary docs
```

### What to always read directly

| File | Why |
|------|-----|
| `STATE.md` | Current position tracking. Small (~4K tokens). Always relevant. |
| Phase `RESEARCH.md` | Phase-specific research. Always fully needed for planning. |
| Phase `CONTEXT.md` | User decisions from `/gsd:discuss-phase`. Locked constraints. |
| Agent definition | The instruction set (`gsd-planner.md`, etc.). Not knowledge. |

### What to query via RAG instead

| File | RAG Query Pattern |
|------|------------------|
| `ROADMAP.md` (30K tokens) | `"Phase {N} {name}: goals, requirements, scope"` |
| Prior SUMMARYs (54K tokens) | `"Prior work on {topic}: patterns, decisions, key files"` |
| `REQUIREMENTS.md` (4K tokens) | `"Requirements for {phase topic}"` |
| Rules (5K tokens) | `"Rules and constraints for {task type}"` |
| Compliance masters (25K+ each) | `"Legal/security requirements for {specific area}"` |

### Practical example: Planning Phase 23

**Without TokenKeeper** — agent loads 141,345 tokens (70.7% of context window)

**With TokenKeeper** — agent loads 26,959 tokens (13.5% of context window)

The agent frees up **114,386 tokens (57.2% of the context window)** for actual work. It stays in the 0-30% "PEAK quality" zone instead of starting at 70% where quality degrades.

---

## Verifying Quality

### Quick check: search relevance

```
Search the knowledge base for "security headers rate limiting CSP middleware" with top_k 10
```

Look at the results:
- Are the source files relevant to the query?
- Are scores in the 0.5-1.0 range for top results?
- Do the chunks contain the information you'd expect?

### Index stats

```
Show me the index statistics
```

Check:
- Total chunks > 0
- Unique source files matches your expectation
- Last indexed timestamp is recent

### Full benchmark

Run the test suite from the TokenKeeper directory:

```bash
cd /path/to/TokenKeeper
export GOOGLE_API_KEY=your-key  # only if using Gemini embeddings

# Practical A/B token savings test
uv run pytest tests/test_practical_token_savings.py -v -s

# Full agent comparison (RAG vs raw grep)
uv run pytest tests/test_agent_comparison.py -v -s

# Full regression (527 tests)
uv run pytest tests/ -v --tb=short
```

---

## Troubleshooting

### Claude Code doesn't show the RAG tools

- Check `.mcp.json` is in the project root (not a subdirectory)
- Restart Claude Code (`/exit` then reopen)
- Check the python path in `.mcp.json` points to TokenKeeper's venv

### "Ollama connection refused"

```bash
ollama serve   # start the server
ollama list    # verify nomic-embed-text is pulled
```

### Indexing takes forever

First index embeds every file. Typical times:
- Small project (50 files): ~30 seconds
- Medium project (500 files): ~3-5 minutes
- Large project (2000+ files): ~10-15 minutes

Subsequent restarts load the cached ChromaDB index in ~5 seconds.

### Search returns irrelevant results

- Try `mode: "keyword"` for exact term matching
- Lower `alpha` to `0.3` for more keyword weight
- Increase `top_k` to `20` for broader recall
- Reindex with `force: true` if files changed while server was off

### `.rag/` directory is huge

ChromaDB stores embeddings on disk. For a 2000-file project, expect ~100-200 MB. Add `.rag/` to `.gitignore`.

---

## File Layout

After setup, your project looks like:

```
your-project/
  .mcp.json              ← MCP server config (tells Claude Code about TokenKeeper)
  .rag/                  ← Auto-created by TokenKeeper (add to .gitignore)
    .rag-config.json     ← Indexing/search configuration
    chroma/              ← ChromaDB persistent storage
    rag.log              ← Server logs (rotating, 5MB max)
  .planning/             ← Your GSD planning docs (indexed)
  .claude/               ← Your Claude rules (indexed)
  src/                   ← Source code (indexed if content_mode includes "code")
```
