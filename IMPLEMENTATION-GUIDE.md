# TokenKeeper Implementation Guide

Integrate TokenKeeper into your Claude Code workflows to reduce prompt token consumption by ~80% on knowledge-heavy projects.

## What TokenKeeper Does

When a GSD agent (planner, executor, verifier) runs, it loads planning documents, roadmaps, prior summaries, rules, and research files into its context window. On a project like HomePayroll with 34 phases, this consumes **141K tokens (70% of the 200K context window)** before the agent even starts working.

TokenKeeper replaces "load everything" with "query for what's relevant." Instead of sending the full ROADMAP.md (30K tokens), the agent queries RAG and gets back only the chunks that match its current task — typically **4-5K tokens**.

**Measured results (Phase 23 planning benchmark):**

| | Traditional | With TokenKeeper |
|---|---|---|
| Prompt tokens | 141,345 | 26,959 |
| Context window used | 70.7% | 13.5% |
| Tokens saved | — | 114,386 (80.9%) |

---

## Prerequisites

1. **Python 3.12+** installed
2. **Ollama** running locally with `nomic-embed-text` model pulled

   ```bash
   ollama pull nomic-embed-text
   ```

   **OR** a Google Generative AI API key for Gemini embeddings (higher quality, no local GPU needed).

3. **TokenKeeper** cloned and dependencies installed:

   ```bash
   git clone <your-tokenkeeper-repo>
   cd TokenKeeper
   uv sync
   ```

---

## Step 1: Add TokenKeeper as an MCP Server

### Option A: Per-Project (Recommended)

Create `.mcp.json` in your project root:

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

> **Linux/Mac**: Replace the python path with your TokenKeeper venv's python binary, e.g., `~/.local/share/TokenKeeper/.venv/bin/python`.

### Option B: Global (All Projects)

Add to `~/.claude/settings.json` under an `mcpServers` key. Same format as above. The `TOKENKEEPER_PROJECT` env var tells TokenKeeper which directory to index.

### Google Gemini Embeddings (Optional)

If you prefer Google Gemini over local Ollama, add the API key to the env block:

```json
"env": {
  "TOKENKEEPER_PROJECT": "${workspaceFolder}",
  "GOOGLE_API_KEY": "your-api-key-here"
}
```

Then modify the server's embed function import in `server.py` to use `embed_texts_google` instead of `embed_texts`. (This will be configurable via config in a future release.)

---

## Step 2: Configure Indexing

TokenKeeper auto-creates `.rag/.rag-config.json` on first run. Defaults work for most projects. Key options:

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

| Setting | Default | Description |
|---------|---------|-------------|
| `content_mode` | `"docs"` | What to index: `"docs"` (markdown/JSON only), `"code"` (source only), `"both"` |
| `chunk_size` | `1000` | Characters per chunk (100-10000) |
| `overlap` | `200` | Overlap between chunks in characters |
| `alpha` | `0.5` | Hybrid weight: 0.0 = pure keyword, 1.0 = pure semantic |
| `mode` | `"hybrid"` | Search mode: `"hybrid"`, `"semantic"`, or `"keyword"` |
| `watch_enabled` | `true` | Auto-reindex when files change |

**Recommended for GSD workflows**: Set `content_mode` to `"both"` to index planning docs, blog content, JSON data, AND source code.

---

## Step 3: First Index

When you start Claude Code in your project, TokenKeeper indexes automatically on startup. You can check status:

```
> Use the indexing_status tool to check if indexing is complete
```

Or trigger a manual reindex:

```
> Use the reindex_documents tool to rebuild the index
```

First indexing takes 3-10 minutes depending on project size. Subsequent starts load the existing ChromaDB index in ~5 seconds.

---

## Step 4: Using TokenKeeper in Your Workflows

### Direct Queries

Ask Claude to search your project knowledge:

```
Search the knowledge base for "tax form filing deadlines and Schedule H requirements"
```

Claude will use the `search_knowledge` tool and return the most relevant chunks from your indexed documents.

### Integrating with GSD Workflows

The key integration point is changing how GSD agents load context. Instead of reading entire files, they query TokenKeeper.

#### Before (Traditional Context Loading)

The GSD planner agent's execution steps load full files:

```
Step 1: Read .planning/STATE.md                  (4,316 tokens)
Step 2: Read .planning/ROADMAP.md                (29,963 tokens)
Step 3: Read .planning/PROJECT.md                (1,781 tokens)
Step 4: Scan all 128 prior SUMMARY.md files      (53,849 tokens)
Step 5: Read .planning/phases/23-*/RESEARCH.md   (7,440 tokens)
Step 6: Load all .claude/rules/*.md              (4,824 tokens)
Step 7: Load supplementary docs                  (25,040 tokens)
                                          TOTAL: 141,345 tokens
```

#### After (RAG-Augmented Context Loading)

Replace bulk file reads with targeted RAG queries:

```
Step 1: Read .planning/STATE.md                  (4,316 tokens)  ← keep (position tracking)
Step 2: Query RAG: "phase 23 goals, scope"       (~1,900 tokens) ← replaces ROADMAP.md
Step 3: Query RAG: "prior work on tax forms"     (~2,200 tokens) ← replaces 128 SUMMARYs
Step 4: Query RAG: "form generation patterns"    (~660 tokens)   ← replaces supplementary
Step 5: Read .planning/phases/23-*/RESEARCH.md   (7,440 tokens)  ← keep (phase-specific)
                                          TOTAL:  26,959 tokens
```

### What to Keep vs What to RAG

| File | Keep or RAG? | Rationale |
|------|-------------|-----------|
| Agent definition (`.md`) | **Keep** | Instruction set, not knowledge |
| `STATE.md` | **Keep** | Current position tracking, small |
| Phase `RESEARCH.md` | **Keep** | Phase-specific, always fully needed |
| Phase `CONTEXT.md` | **Keep** | User decisions, always fully needed |
| `ROADMAP.md` | **RAG** | 30K tokens but only need the current phase section |
| `PROJECT.md` | **RAG** | Can be queried for relevant architecture info |
| `REQUIREMENTS.md` | **RAG** | Query for phase-specific requirements only |
| Prior `SUMMARY.md` files | **RAG** | 128 files at 54K tokens, but only 2-4 are relevant |
| `.claude/rules/*.md` | **RAG** | Query for task-relevant rules only |
| Supplementary masters | **RAG** | Large compliance docs, query for relevant sections |

### Modifying GSD Agent Context Assembly

To integrate with existing GSD workflows, modify the context loading steps in your agent definitions. Here's the pattern:

**In the planner agent's `load_project_state` step**, replace:

```markdown
Read `.planning/ROADMAP.md` to identify phase goals
```

With:

```markdown
Query TokenKeeper: search_knowledge("Phase {N} {phase-name}: goals, requirements, scope, dependencies")
Use the returned chunks instead of reading the full ROADMAP.md
```

**In the `read_project_history` step**, replace:

```markdown
Scan all SUMMARY.md frontmatter, then read full summaries for relevant phases
```

With:

```markdown
Query TokenKeeper: search_knowledge("Prior work related to {phase-topic}: patterns established, decisions made, key files created")
```

**In the `gather_phase_context` step**, keep direct reads for:
- `RESEARCH.md` (always fully needed)
- `CONTEXT.md` (always fully needed)
- `STATE.md` (always fully needed)

---

## Step 5: Verify It's Working

### Check Index Stats

```
> Use the get_index_stats tool
```

Should show: total chunks, unique files, search config.

### Test a Query

```
> Use search_knowledge with query "security headers rate limiting CSP" and top_k 10
```

Should return relevant chunks from security docs, middleware files, and compliance plans.

### Run the Benchmark Suite

```bash
export GOOGLE_API_KEY=your-key  # if using Gemini embeddings
cd TokenKeeper
uv run pytest tests/test_practical_token_savings.py -v -s
```

This runs the A/B comparison for Phase 23 planning and shows exact token savings.

---

## Available MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|---------------|
| `search_knowledge` | Query the RAG index | `query` (str), `top_k` (1-50, default 10), `mode` (hybrid/semantic/keyword) |
| `indexing_status` | Check indexing progress | None |
| `reindex_documents` | Trigger reindexing | `paths` (optional file list), `force` (bool) |
| `get_index_stats` | Index statistics | None |

### search_knowledge Parameters

- **`query`**: Natural language question. Be specific: "Phase 23 form wizard tax requirements" beats "forms".
- **`top_k`**: Results to return. Use 10 for focused queries, 20 for broad exploration, 5 for tight precision.
- **`alpha`**: Hybrid weight override. 0.5 is balanced. Use 0.7+ for conceptual queries, 0.3 for keyword-heavy queries.
- **`mode`**: `"hybrid"` (default, best for most queries), `"semantic"` (conceptual similarity), `"keyword"` (exact term matching).

---

## Tuning for Your Project

### Search Quality

| top_k | Recall | Precision | Best For |
|-------|--------|-----------|----------|
| 5 | 49% | 81% | Tight, focused answers |
| 10 | 75% | 76% | General purpose (best F1) |
| 15 | 86% | 62% | Broad context gathering |
| 20 | 100% | 62% | Maximum recall, planning tasks |

**Recommendation**: Use `top_k=15` for GSD planning queries (need broad context), `top_k=10` for execution queries (need focused answers).

### Chunk Size

- **1000 chars** (default): Good balance for markdown docs and planning files
- **500 chars**: Better for dense code files, more granular retrieval
- **2000 chars**: Better for long-form narrative docs (blog posts, legal text)

### Content Mode

- **`"docs"`**: Index only `.md`, `.mdx`, `.json` — fastest, smallest index
- **`"code"`**: Index only source code files — for code search use cases
- **`"both"`**: Index everything — recommended for full-project RAG

---

## Architecture

```
Your Project/
  .rag/
    .rag-config.json     # Configuration
    chroma/              # ChromaDB persistent storage (vectors + metadata)
    rag.log              # Rotating log file
  .mcp.json              # MCP server config (tells Claude Code about TokenKeeper)
  .planning/             # GSD planning docs (indexed by TokenKeeper)
  src/                   # Source code (indexed if content_mode includes "code")
```

**How search works:**

1. Query → embedded into vector via Ollama/Gemini
2. Vector search → ChromaDB cosine similarity (semantic matches)
3. Keyword search → BM25 index (exact term matches)
4. Hybrid fusion → Reciprocal Rank Fusion combines both, weighted by alpha
5. Results → top_k chunks with scores, source files, metadata

**Data flow:**

```
File changes → File watcher → Ingest pipeline → Chunks
                                                    ↓
                                          ChromaDB (vectors + metadata)
                                          BM25 Index (keyword tokens)
                                                    ↓
                              search_knowledge query → Hybrid search → Ranked results
```

---

## Troubleshooting

### "Ollama connection refused"

Ollama isn't running. Start it:

```bash
ollama serve
```

### "GOOGLE_API_KEY not set"

If using Gemini embeddings, set the env var in `.mcp.json` or export it:

```bash
export GOOGLE_API_KEY=your-key
```

### "Index is empty" / "0 chunks"

The indexer couldn't find files. Check:
- `TOKENKEEPER_PROJECT` env var points to your project root
- `content_mode` in `.rag/.rag-config.json` matches your file types
- Files aren't in `.gitignore`-excluded directories

### Slow first indexing

First index embeds all files (3-10 min). Subsequent runs only re-embed changed files. The ChromaDB index persists to disk at `.rag/chroma/`.

### Low recall on JSON files

JSON files are now chunked structure-aware (arrays by item, objects by key). If you have JSON data that isn't being found, try adding descriptive keys or restructuring nested data.

---

## Cost Impact

Based on HomePayroll (34 phases, 127 prior summaries, 11K+ indexed chunks):

| Scenario | Tokens/Call | Calls/Phase | Total |
|----------|------------|-------------|-------|
| Traditional planner | 141,345 | 1 | 141,345 |
| Traditional executor (per plan) | ~31,250 | 4-6 plans | ~156,250 |
| Traditional verifier | ~36,250 | 1 | 36,250 |
| **Traditional total per phase** | | | **~333,845** |
| | | | |
| RAG planner | 26,959 | 1 | 26,959 |
| RAG executor (per plan) | ~12,000 | 4-6 plans | ~60,000 |
| RAG verifier | ~15,000 | 1 | 15,000 |
| **RAG total per phase** | | | **~101,959** |
| | | | |
| **Savings per phase** | | | **~231,886 (69%)** |

Over 34 phases, that's roughly **7.8M tokens saved** — a significant reduction in API costs.

At Anthropic's Opus pricing (~$15/M input tokens), that's approximately **$117 saved per full project build**.

At Sonnet pricing (~$3/M input tokens), that's approximately **$23 saved per full project build**.

These are conservative estimates — actual savings compound because smaller context windows also improve output quality (the agent stays in the "PEAK" 0-30% quality zone longer).
