# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-20

### Added

- Initial release of TokenKeeper (tokenkeeper)
- Hybrid search: semantic (vector) + keyword (BM25) with Reciprocal Rank Fusion
- Local-first architecture: Ollama embeddings, ChromaDB persistent storage
- FastMCP server with 4 tools: search_knowledge, indexing_status, reindex_documents, get_index_stats
- File watcher for automatic reindexing on changes
- Per-project isolation via .rag/ directory
- Content modes: docs, code, both
- Heading-aware markdown chunking with structure protection
- Language-aware code chunking (Python AST, TypeScript/JavaScript regex)
- Google Gemini embedding provider (optional cloud alternative)
- 445 tests passing

[Unreleased]: https://github.com/admin-sosys/TokenKeeper/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/admin-sosys/TokenKeeper/releases/tag/v0.1.0
