"""Knowledge RAG - Local RAG system as MCP server for Claude Code.

Thin wrapper that delegates to the FastMCP server entry point.
Prefer ``python -m knowledge_rag`` for production use.
"""

from knowledge_rag.server import mcp


def main() -> None:
    """Run the Knowledge RAG MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
