"""Entry point for ``python -m knowledge_rag`` and ``knowledge-rag`` CLI."""

from knowledge_rag.server import mcp


def main() -> None:
    """Run the Knowledge RAG MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
