"""TokenKeeper - Local RAG system as MCP server for Claude Code.

Thin wrapper that delegates to the FastMCP server entry point.
Prefer ``python -m tokenkeeper`` for production use.
"""

from tokenkeeper.server import mcp


def main() -> None:
    """Run the TokenKeeper MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
