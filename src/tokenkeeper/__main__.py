"""Entry point for ``python -m tokenkeeper`` and ``tokenkeeper`` CLI."""

from tokenkeeper.server import mcp


def main() -> None:
    """Run the TokenKeeper MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
