"""Entry point for ``python -m knowledge_rag``.

This is the SOLE entry point for the MCP server. It imports the
FastMCP server instance and calls ``mcp.run()``.
"""

from knowledge_rag.server import mcp

mcp.run()
