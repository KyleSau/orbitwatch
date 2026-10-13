"""MCP server exposing the orbitwatch tools to Claude Code, Claude Desktop or any MCP client.

Built on the official MCP Python SDK. In SDK v2 the high-level ``FastMCP`` class was
renamed ``MCPServer``; it derives each tool's JSON Schema from the function signature
and uses the docstring as the tool description, so we register the ``Toolbox`` bound
methods as-is.

Run it over stdio::

    orbitwatch mcp                  # live data
    orbitwatch mcp --data fixtures  # offline, recorded fixtures

Logs go to stderr; stdout is reserved for the protocol.
"""

from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer

from orbitwatch.context import DataContext
from orbitwatch.tools import TOOL_NAMES, Toolbox

INSTRUCTIONS = (
    "Tools for orbital and Earth-observation analysis around Lompoc / Vandenberg SFB, "
    "California. Use predict_passes for satellite pass times, search_sentinel2_scenes then "
    "compare_index_between_scenes for vegetation (NDVI) or burn (NBR) change. Quote numbers, "
    "scene IDs, dates and TLE epochs exactly as the tools return them."
)


def build_server(ctx: DataContext | None = None) -> MCPServer:
    toolbox = Toolbox(ctx or DataContext.live())
    server = MCPServer(name="orbitwatch", instructions=INSTRUCTIONS, version="0.1.0")
    for name in TOOL_NAMES:
        server.add_tool(getattr(toolbox, name), name=name)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the orbitwatch MCP server over stdio.")
    parser.add_argument("--data", choices=["live", "fixtures"], default="live")
    args = parser.parse_args(argv)
    ctx = DataContext.fixtures() if args.data == "fixtures" else DataContext.live()
    build_server(ctx).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
