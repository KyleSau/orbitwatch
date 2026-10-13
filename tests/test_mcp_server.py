import asyncio
import json
import sys

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from orbitwatch.mcp_server import build_server
from orbitwatch.tools import TOOL_NAMES, anthropic_tool_definitions


def test_lists_all_tools_with_matching_schemas(fixture_ctx):
    async def main():
        async with Client(build_server(fixture_ctx)) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(main())}
    assert set(tools) == set(TOOL_NAMES)
    for definition in anthropic_tool_definitions():
        mcp_schema = tools[definition["name"]].input_schema
        ours = definition["input_schema"]
        # Same parameters and same required set: MCP and the Claude agent cannot drift.
        assert set(mcp_schema.get("properties", {})) == set(ours["properties"])
        assert set(mcp_schema.get("required", [])) == set(ours.get("required", []))
        assert (
            tools[definition["name"]].description.split()[:5]
            == definition["description"].split()[:5]
        )


def test_call_tool_over_mcp(fixture_ctx):
    async def main():
        async with Client(build_server(fixture_ctx)) as client:
            return await client.call_tool("list_places", {})

    result = asyncio.run(main())
    assert not result.is_error
    text = "".join(c.text for c in result.content if c.type == "text")
    assert "lompoc" in json.loads(text)["ground_sites"][0]["name"]


def test_stdio_server_starts_and_lists_tools(tmp_path):
    """Launch `python -m orbitwatch.mcp_server --data fixtures` as a real subprocess."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "orbitwatch.mcp_server", "--data", "fixtures"],
        env={"ORBITWATCH_CACHE_DIR": str(tmp_path)},  # merged over the SDK's safe defaults
    )

    async def main():
        async with Client(params) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert set(asyncio.run(main())) == set(TOOL_NAMES)
