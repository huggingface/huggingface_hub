import contextlib
from typing import Optional
from unittest.mock import patch

import pytest

from huggingface_hub import MCPClient


mcp = pytest.importorskip("mcp")
mcp_types = pytest.importorskip("mcp.types")


def _tool(name: str) -> "mcp_types.Tool":
    return mcp_types.Tool(name=name, description=name, inputSchema={"type": "object", "properties": {}})


class _PaginatedSession:
    """A server whose tools/list answers one tool per page."""

    def __init__(self, pages: list[list[str]]) -> None:
        self.pages = pages
        self.cursors_seen: list[Optional[str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def initialize(self):
        return None

    async def list_tools(self, cursor: Optional[str] = None):
        self.cursors_seen.append(cursor)
        index = 0 if cursor is None else int(cursor)
        next_index = index + 1
        return mcp_types.ListToolsResult(
            tools=[_tool(name) for name in self.pages[index]],
            nextCursor=str(next_index) if next_index < len(self.pages) else None,
        )


@contextlib.asynccontextmanager
async def _fake_stdio_client(server_params):
    yield None, None


@pytest.mark.asyncio
async def test_add_mcp_server_follows_next_cursor():
    session = _PaginatedSession([["alpha"], ["beta"], ["gamma"]])
    with (
        patch("mcp.client.stdio.stdio_client", _fake_stdio_client),
        patch("mcp.ClientSession", lambda **kwargs: session),
    ):
        async with MCPClient(model="test-model") as client:
            await client.add_mcp_server(type="stdio", command="true")

    assert sorted(client.sessions) == ["alpha", "beta", "gamma"]
    assert session.cursors_seen == [None, "1", "2"]


@pytest.mark.asyncio
async def test_add_mcp_server_single_page_asks_once():
    session = _PaginatedSession([["alpha", "beta"]])
    with (
        patch("mcp.client.stdio.stdio_client", _fake_stdio_client),
        patch("mcp.ClientSession", lambda **kwargs: session),
    ):
        async with MCPClient(model="test-model") as client:
            await client.add_mcp_server(type="stdio", command="true")

    assert session.cursors_seen == [None]
    assert sorted(client.sessions) == ["alpha", "beta"]
