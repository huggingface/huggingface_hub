import asyncio

import pytest


class DummyAgent:
    """Minimal async-context stub that satisfies the CLI contract."""

    def __init__(self, *_, **__):
        self.available_tools = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def load_tools(self):
        pass

    async def run(self, *_args, **_kwargs):
        if False:
            yield


@pytest.mark.asyncio
async def test_run_agent_handles_missing_signal_handler(monkeypatch):
    import huggingface_hub.inference._mcp.cli as cli

    monkeypatch.setattr(
        cli, "_load_agent_config", lambda _path: ({"provider": "dummy", "model": "dummy", "servers": []}, None)
    )

    async def fake_input(prompt: str = "» "):  # noqa: D401 – simple name
        raise EOFError

    monkeypatch.setattr(cli, "_ainput", fake_input)
    monkeypatch.setattr(cli, "Agent", DummyAgent)

    def raise_not_implemented(self, *_a, **_kw):
        raise NotImplementedError

    monkeypatch.setattr(asyncio.AbstractEventLoop, "add_signal_handler", raise_not_implemented, raising=True)

    def dummy_remove(self, *_a, **_kw):
        return False

    monkeypatch.setattr(asyncio.AbstractEventLoop, "remove_signal_handler", dummy_remove, raising=False)

    await cli.run_agent(None)
