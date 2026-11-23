import asyncio
import os
import sys
from typing import IO

import pytest

from huggingface_hub.inference._mcp._cli_hacks import _async_prompt


pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(
        sys.platform == "win32", reason="stdin is read with `loop.add_reader`, which Windows doesn't support"
    ),
]


def _pipe_stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> IO[str]:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, data)
    os.close(write_fd)
    stdin = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", stdin)
    return stdin


async def _prompt_until_eof() -> list[str]:
    lines = []
    for _ in range(5):
        try:
            lines.append(await _async_prompt(exit_event=asyncio.Event()))
        except EOFError:
            return lines
    raise AssertionError(f"No EOFError after reading {lines!r}")


def test_async_prompt_raises_eof_error_at_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # e.g. `echo "hi" | tiny-agents run ...`: once stdin is exhausted, there is nothing more to read
    with _pipe_stdin(monkeypatch, b"hi\n"):
        assert asyncio.run(_prompt_until_eof()) == ["hi"]


def test_async_prompt_empty_line_is_not_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    with _pipe_stdin(monkeypatch, b"\nhi"):
        assert asyncio.run(_prompt_until_eof()) == ["", "hi"]
