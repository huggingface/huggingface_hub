# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dual-mode CLI output framework: human-friendly vs agent-friendly.

When ``HF_CLI_AGENT_OUTPUT=1`` (or detected automatically), all output helpers
produce compact, structured text with no ANSI escapes, no decorative formatting,
and no interactive elements. The goal is to save tokens and be unambiguous for
LLM-based agents while preserving a rich experience for human users.

Developer guide
---------------
1. Use :func:`cli_print` instead of bare ``print()`` for any line that should
   differ between human and agent mode.
2. Use :class:`CLIOutput` to build structured output blocks (key-value, tables,
   success/error messages) that render appropriately in each mode.
3. For complex commands, use the ``@with_cli_output`` decorator to inject a
   ``CLIOutput`` instance into the command function.

Example::

    from huggingface_hub.cli._output import CLIOutput, cli_print

    # Simple one-liner that adapts automatically:
    cli_print(
        human=f"Successfully created {ANSI.bold(repo_id)} on the Hub.",
        agent=f"created repo_id={repo_id}",
    )

    # Structured block:
    out = CLIOutput()
    out.success("Repo created", repo_id=repo_id, url=str(repo_url))
    out.flush()
"""

import json
import os
import sys
from typing import Any, Optional, Sequence, Union, cast

from huggingface_hub import constants
from huggingface_hub.utils import ANSI, tabulate


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_AGENT_ENV_HINTS = (
    "CURSOR_AGENT",
    "CLAUDE_CODE",
    "CODEX_CLI",
    "OPENCODE",
    "AIDER",
    "CLINE",
    "WINDSURF",
)


def is_agent_output() -> bool:
    """Return ``True`` when CLI output should target LLM agents.

    Checks (in order):

    1. **Explicit opt-in** — ``HF_CLI_AGENT_OUTPUT=1`` forces agent mode.
    2. **Explicit opt-out** — ``HF_CLI_AGENT_OUTPUT=0`` forces human mode.
    3. **Auto-detect** — ``HF_CLI_AGENT_OUTPUT=auto`` (or unset) inspects
       well-known agent environment variables (``CURSOR_AGENT``,
       ``CLAUDE_CODE``, ``CODEX_CLI``, …). If any are set, agent mode is
       enabled.

    By default (unset), auto-detection is **off** to avoid surprises. Set
    ``HF_CLI_AGENT_OUTPUT=auto`` to enable it, or ``HF_CLI_AGENT_OUTPUT=1``
    to force it unconditionally.
    """
    if constants.HF_CLI_AGENT_OUTPUT:
        return True
    raw = os.environ.get("HF_CLI_AGENT_OUTPUT", "").upper()
    if raw in ("0", "FALSE", "NO", "OFF", ""):
        return False
    if raw == "AUTO":
        return any(os.environ.get(v) for v in _AGENT_ENV_HINTS)
    return False


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def cli_print(
    human: str,
    agent: Optional[str] = None,
    *,
    file: Any = None,
) -> None:
    """Print *human* or *agent* text depending on the current mode.

    If *agent* is ``None`` the *human* text is used in both modes after stripping
    ANSI escape sequences.
    """
    if file is None:
        file = sys.stdout
    if is_agent_output():
        print(_strip_ansi(agent if agent is not None else human), file=file)
    else:
        print(human, file=file)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# CLIOutput — structured output builder
# ---------------------------------------------------------------------------


class CLIOutput:
    """Accumulate structured output and flush it at the end.

    Human mode renders nicely formatted tables, coloured status lines, and
    progress hints.  Agent mode renders a compact, unambiguous text block that
    an LLM can parse reliably.

    Typical workflow inside a command function::

        out = CLIOutput()
        out.success("Repo created", repo_id="org/model", url="https://…")
        out.flush()
    """

    def __init__(self, agent: Optional[bool] = None):
        self._agent = agent if agent is not None else is_agent_output()
        self._blocks: list[str] = []

    @property
    def agent_mode(self) -> bool:
        return self._agent

    # -- High-level helpers ------------------------------------------------

    def success(self, message: str, **fields: Any) -> "CLIOutput":
        """Record a success message with optional structured fields."""
        if self._agent:
            parts = [f"OK: {message}"]
            for k, v in fields.items():
                parts.append(f"  {k}={v}")
            self._blocks.append("\n".join(parts))
        else:
            line = ANSI.green(f"✓ {message}")
            for k, v in fields.items():
                line += f"\n  {ANSI.bold(k)}: {v}"
            self._blocks.append(line)
        return self

    def error(self, message: str, **fields: Any) -> "CLIOutput":
        """Record an error message with optional structured fields."""
        if self._agent:
            parts = [f"ERROR: {message}"]
            for k, v in fields.items():
                parts.append(f"  {k}={v}")
            self._blocks.append("\n".join(parts))
        else:
            line = ANSI.red(f"✗ {message}")
            for k, v in fields.items():
                line += f"\n  {ANSI.bold(k)}: {v}"
            self._blocks.append(line)
        return self

    def info(self, message: str, **fields: Any) -> "CLIOutput":
        """Record an informational message with optional structured fields."""
        if self._agent:
            parts = [message]
            for k, v in fields.items():
                parts.append(f"  {k}={v}")
            self._blocks.append("\n".join(parts))
        else:
            line = message
            for k, v in fields.items():
                line += f"\n  {ANSI.bold(k)}: {v}"
            self._blocks.append(line)
        return self

    def warning(self, message: str) -> "CLIOutput":
        """Record a warning."""
        if self._agent:
            self._blocks.append(f"WARN: {message}")
        else:
            self._blocks.append(ANSI.yellow(f"⚠ {message}"))
        return self

    def key_value(self, data: dict[str, Any], *, title: Optional[str] = None) -> "CLIOutput":
        """Render a dict as key-value pairs.

        Agent mode uses ``key=value`` lines.  Human mode uses aligned, styled
        output.
        """
        if self._agent:
            lines: list[str] = []
            if title:
                lines.append(title)
            for k, v in data.items():
                lines.append(f"  {k}={_serialize_value(v)}")
            self._blocks.append("\n".join(lines))
        else:
            lines = []
            if title:
                lines.append(ANSI.bold(title))
            max_key_len = max(len(str(k)) for k in data) if data else 0
            for k, v in data.items():
                lines.append(f"  {str(k).ljust(max_key_len)}  {v}")
            self._blocks.append("\n".join(lines))
        return self

    def table(
        self,
        items: Sequence[dict[str, Any]],
        columns: list[str],
        *,
        title: Optional[str] = None,
        row_fn: Optional[Any] = None,
        alignments: Optional[dict[str, str]] = None,
    ) -> "CLIOutput":
        """Render a list of dicts as a table.

        In agent mode, rows are rendered as compact TSV (tab-separated).
        In human mode, the existing ``tabulate`` helper is used.
        """
        if not items:
            self._blocks.append("No results found." if not self._agent else "EMPTY")
            return self

        def _default_row(item: dict[str, Any]) -> list[str]:
            return [_format_cell(item.get(col, "")) for col in columns]

        extract = row_fn or _default_row
        rows = [extract(item) for item in items]

        if self._agent:
            lines: list[str] = []
            if title:
                lines.append(title)
            lines.append("\t".join(columns))
            for row in rows:
                lines.append("\t".join(str(c) for c in row))
            self._blocks.append("\n".join(lines))
        else:
            from huggingface_hub.cli._cli_utils import _to_header

            screaming = [_to_header(h) for h in columns]
            tbl = tabulate(
                cast(list[list[Union[str, int]]], rows),
                headers=screaming,
                alignments={_to_header(k): v for k, v in (alignments or {}).items()},
            )
            block = f"{ANSI.bold(title)}\n{tbl}" if title else tbl
            self._blocks.append(block)
        return self

    def json(self, data: Any, *, compact: bool = False) -> "CLIOutput":
        """Render arbitrary data as JSON.

        Agent mode always uses compact (no indent) JSON. Human mode uses
        indented JSON unless *compact* is ``True``.
        """
        if self._agent:
            self._blocks.append(json.dumps(data, default=str, separators=(",", ":")))
        else:
            indent = None if compact else 2
            self._blocks.append(json.dumps(data, indent=indent, default=str))
        return self

    def raw(self, text: str) -> "CLIOutput":
        """Add a pre-formatted text block (used as-is in both modes)."""
        self._blocks.append(text)
        return self

    # -- Rendering ---------------------------------------------------------

    def render(self) -> str:
        """Return the accumulated output as a single string."""
        return "\n".join(self._blocks)

    def flush(self, *, file: Any = None) -> None:
        """Print accumulated output and reset the buffer."""
        if file is None:
            file = sys.stdout
        text = self.render()
        if text:
            print(text, file=file)
        self._blocks.clear()


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------


def _format_cell(value: Any, max_len: int = 35) -> str:
    """Format and truncate a value for table cells."""
    if value is None or value == "":
        return ""
    s = str(value)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _serialize_value(v: Any) -> str:
    """Serialize a value for agent key=value output."""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=str, separators=(",", ":"))
    return str(v)


# ---------------------------------------------------------------------------
# Decorator for commands
# ---------------------------------------------------------------------------


def with_cli_output(fn: Any) -> Any:
    """Decorator that injects a ``CLIOutput`` as the first positional arg.

    Usage::

        @app.command()
        @with_cli_output
        def my_command(out: CLIOutput, repo_id: str, ...):
            out.success("Done", repo_id=repo_id)
            out.flush()

    The decorator creates a ``CLIOutput()`` and passes it as the first
    argument. The original Typer signature (minus *out*) is preserved so that
    Typer still sees the correct parameters.
    """
    import functools
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if params and params[0].name == "out":
        params = params[1:]
    new_sig = sig.replace(parameters=params)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = CLIOutput()
        return fn(out, *args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    return wrapper
