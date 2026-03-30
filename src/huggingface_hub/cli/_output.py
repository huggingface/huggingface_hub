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
"""Output framework for the `hf` CLI."""

import datetime
import json
import os
import re
import sys
from enum import Enum
from typing import Any, Optional, Sequence, Union

import typer

from huggingface_hub.utils import ANSI, is_agent, tabulate


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

_MAX_CELL_LENGTH = 35


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _to_header(name: str) -> str:
    """Convert a camelCase or PascalCase string to SCREAMING_SNAKE_CASE."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return s.upper()


def _format_value(value: Any) -> str:
    """Convert a value to string for terminal display."""
    if not value:
        return ""
    if isinstance(value, bool):
        return "✔" if value else ""
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", value):
        return value[:10]
    if isinstance(value, list):
        return ", ".join(_format_value(v) for v in value)
    elif isinstance(value, dict):
        if "name" in value:
            return str(value["name"])
        return json.dumps(value)
    return str(value)


def _format_cell(value: Any, max_len: int = _MAX_CELL_LENGTH) -> str:
    """Format a value + truncate it for table display."""
    cell = _format_value(value)
    if len(cell) > max_len:
        cell = cell[: max_len - 3] + "..."
    return cell


class OutputMode(str, Enum):
    """Output mode for the `hf` CLI."""

    human = "human"
    agent = "agent"
    json = "json"


class Output:
    """Output sink for the `hf` CLI.

    Mode resolution (first match wins):
    1. Explicit `set_mode()` call (from a `--output` flag).
    2. `HF_OUTPUT` environment variable.
    3. `is_agent()` auto-detection → `agent` if detected, else `human`.

    The resolved mode is cached after first access.
    """

    def __init__(self) -> None:
        self._mode: Optional[OutputMode] = None

    @property
    def mode(self) -> OutputMode:
        if self._mode is None:
            self._mode = self._resolve_mode()
        return self._mode

    def set_mode(self, mode: Optional[OutputMode] = None) -> None:
        """Set the output mode.  Pass `None` to reset to auto-detection."""
        self._mode = mode

    @staticmethod
    def _resolve_mode() -> OutputMode:
        env = os.environ.get("HF_OUTPUT", "").strip().lower()
        if env in OutputMode.__members__:
            return OutputMode(env)
        return OutputMode.agent if is_agent() else OutputMode.human

    def text(self, human: str, agent: Optional[str] = None) -> None:
        """Print a free-form text message to stdout."""
        if self.mode == OutputMode.json:
            return
        if self.mode == OutputMode.agent:
            msg = agent if agent is not None else _strip_ansi(human)
        else:
            msg = human
        print(msg)

    def table(
        self,
        headers: list[str],
        rows: Sequence[list[Any]],
        alignments: Optional[dict[str, str]] = None,
    ) -> None:
        """Print tabular data to stdout.

        Args:
            headers: Column names.
            rows: List of rows, each a list of raw values.
            alignments: Optional mapping of header name to "left" or "right". Defaults to "left".
        """
        if not rows:
            if self.mode == OutputMode.human:
                print("No results found.")
            elif self.mode == OutputMode.json:
                print("[]")
            return

        if self.mode == OutputMode.human:
            formatted_rows: list[list[Union[str, int]]] = [[_format_cell(v) for v in row] for row in rows]
            screaming_headers = [_to_header(h) for h in headers]
            screaming_alignments = {_to_header(k): v for k, v in (alignments or {}).items()}
            print(tabulate(formatted_rows, headers=screaming_headers, alignments=screaming_alignments))
        elif self.mode == OutputMode.agent:
            print("\t".join(headers))
            for row in rows:
                print("\t".join(self._format_agent_cell(v) for v in row))
        elif self.mode == OutputMode.json:
            items = [dict(zip(headers, row)) for row in rows]
            print(json.dumps(items, default=str))

    def dict(self, data: dict[str, Any]) -> None:
        """Print structured data as JSON to stdout."""
        indent = 2 if self.mode == OutputMode.human else None
        print(json.dumps(data, indent=indent, default=str))

    def result(self, message: str, **data: Any) -> None:
        """Print a success summary to stdout."""
        if self.mode == OutputMode.human:
            parts = [ANSI.green(f"✓ {message}")]
            for k, v in data.items():
                if v is not None:
                    parts.append(f"  {k}: {v}")
            print("\n".join(parts))
        elif self.mode == OutputMode.agent:
            parts = [f"{k}={v}" for k, v in data.items() if v is not None]
            print(" ".join(parts) if parts else message)
        elif self.mode == OutputMode.json:
            print(json.dumps(data, default=str) if data else "")

    def warning(self, message: str) -> None:
        """Print a non-fatal warning to stderr."""
        if self.mode == OutputMode.human:
            print(ANSI.yellow(f"  Warning: {message}"), file=sys.stderr)
        else:
            print(f"Warning: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Print an error to stderr."""
        if self.mode == OutputMode.human:
            print(ANSI.red(f"  Error: {message}"), file=sys.stderr)
        elif self.mode == OutputMode.json:
            print(json.dumps({"error": message}), file=sys.stderr)
        else:
            print(f"Error: {message}", file=sys.stderr)

    def confirm(self, message: str, *, yes: bool = False) -> None:
        """Interactive confirmation. Raises if in agent mode without ``--yes``."""
        if yes:
            return
        if self.mode == OutputMode.human and sys.stdin.isatty():
            choice = input(f"{message} [Y/n] ")
            if choice.lower() not in ("", "y", "yes"):
                raise typer.Abort()
        else:
            from huggingface_hub.errors import CLIError

            raise CLIError(f"{message} Use --yes to skip confirmation.")

    def hint(self, message: str) -> None:
        """Print a helpful hint to stderr."""
        if self.mode == OutputMode.human:
            print(ANSI.gray(f"  {message}"), file=sys.stderr)
        else:
            print(f"Hint: {message}", file=sys.stderr)

    @staticmethod
    def _format_agent_cell(value: Any) -> str:
        """Format a cell value for agent TSV output (lowercase bools, ISO timestamps)."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        return str(value)


out = Output()
