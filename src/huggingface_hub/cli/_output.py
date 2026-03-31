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
import re
import sys
from collections.abc import Sequence
from typing import Any

import typer

from huggingface_hub.errors import CLIError
from huggingface_hub.utils import ANSI, is_agent, tabulate

from ._cli_utils import AutoOutputFormat


class Output:
    """Output sink for the `hf` CLI.

    Mode is resolved once at init time based on `is_agent()` auto-detection
    and can be overridden per-command via `set_mode()`.
    """

    mode: AutoOutputFormat

    def __init__(self) -> None:
        self.set_mode()

    def set_mode(self, mode: AutoOutputFormat = AutoOutputFormat.auto) -> None:
        """Override the output mode (called by commands that receive ``--format``)."""
        if mode == AutoOutputFormat.auto:
            mode = AutoOutputFormat.agent if is_agent() else AutoOutputFormat.human
        self.mode = mode

    def text(self, human: str, agent: str | None = None) -> None:
        """Print a free-form text message to stdout."""
        match self.mode:
            case AutoOutputFormat.agent:
                print(agent if agent is not None else _strip_ansi(human))
            case AutoOutputFormat.human:
                print(human)

    def table(
        self,
        headers: list[str],
        rows: Sequence[list[Any]],
        alignments: dict[str, str] | None = None,
    ) -> None:
        """Print tabular data to stdout.

        Args:
            headers: Column names.
            rows: List of rows, each a list of raw values.
            alignments: Optional mapping of header name to "left" or "right". Defaults to "left".
        """
        if not rows:
            match self.mode:
                case AutoOutputFormat.agent | AutoOutputFormat.human:
                    print("No results found.")
                case AutoOutputFormat.json:
                    print("[]")
            return

        match self.mode:
            case AutoOutputFormat.human:
                formatted_rows: list[list[str | int]] = [[_format_table_cell_human(v) for v in row] for row in rows]
                screaming_headers = [_to_header(h) for h in headers]
                screaming_alignments = {_to_header(k): v for k, v in (alignments or {}).items()}
                print(tabulate(formatted_rows, headers=screaming_headers, alignments=screaming_alignments))
            case AutoOutputFormat.agent:
                print("\t".join(headers))
                for row in rows:
                    print("\t".join(_format_table_cell_agent(v) for v in row))
            case AutoOutputFormat.json:
                items = [dict(zip(headers, row)) for row in rows]
                print(json.dumps(items, default=str))

    def dict(self, data: dict[str, Any]) -> None:
        """Print structured data as JSON in all modes"""
        indent = 2 if self.mode == AutoOutputFormat.human else None
        print(json.dumps(data, indent=indent, default=str))

    def result(self, message: str, **data: Any) -> None:
        """Print a success summary to stdout."""
        match self.mode:
            case AutoOutputFormat.human:
                parts = [ANSI.green(f"✓ {message}")]
                for k, v in data.items():
                    if v is not None:
                        parts.append(f"  {k}: {v}")
                print("\n".join(parts))
            case AutoOutputFormat.agent:
                parts = [f"{k}={v}" for k, v in data.items() if v is not None]
                print(" ".join(parts) if parts else message)
            case AutoOutputFormat.json:
                print(json.dumps(data, default=str) if data else "")

    def warning(self, message: str) -> None:
        """Print a non-fatal warning to stderr."""
        if self.mode == AutoOutputFormat.human:
            print(ANSI.yellow(f"  Warning: {message}"), file=sys.stderr)
        else:
            print(f"Warning: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Print an error to stderr."""
        if self.mode == AutoOutputFormat.human:
            print(ANSI.red(f"  Error: {message}"), file=sys.stderr)
        else:
            print(f"Error: {message}", file=sys.stderr)

    def confirm(self, message: str, *, yes: bool = False) -> None:
        """Interactive confirmation. Raises if in agent mode without ``--yes``."""
        if yes:
            return
        if self.mode == AutoOutputFormat.human and sys.stdin.isatty():
            choice = input(f"{message} [Y/n] ")
            if choice.lower() not in ("", "y", "yes"):
                raise typer.Abort()
        else:
            raise CLIError(f"{message} Use --yes to skip confirmation.")

    def hint(self, message: str) -> None:
        """Print a helpful hint to stderr."""
        if self.mode == AutoOutputFormat.human:
            print(ANSI.gray(f"  {message}"), file=sys.stderr)
        else:
            print(f"Hint: {message}", file=sys.stderr)


# HELPERS

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_MAX_CELL_LENGTH = 35


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _to_header(name: str) -> str:
    """Convert a camelCase or PascalCase string to SCREAMING_SNAKE_CASE."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return s.upper()


def _format_table_value_human(value: Any) -> str:
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
        return ", ".join(_format_table_value_human(v) for v in value)
    elif isinstance(value, dict):
        if "name" in value:  # Likely to be a user or org => print name
            return str(value["name"])
        return json.dumps(value)
    return str(value)


def _format_table_cell_human(value: Any, max_len: int = _MAX_CELL_LENGTH) -> str:
    """Format a value + truncate it for table display."""
    cell = _format_table_value_human(value)
    if len(cell) > max_len:
        cell = cell[: max_len - 3] + "..."
    return cell


def _format_table_cell_agent(value: Any) -> str:
    """Format a cell value for agent TSV output (ISO timestamps, tabs escaped)."""
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return str(value).replace("\t", " ")


out = Output()
