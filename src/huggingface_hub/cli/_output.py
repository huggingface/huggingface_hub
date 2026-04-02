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

from huggingface_hub.utils import ANSI, is_agent, tabulate

from ._cli_utils import OutputFormatWithAuto


class Output:
    """Output sink for the `hf` CLI.

    Mode is resolved once at init time based on `is_agent()` auto-detection
    and can be overridden per-command via `set_mode()`.
    """

    mode: OutputFormatWithAuto

    def __init__(self) -> None:
        self.set_mode()

    def set_mode(self, mode: OutputFormatWithAuto = OutputFormatWithAuto.auto) -> None:
        """Override the output mode (called by commands that receive ``--format``)."""
        if mode == OutputFormatWithAuto.auto:
            mode = OutputFormatWithAuto.agent if is_agent() else OutputFormatWithAuto.human
        self.mode = mode

    def text(self, human: str, agent: str | None = None) -> None:
        """Print a free-form text message to stdout."""
        match self.mode:
            case OutputFormatWithAuto.agent:  # agent alt or ANSI-stripped human
                print(agent if agent is not None else _strip_ansi(human))
            case OutputFormatWithAuto.human:  # as-is, may contain ANSI
                print(human)
            # json/quiet: no-op

    def table(
        self,
        items: Sequence[dict[str, Any]],
        *,
        headers: list[str] | None = None,
        alignments: dict[str, str] | None = None,
    ) -> None:
        """Print tabular data to stdout.

        Args:
            items: List of dicts. Headers are auto-detected from keys if not provided.
            headers: Explicit column names. If None, derived from dict keys (empty columns filtered).
            alignments: Optional mapping of header name to "left" or "right". Defaults to "left".
        """
        if not items:
            match self.mode:
                case OutputFormatWithAuto.agent | OutputFormatWithAuto.human:
                    print("No results found.")
                case OutputFormatWithAuto.json:
                    print("[]")
            return

        if headers is None:
            all_columns = list(items[0].keys())
            headers = [col for col in all_columns if any(item.get(col) for item in items)]
        rows = [[item.get(h) for h in headers] for item in items]

        match self.mode:
            case OutputFormatWithAuto.human:  # padded table, truncated cells, SCREAMING_SNAKE headers
                formatted_rows: list[list[str | int]] = [[_format_table_cell_human(v) for v in row] for row in rows]
                screaming_headers = [_to_header(h) for h in headers]
                screaming_alignments = {_to_header(k): v for k, v in (alignments or {}).items()}
                print(tabulate(formatted_rows, headers=screaming_headers, alignments=screaming_alignments))
            case OutputFormatWithAuto.agent:  # TSV, no truncation, full timestamps
                print("\t".join(headers))
                for row in rows:
                    print("\t".join(_format_table_cell_agent(v) for v in row))
            case OutputFormatWithAuto.json:  # compact JSON array
                print(json.dumps(list(items), default=str))
            case OutputFormatWithAuto.quiet:  # first column only, one per line
                for row in rows:
                    print(row[0])

    def dict(self, data: dict[str, Any]) -> None:
        """Print structured data as JSON in all modes (indented for human, compact otherwise)."""
        indent = 2 if self.mode == OutputFormatWithAuto.human else None
        print(json.dumps(data, indent=indent, default=str))

    def result(self, message: str, **data: Any) -> None:
        """Print a success summary to stdout."""
        match self.mode:
            case OutputFormatWithAuto.human:  # ✓ message + key: value lines
                parts = [ANSI.green(f"✓ {message}")]
                for k, v in data.items():
                    if v is not None:
                        parts.append(f"  {k}: {v}")
                print("\n".join(parts))
            case OutputFormatWithAuto.agent:  # key=val pairs, space-separated
                parts = [f"{k}={v}" for k, v in data.items() if v is not None]
                print(" ".join(parts) if parts else message)
            case OutputFormatWithAuto.json:  # json.dumps(data), message ignored
                print(json.dumps(data, default=str) if data else "")
            case OutputFormatWithAuto.quiet:  # first value only
                values = list(data.values())
                if values:
                    print(values[0])

    def warning(self, message: str) -> None:
        """Print a non-fatal warning to stderr (all modes)."""
        if self.mode == OutputFormatWithAuto.human:
            print(ANSI.yellow(f"  Warning: {message}"), file=sys.stderr)
        else:
            print(f"Warning: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Print an error to stderr (all modes)."""
        if self.mode == OutputFormatWithAuto.human:
            print(ANSI.red(f"  Error: {message}"), file=sys.stderr)
        else:
            print(f"Error: {message}", file=sys.stderr)

    def hint(self, message: str) -> None:
        """Print a helpful hint to stderr (human: gray, agent/json: plain text)."""
        if self.mode == OutputFormatWithAuto.human:
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
