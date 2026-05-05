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
"""Pretty ANSI help formatter for the ``hf`` CLI.

Subclasses Click's ``HelpFormatter`` to add bold section headings, a styled
usage line, and dimmed meta text.  ANSI codes are only emitted when stdout
is a TTY and the process is not driven by an AI agent (``is_agent()``);
otherwise the output is identical to the default Click formatter.
"""

import os
import re
import sys

import click

from huggingface_hub.utils import is_agent


def _use_ansi() -> bool:
    """Return True when ANSI escape codes should be emitted."""
    if os.environ.get("NO_COLOR"):
        return False
    if is_agent():
        return False
    if not sys.stdout.isatty():
        return False
    return True


# ANSI codes
_BOLD = "\033[1m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_EXAMPLE_RE = re.compile(r"^(\s*\$ .+)$", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^[A-Z][A-Za-z ]+$")


class StyledHelpFormatter(click.HelpFormatter):
    """Click ``HelpFormatter`` with ANSI styling for human-friendly output.

    When ANSI is disabled (pipe, agent, ``NO_COLOR``) it behaves identically
    to the default ``HelpFormatter``.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.ansi = _use_ansi()

    # -- helpers ---------------------------------------------------------------

    def _bold(self, text: str) -> str:
        return f"{_BOLD}{text}{_RESET}" if self.ansi else text

    def _dim(self, text: str) -> str:
        return f"{_DIM}{text}{_RESET}" if self.ansi else text

    def _yellow(self, text: str) -> str:
        return f"{_YELLOW}{text}{_RESET}" if self.ansi else text

    def _cyan(self, text: str) -> str:
        return f"{_CYAN}{text}{_RESET}" if self.ansi else text

    # -- overrides -------------------------------------------------------------

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        if prefix is None:
            prefix = f"{self._yellow('Usage:')} " if self.ansi else "Usage: "
        if self.ansi:
            prog = self._bold(prog)
        super().write_usage(prog, args, prefix=prefix)

    def write_heading(self, heading: str) -> None:
        self.write(f"{'':>{self.current_indent}}{self._bold(heading + ':')}\n")

    def write_dl(self, rows, col_max=30, col_spacing=2):  # type: ignore[override]
        if not self.ansi:
            return super().write_dl(rows, col_max=col_max, col_spacing=col_spacing)
        styled_rows = [(self._cyan(first), second) for first, second in rows]
        super().write_dl(styled_rows, col_max=col_max, col_spacing=col_spacing)

    def write_text(self, text: str) -> None:
        if self.ansi:
            text = _EXAMPLE_RE.sub(lambda m: self._dim(m.group(1)), text)
            if _HEADING_LINE_RE.match(text):
                text = self._bold(text)
        super().write_text(text)


class StyledContext(click.Context):
    """Click ``Context`` that uses :class:`StyledHelpFormatter`."""

    formatter_class = StyledHelpFormatter
