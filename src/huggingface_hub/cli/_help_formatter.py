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

import re
import sys

import click

from huggingface_hub.utils import ANSI, is_agent


def _use_ansi() -> bool:
    """Return True when ANSI escape codes should be emitted."""
    if is_agent():
        return False
    if not sys.stdout.isatty():
        return False
    return True


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

    # -- overrides -------------------------------------------------------------

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        if self.ansi and prefix is None:
            prefix = f"{ANSI.yellow('Usage:')} "
            prog = ANSI.bold(prog)
        super().write_usage(prog, args, prefix=prefix)

    def write_heading(self, heading: str) -> None:
        self.write(f"{'':>{self.current_indent}}{ANSI.bold(heading + ':') if self.ansi else heading + ':'}\n")

    def write_text(self, text: str) -> None:
        if self.ansi:
            text = _EXAMPLE_RE.sub(lambda m: ANSI.dim(m.group(1)), text)
            if self.current_indent == 0 and _HEADING_LINE_RE.match(text):
                text = ANSI.bold(text)
        super().write_text(text)


class StyledContext(click.Context):
    """Click ``Context`` that uses :class:`StyledHelpFormatter`."""

    formatter_class = StyledHelpFormatter
