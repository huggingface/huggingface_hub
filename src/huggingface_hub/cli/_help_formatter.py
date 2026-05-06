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
"""Pretty ANSI help formatter for the `hf` CLI.

Subclasses Click's `HelpFormatter` to add underlined section headings, a styled usage line, and bold
commands/options. ANSI codes are only emitted when stdout is a TTY and the process is not driven by an AI agent
(`is_agent()`). Otherwise the output is identical to the default Click formatter.
"""

from collections.abc import Sequence

import click

from huggingface_hub.utils import ANSI


class StyledHelpFormatter(click.HelpFormatter):
    """Click ``HelpFormatter`` with ANSI styling for human-friendly output.

    ANSI is disabled when the output is piped, driven by an AI agent, or the `NO_COLOR` environment variable is set.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    def write_heading(self, heading: str) -> None:
        styled = ANSI.underline(heading + ":")
        self.write(f"{'':>{self.current_indent}}{styled}\n")

    def write_dl(self, rows: Sequence[tuple[str, str]], col_max: int = 30, col_spacing: int = 2) -> None:
        rows = [(ANSI.bold(first), second) for first, second in rows]
        super().write_dl(rows, col_max=col_max, col_spacing=col_spacing)


class StyledContext(click.Context):
    """Click ``Context`` that uses :class:`StyledHelpFormatter`."""

    formatter_class = StyledHelpFormatter
