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
"""Shell completion for the ``hf`` CLI, built on Click's native completion.

Click generates completion scripts activated by the ``_HF_COMPLETE`` env var (this
works out of the box because ``hf`` is a Click command). This module exposes the two
conveniences Typer used to provide — ``--install-completion`` / ``--show-completion``
— as thin eager options over that machinery.
"""

import os
from pathlib import Path
from typing import Annotated

import click
from click.shell_completion import get_completion_class

from ._framework import Option


_COMPLETE_VAR = "_HF_COMPLETE"

# Per supported shell: the rc file to append to and the line that activates completion.
_SHELL_ACTIVATION: dict[str, tuple[Path, str]] = {
    "bash": (Path.home() / ".bashrc", f'eval "$({_COMPLETE_VAR}=bash_source hf)"'),
    "zsh": (Path.home() / ".zshrc", f'eval "$({_COMPLETE_VAR}=zsh_source hf)"'),
    "fish": (Path.home() / ".config" / "fish" / "completions" / "hf.fish", f"{_COMPLETE_VAR}=fish_source hf | source"),
}


def _detect_shell() -> str:
    return Path(os.environ.get("SHELL", "")).name or "bash"


def _completion_script(shell: str) -> str:
    # Imported lazily to avoid a circular import (hf.py imports the options below).
    from .hf import app

    completion_cls = get_completion_class(shell)
    if completion_cls is None:
        raise click.ClickException(f"Shell '{shell}' is not supported for completion.")
    return completion_cls(app, {}, "hf", _COMPLETE_VAR).source()


def _show_completion(value: bool) -> None:
    if not value:
        return
    click.echo(_completion_script(_detect_shell()))
    raise click.exceptions.Exit()


def _install_completion(value: bool) -> None:
    if not value:
        return
    shell = _detect_shell()
    if shell not in _SHELL_ACTIVATION:
        raise click.ClickException(f"Shell '{shell}' is not supported for completion.")
    rc_path, activation = _SHELL_ACTIVATION[shell]
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = rc_path.read_text() if rc_path.exists() else ""
    if activation not in existing:
        with rc_path.open("a") as file:
            file.write(f"\n{activation}\n")
    click.echo(f"{shell} completion installed in {rc_path}. Restart your shell for it to take effect.")
    raise click.exceptions.Exit()


InstallCompletionOpt = Annotated[
    bool,
    Option(
        "--install-completion",
        callback=_install_completion,
        is_eager=True,
        help="Install completion for the current shell.",
    ),
]

ShowCompletionOpt = Annotated[
    bool,
    Option(
        "--show-completion",
        callback=_show_completion,
        is_eager=True,
        help="Show completion for the current shell, to copy it or customize the installation.",
    ),
]
