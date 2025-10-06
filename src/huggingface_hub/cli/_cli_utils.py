# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Contains CLI utilities (styling, helpers)."""

import importlib.metadata
import os
import time
from enum import Enum
from typing import Annotated, Optional, Union

import click
import typer

from huggingface_hub import __version__, constants
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import get_session, hf_raise_for_status, installation_method, logging


logger = logging.get_logger()


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _gray = "\u001b[90m"
    _red = "\u001b[31m"
    _reset = "\u001b[0m"
    _yellow = "\u001b[33m"

    @classmethod
    def bold(cls, s: str) -> str:
        return cls._format(s, cls._bold)

    @classmethod
    def gray(cls, s: str) -> str:
        return cls._format(s, cls._gray)

    @classmethod
    def red(cls, s: str) -> str:
        return cls._format(s, cls._bold + cls._red)

    @classmethod
    def yellow(cls, s: str) -> str:
        return cls._format(s, cls._yellow)

    @classmethod
    def _format(cls, s: str, code: str) -> str:
        if os.environ.get("NO_COLOR"):
            # See https://no-color.org/
            return s
        return f"{code}{s}{cls._reset}"


def tabulate(rows: list[list[Union[str, int]]], headers: list[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


#### TYPER UTILS


class AlphabeticalMixedGroup(typer.core.TyperGroup):
    """
    Typer Group that lists commands and sub-apps mixed and alphabetically.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:  # type: ignore[name-defined]
        # click.Group stores both commands and sub-groups in `self.commands`
        return sorted(self.commands.keys())


def typer_factory(help: str) -> typer.Typer:
    return typer.Typer(
        help=help,
        add_completion=True,
        rich_markup_mode=None,
        no_args_is_help=True,
        cls=AlphabeticalMixedGroup,
    )


class RepoType(str, Enum):
    model = "model"
    dataset = "dataset"
    space = "space"


RepoIdArg = Annotated[
    str,
    typer.Argument(
        help="The ID of the repo (e.g. `username/repo-name`).",
    ),
]


RepoTypeOpt = Annotated[
    RepoType,
    typer.Option(
        help="The type of repository (model, dataset, or space).",
    ),
]

TokenOpt = Annotated[
    Optional[str],
    typer.Option(
        help="A User Access Token generated from https://huggingface.co/settings/tokens.",
    ),
]

PrivateOpt = Annotated[
    bool,
    typer.Option(
        help="Whether to create a private repo if repo doesn't exist on the Hub. Ignored if the repo already exists.",
    ),
]

RevisionOpt = Annotated[
    Optional[str],
    typer.Option(
        help="Git revision id which can be a branch name, a tag, or a commit hash.",
    ),
]


def get_hf_api(token: Optional[str] = None) -> HfApi:
    return HfApi(token=token, library_name="hf", library_version=__version__)


### PyPI VERSION CHECKER


def check_cli_update() -> None:
    """
    Check whether a newer version of `huggingface_hub` is available on PyPI.

    If a newer version is found, notify the user and suggest updating.
    The latest PyPI version is cached locally in `$HF_HOME/pypi_latest_version` for 24 hours to prevent repeated notifications.
    If current version is a pre-release (e.g. `1.0.0.rc1`), or a dev version (e.g. `1.0.0.dev1`), no check is performed.

    This function is called at the entry point of the CLI.
    """
    try:
        _check_cli_update()
    except Exception:
        # We don't want the CLI to fail on version checks, no matter the reason.
        logger.debug("Error while checking for CLI update.", exc_info=True)


def _check_cli_update() -> None:
    current_version = importlib.metadata.version("huggingface_hub")

    if any(tag in current_version for tag in ["a", "b", "rc", "dev"]):
        # Don't check for pre-releases or dev versions
        return

    cached_version = _get_cached_pypi_version()
    if cached_version is None:
        latest_version = _get_pypi_version()
        _cache_pypi_version(latest_version)
    else:
        latest_version = cached_version

    if current_version != latest_version:
        method = installation_method()
        if method == "brew":
            update_command = "brew upgrade huggingface-cli"
        elif method == "hf_installer" and os.name == "nt":
            update_command = "curl -LsSf https://hf.co/cli/install.ps1 | pwsh -"
        elif method == "hf_installer":
            update_command = "curl -LsSf https://hf.co/cli/install.sh | sh -"
        else:  # unknown => likely pip
            update_command = "pip install -U huggingface_hub"

        click.echo(
            ANSI.yellow(
                f"A new version of huggingface_hub ({latest_version}) is available! "
                f"You are using version {current_version}.\n"
                f"To update, run: {ANSI.bold(update_command)}\n",
            )
        )


def _get_pypi_version() -> str:
    response = get_session().get("https://pypi.org/pypi/huggingface_hub/json", timeout=2)
    hf_raise_for_status(response)
    data = response.json()
    return data["info"]["version"]


def _get_cached_pypi_version() -> Optional[str]:
    if os.path.exists(constants.PYPI_LATEST_VERSION_PATH):
        mtime = os.path.getmtime(constants.PYPI_LATEST_VERSION_PATH)
        # If the file is older than 24h, we don't use it
        if (time.time() - mtime) < 24 * 3600:
            with open(constants.PYPI_LATEST_VERSION_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None


def _cache_pypi_version(version: str) -> None:
    with open(constants.PYPI_LATEST_VERSION_PATH, "w", encoding="utf-8") as f:
        f.write(version)
