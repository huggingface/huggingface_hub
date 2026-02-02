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

import dataclasses
import datetime
import importlib.metadata
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Optional, Sequence, Union, cast

import click
import typer

from huggingface_hub import DatasetInfo, ModelInfo, SpaceInfo, __version__, constants
from huggingface_hub.hf_api import PaperInfo
from huggingface_hub.utils import ANSI, get_session, hf_raise_for_status, installation_method, logging, tabulate


logger = logging.get_logger()

# Arbitrary maximum length of a cell in a table output
_MAX_CELL_LENGTH = 35

if TYPE_CHECKING:
    from huggingface_hub.hf_api import HfApi


def get_hf_api(token: Optional[str] = None) -> "HfApi":
    # Import here to avoid circular import
    from huggingface_hub.hf_api import HfApi

    return HfApi(token=token, library_name="huggingface-cli", library_version=__version__)


#### TYPER UTILS


class AlphabeticalMixedGroup(typer.core.TyperGroup):
    """
    Typer Group that lists commands and sub-apps mixed and alphabetically.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:  # type: ignore[name-defined]
        # click.Group stores both commands and subgroups in `self.commands`
        return sorted(self.commands.keys())


def typer_factory(help: str) -> typer.Typer:
    return typer.Typer(
        help=help,
        add_completion=True,
        no_args_is_help=True,
        cls=AlphabeticalMixedGroup,
        # Disable rich completely for consistent experience
        rich_markup_mode=None,
        rich_help_panel=None,
        pretty_exceptions_enable=False,
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
    Optional[bool],
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


LimitOpt = Annotated[
    int,
    typer.Option(help="Limit the number of results."),
]

AuthorOpt = Annotated[
    Optional[str],
    typer.Option(help="Filter by author or organization."),
]

FilterOpt = Annotated[
    Optional[list[str]],
    typer.Option(help="Filter by tags (e.g. 'text-classification'). Can be used multiple times."),
]

SearchOpt = Annotated[
    Optional[str],
    typer.Option(help="Search query."),
]


class OutputFormat(str, Enum):
    """Output format for CLI list commands."""

    table = "table"
    json = "json"


FormatOpt = Annotated[
    OutputFormat,
    typer.Option(
        help="Output format (table or json).",
    ),
]

QuietOpt = Annotated[
    bool,
    typer.Option(
        "-q",
        "--quiet",
        help="Print only IDs (one per line).",
    ),
]


def _to_header(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return s.upper()


def _format_cell(value: object, max_len: int = _MAX_CELL_LENGTH) -> str:
    """Format a value for table display with truncation."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "✔" if value else ""
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", value):
        return value[:10]
    if isinstance(value, list):
        if not value:
            return ""
        cell = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        return ""
    else:
        cell = str(value)
    if len(cell) > max_len:
        cell = cell[: max_len - 3] + "..."
    return cell


def print_as_table(
    items: Sequence[dict[str, Any]],
    headers: list[str],
    row_fn: Callable[[dict[str, Any]], list[str]],
) -> None:
    """Print items as a formatted table.

    Args:
        items: Sequence of dictionaries representing the items to display.
        headers: List of column headers.
        row_fn: Function that takes an item dict and returns a list of string values for each column.
    """
    if not items:
        print("No results found.")
        return
    rows = cast(list[list[Union[str, int]]], [row_fn(item) for item in items])
    print(tabulate(rows, headers=[_to_header(h) for h in headers]))


def print_list_output(
    items: Sequence[dict[str, Any]],
    format: OutputFormat,
    quiet: bool,
    id_key: str = "id",
    headers: Optional[list[str]] = None,
    row_fn: Optional[Callable[[dict[str, Any]], list[str]]] = None,
) -> None:
    """Print list command output in the specified format.

    Args:
        items: Sequence of dictionaries representing the items to display.
        format: Output format (table or json).
        quiet: If True, print only IDs (one per line).
        id_key: Key to use for extracting IDs in quiet mode.
        headers: Optional list of column names for headers. If not provided, auto-detected from keys.
        row_fn: Optional function to extract row values. If not provided, uses _format_cell on each column.
    """
    if quiet:
        for item in items:
            print(item[id_key])
        return

    if format == OutputFormat.json:
        print(json.dumps(list(items), indent=2))
        return

    if headers is None:
        all_columns = list(items[0].keys()) if items else [id_key]
        headers = [col for col in all_columns if any(_format_cell(item.get(col)) for item in items)]

    if row_fn is None:

        def row_fn(item: dict[str, Any]) -> list[str]:
            return [_format_cell(item.get(col)) for col in headers]  # type: ignore[union-attr]

    print_as_table(items, headers=headers, row_fn=row_fn)


def _serialize_value(v: object) -> object:
    """Recursively serialize a value to be JSON-compatible."""
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    elif isinstance(v, dict):
        return {key: _serialize_value(val) for key, val in v.items() if val is not None}
    elif isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def api_object_to_dict(
    info: Union[ModelInfo, DatasetInfo, SpaceInfo, PaperInfo],
) -> dict[str, object]:
    """Convert repo info dataclasses to json-serializable dicts."""
    return {k: _serialize_value(v) for k, v in dataclasses.asdict(info).items() if v is not None}


def make_expand_properties_parser(valid_properties: list[str]):
    """Create a callback to parse and validate comma-separated expand properties."""

    def _parse_expand_properties(value: Optional[str]) -> Optional[list[str]]:
        if value is None:
            return None
        properties = [p.strip() for p in value.split(",")]
        for prop in properties:
            if prop not in valid_properties:
                raise typer.BadParameter(
                    f"Invalid expand property: '{prop}'. Valid values are: {', '.join(valid_properties)}"
                )
        return properties

    return _parse_expand_properties


### PyPI VERSION CHECKER


def check_cli_update(library: Literal["huggingface_hub", "transformers"]) -> None:
    """
    Check whether a newer version of a library is available on PyPI.

    If a newer version is found, notify the user and suggest updating.
    If current version is a pre-release (e.g. `1.0.0.rc1`), or a dev version (e.g. `1.0.0.dev1`), no check is performed.

    This function is called at the entry point of the CLI. It only performs the check once every 24 hours, and any error
    during the check is caught and logged, to avoid breaking the CLI.

    Args:
        library: The library to check for updates. Currently supports "huggingface_hub" and "transformers".
    """
    try:
        _check_cli_update(library)
    except Exception:
        # We don't want the CLI to fail on version checks, no matter the reason.
        logger.debug("Error while checking for CLI update.", exc_info=True)


def _check_cli_update(library: Literal["huggingface_hub", "transformers"]) -> None:
    current_version = importlib.metadata.version(library)

    # Skip if current version is a pre-release or dev version
    if any(tag in current_version for tag in ["rc", "dev"]):
        return

    # Skip if already checked in the last 24 hours
    if os.path.exists(constants.CHECK_FOR_UPDATE_DONE_PATH):
        mtime = os.path.getmtime(constants.CHECK_FOR_UPDATE_DONE_PATH)
        if (time.time() - mtime) < 24 * 3600:
            return

    # Touch the file to mark that we did the check now
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).touch()

    # Check latest version from PyPI
    response = get_session().get(f"https://pypi.org/pypi/{library}/json", timeout=2)
    hf_raise_for_status(response)
    data = response.json()
    latest_version = data["info"]["version"]

    # If latest version is different from current, notify user
    if current_version != latest_version:
        if library == "huggingface_hub":
            update_command = _get_huggingface_hub_update_command()
        else:
            update_command = _get_transformers_update_command()

        click.echo(
            ANSI.yellow(
                f"A new version of {library} ({latest_version}) is available! "
                f"You are using version {current_version}.\n"
                f"To update, run: {ANSI.bold(update_command)}\n",
            )
        )


def _get_huggingface_hub_update_command() -> str:
    """Return the command to update huggingface_hub."""
    method = installation_method()
    if method == "brew":
        return "brew upgrade huggingface-cli"
    elif method == "hf_installer" and os.name == "nt":
        return 'powershell -NoProfile -Command "iwr -useb https://hf.co/cli/install.ps1 | iex"'
    elif method == "hf_installer":
        return "curl -LsSf https://hf.co/cli/install.sh | bash -"
    else:  # unknown => likely pip
        return "pip install -U huggingface_hub"


def _get_transformers_update_command() -> str:
    """Return the command to update transformers."""
    method = installation_method()
    if method == "hf_installer" and os.name == "nt":
        return 'powershell -NoProfile -Command "iwr -useb https://hf.co/cli/install.ps1 | iex" -WithTransformers'
    elif method == "hf_installer":
        return "curl -LsSf https://hf.co/cli/install.sh | bash -s -- --with-transformers"
    else:  # brew/unknown => likely pip
        return "pip install -U transformers"
