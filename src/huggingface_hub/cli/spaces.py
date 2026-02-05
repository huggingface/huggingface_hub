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
"""Contains commands to interact with spaces on the Hugging Face Hub.

Usage:
    # list spaces on the Hub
    hf spaces ls

    # list spaces with a search query
    hf spaces ls --search "chatbot"

    # get info about a space
    hf spaces info enzostvs/deepsite
"""

import enum
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any, Optional, get_args

import typer
from packaging import version

from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import ExpandSpaceProperty_T, SpaceSort_T
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

from ._cli_utils import (
    AuthorOpt,
    FilterOpt,
    FormatOpt,
    LimitOpt,
    OutputFormat,
    QuietOpt,
    RevisionOpt,
    SearchOpt,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    make_expand_properties_parser,
    print_list_output,
    typer_factory,
)


HOT_RELOADING_MIN_GRADIO = "6.0.0"
HOT_RELOADING_MIN_PYSPACES = "0.44.0"


_EXPAND_PROPERTIES = sorted(get_args(ExpandSpaceProperty_T))
_SORT_OPTIONS = get_args(SpaceSort_T)
SpaceSortEnum = enum.Enum("SpaceSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    Optional[str],
    typer.Option(
        help=f"Comma-separated properties to expand. Example: '--expand=likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


spaces_cli = typer_factory(help="Interact with spaces on the Hub.")
spaces_hot_reloading_cli = typer_factory(help="Low-level hot-reloading commands")

spaces_cli.add_typer(spaces_hot_reloading_cli, name="hot-reloading")


@spaces_cli.command("ls")
def spaces_ls(
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        Optional[SpaceSortEnum],
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 10,
    expand: ExpandOpt = None,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List spaces on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(space_info)
        for space_info in api.list_spaces(
            filter=filter, author=author, search=search, sort=sort_key, limit=limit, expand=expand
        )
    ]

    def row_fn(item: dict[str, Any]) -> list[str]:
        repo_id = str(item.get("id", ""))
        author = str(item.get("author", "")) or (repo_id.split("/")[0] if "/" in repo_id else "")
        return [
            repo_id,
            author,
            str(item.get("sdk", "") or ""),
            str(item.get("likes", "") or ""),
        ]

    print_list_output(
        items=results,
        format=format,
        quiet=quiet,
        id_key="id",
        headers=["ID", "AUTHOR", "SDK", "LIKES"],
        row_fn=row_fn,
    )


@spaces_cli.command("info")
def spaces_info(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a space on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.space_info(repo_id=space_id, revision=revision, expand=expand)  # type: ignore[arg-type]
    except RepositoryNotFoundError as e:
        raise CLIError(f"Space '{space_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{space_id}'.") from e
    print(json.dumps(api_object_to_dict(info), indent=2))


@spaces_cli.command("hot-reload")
def spaces_hot_reload(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    filename: Annotated[str, typer.Argument(help="Path to the Python file in the Space repository.")],
    skip_checks: Annotated[bool, typer.Option(help="Skip hot-reload compatibility checks.")] = False,
    token: TokenOpt = None,
) -> None:
    """Perform a hot-reloaded update on any Python file of a Space"""

    api = get_hf_api(token=token)

    if not skip_checks:
        space_info = api.space_info(space_id, token=token)
        if space_info.sdk != "gradio":
            raise CLIError(f"Hot-reloading is only available on Gradio SDK. Found {space_info.sdk} SDK")
        if (card_data := space_info.card_data) is None:
            raise CLIError(f"Unable to read cardData for Space {space_id}")
        if (sdk_version := card_data.sdk_version) is None:
            raise CLIError(f"Unable to read sdk_version from {space_id} cardData")
        if (sdk_version := version.parse(sdk_version)) < version.Version(HOT_RELOADING_MIN_GRADIO):
            raise CLIError(f"Hot-reloading requires Gradio 6+ (found {sdk_version})")
        if (runtime := space_info.runtime) is None:
            raise CLIError(f"Unable to read SpaceRuntime for Space {space_id}")
        if (spaces_version := runtime.pyspaces_version) is None:
            raise CLIError(f"Unable to read pySpacesVersion from {space_id} SpaceRuntime")
        if (spaces_version := version.parse(spaces_version)) < version.Version(HOT_RELOADING_MIN_PYSPACES):
            raise CLIError(f"Hot-reloading requires spaces >= 0.44.0 (found {spaces_version})")

    with tempfile.TemporaryDirectory() as local_dir:
        filepath = Path(local_dir) / filename
        if not (pbar_disabled := are_progress_bars_disabled()):
            disable_progress_bars()
        hf_hub_download(
            repo_type="space",
            repo_id=space_id,
            filename=filename,
            local_dir=local_dir,
        )
        if not pbar_disabled:
            enable_progress_bars()
        subprocess.run(['code', '--wait', filepath]) # TODO: $EDITOR
        api.upload_file(
            repo_type="space",
            repo_id=space_id,
            path_or_fileobj=filepath,
            path_in_repo=filename,
            hot_reload=True,
        )

    # hot-reloading summary
    # TODO
