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
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Optional, get_args

import typer
from packaging import version
from typing_extensions import assert_never

from huggingface_hub.cli import _cli_utils
from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import ExpandSpaceProperty_T, HfApi, SpaceSort_T
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


HOT_RELOADING_MIN_GRADIO = "6.1.0"


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
    local: Annotated[bool, typer.Option(help="Whether file should be taken locally instead of EDITOR open.")] = False,
    local_path: Annotated[Optional[str], typer.Option(help="Path of local file. Only relevant with --local flag")] = None,
    skip_checks: Annotated[bool, typer.Option(help="Skip hot-reload compatibility checks.")] = False,
    skip_summary: Annotated[bool, typer.Option(help="Skip summary display after hot-reloaded triggered")] = False,
    token: TokenOpt = None,
) -> None:
    """
    Perform a hot-reloaded update on any Python file of a Space.
    Opens an interactive editor unless --local flag or --local-path option is used.
    """

    api = get_hf_api(token=token)

    if not skip_checks:
        space_info = api.space_info(space_id)
        if space_info.sdk != "gradio":
            raise CLIError(f"Hot-reloading is only available on Gradio SDK. Found {space_info.sdk} SDK")
        if (card_data := space_info.card_data) is None:
            raise CLIError(f"Unable to read cardData for Space {space_id}")
        if (sdk_version := card_data.sdk_version) is None:
            raise CLIError(f"Unable to read sdk_version from {space_id} cardData")
        if (sdk_version := version.parse(sdk_version)) < version.Version(HOT_RELOADING_MIN_GRADIO):
            raise CLIError(f"Hot-reloading requires Gradio >= {HOT_RELOADING_MIN_GRADIO} (found {sdk_version})")

    if local_path:
        filepath = local_path
    elif local:
        filepath = filename
    else:
        if not skip_checks:
            api.auth_check(
                repo_type="spaces",
                repo_id=space_id,
                write=True,
            )
        temp_dir = tempfile.TemporaryDirectory()
        filepath = Path(temp_dir.name) / filename
        if not (pbar_disabled := are_progress_bars_disabled()):
            disable_progress_bars()
        hf_hub_download(
            repo_type="space",
            repo_id=space_id,
            filename=filename,
            local_dir=temp_dir.name,
        )
        if not pbar_disabled:
            enable_progress_bars()
        editor_res = _cli_utils.editor_open(str(filepath))
        if editor_res == "no-tty":
            raise CLIError("Cannot open an editor (no TTY). Use --local flag to ho-reload from local path")
        if editor_res == "no-editor":
            raise CLIError("No editor found in local environment. Use --local flag to ho-reload from local path")
        if editor_res != 0:
            raise CLIError(f"Editor returned a non-zero exit code while attempting to edit {filepath}")

    commit_info = api.upload_file(
        repo_type="space",
        repo_id=space_id,
        path_or_fileobj=filepath,
        path_in_repo=filename,
        hot_reload=True,
    )

    # TODO: Fetch / display post-commit runtime.hotReloading here instead of inside _spaces_hot_reloading_summary?
    # So this is displayed even with --skip-summary and gives enough information to perorm a `hf spaces hot-reloading get` call

    if not skip_summary:
        _spaces_hot_reloading_summary(
            api=api,
            space_id=space_id,
            commit_sha=commit_info.oid,
            token=token,
        )


def _spaces_hot_reloading_summary(
    api: HfApi,
    space_id: str,
    commit_sha: str,
    token: Optional[str],
) -> None:
    from huggingface_hub._hot_reloading_client import ReloadClient
    from huggingface_hub._hot_reloading_types import ApiGetReloadEventSourceData

    space_info = api.space_info(space_id)
    if (runtime := space_info.runtime) is None:
        raise CLIError(f"Unable to read SpaceRuntime from {space_id} infos")
    if (hot_reloading := runtime.hot_reloading) is None:
        raise CLIError(f"Space {space_id} current running version has not been hot-reloaded")
    if hot_reloading.status != "created":
        typer.echo("...")
        return

    if (space_host := space_info.host) is None:
        raise CLIError(f"Unexpected None host on hotReloaded Space")
    if (space_subdomain := space_info.subdomain) is None:
        raise CLIError(f"Unexpected None subdomain on hotReloaded Space")

    clients = [ReloadClient(
        host=space_host,
        subdomain=space_subdomain,
        replica_hash=hash,
        token=token,
    ) for hash, _ in hot_reloading.replica_statuses]

    # Quand on fait --local : display local_path:line:col (code region). Comme ça on peut link-click! Sans --local, peu de chances que le fichier existe
    # Faut que ça marche aussi bien avec la summary command qu'on lancerait éventuellement après un git push -o hot-reload (donc peut-être par défaut afficher code region dans la summary command)

    previous_events: dict[int, ApiGetReloadEventSourceData] = defaultdict()
    for client in clients:
        events: dict[int, ApiGetReloadEventSourceData] = defaultdict()
        full_match = True
        for index, event in enumerate(client.get_reload(commit_sha)):
            events[index] = event
            if full_match and previous_events[index] == event:
                continue
            # TODO: replay in case of full_match flip with index > 0
            full_match = False
            if False:
                pass
            elif event.data.kind == "error":
                ...
            elif event.data.kind == "exception":
                ...
            elif event.data.kind == "add":
                ...
            elif event.data.kind == "delete":
                ...
            elif event.data.kind == "update":
                ...
            elif event.data.kind == "run":
                ...
            elif event.data.kind == "ui":
                ...
            else:
                assert_never(event.data.kind)
        if full_match:
            ...
        previous_events = events
    ...
    # Fetch first client (display replica hash if multiple)
    # Compare others
    # Display final info (Success, Hot-reloading contains errors, etc.)



@spaces_hot_reloading_cli.command("summary")
def spaces_hot_reloading_summary(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    commit_sha: Annotated[str, typer.Argument(help="...")],
    token: TokenOpt = None,
):
    """ Description """
    api = get_hf_api(token=token)
    _spaces_hot_reloading_summary(api, space_id, commit_sha, token)
