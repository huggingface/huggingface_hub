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
import functools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Annotated, Literal, Optional, Union, get_args

import typer
from packaging import version
from typing_extensions import assert_never

from huggingface_hub import constants
from huggingface_hub._hot_reload.client import ReloadClient
from huggingface_hub._hot_reload.types import ApiGetReloadEventSourceData, ReloadRegion
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


@spaces_cli.command(
    "ls",
    examples=[
        "hf spaces ls --limit 10",
        'hf spaces ls --search "chatbot" --author huggingface',
    ],
)
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
    print_list_output(results, format=format, quiet=quiet)


@spaces_cli.command(
    "info",
    examples=[
        "hf spaces info enzostvs/deepsite",
        "hf spaces info gradio/theme_builder --expand sdk,runtime,likes",
    ],
)
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
    space_id: Annotated[
        str,
        typer.Argument(
            help="The space ID (e.g. `username/repo-name`).",
        ),
    ],
    filename: Annotated[
        Optional[str],
        typer.Argument(
            help="Path to the Python file in the Space repository. Can be omitted when --local-file is specified and path in repository matches."
        ),
    ] = None,
    local_file: Annotated[
        Optional[str],
        typer.Option(
            "--local-file",
            "-f",
            help="Path of local file. Interactive editor mode if not specified",
        ),
    ] = None,
    skip_checks: Annotated[bool, typer.Option(help="Skip hot-reload compatibility checks.")] = False,
    skip_summary: Annotated[bool, typer.Option(help="Skip summary display after hot-reloaded triggered")] = False,
    token: TokenOpt = None,
) -> None:
    """
    ⚠ This feature is experimental ⚠

    Hot-reload any Python file of a Space without a full rebuild + restart.

    Only works with Gradio SDK (6.1+)
    Opens an interactive editor unless --local-file/-f is specified.

    This command patches the live Python process using https://github.com/breuleux/jurigged
    (AST-based diffing, in-place function updates, etc.), integrated with Gradio's native hot-reload support
    (meaning that Gradio demo object changes are reflected in the UI)

    Usage examples:

    \b
    hf spaces hot-reload username/repo-name app.py               # Open an interactive editor to the remote app.py file
    hf spaces hot-reload username/repo-name -f app.py            # Take local version from ./app.py and patch app.py in remote repo
    hf spaces hot-reload username/repo-name app.py -f src/app.py # Take local version from ./src/app.py and patch app.py in remote repo
    """

    typer.secho("This feature is experimental and subject to change", fg=typer.colors.BRIGHT_BLACK)

    api = get_hf_api(token=token)

    if not skip_checks:
        space_info = api.space_info(space_id)
        if space_info.sdk != "gradio":
            raise CLIError(f"Hot-reloading is only available on Gradio SDK. Found {space_info.sdk} SDK")
        if (card_data := space_info.card_data) is None:
            raise CLIError(f"Unable to read cardData for Space {space_id}")
        if (sdk_version := card_data.sdk_version) is None:
            raise CLIError(f"Unable to read sdk_version from {space_id} cardData")
        if version.parse(sdk_version) < version.Version(HOT_RELOADING_MIN_GRADIO):
            raise CLIError(f"Hot-reloading requires Gradio >= {HOT_RELOADING_MIN_GRADIO} (found {sdk_version})")

    if local_file:
        filepath = local_file
        filename = local_file if filename is None else filename
    elif filename:
        if not skip_checks:
            api.auth_check(
                repo_type="space",
                repo_id=space_id,
                write=True,
            )
        temp_dir = tempfile.TemporaryDirectory()
        filepath = os.path.join(temp_dir.name, filename)
        if not (pbar_disabled := are_progress_bars_disabled()):
            disable_progress_bars()
        try:
            hf_hub_download(
                repo_type="space",
                repo_id=space_id,
                filename=filename,
                local_dir=temp_dir.name,
            )
        finally:
            if not pbar_disabled:
                enable_progress_bars()
        editor_res = _editor_open(filepath)
        if editor_res == "no-tty":
            raise CLIError("Cannot open an editor (no TTY). Use -f flag to hot-reload from local path")
        if editor_res == "no-editor":
            raise CLIError("No editor found in local environment. Use -f flag to hot-reload from local path")
        if editor_res != 0:
            raise CLIError(f"Editor returned a non-zero exit code while attempting to edit {filepath}")
    else:
        raise CLIError("Either filename or --local-file/-f must be specified")

    commit_info = api.upload_file(
        repo_type="space",
        repo_id=space_id,
        path_or_fileobj=filepath,
        path_in_repo=filename,
        _hot_reload=True,
    )

    if not skip_summary:
        _spaces_hot_reloading_summary(
            api=api,
            space_id=space_id,
            commit_sha=commit_info.oid,
            filepath=filepath if local_file else os.path.basename(filepath),
            token=token,
        )


def _spaces_hot_reloading_summary(
    api: HfApi,
    space_id: str,
    commit_sha: str,
    filepath: Optional[str],
    token: Optional[str],
) -> None:
    space_info = api.space_info(space_id)
    if (runtime := space_info.runtime) is None:
        raise CLIError(f"Unable to read SpaceRuntime from {space_id} infos")
    if (hot_reloading := runtime.hot_reloading) is None:
        raise CLIError(f"Space {space_id} current running version has not been hot-reloaded")
    if hot_reloading.status != "created":
        typer.echo(f"Failed creating hot-reloaded commit. {hot_reloading.replica_statuses=}")
        return

    if (space_host := space_info.host) is None:
        raise CLIError("Unexpected None host on hotReloaded Space")
    if (space_subdomain := space_info.subdomain) is None:
        raise CLIError("Unexpected None subdomain on hotReloaded Space")

    clients = [
        ReloadClient(
            host=space_host,
            subdomain=space_subdomain,
            replica_hash=hash,
            token=token,
        )
        for hash, _ in hot_reloading.replica_statuses
    ]

    def render_region(region: ReloadRegion) -> str:
        res = ""
        if filepath is not None:
            res += f"{filepath}, "
        if region["startLine"] == region["endLine"]:
            res += f"line {region['startLine'] - 1}"
        else:
            res += f"lines {region['startLine'] - 1}-{region['endLine'] - 1}"
        return res

    def display_event(event: ApiGetReloadEventSourceData) -> None:
        if event["data"]["kind"] == "error":
            typer.secho("✘ Unexpected hot-reloading error", bold=True)
            typer.secho(event["data"]["traceback"], italic=True)
        elif event["data"]["kind"] == "exception":
            typer.secho(f"✘ Exception at {render_region(event['data']['region'])}", bold=True)
            typer.secho(event["data"]["traceback"], italic=True)
        elif event["data"]["kind"] == "add":
            typer.secho(f"✔︎ Created {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "delete":
            typer.secho(f"∅ Deleted {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "update":
            typer.secho(f"✔︎ Updated {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "run":
            typer.secho(f"▶ Run {render_region(event['data']['region'])}", bold=True)
            typer.secho(event["data"]["codeLines"], italic=True)
        elif event["data"]["kind"] == "ui":
            if event["data"]["updated"]:
                typer.secho("⟳ UI updated", bold=True)
            else:
                typer.secho("∅ UI untouched", bold=True)
        else:
            assert_never(event["data"]["kind"])

    first_client_events: dict[int, ApiGetReloadEventSourceData] = {}
    for client_index, client in enumerate(clients):
        if len(clients) > 1:
            typer.secho(f"---- Replica {client.replica_hash} ----")
        full_match = True
        replay: list[ApiGetReloadEventSourceData] = []
        for event_index, event in enumerate(client.get_reload(commit_sha)):
            if client_index == 0:
                first_client_events[event_index] = event
            elif full_match := full_match and first_client_events.get(event_index) == event:
                replay += [event]
                continue
            if replay:
                for replay_event in replay:
                    display_event(replay_event)
                replay = []
            display_event(event)
        if client_index > 0 and full_match:
            typer.echo("✔︎ Same as first replica")


@functools.cache
def _get_editor_command() -> Optional[str]:
    for env in ("HF_EDITOR", "VISUAL", "EDITOR"):
        if command := os.getenv(env, "").strip():
            return command
    for binary_path, editor_command in constants.PREFERRED_EDITORS:
        if shutil.which(binary_path) is not None:
            return editor_command
    return None


def _editor_open(filepath: str) -> Union[int, Literal["no-tty", "no-editor"]]:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "no-tty"
    if (editor_command := _get_editor_command()) is None:
        return "no-editor"
    command = [*shlex.split(editor_command), filepath]
    res = subprocess.run(command, start_new_session=True)
    return res.returncode
