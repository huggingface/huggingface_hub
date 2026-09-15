# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import sys
import traceback
from typing import Annotated, Any

import click

from huggingface_hub import __version__, constants
from huggingface_hub.cli._cli_utils import (
    LazyHfCommand,
    LazyHfGroup,
    check_cli_update,
    fallback_typer_group_factory,
    typer_factory,
)
from huggingface_hub.cli._output import out
from huggingface_hub.utils import logging

from ._completion import _COMPLETE_VAR, InstallCompletionOpt, ShowCompletionOpt
from ._framework import Option


def _dispatch_unknown_top_level_extension(args: list[str], built_in_commands: set[str]) -> int | None:
    from .extensions import dispatch_unknown_top_level_extension

    return dispatch_unknown_top_level_extension(args, built_in_commands)


def _list_installed_extensions_for_help() -> list[tuple[str, str]]:
    from .extensions import list_installed_extensions_for_help

    return list_installed_extensions_for_help()


def check_skill_update() -> None:
    from ._skills import check_skill_update as _check_skill_update

    _check_skill_update()


app = typer_factory(
    help="Hugging Face Hub CLI",
    cls=fallback_typer_group_factory(
        _dispatch_unknown_top_level_extension,
        extra_commands_provider=_list_installed_extensions_for_help,
    ),
)


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise click.exceptions.Exit()


def _skills_callback(value: bool) -> None:
    if value:
        from .skills import skills_preview

        skills_preview()
        raise click.exceptions.Exit()


@app.group_callback(invoke_without_command=True)
def app_callback(
    version: Annotated[
        bool | None, Option("-v", "--version", callback=_version_callback, is_eager=True, hidden=True)
    ] = None,
    skills: Annotated[
        bool,
        Option(
            "--skills",
            callback=_skills_callback,
            is_eager=True,
            help="Print the `hf-cli` SKILL.md to stdout (alias for `hf skills preview`).",
        ),
    ] = False,
    install_completion: InstallCompletionOpt = False,
    show_completion: ShowCompletionOpt = False,
) -> None:
    pass


_LAZY_COMMANDS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("sync", "buckets", "sync", {}),
    ("cp", "_cp", "make_cp", {"examples_attribute": "CP_EXAMPLES", "is_factory": True}),
    ("download", "download", "download", {"examples_attribute": "DOWNLOAD_EXAMPLES"}),
    ("upload", "upload", "upload", {"examples_attribute": "UPLOAD_EXAMPLES"}),
    (
        "upload-large-folder",
        "upload_large_folder",
        "upload_large_folder",
        {"examples_attribute": "UPLOAD_LARGE_FOLDER_EXAMPLES"},
    ),
    ("env", "system", "env", {"topic": "help"}),
    ("update", "system", "update", {"topic": "help"}),
    ("version", "system", "version", {"topic": "help"}),
    ("lfs-enable-largefiles", "lfs", "lfs_enable_largefiles", {"hidden": True}),
    ("lfs-multipart-upload", "lfs", "lfs_multipart_upload", {"hidden": True}),
]


_LAZY_GROUPS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("auth", "auth", "auth_cli", {}),
    ("buckets", "buckets", "buckets_cli", {}),
    ("cache", "cache", "cache_cli", {}),
    ("collections", "collections", "collections_cli", {}),
    ("datasets", "datasets", "datasets_cli", {}),
    ("discussions", "discussions", "discussions_cli", {}),
    ("jobs", "jobs", "jobs_cli", {}),
    ("models", "models", "models_cli", {}),
    ("papers", "papers", "papers_cli", {}),
    ("repos | repo", "repos", "repos_cli", {}),
    ("sandbox", "sandbox", "sandbox_cli", {}),
    ("skills", "skills", "skills_cli", {}),
    ("spaces", "spaces", "spaces_cli", {}),
    ("webhooks", "webhooks", "webhooks_cli", {}),
    ("endpoints", "inference_endpoints", "ie_cli", {}),
    ("extensions | ext", "extensions", "extensions_cli", {}),
    ("repo-files", "repo_files", "repo_files_cli", {"hidden": True}),
]

for name, module, attribute, extras in _LAZY_COMMANDS:
    app.add_command(
        LazyHfCommand(
            name,
            module=f"huggingface_hub.cli.{module}",
            attribute=attribute,
            **extras,
        ),
        name,
    )

for name, module, attribute, extras in _LAZY_GROUPS:
    app.add_command(
        LazyHfGroup(
            name,
            module=f"huggingface_hub.cli.{module}",
            attribute=attribute,
            **extras,
        ),
        name,
    )


def main():
    # Shell-completion requests must stay fast and emit nothing but candidates:
    # skip the startup work and let click handle the env var inside `app()`.
    if _COMPLETE_VAR not in os.environ:
        if not constants.HF_DEBUG:
            logging.set_verbosity_info()
        check_cli_update("huggingface_hub")
        # Don't nag while the user is already managing skills, nor on `hf update` which handles the
        # skill itself (it would print a redundant or contradictory hint before doing so).
        if sys.argv[1:2] not in (["skills"], ["update"], ["--skills"]):
            check_skill_update()

    try:
        app()
    except Exception as e:
        from ._errors import format_known_exception

        message = format_known_exception(e)
        if message:
            out.error(message)
            if constants.HF_DEBUG:
                traceback.print_exc()
            else:
                out.hint("set HF_DEBUG=1 as environment variable for full traceback.")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
