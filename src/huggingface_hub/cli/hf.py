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

import sys
import traceback
from typing import Annotated, Optional

import typer

from huggingface_hub import __version__, constants
from huggingface_hub.cli._cli_utils import check_cli_update, typer_factory
from huggingface_hub.cli._errors import format_known_exception
from huggingface_hub.cli.auth import auth_cli
from huggingface_hub.cli.cache import cache_cli
from huggingface_hub.cli.collections import collections_cli
from huggingface_hub.cli.datasets import datasets_cli
from huggingface_hub.cli.download import DOWNLOAD_EXAMPLES, download
from huggingface_hub.cli.extensions import _execute_extension_binary, _get_extension_executable_path, extensions_cli
from huggingface_hub.cli.inference_endpoints import ie_cli
from huggingface_hub.cli.jobs import jobs_cli
from huggingface_hub.cli.lfs import lfs_enable_largefiles, lfs_multipart_upload
from huggingface_hub.cli.models import models_cli
from huggingface_hub.cli.papers import papers_cli
from huggingface_hub.cli.repo import repo_cli
from huggingface_hub.cli.repo_files import repo_files_cli
from huggingface_hub.cli.skills import skills_cli
from huggingface_hub.cli.spaces import spaces_cli
from huggingface_hub.cli.system import env, version
from huggingface_hub.cli.upload import UPLOAD_EXAMPLES, upload
from huggingface_hub.cli.upload_large_folder import UPLOAD_LARGE_FOLDER_EXAMPLES, upload_large_folder
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import ANSI, logging


app = typer_factory(help="Hugging Face Hub CLI")


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def app_callback(
    version: Annotated[
        Optional[bool], typer.Option("--version", callback=_version_callback, is_eager=True, hidden=True)
    ] = None,
) -> None:
    pass


# top level single commands (defined in their respective files)
app.command(examples=DOWNLOAD_EXAMPLES)(download)
app.command(examples=UPLOAD_EXAMPLES)(upload)
app.command(examples=UPLOAD_LARGE_FOLDER_EXAMPLES)(upload_large_folder)

app.command(topic="help")(env)
app.command(topic="help")(version)

app.command(hidden=True)(lfs_enable_largefiles)
app.command(hidden=True)(lfs_multipart_upload)

# command groups
app.add_typer(auth_cli, name="auth")
app.add_typer(cache_cli, name="cache")
app.add_typer(collections_cli, name="collections")
app.add_typer(datasets_cli, name="datasets")
app.add_typer(jobs_cli, name="jobs")
app.add_typer(models_cli, name="models")
app.add_typer(papers_cli, name="papers")
app.add_typer(repo_cli, name="repo")
app.add_typer(repo_files_cli, name="repo-files")
app.add_typer(skills_cli, name="skills")
app.add_typer(spaces_cli, name="spaces")
app.add_typer(ie_cli, name="endpoints")
app.add_typer(extensions_cli, name="extensions")


def _get_top_level_command_names() -> set[str]:
    click_app = typer.main.get_command(app)
    return set(click_app.commands.keys())  # type: ignore[attr-defined]


def _dispatch_installed_extension(argv: list[str]) -> Optional[int]:
    if not argv:
        return None

    command_name = argv[0]
    if command_name.startswith("-"):
        return None
    if command_name in _get_top_level_command_names():
        return None

    short_name = command_name[3:] if command_name.startswith("hf-") else command_name
    if not short_name:
        return None

    executable_path = _get_extension_executable_path(short_name)
    if not executable_path.is_file():
        return None

    return _execute_extension_binary(executable_path=executable_path, args=argv[1:])


def main():
    if not constants.HF_DEBUG:
        logging.set_verbosity_info()
    check_cli_update("huggingface_hub")

    try:
        extension_exit_code = _dispatch_installed_extension(sys.argv[1:])
        if extension_exit_code is not None:
            sys.exit(extension_exit_code)
        app()
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        if constants.HF_DEBUG:
            traceback.print_exc()
        else:
            print(ANSI.gray("Set HF_DEBUG=1 as environment variable for full traceback."))
        sys.exit(1)
    except Exception as e:
        message = format_known_exception(e)
        if message:
            print(f"Error: {message}", file=sys.stderr)
            if constants.HF_DEBUG:
                traceback.print_exc()
            else:
                print(ANSI.gray("Set HF_DEBUG=1 as environment variable for full traceback."))
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
