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


import typer

from huggingface_hub import constants
from huggingface_hub.cli._cli_utils import GroupedTyperGroup, TyperCommandWithEpilog, check_cli_update
from huggingface_hub.cli.auth import auth_cli
from huggingface_hub.cli.cache import cache_cli
from huggingface_hub.cli.datasets import datasets_cli
from huggingface_hub.cli.download import DOWNLOAD_EPILOG, download
from huggingface_hub.cli.inference_endpoints import ie_cli
from huggingface_hub.cli.jobs import jobs_cli
from huggingface_hub.cli.lfs import lfs_enable_largefiles, lfs_multipart_upload
from huggingface_hub.cli.models import models_cli
from huggingface_hub.cli.papers import papers_cli
from huggingface_hub.cli.repo import repo_cli
from huggingface_hub.cli.repo_files import repo_files_cli
from huggingface_hub.cli.spaces import spaces_cli
from huggingface_hub.cli.system import env, version
from huggingface_hub.cli.upload import UPLOAD_EPILOG, upload
from huggingface_hub.cli.upload_large_folder import UPLOAD_LARGE_FOLDER_EPILOG, upload_large_folder
from huggingface_hub.utils import logging


app = typer.Typer(
    help="Hugging Face Hub CLI",
    add_completion=True,
    no_args_is_help=True,
    cls=GroupedTyperGroup,
    rich_markup_mode=None,
    rich_help_panel=None,
    pretty_exceptions_enable=False,
)


_HF_EPILOG = """\
EXAMPLES
  $ hf auth login
  $ hf repo create Wauplin/my-cool-model --private
  $ hf download meta-llama/Llama-3.2-1B-Instruct
  $ hf upload Wauplin/my-cool-model ./model.safetensors
  $ hf cache ls
  $ hf models ls --filter "text-generation"
  $ hf jobs ps
  $ hf jobs run python:3.12 python -c "print('Hello')"

LEARN MORE
  Use `hf <command> --help` for more information about a command.
  Read the documentation at https://huggingface.co/docs/huggingface_hub/guides/cli
"""


@app.callback(epilog=_HF_EPILOG, invoke_without_command=True)
def hf_callback(ctx: typer.Context) -> None:
    """Hugging Face Hub CLI"""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# top level single commands (defined in their respective files)
app.command(
    cls=TyperCommandWithEpilog,
    help="Download files from the Hub to local cache or a specific directory.",
    epilog=DOWNLOAD_EPILOG,
)(download)
app.command(
    cls=TyperCommandWithEpilog,
    help="Upload a file or a folder to the Hub.",
    epilog=UPLOAD_EPILOG,
)(upload)
app.command(
    cls=TyperCommandWithEpilog,
    help="Upload a large folder to the Hub. Recommended for resumable uploads.",
    epilog=UPLOAD_LARGE_FOLDER_EPILOG,
)(upload_large_folder)

app.command()(env)
app.command()(version)
app.command(hidden=False)(lfs_enable_largefiles)
app.command(hidden=True)(lfs_multipart_upload)


# command groups
app.add_typer(auth_cli, name="auth")
app.add_typer(cache_cli, name="cache")
app.add_typer(datasets_cli, name="datasets")
app.add_typer(jobs_cli, name="jobs")
app.add_typer(models_cli, name="models")
app.add_typer(papers_cli, name="papers")
app.add_typer(repo_cli, name="repo")
app.add_typer(repo_files_cli, name="repo-files")
app.add_typer(spaces_cli, name="spaces")
app.add_typer(ie_cli, name="endpoints")


def main():
    if not constants.HF_DEBUG:
        logging.set_verbosity_info()
    check_cli_update("huggingface_hub")
    app()


if __name__ == "__main__":
    main()
