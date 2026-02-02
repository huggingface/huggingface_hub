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

from huggingface_hub import constants
from huggingface_hub.cli._cli_utils import (
    TyperCommandWithEpilog,
    TyperHelpTopicCommand,
    check_cli_update,
    generate_epilog,
    typer_factory,
)
from huggingface_hub.cli._errors import format_known_exception
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
from huggingface_hub.cli.skills import skills_cli
from huggingface_hub.cli.spaces import spaces_cli
from huggingface_hub.cli.system import env, version
from huggingface_hub.cli.upload import UPLOAD_EPILOG, upload
from huggingface_hub.cli.upload_large_folder import UPLOAD_LARGE_FOLDER_EPILOG, upload_large_folder
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import ANSI, logging


app = typer_factory(
    help="Hugging Face Hub CLI",
    epilog=generate_epilog(
        examples=[
            "hf auth login",
            "hf repo create Wauplin/my-cool-model --private",
            "hf download meta-llama/Llama-3.2-1B-Instruct",
            "hf upload Wauplin/my-cool-model ./model.safetensors",
            "hf cache ls",
            'hf models ls --filter "text-generation"',
            "hf jobs ps",
            "hf jobs run python:3.12 python -c \"print('Hello')\"",
        ],
    ),
    grouped=True,
)


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
app.command(name="env", cls=TyperHelpTopicCommand, help="Print information about the environment.")(env)
app.command(cls=TyperHelpTopicCommand, help="Print information about the hf version.")(version)
app.command(help="Configure your repository to enable upload of files > 5GB.", hidden=True)(lfs_enable_largefiles)
app.command(help="Upload large files to the Hub.", hidden=True)(lfs_multipart_upload)


# command groups
app.add_typer(auth_cli, name="auth")
app.add_typer(cache_cli, name="cache")
app.add_typer(datasets_cli, name="datasets")
app.add_typer(jobs_cli, name="jobs")
app.add_typer(models_cli, name="models")
app.add_typer(papers_cli, name="papers")
app.add_typer(repo_cli, name="repo")
app.add_typer(repo_files_cli, name="repo-files")
app.add_typer(skills_cli, name="skills")
app.add_typer(spaces_cli, name="spaces")
app.add_typer(ie_cli, name="endpoints")


def main():
    if not constants.HF_DEBUG:
        logging.set_verbosity_info()
    check_cli_update("huggingface_hub")

    try:
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
