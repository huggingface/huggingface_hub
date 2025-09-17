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

from huggingface_hub.cli.auth import auth_cli
from huggingface_hub.cli.cache import cache_cli
from huggingface_hub.cli.download import download
from huggingface_hub.cli.jobs import jobs_cli
from huggingface_hub.cli.lfs import lfs_enable_largefiles, lfs_multipart_upload
from huggingface_hub.cli.repo import repo_cli
from huggingface_hub.cli.repo_files import repo_files_cli
from huggingface_hub.cli.system import env, version

# from huggingface_hub.cli.jobs import jobs_app
from huggingface_hub.cli.upload import upload
from huggingface_hub.cli.upload_large_folder import upload_large_folder
from huggingface_hub.utils import logging


app = typer.Typer(add_completion=False, no_args_is_help=True, help="Hugging Face Hub CLI", rich_markup_mode=None)


# top level single commands (defined in their respective files)
app.command(
    name="download",
    help="Download files from the Hub.",
)(download)
app.command(
    name="upload",
    help="Upload a file or a folder to the Hub.",
)(upload)
app.command(
    name="upload-large-folder",
    help="Upload a large folder to the Hub. Recommended for resumable uploads.",
)(upload_large_folder)
app.command(
    name="env",
    help="Print information about the environment.",
)(env)
app.command(
    name="version",
    help="Print information about the hf version.",
)(version)
app.command(
    name="lfs-enable-largefiles",
    help="Configure your repository to enable upload of files > 5GB.",
    hidden=True,
)(lfs_enable_largefiles)
app.command(
    name="lfs-multipart-upload",
    help="Upload large files to the Hub.",
    hidden=True,
)(lfs_multipart_upload)


# command groups
app.add_typer(auth_cli, name="auth")
app.add_typer(cache_cli, name="cache")
app.add_typer(repo_cli, name="repo")
app.add_typer(repo_files_cli, name="repo-files")
app.add_typer(jobs_cli, name="jobs")


def main():
    logging.set_verbosity_info()
    app()


if __name__ == "__main__":
    main()
