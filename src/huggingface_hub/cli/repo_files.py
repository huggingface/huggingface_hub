# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
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
"""Contains command to update or delete files in a repository using the CLI.

Usage:
    # delete all
    hf repo-files delete <repo_id> "*"

    # delete single file
    hf repo-files delete <repo_id> file.txt

    # delete single folder
    hf repo-files delete <repo_id> folder/

    # delete multiple
    hf repo-files delete <repo_id> file.txt folder/ file2.txt

    # delete multiple patterns
    hf repo-files delete <repo_id> file.txt "*.json" "folder/*.parquet"

    # delete from different revision / repo-type
    hf repo-files delete <repo_id> file.txt --revision=refs/pr/1 --repo-type=dataset
"""

from typing import Optional

import typer

from huggingface_hub import logging
from huggingface_hub.hf_api import HfApi

from ._cli_utils import validate_repo_type


logger = logging.get_logger(__name__)


repo_files_app = typer.Typer(help="Manage files in a repo on the Hub.")


@repo_files_app.command("delete")
def repo_files_delete(
    repo_id: str = typer.Argument(
        ...,
        help="The ID of the repo (e.g. username/repo-name).",
    ),
    patterns: list[str] = typer.Argument(
        ...,
        help="Glob patterns to match files to delete.",
    ),
    repo_type: Optional[str] = typer.Option(
        "model",
        "--repo-type",
        help="Type of the repo to upload to (e.g. `dataset`).",
    ),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        help="An optional Git revision to push to. It can be a branch name or a PR reference. If revision does not exist and `--create-pr` is not set, a branch will be automatically created.",
    ),
    commit_message: Optional[str] = typer.Option(
        None, "--commit-message", help="The summary / title / first line of the generated commit."
    ),
    commit_description: Optional[str] = typer.Option(
        None, "--commit-description", help="The description of the generated commit."
    ),
    create_pr: bool = typer.Option(
        False, "--create-pr", help="Whether to create a new Pull Request for these changes."
    ),
    token: Optional[str] = typer.Option(
        None, "--token", help="A User Access Token generated from https://huggingface.co/settings/tokens"
    ),
) -> None:
    logging.set_verbosity_info()
    repo_type = validate_repo_type(repo_type)
    api = HfApi(token=token, library_name="hf")
    url = api.delete_files(
        delete_patterns=patterns,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
    )
    print(f"Files correctly deleted from repo. Commit: {url}.")
    logging.set_verbosity_warning()
