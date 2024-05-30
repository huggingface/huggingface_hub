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
    huggingface-cli repo-files <repo_id> delete *

    # delete single file
    huggingface-cli repo-files <repo_id> delete file.txt

    # delete single folder
    huggingface-cli repo-files <repo_id> delete folder/

    # delete multiple
    huggingface-cli repo-files <repo_id> delete file.txt folder/ file2.txt

    # delete multiple patterns
    huggingface-cli repo-files <repo_id> delete file.txt *.json folder/*.parquet

    # delete from different revision / repo-type
    huggingface-cli repo-files <repo_id> delete file.txt --revision=refs/pr/1 --repo-type=dataset
"""

from argparse import _SubParsersAction
from typing import List, Optional

from huggingface_hub import logging
from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.hf_api import HfApi


logger = logging.get_logger(__name__)


class DeleteFilesSubCommand:
    def __init__(self, args):
        self.args = args
        self.repo_id: str = args.repo_id
        self.token: str = args.token
        self.repo_type: Optional[str] = args.repo_type
        self.revision: Optional[str] = args.revision
        self.api: HfApi = HfApi(token=args.token, library_name="huggingface-cli")
        self.patterns: List[str] = args.patterns

    def run(self) -> None:
        logging.set_verbosity_info()
        print(self._delete())
        logging.set_verbosity_warning()

    def _delete(self):
        return self.api.delete_files(
            patterns=self.patterns,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            revision=self.revision,
        )


class RepoFilesCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        repo_files_parser = parser.add_parser("repo-files", help="Manage files in a repo on the Hub")
        repo_files_parser.add_argument(
            "repo_id", type=str, help="The ID of the repo to manage (e.g. `username/repo-name`)."
        )
        repo_files_subparsers = repo_files_parser.add_subparsers(
            help="Action to execute against the files.",
            required=True,
        )
        delete_subparser = repo_files_subparsers.add_parser(
            "delete",
            help="Delete files from a repo on the Hub",
        )
        delete_subparser.set_defaults(func=lambda args: DeleteFilesSubCommand(args))
        delete_subparser.add_argument(
            "patterns",
            nargs="+",
            type=str,
            help="Glob patterns to match files to delete.",
        )
        delete_subparser.add_argument(
            "--repo-type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Type of the repo to upload to (e.g. `dataset`).",
        )
        delete_subparser.add_argument(
            "--revision",
            type=str,
            help=(
                "An optional Git revision to push to. It can be a branch name "
                "or a PR reference. If revision does not"
                " exist and `--create-pr` is not set, a branch will be automatically created."
            ),
        )
        repo_files_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
            required=False,
        )

        repo_files_parser.set_defaults(func=RepoFilesCommand)
