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
"""Contains command to upload a repo or file with the CLI.

Usage:
    huggingface-cli upload repo_id
    huggingface-cli upload repo_id [path] [path-in-repo]
"""
import os
import warnings
from argparse import Namespace, _SubParsersAction
from typing import List, Optional

from huggingface_hub import HfApi
from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars


class UploadCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        upload_parser = parser.add_parser(
            "upload",
            help="Upload a repo or a repo file to huggingface.co",
        )
        upload_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to upload to (e.g. `username/repo-name`).",
        )
        upload_parser.add_argument(
            "path",
            nargs="?",
            help="Local path. (optional)",
        )
        upload_parser.add_argument(
            "path_in_repo",
            nargs="?",
            help="Path in repo. (optional)",
        )
        upload_parser.add_argument(
            "--repo-type",
            type=str,
            help="The type of the repo to upload to (e.g. `dataset`).",
        )
        upload_parser.add_argument(
            "--revision",
            type=str,
            help="The revision of the repo to upload.",
        )
        upload_parser.add_argument(
            "--include",
            nargs="+",
            type=str,
            help="Glob patterns to match files to upload.",
        )
        upload_parser.add_argument(
            "--exclude",
            nargs="+",
            type=str,
            help="Glob patterns to exclude from files to upload.",
        )
        upload_parser.add_argument(
            "--delete",
            nargs="+",
            type=str,
            help="Glob patterns for file to be deleted from the repo while committing.",
        )
        upload_parser.add_argument(
            "--commit-message",
            type=str,
            help="The summary / title / first line of the generated commit.",
        )
        upload_parser.add_argument(
            "--commit-description",
            type=str,
            help="The description of the generated commit.",
        )
        upload_parser.add_argument(
            "--create-pr",
            action="store_true",
            help="Whether to create a PR.",
        )
        upload_parser.add_argument(
            "--every",
            action="store_true",
            help="Run a CommitScheduler instead of a single commit.",
        )
        upload_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        upload_parser.add_argument(
            "--quiet",
            action="store_true",
            help="If True, progress bars are disabled and only the path to the uploaded files is printed.",
        )
        upload_parser.set_defaults(func=UploadCommand)

    def __init__(self, args: Namespace) -> None:
        self.api = HfApi(token=args.token)
        self.repo_id: str = args.repo_id
        self.path: str = args.path
        self.path_in_repo: str = args.path_in_repo
        self.repo_type: Optional[str] = args.repo_type
        self.revision: Optional[str] = args.revision
        self.include: List[str] = args.include
        self.exclude: List[str] = args.exclude
        self.delete: List[str] = args.delete
        self.commit_message: Optional[str] = args.commit_message
        self.commit_description: Optional[str] = args.commit_description
        self.create_pr: bool = args.create_pr
        self.every: bool = args.every
        self.token: Optional[str] = args.token
        self.quiet: bool = args.quiet

    def run(self) -> None:
        if self.quiet:
            disable_progress_bars()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(self._upload())  # Print path to uploaded files
            enable_progress_bars()
        else:
            print(self._upload())  # Print path to uploaded files

    def _upload(self) -> str:
        self.path = "." if self.path is None else self.path

        self.path_in_repo = (
            self.path_in_repo
            if self.path_in_repo
            else (os.path.relpath(self.path).replace("\\", "/") if self.path != "." else "/")
        )

        # File or Folder based uploading
        if os.path.isfile(self.path):
            if self.include or self.exclude or self.delete:
                raise ValueError("--include / --exclude / --delete cannot be used with a file path.")

            return self.api.upload_file(
                path_or_fileobj=self.path,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                token=self.token,
                repo_type=self.repo_type,
                revision=self.revision,
                commit_message=self.commit_message,
                commit_description=self.commit_description,
                create_pr=self.create_pr,
                run_as_future=self.every,
            )

        elif os.path.isdir(self.path):
            return self.api.upload_folder(
                folder_path=self.path,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                token=self.token,
                repo_type=self.repo_type,
                revision=self.revision,
                commit_message=self.commit_message,
                commit_description=self.commit_description,
                create_pr=self.create_pr,
                allow_patterns=self.include,
                ignore_patterns=self.exclude,
                delete_patterns=self.delete,
                run_as_future=self.every,
            )

        else:
            raise ValueError(f"Provided PATH: {self.path} does not exist.")
