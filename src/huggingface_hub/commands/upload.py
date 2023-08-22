"""Contains command to upload a repo or file with the CLI.
Usage:
    huggingface-cli upload repo_id
    huggingface-cli upload repo_id [path] [path-in-repo]
"""
import os
from argparse import _SubParsersAction

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    REPO_TYPES,
)
from huggingface_hub.hf_api import HfApi

from ..utils import HfFolder


class UploadCommand(BaseHuggingfaceCLICommand):
    def __init__(self, args):
        self.args = args
        self._api = HfApi()

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        upload_parser = parser.add_parser(
            "upload",
            help="Upload a repo or a repo file to huggingface.co",
        )

        upload_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to upload.",
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
            "--token",
            type=str,
            help="Token generated from https://huggingface.co/settings/tokens",
        )
        upload_parser.add_argument(
            "--type",
            type=str,
            help=(
                "The type of the repo to upload. Can be one of:"
                f" {', '.join([item for item in REPO_TYPES if isinstance(item, str)])}"
            ),
        )
        upload_parser.add_argument(
            "--revision",
            type=str,
            help="The revision of the repo to upload.",
        )
        upload_parser.add_argument(
            "--allow-patterns",
            nargs="+",
            type=str,
            help="Glob patterns to match files to upload.",
        )
        upload_parser.add_argument(
            "--ignore-patterns",
            nargs="+",
            type=str,
            help="Glob patterns to exclude from files to upload.",
        )
        upload_parser.add_argument(
            "--delete-patterns",
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

        upload_parser.set_defaults(func=UploadCommand)

    def run(self):
        if self.args.token:
            self.token = self.args.token
            HfFolder.save_token(self.args.token)
        else:
            self.token = HfFolder.get_token()

        if self.token is None:
            raise ValueError("Not logged in or token is not provided. Consider running `huggingface-cli login`.")

        if self.args.type not in REPO_TYPES:
            raise ValueError(
                f"Invalid repo --type: {self.args.type}. "
                f"Can be one of: {', '.join([item for item in REPO_TYPES if isinstance(item, str)])}."
            )

        self.path = "." if self.args.path is None else self.args.path

        self.path_in_repo = (
            self.args.path_in_repo
            if self.args.path_in_repo
            else (os.path.relpath(self.path).replace("\\", "/") if self.path != "." else "/")
        )

        # File or Folder based uploading
        if os.path.isfile(self.path):
            if self.args.allow_patterns or self.args.ignore_patterns:
                raise ValueError("--allow-patterns / --ignore-patterns cannot be used with a file path.")

            self._api.upload_file(
                path_or_fileobj=self.path,
                path_in_repo=self.path_in_repo,
                repo_id=self.args.repo_id,
                token=self.token,
                repo_type=self.args.type,
                revision=self.args.revision,
                commit_message=self.args.commit_message,
                commit_description=self.args.commit_description,
                create_pr=self.args.create_pr,
                run_as_future=self.args.every,
            )
            print(f"Successfully uploaded selected file to repo {self.args.repo_id}")

        elif os.path.isdir(self.path):
            self._api.upload_folder(
                folder_path=self.path,
                path_in_repo=self.path_in_repo,
                repo_id=self.args.repo_id,
                token=self.token,
                repo_type=self.args.type,
                revision=self.args.revision,
                commit_message=self.args.commit_message,
                commit_description=self.args.commit_description,
                create_pr=self.args.create_pr,
                allow_patterns=self.args.allow_patterns,
                ignore_patterns=self.args.ignore_patterns,
                delete_patterns=self.args.delete_patterns,
                run_as_future=self.args.every,
            )
            print(f"Successfully uploaded selected folder to repo {self.args.repo_id}")

        else:
            raise ValueError(f"Provided PATH: {self.args.path} does not exist.")
