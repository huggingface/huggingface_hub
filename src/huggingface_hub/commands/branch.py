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
"""Contains commands to perform branch management with the CLI.

Usage Examples:
    # Create a new branch in a repository on huggingface.co
    huggingface-cli branch create my-cool-model my-great-branch

    # List branches in a repositories on huggingface.co
    huggingface-cli branch list my-cool-model

    # Delete a branch in a repository on huggingface.co
    huggingface-cli branch delete my-cool-model my-great-branch
"""
import subprocess
from argparse import Namespace, _SubParsersAction

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
)
from huggingface_hub.hf_api import HfApi

from ..utils import HfFolder
from ._cli_utils import ANSI


class BranchCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        branch_parser = parser.add_parser(
            "branch",
            help="{create, list, delete} commands to interact with your huggingface.co repo branches.",
        )
        branch_subparsers = branch_parser.add_subparsers(help="huggingface.co repo branch related commands")
        branch_create_parser = branch_subparsers.add_parser(
            "create", help="Create a new branch for your repo on huggingface.co"
        )
        branch_create_parser.add_argument(
            "repo_id",
            type=str,
            help="The repository in which the branch will be created.",
        )
        branch_create_parser.add_argument(
            "branch",
            type=str,
            help="The name of the branch to create.",
        )
        branch_create_parser.add_argument(
            "--revision",
            type=str,
            help="The git revision to create the branch from.",
        )
        branch_create_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help=(
                'Optional: type: set to "dataset" or "space" if creating a branch in a dataset or space, default is'
                " model."
            ),
        )
        branch_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        branch_create_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        branch_create_parser.set_defaults(func=lambda args: BranchCreateCommand(args))
        branch_list_parser = branch_subparsers.add_parser("list", help="List branches of the repo on huggingface.co")
        branch_list_parser.add_argument(
            "repo_id",
            type=str,
            help="The repository for which to list branches.",
        )
        branch_list_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Type of repo to list branches for, default is model.",
        )
        branch_list_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        branch_list_parser.set_defaults(func=lambda args: BranchListCommand(args))
        branch_delete_parser = branch_subparsers.add_parser(
            "delete", help="Delete a branch for your repo on huggingface.co"
        )
        branch_delete_parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repo in which to delete the branch.",
        )
        branch_delete_parser.add_argument(
            "branch",
            type=str,
            help="The name of the branch to delete.",
        )
        branch_delete_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Type of the repo to delete the branch in, default is model.",
        )
        branch_delete_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        branch_delete_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        branch_delete_parser.set_defaults(func=lambda args: BranchDeleteCommand(args))


class BaseBranchCommand:
    def __init__(self, args: Namespace):
        self.args = args
        self._api = HfApi()
        self.token = HfFolder.get_token()
        if self.token is None:
            print("Not logged in")
            exit(1)
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")


class BranchCreateCommand(BaseBranchCommand):
    def run(self):
        try:
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

        user = self._api.whoami(self.token)["name"]
        namespace = self.args.organization if self.args.organization is not None else user

        repo_id = f"{namespace}/{self.args.repo_id}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to create branch {ANSI.bold(self.args.branch)} on {ANSI.bold(prefixed_repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            self._api.create_branch(
                repo_id=repo_id,
                branch=self.args.branch,
                revision=self.args.revision,
                token=self.token,
                repo_type=self.args.type,
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour branch now lives at:")
        print(f"  {prefixed_repo_id}/branch/{ANSI.bold(self.args.branch)}")


class BranchListCommand(BaseBranchCommand):
    def run(self):
        self.type = self.args.type
        user = self._api.whoami(self.token)["name"]
        namespace = self.args.organization if self.args.organization is not None else user

        repo_id = f"{namespace}/{self.args.repo_id}"

        try:
            refs = self._api.list_repo_refs(
                repo_id=repo_id,
                repo_type=self.type,
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour branches:")
        for branch in refs.branches:
            print(f"  {ANSI.bold(branch.name)}")
        print("")


class BranchDeleteCommand(BaseBranchCommand):
    def run(self):
        user = self._api.whoami(self.token)["name"]
        namespace = self.args.organization if self.args.organization is not None else user

        repo_id = f"{namespace}/{self.args.repo_id}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to delete branch {ANSI.bold(self.args.branch)} on {ANSI.bold(prefixed_repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            self._api.delete_branch(
                repo_id=repo_id, branch=self.args.branch, token=self.token, repo_type=self.args.type
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour branch has been deleted.")
        print("")
