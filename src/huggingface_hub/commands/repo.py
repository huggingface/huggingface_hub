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
"""Contains commands to perform repo management with the CLI.

Usage Examples:
    # Create a new repository on huggingface.co
    huggingface-cli repo create my-cool-model

    # List repositories on huggingface.co
    huggingface-cli repo list

    # Delete a repository on huggingface.co
    huggingface-cli repo delete my-cool-model

    # Toggle the visibility of a repository to private
    huggingface-cli repo toggle my-cool-model private
"""
import subprocess
from argparse import Namespace, _SubParsersAction

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
)
from huggingface_hub.hf_api import HfApi

from ..utils import HfFolder
from ._cli_utils import ANSI


class RepoCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        repo_parser = parser.add_parser(
            "repo",
            help=(
                "{create, ls-files, list, delete, toggle visibility} commands to interact with your huggingface.co"
                " repos."
            ),
        )
        repo_subparsers = repo_parser.add_subparsers(help="huggingface.co repos related commands")
        repo_create_parser = repo_subparsers.add_parser("create", help="Create a new repo on huggingface.co")
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your repo. Will be namespaced under your username to build the repo id.",
        )
        repo_create_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help='Optional: type: set to "dataset" or "space" if creating a dataset or space, default is model.',
        )
        repo_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_create_parser.add_argument(
            "--space_sdk",
            type=str,
            help='Optional: Hugging Face Spaces SDK type. Required when --type is set to "space".',
            choices=SPACES_SDK_TYPES,
        )
        repo_create_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))
        repo_list_parser = repo_subparsers.add_parser("list", help="List repos on huggingface.co")
        repo_list_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Type of repos to list, default is model.",
        )
        repo_list_parser.add_argument(
            "--author",
            type=str,
            help="Optional: author namespace.",
        )
        repo_list_parser.add_argument(
            "--search",
            type=str,
            help="Optional: A string that will be contained in the returned repo ids.",
        )
        repo_list_parser.set_defaults(func=lambda args: RepoListCommand(args))
        repo_delete_parser = repo_subparsers.add_parser("delete", help="Delete a repo on huggingface.co")
        repo_delete_parser.add_argument(
            "name",
            type=str,
            help="Name for your repo.",
        )
        repo_delete_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Type of repos to list, default is model.",
        )
        repo_delete_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_delete_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        repo_delete_parser.set_defaults(func=lambda args: RepoDeleteCommand(args))
        repo_toggle_parser = repo_subparsers.add_parser(
            "toggle", help="Toggle a repo on huggingface.co private or public"
        )
        repo_toggle_parser.add_argument(
            "name",
            type=str,
            help="Name for your repo.",
        )
        repo_toggle_parser.add_argument(
            "private",
            choices=["public", "private"],
            default="public",
            help="Name for your repo.",
        )
        repo_toggle_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Type of repos to list, default is model.",
        )
        repo_toggle_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_toggle_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        repo_toggle_parser.set_defaults(func=lambda args: RepoToggleCommand(args))


class BaseRepoCommand:
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


class RepoCreateCommand(BaseRepoCommand):
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

        repo_id = f"{namespace}/{self.args.name}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to create {ANSI.bold(prefixed_repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            url = self._api.create_repo(
                repo_id=repo_id,
                token=self.token,
                repo_type=self.args.type,
                space_sdk=self.args.space_sdk,
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour repo now lives at:")
        print(f"  {ANSI.bold(url)}")
        print("\nYou can clone it locally with the command below, and commit/push as usual.")
        print(f"\n  git clone {url}")
        print("")


class RepoListCommand(BaseRepoCommand):
    def run(self):
        self.type = self.args.type
        self.author = self.args.author
        self.search = self.args.search
        try:
            if self.type is None or self.type == "model":
                repos = self._api.list_models(token=self.token, author=self.author, search=self.search)
            elif self.type == "dataset":
                repos = self._api.list_datasets(token=self.token, author=self.author, search=self.search)
            elif self.type == "space":
                repos = self._api.list_spaces(token=self.token, author=self.author, search=self.search)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour repos:")
        for repo in repos:
            print(f"  {ANSI.bold(repo.id)}")
        print("")


class RepoDeleteCommand(BaseRepoCommand):
    def run(self):
        user = self._api.whoami(self.token)["name"]
        namespace = self.args.organization if self.args.organization is not None else user

        repo_id = f"{namespace}/{self.args.name}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to delete {ANSI.bold(prefixed_repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            self._api.delete_repo(repo_id=repo_id, token=self.token, repo_type=self.args.type)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour repo has been deleted.")
        print("")


class RepoToggleCommand(BaseRepoCommand):
    def run(self):
        self.private = self.args.private
        user = self._api.whoami(self.token)["name"]
        namespace = self.args.organization if self.args.organization is not None else user

        repo_id = f"{namespace}/{self.args.name}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to toggle {ANSI.bold(prefixed_repo_id)} to {self.private}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
            self.privateBool = False if self.private == "public" else True
        try:
            self._api.update_repo_visibility(
                repo_id=repo_id, private=self.privateBool, token=self.token, repo_type=self.args.type
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print(f"\nYour repo has been toggled to {self.private}.")
        print("")
