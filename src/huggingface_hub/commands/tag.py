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

"""Contains commands to perform tag management with the CLI.

Usage Examples:
    - Create a tag:
        $ huggingface-cli tag user/my-model 1.0 --message "First release"
        $ huggingface-cli tag user/my-model 1.0 -m "First release" --revision develop
        $ huggingface-cli tag user/my-dataset 1.0 -m "First release" --repo-type dataset
    - List all tags:
        $ huggingface-cli tag -l user/my-model
        $ huggingface-cli tag --list user/my-dataset --repo-type dataset
    - Delete a tag:
        $ huggingface-cli tag -d user/my-model 1.0
        $ huggingface-cli tag --delete user/my-dataset 1.0 --repo-type dataset
"""

import subprocess
from argparse import Namespace, _SubParsersAction

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    REPO_TYPES,
)
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import get_token

from ..utils import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from ._cli_utils import ANSI


class TagCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        tag_parser = parser.add_parser("tag", help="(create, list, delete) tags for a model in the hub")

        tag_parser.add_argument(
            "repo_id", type=str, help="The repository (model, dataset, or space) for the operation."
        )
        tag_parser.add_argument("tag", nargs="?", type=str, help="The name of the tag for creation or deletion.")
        tag_parser.add_argument("-m", "--message", type=str, help="The description of the tag to create.")
        tag_parser.add_argument("--revision", type=str, help="The git revision to tag.")
        tag_parser.add_argument("--token", type=str, help="Authentication token.")
        tag_parser.add_argument(
            "--repo-type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Set the type of repository (model, dataset, or space).",
        )
        tag_parser.add_argument("-y", "--yes", action="store_true", help="Answer Yes to prompts automatically.")
        tag_parser.add_argument("--force", action="store_true", help="Force tag creation or deletion.")

        tag_parser.add_argument("-l", "--list", action="store_true", help="List tags for a repository.")
        tag_parser.add_argument("-d", "--delete", action="store_true", help="Delete a tag for a repository.")

        tag_parser.set_defaults(func=lambda args: handle_commands(args))


def handle_commands(args: Namespace):
    if args.list:
        return TagListCommand(args)
    elif args.delete:
        return TagDeleteCommand(args)
    else:
        return TagCreateCommand(args)


class TagCommand:
    def __init__(self, args: Namespace):
        self.args = args
        self._api = HfApi()
        self.token = self.args.token if self.args.token is not None else get_token()
        self.user = self._api.whoami(self.token)["name"]
        self.repo_id = self.args.repo_id
        self.check_git_installed()
        self.check_fields()

    def check_git_installed(self):
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

    def check_fields(self):
        self.token = self.args.token if self.args.token is not None else get_token()
        self.user = self._api.whoami(self.token)["name"]
        if self.args.repo_type not in REPO_TYPES:
            print("Invalid repo --repo-type")
            exit(1)
        if self.token is None:
            print("Not logged in")
            exit(1)


class TagCreateCommand(TagCommand):
    def run(self):
        if self.args.message is None:
            print("Tag message cannot be empty. Please provide a message with `--tag-message`.")
            exit(1)

        print(f"You are about to create tag {ANSI.bold(self.args.tag)} on {ANSI.bold(self.repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            self._api.create_tag(
                repo_id=self.repo_id,
                tag=self.args.tag,
                tag_message=self.args.message,
                revision=self.args.revision,
                token=self.token,
                exist_ok=self.args.force,
                repo_type=self.args.repo_type,
            )
        except RepositoryNotFoundError:
            print(f"Repository {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except RevisionNotFoundError:
            print(f"Revision {ANSI.bold(self.args.revision)} not found.")
            exit(1)
        except HfHubHTTPError as e:
            if e.response.status_code == 409:
                print(f"Tag {ANSI.bold(self.args.tag)} already exists on {ANSI.bold(self.repo_id)}")
                print("Use `--force` to overwrite the existing tag.")
                exit(1)
            raise e

        print(f"Tag {ANSI.bold(self.args.tag)} created on {ANSI.bold(self.repo_id)}")
        print("")


class TagListCommand(TagCommand):
    def run(self):
        try:
            refs = self._api.list_repo_refs(
                repo_id=self.repo_id,
                repo_type=self.args.repo_type,
            )
        except RepositoryNotFoundError:
            print(f"Repository {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        if len(refs.tags) == 0:
            print("  No tags found")
            exit(0)
        print("\nYour tags:")
        for tag in refs.tags:
            print(f"  {ANSI.bold(tag.name)}")
        print("")


class TagDeleteCommand(TagCommand):
    def run(self):
        print(f"You are about to delete tag {ANSI.bold(self.args.tag)} on {ANSI.bold(self.repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            self._api.delete_tag(
                repo_id=self.repo_id, tag=self.args.tag, token=self.token, repo_type=self.args.repo_type
            )
        except RepositoryNotFoundError:
            print(f"Repository {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except RevisionNotFoundError:
            print(f"Tag {ANSI.bold(self.args.tag)} not found on {ANSI.bold(self.repo_id)}")
            exit(1)
        print(f"Tag {ANSI.bold(self.args.tag)} deleted on {ANSI.bold(self.repo_id)}")
        print("")
