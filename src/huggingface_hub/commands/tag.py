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
        $ huggingface-cli tag create my-model 1.0 --tag-message "First release"
        $ huggingface-cli tag create my-model 1.0 --tag-message "First release" --revision develop
        $ huggingface-cli tag create my-model 1.0 --tag-message "First release" --type dataset
    - List all tags:
        $ huggingface-cli tag list my-model
    - Delete a tag:
        $ huggingface-cli tag delete my-model 1.0
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
        tag_subparsers = tag_parser.add_subparsers(help="tag commands")

        tag_create_parser = tag_subparsers.add_parser("create", help="create a tag for a model in the hub")

        tag_create_parser.add_argument("repo_id", type=str, help="The repository in which a commit will be tagged.")
        tag_create_parser.add_argument("tag", type=str, help="The name of the tag to create.")
        tag_create_parser.add_argument("-m", "--tag-message", type=str, help="The description of the tag to create.")
        tag_create_parser.add_argument("--revision", type=str, help="The git revision to tag.")
        tag_create_parser.add_argument("--token", type=str, help="Authentication token.")
        tag_create_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Optional: Set to 'dataset' or 'space' if tagging a dataset or space, 'model' if tagging a model.",
        )
        tag_create_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        tag_create_parser.add_argument("--force", action="store_true", help="Optional: force tag creation.")
        tag_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        tag_create_parser.set_defaults(func=lambda args: TagCreateCommand(args))

        tag_list_parser = tag_subparsers.add_parser("list", help="list tags for a model in the hub")

        tag_list_parser.add_argument("repo_id", type=str, help="The repository for which to list tags.")
        tag_list_parser.add_argument("--token", type=str, help="Authentication token.")
        tag_list_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Optional: Set to 'dataset' or 'space' if listing tags for a dataset or space, 'model' if listing tags for a model.",
        )
        tag_list_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        tag_list_parser.set_defaults(func=lambda args: TagListCommand(args))

        tag_delete_parser = tag_subparsers.add_parser("delete", help="delete a tag for a model in the hub")

        tag_delete_parser.add_argument("repo_id", type=str, help="The repository in which a commit will be tagged.")
        tag_delete_parser.add_argument("tag", type=str, help="The name of the tag to delete.")
        tag_delete_parser.add_argument("--token", type=str, help="Authentication token.")
        tag_delete_parser.add_argument(
            "--type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Optional: Set to 'dataset' or 'space' if tag is in a dataset or space, 'model' if deleting tag for a model.",
        )
        tag_delete_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        tag_delete_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        tag_delete_parser.set_defaults(func=lambda args: TagDeleteCommand(args))


class BaseTagCommand:
    def __init__(self, args: Namespace):
        self.args = args
        self._api = HfApi()
        self.check_fields()
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

    def check_fields(self):
        self.token = self.args.token if self.args.token is not None else get_token()
        self.user = self._api.whoami(self.token)["name"]
        self.namespace = self.args.organization if self.args.organization is not None else self.user
        self.repo_id = f"{self.namespace}/{self.args.repo_id}"
        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)
        if self.token is None:
            print("Not logged in")
            exit(1)


class TagCreateCommand(BaseTagCommand):
    def run(self):
        if self.args.tag_message is None:
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
                tag_message=self.args.tag_message,
                revision=self.args.revision,
                token=self.token,
                exist_ok=self.args.force,
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


class TagListCommand(BaseTagCommand):
    def run(self):
        try:
            refs = self._api.list_repo_refs(
                repo_id=self.repo_id,
                repo_type=self.args.type,
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


class TagDeleteCommand(BaseTagCommand):
    def run(self):
        print(f"You are about to delete tag {ANSI.bold(self.args.tag)} on {ANSI.bold(self.repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            self._api.delete_tag(repo_id=self.repo_id, tag=self.args.tag, token=self.token, repo_type=self.args.type)
        except RepositoryNotFoundError:
            print(f"Repository {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except RevisionNotFoundError:
            print(f"Tag {ANSI.bold(self.args.tag)} not found on {ANSI.bold(self.repo_id)}")
            exit(1)
        print(f"Tag {ANSI.bold(self.args.tag)} deleted on {ANSI.bold(self.repo_id)}")
        print("")
