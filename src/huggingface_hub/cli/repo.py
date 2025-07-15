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
"""Contains commands for repository management.

Usage:
    hf repo create my-model
    hf repo tag create my-model v1.0 --message="First release"
    hf repo tag delete my-model v1.0
    hf repo tag list my-model
"""

from argparse import _SubParsersAction

from huggingface_hub.cli import BaseHfCLICommand
from huggingface_hub.commands.repo import RepoCreateCommand as OldRepoCreateCommand
from huggingface_hub.commands.tag import TagCreateCommand as OldTagCreateCommand
from huggingface_hub.commands.tag import TagDeleteCommand as OldTagDeleteCommand
from huggingface_hub.commands.tag import TagListCommand as OldTagListCommand


class RepoCommand(BaseHfCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        repo_parser = parser.add_parser("repo", help="Repository commands")
        repo_subparsers = repo_parser.add_subparsers(help="Repo subcommands")

        # Create command
        create_parser = repo_subparsers.add_parser("create", help="Create a new repository")
        create_parser.add_argument("name", type=str, help="Name of the repository to create")
        create_parser.add_argument("--type", type=str, default="model", choices=["model", "dataset", "space"], help="Type of the repository")
        create_parser.add_argument("--organization", type=str, help="Organization under which to create the repository")
        create_parser.add_argument("--private", action="store_true", help="Whether the repository should be private")
        create_parser.add_argument("--sdk", type=str, help="SDK to use for the repository (if space)")
        create_parser.add_argument("--template", type=str, help="Template to use for the repository (if space)")
        create_parser.add_argument("--yes", action="store_true", help="Assume yes to all prompts")
        create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))

        # Tag commands
        tag_parser = repo_subparsers.add_parser("tag", help="Tag commands")
        tag_subparsers = tag_parser.add_subparsers(help="Tag subcommands")

        # Tag create
        tag_create_parser = tag_subparsers.add_parser("create", help="Create a new tag")
        tag_create_parser.add_argument("repo_id", type=str, help="Repository ID")
        tag_create_parser.add_argument("tag", type=str, help="Tag name")
        tag_create_parser.add_argument("--message", type=str, help="Tag message")
        tag_create_parser.add_argument("--revision", type=str, help="Git revision to tag")
        tag_create_parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of the repository")
        tag_create_parser.add_argument("--token", type=str, help="Hugging Face token")
        tag_create_parser.set_defaults(func=lambda args: RepoTagCreateCommand(args))

        # Tag delete
        tag_delete_parser = tag_subparsers.add_parser("delete", help="Delete a tag")
        tag_delete_parser.add_argument("repo_id", type=str, help="Repository ID")
        tag_delete_parser.add_argument("tag", type=str, help="Tag name")
        tag_delete_parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of the repository")
        tag_delete_parser.add_argument("--token", type=str, help="Hugging Face token")
        tag_delete_parser.set_defaults(func=lambda args: RepoTagDeleteCommand(args))

        # Tag list
        tag_list_parser = tag_subparsers.add_parser("list", help="List tags")
        tag_list_parser.add_argument("repo_id", type=str, help="Repository ID")
        tag_list_parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of the repository")
        tag_list_parser.add_argument("--token", type=str, help="Hugging Face token")
        tag_list_parser.set_defaults(func=lambda args: RepoTagListCommand(args))

    def run(self):
        # This should never be called since we have subcommands
        raise NotImplementedError("Repo command requires a subcommand")


class RepoCreateCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent RepoCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldRepoCreateCommand(self.args)
        old_command.run()


class RepoTagCreateCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent RepoCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldTagCreateCommand(self.args)
        old_command.run()


class RepoTagDeleteCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent RepoCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldTagDeleteCommand(self.args)
        old_command.run()


class RepoTagListCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent RepoCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldTagListCommand(self.args)
        old_command.run()