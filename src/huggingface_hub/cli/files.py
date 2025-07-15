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
"""Contains commands for file management.

Usage:
    hf files download gpt2 config.json
    hf files upload my-model ./config.json
    hf files delete my-model config.json
    hf download gpt2 config.json  # alias
    hf upload my-model ./config.json  # alias
"""

from argparse import _SubParsersAction

from huggingface_hub.cli import BaseHfCLICommand
from huggingface_hub.commands.download import DownloadCommand as OldDownloadCommand
from huggingface_hub.commands.repo_files import DeleteFilesSubCommand as OldDeleteFilesSubCommand
from huggingface_hub.commands.upload import UploadCommand as OldUploadCommand


class FilesCommand(BaseHfCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        files_parser = parser.add_parser("files", help="Files commands")
        files_subparsers = files_parser.add_subparsers(help="Files subcommands")

        # Download command
        download_parser = files_subparsers.add_parser("download", help="Download files from the Hub")
        download_parser.add_argument("repo_id", type=str, help="Repository ID")
        download_parser.add_argument("filename", type=str, nargs="?", help="Filename to download")
        download_parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of the repository")
        download_parser.add_argument("--revision", type=str, help="Git revision to download from")
        download_parser.add_argument("--include", type=str, action="append", help="Patterns to include")
        download_parser.add_argument("--exclude", type=str, action="append", help="Patterns to exclude")
        download_parser.add_argument("--cache-dir", type=str, help="Cache directory")
        download_parser.add_argument("--local-dir", type=str, help="Local directory to download to")
        download_parser.add_argument("--local-dir-use-symlinks", action="store_true", help="Use symlinks in local dir")
        download_parser.add_argument("--token", type=str, help="Hugging Face token")
        download_parser.add_argument("--quiet", action="store_true", help="Quiet mode")
        download_parser.set_defaults(func=lambda args: FilesDownloadCommand(args))

        # Upload command
        upload_parser = files_subparsers.add_parser("upload", help="Upload a file or folder to the Hub")
        upload_parser.add_argument("repo_id", type=str, help="Repository ID")
        upload_parser.add_argument("local_path", type=str, help="Local path to upload")
        upload_parser.add_argument("path_in_repo", type=str, nargs="?", help="Path in the repository")
        upload_parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of the repository")
        upload_parser.add_argument("--revision", type=str, help="Git revision to upload to")
        upload_parser.add_argument("--commit-message", type=str, help="Commit message")
        upload_parser.add_argument("--commit-description", type=str, help="Commit description")
        upload_parser.add_argument("--private", action="store_true", help="Whether the repository should be private")
        upload_parser.add_argument("--token", type=str, help="Hugging Face token")
        upload_parser.add_argument("--quiet", action="store_true", help="Quiet mode")
        upload_parser.add_argument("--include", type=str, action="append", help="Patterns to include")
        upload_parser.add_argument("--exclude", type=str, action="append", help="Patterns to exclude")
        upload_parser.add_argument("--delete", type=str, action="append", help="Patterns to delete")
        upload_parser.set_defaults(func=lambda args: FilesUploadCommand(args))

        # Delete command
        delete_parser = files_subparsers.add_parser("delete", help="Delete files from the Hub")
        delete_parser.add_argument("repo_id", type=str, help="Repository ID")
        delete_parser.add_argument("path_in_repo", type=str, help="Path in the repository")
        delete_parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of the repository")
        delete_parser.add_argument("--revision", type=str, help="Git revision to delete from")
        delete_parser.add_argument("--commit-message", type=str, help="Commit message")
        delete_parser.add_argument("--commit-description", type=str, help="Commit description")
        delete_parser.add_argument("--token", type=str, help="Hugging Face token")
        delete_parser.set_defaults(func=lambda args: FilesDeleteCommand(args))

        # Add direct aliases at the parent level
        # These will be added by the UtilsCommand to maintain the same behavior as existing CLI

    def run(self):
        # This should never be called since we have subcommands
        raise NotImplementedError("Files command requires a subcommand")


class FilesDownloadCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent FilesCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldDownloadCommand(self.args)
        old_command.run()


class FilesUploadCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent FilesCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldUploadCommand(self.args)
        old_command.run()


class FilesDeleteCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent FilesCommand
        pass

    def run(self):
        # Reuse the existing implementation  
        old_command = OldDeleteFilesSubCommand(self.args)
        old_command.run()