"""Contains command to download a repo or file with the CLI.

Usage:
    huggingface-cli download repo_id
    huggingface-cli download repo_id filename
    huggingface-cli download repo_id --token <token> --type <repo_type> --revision <revision> --allow-patterns "<pattern>" ... --ignore-patterns "<pattern">" ... --to-local-dir <local-dir> --local-dir-use-symlinks <bool> --proxies <protocol:url> --force-download <bool> --resume-download <bool>
    huggingface-cli download repo_id filename --token <token> --type <repo_type> --revision <revision> --to-local-dir <local-dir> --local-dir-use-symlinks <bool> --proxies <protocol:url> --force-download <bool> --resume-download <bool>
"""
import os
from argparse import _SubParsersAction

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    REPO_TYPES,
)
from huggingface_hub.hf_api import HfApi

from ..utils import HfFolder


def _parse_proxy_string(proxy_str):
    protocol, url = proxy_str.split(":", 1)
    return {protocol: proxy_str}


class DownloadCommand(BaseHuggingfaceCLICommand):
    def __init__(self, args):
        self.args = args
        self._api = HfApi()

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        download_parser = parser.add_parser(
            "download",
            help="Download a repo or a repo file from huggingface.co",
        )

        download_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to download.",
        )
        download_parser.add_argument(
            "filename",
            nargs="?",
            help="Name of the file to download. (optional)",
        )
        download_parser.add_argument(
            "--token",
            type=str,
            help="Token generated from https://huggingface.co/settings/tokens",
        )
        download_parser.add_argument(
            "--type",
            type=str,
            help=(
                "The type of the repo to download. Can be one of:"
                f" {', '.join([item for item in REPO_TYPES if isinstance(item, str)])}"
            ),
        )
        download_parser.add_argument(
            "--revision",
            type=str,
            help="The revision of the repo to download.",
        )
        download_parser.add_argument(
            "--allow-patterns",
            nargs="+",
            type=str,
            help="Glob patterns to match files to download.",
        )
        download_parser.add_argument(
            "--ignore-patterns",
            nargs="+",
            type=str,
            help="Glob patterns to exclude from files to download.",
        )
        download_parser.add_argument(
            "--to-local-dir",
            type=str,
            help=(
                "The local directory to download the repo to. If not given, the repo will be downloaded to the current"
                " directory."
            ),
        )
        download_parser.add_argument(
            "--local-dir-use-symlinks",
            action="store_true",
            help="Whether to use symlinks for the downloaded files.",
        )
        download_parser.add_argument(
            "--proxies",
            type=_parse_proxy_string,
            help="A list of proxies. For example, `http://127.0.0.1:8080'`.",
        )
        download_parser.add_argument(
            "--force-download",
            action="store_true",
            help="Whether the file should be downloaded even if it already exists in the local cache.",
        )
        download_parser.add_argument(
            "--resume-download",
            action="store_true",
            help="If `True`, resume a previously interrupted download.",
        )

        download_parser.set_defaults(func=DownloadCommand)

    def run(self):
        if self.args.token:
            HfFolder.save_token(self.args.token)
        else:
            self.token = HfFolder.get_token()
        if self.token is None:
            print("Not logged in")

        if self.args.type not in REPO_TYPES:
            print(
                f"Invalid repo --type: {self.args.type}.",
                "Can be one of:",
                f"{', '.join([item for item in REPO_TYPES if isinstance(item, str)])}.",
            )
            exit(1)

        local_dir = os.path.abspath(self.args.to_local_dir or "")

        if self.args.filename:
            self._api.hf_hub_download(
                repo_id=self.args.repo_id,
                filename=self.args.filename,
                repo_type=self.args.type,
                revision=self.args.revision,
                local_dir=local_dir,
                local_dir_use_symlinks=self.args.local_dir_use_symlinks,
                proxies=self.args.proxies,
                force_download=self.args.force_download,
                resume_download=self.args.resume_download,
            )
            print(f"Successfully downloaded selected file from repo {self.args.repo_id} to {local_dir}")
        else:
            self._api.snapshot_download(
                repo_id=self.args.repo_id,
                repo_type=self.args.type,
                revision=self.args.revision,
                local_dir=local_dir,
                local_dir_use_symlinks=self.args.local_dir_use_symlinks,
                allow_patterns=self.args.allow_patterns,
                ignore_patterns=self.args.ignore_patterns,
                proxies=self.args.proxies,
                force_download=self.args.force_download,
                resume_download=self.args.resume_download,
            )
            print(f"Successfully downloaded repo {self.args.repo_id} to {local_dir}")
