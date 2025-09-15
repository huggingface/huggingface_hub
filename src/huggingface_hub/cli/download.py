# coding=utf-8
# Copyright 202-present, the HuggingFace Inc. team.
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
"""Contains command to download files from the Hub with the CLI.

Usage:
    hf download --help

    # Download file
    hf download gpt2 config.json

    # Download entire repo
    hf download fffiloni/zeroscope --repo-type=space --revision=refs/pr/78

    # Download repo with filters
    hf download gpt2 --include="*.safetensors"

    # Download with token
    hf download Wauplin/private-model --token=hf_***

    # Download quietly (no progress bar, no warnings, only the returned path)
    hf download gpt2 config.json --quiet

    # Download to local dir
    hf download gpt2 --local-dir=./models/gpt2
"""

import warnings
from typing import Optional

import typer

from huggingface_hub import logging
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from ._cli_utils import validate_repo_type


logger = logging.get_logger(__name__)


def download(
    repo_id: str = typer.Argument(
        ...,
        help="ID of the repo to download from (e.g. `username/repo-name`).",
    ),
    filenames: Optional[list[str]] = typer.Argument(
        None,
        help="Files to download (e.g. `config.json`, `data/metadata.jsonl`).",
    ),
    repo_type: Optional[str] = typer.Option(
        "model",
        "--repo-type",
        help="Type of repo to download from.",
    ),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        help="Git revision id which can be a branch name, a tag, or a commit hash.",
    ),
    include: Optional[list[str]] = typer.Option(
        None,
        "--include",
        help="Glob patterns to include from files to download. eg: *.json",
    ),
    exclude: Optional[list[str]] = typer.Option(
        None,
        "--exclude",
        help="Glob patterns to exclude from files to download.",
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Directory where to save files.",
    ),
    local_dir: Optional[str] = typer.Option(
        None,
        "--local-dir",
        help="If set, the downloaded file will be placed under this directory. Check out https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-local-folder for more details.",
    ),
    force_download: Optional[bool] = typer.Option(
        False,
        "--force-download",
        help="If True, the files will be downloaded even if they are already cached.",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="A User Access Token generated from https://huggingface.co/settings/tokens",
    ),
    quiet: Optional[bool] = typer.Option(
        False,
        "--quiet",
        help="If True, progress bars are disabled and only the path to the download files is printed.",
    ),
    max_workers: Optional[int] = typer.Option(
        8,
        "--max-workers",
        help="Maximum number of workers to use for downloading files. Default is 8.",
    ),
) -> None:
    """Download files from the Hub."""
    # Validate repo_type if provided
    repo_type = validate_repo_type(repo_type)

    if quiet:
        disable_progress_bars()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(
                _download_impl(
                    repo_id=repo_id,
                    filenames=filenames or [],
                    repo_type=repo_type,
                    revision=revision,
                    include=include,
                    exclude=exclude,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                    force_download=force_download,
                    token=token,
                    max_workers=max_workers,
                )
            )
        enable_progress_bars()
    else:
        logging.set_verbosity_info()
        print(
            _download_impl(
                repo_id=repo_id,
                filenames=filenames or [],
                repo_type=repo_type,
                revision=revision,
                include=include,
                exclude=exclude,
                cache_dir=cache_dir,
                local_dir=local_dir,
                force_download=force_download,
                token=token,
                max_workers=max_workers,
            )
        )
        logging.set_verbosity_warning()


def _download_impl(
    *,
    repo_id: str,
    filenames: list[str],
    repo_type: str,
    revision: Optional[str],
    include: Optional[list[str]],
    exclude: Optional[list[str]],
    cache_dir: Optional[str],
    local_dir: Optional[str],
    force_download: bool,
    token: Optional[str],
    max_workers: int,
) -> str:
    # Warn user if patterns are ignored
    if len(filenames) > 0:
        if include is not None and len(include) > 0:
            warnings.warn("Ignoring `--include` since filenames have being explicitly set.")
        if exclude is not None and len(exclude) > 0:
            warnings.warn("Ignoring `--exclude` since filenames have being explicitly set.")

    # Single file to download: use `hf_hub_download`
    if len(filenames) == 1:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=filenames[0],
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_dir=local_dir,
            library_name="hf",
        )

    # Otherwise: use `snapshot_download` to ensure all files comes from same revision
    elif len(filenames) == 0:
        allow_patterns = include
        ignore_patterns = exclude
    else:
        allow_patterns = filenames
        ignore_patterns = None

    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        force_download=force_download,
        cache_dir=cache_dir,
        token=token,
        local_dir=local_dir,
        library_name="hf",
        max_workers=max_workers,
    )
