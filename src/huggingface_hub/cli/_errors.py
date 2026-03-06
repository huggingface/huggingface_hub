# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""CLI error handling utilities."""

import traceback
from typing import Callable, Optional

from huggingface_hub.errors import (
    BucketNotFoundError,
    CLIError,
    CLIExtensionInstallError,
    GatedRepoError,
    HfHubHTTPError,
    LocalTokenNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


def _format_cli_error(error: CLIError) -> str:
    """No traceback, just the error message."""
    return str(error)


def _format_cli_extension_install_error(error: CLIExtensionInstallError) -> str:
    """Format a CLI extension installation error.

    The error is likely to be a tricky subprocess error to investigate. In this specific case we want to format the
    traceback of the root cause while keeping the "nicely formatted" error message of the CLIExtensionInstallError
    as a 1-line message.
    """
    cause_tb = "".join(traceback.format_exception(error.__cause__)) if error.__cause__ is not None else ""  # type: ignore
    return f"{cause_tb}\n{error}"


CLI_ERROR_MAPPINGS: dict[type[Exception], Callable[..., str]] = {
    BucketNotFoundError: lambda e: (
        "Bucket not found. Check the bucket id (namespace/name). If the bucket is private, make sure you are authenticated."
    ),
    RepositoryNotFoundError: lambda e: (
        "Repository not found. Check the `repo_id` and `repo_type` parameters. If the repo is private, make sure you are authenticated."
    ),
    RevisionNotFoundError: lambda e: "Revision not found. Check the `revision` parameter.",
    GatedRepoError: lambda e: "Access denied. This repository requires approval.",
    LocalTokenNotFoundError: lambda e: "Not logged in. Run 'hf auth login' first.",
    RemoteEntryNotFoundError: lambda e: "File not found in repository.",
    HfHubHTTPError: lambda e: str(e),
    ValueError: lambda e: f"Invalid value. {e}",
    CLIExtensionInstallError: _format_cli_extension_install_error,
    CLIError: _format_cli_error,
}


def format_known_exception(e: Exception) -> Optional[str]:
    for exc_type, formatter in CLI_ERROR_MAPPINGS.items():
        if isinstance(e, exc_type):
            return formatter(e)
    return None
