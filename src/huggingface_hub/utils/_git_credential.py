# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to manage Git credentials."""
import subprocess
from typing import Optional, Tuple, Union

from ..constants import ENDPOINT
from ._deprecation import _deprecate_method
from ._runtime import is_google_colab, is_notebook
from ._subprocess import run_interactive_subprocess, run_subprocess


@_deprecate_method(
    version="0.14",
    message=(
        "Please use `huggingface_hub.utils.git_credential_approve` instead as it allows"
        " the user to chose which git-credential tool to use."
    ),
)
def write_to_credential_store(username: str, password: str) -> None:
    with run_interactive_subprocess("git credential-store store") as (stdin, _):
        input_username = f"username={username.lower()}"
        input_password = f"password={password}"
        stdin.write(f"url={ENDPOINT}\n{input_username}\n{input_password}\n\n")
        stdin.flush()


@_deprecate_method(
    version="0.14",
    message=(
        "Please use `huggingface_hub.utils.git_credential_fill` instead as it allows"
        " the user to chose which git-credential tool to use."
    ),
)
def read_from_credential_store(
    username: Optional[str] = None,
) -> Union[Tuple[str, str], Tuple[None, None]]:
    """
    Reads the credential store relative to huggingface.co.

    Args:
        username (`str`, *optional*):
            A username to filter to search. If not specified, the first entry under
            `huggingface.co` endpoint is returned.

    Returns:
        `Tuple[str, str]` or `Tuple[None, None]`: either a username/password pair or
        None/None if credential has not been found. The returned username is always
        lowercase.
    """
    with run_interactive_subprocess("git credential-store get") as (stdin, stdout):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input)
        stdin.flush()
        output = stdout.read()

    if len(output) == 0:
        return None, None

    username, password = [line for line in output.split("\n") if len(line) != 0]
    return username.split("=")[1], password.split("=")[1]


@_deprecate_method(
    version="0.14",
    message=(
        "Please use `huggingface_hub.utils.git_credential_reject` instead as it allows"
        " the user to chose which git-credential tool to use."
    ),
)
def erase_from_credential_store(username: Optional[str] = None) -> None:
    """
    Erases the credential store relative to huggingface.co.

    Args:
        username (`str`, *optional*):
            A username to filter to search. If not specified, all entries under
            `huggingface.co` endpoint is erased.
    """
    with run_interactive_subprocess("git credential-store erase") as (stdin, _):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input)
        stdin.flush()


def git_credential_approve(username: str, password: str) -> None:
    with run_interactive_subprocess("git credential approve") as (stdin, _):
        stdin.write(
            f"url={ENDPOINT}\nusername={username.lower()}\npassword={password}\n\n"
        )
        stdin.flush()


def git_credential_fill(username: Optional[str] = None) -> None:
    with run_interactive_subprocess("git credential fill") as (stdin, stdout):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input)
        stdin.flush()
        output = stdout.read()

    if len(output) == 0:
        return None, None

    username, password = [line for line in output.split("\n") if len(line) != 0]
    return username.split("=")[1], password.split("=")[1]


def git_credential_reject(username: Optional[str] = None) -> None:
    with run_interactive_subprocess("git credential reject") as (stdin, _):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input)
        stdin.flush()


def _is_git_credential_helper_configured() -> bool:
    """Return True if `git credential` has at least 1 helper configured."""
    try:
        output = run_subprocess("git config --list").stdout.split("\n")
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    for line in output:
        if "credential.helper" in line:
            return True
    return False
