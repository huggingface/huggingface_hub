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
from typing import List, Optional, Tuple, Union

from ..constants import ENDPOINT
from ._deprecation import _deprecate_method
from ._subprocess import run_interactive_subprocess, run_subprocess


def list_credential_helpers(folder: Optional[str] = None) -> List[str]:
    """Return the list of git credential helpers configured.

    See https://git-scm.com/docs/gitcredentials.

    Credentials are saved in all configured helpers (store, cache, macos keychain,...).
    Calls "`git credential approve`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    try:
        output = run_subprocess("git config --list", folder=folder).stdout
        # NOTE: If user has set an helper for a custom URL, it will not we caught here.
        #       Example: `credential.https://huggingface.co.helper=store`
        #       See: https://github.com/huggingface/huggingface_hub/pull/1138#discussion_r1013324508
        return sorted(  # Sort for nice printing
            {  # Might have some duplicates
                line.split("=")[-1].split()[0]
                for line in output.split("\n")
                if "credential.helper" in line
            }
        )
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)


def set_git_credential(
    token: str, username: str = "hf_user", folder: Optional[str] = None
) -> None:
    """Save a username/token pair in git credential for HF Hub registry.

    Credentials are saved in all configured helpers (store, cache, macos keychain,...).
    Calls "`git credential approve`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        username (`str`, defaults to `"hf_user"`):
            A git username. Defaults to `"hf_user"`, the default user used in the Hub.
        token (`str`, defaults to `"hf_user"`):
            A git password. In practice, the User Access Token for the Hub.
            See https://huggingface.co/settings/tokens.
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    with run_interactive_subprocess("git credential approve", folder=folder) as (
        stdin,
        _,
    ):
        stdin.write(
            f"url={ENDPOINT}\nusername={username.lower()}\npassword={token}\n\n"
        )
        stdin.flush()


def unset_git_credential(
    username: str = "hf_user", folder: Optional[str] = None
) -> None:
    """Erase credentials from git credential for HF Hub registry.

    Credentials are erased from the configured helpers (store, cache, macos
    keychain,...), if any. If `username` is not provided, any credential configured for
    HF Hub endpoint is erased.
    Calls "`git credential erase`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        username (`str`, defaults to `"hf_user"`):
            A git username. Defaults to `"hf_user"`, the default user used in the Hub.
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    with run_interactive_subprocess("git credential reject", folder=folder) as (
        stdin,
        _,
    ):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input)
        stdin.flush()


@_deprecate_method(
    version="0.14",
    message=(
        "Please use `huggingface_hub.set_git_credential` instead as it allows"
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
        "Please open an issue on https://github.com/huggingface/huggingface_hub if this"
        " a useful feature for you."
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
        "Please use `huggingface_hub.unset_git_credential` instead as it allows"
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
