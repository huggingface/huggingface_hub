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
"""Contains utilities to handle headers to send in calls to Huggingface Hub."""
from functools import lru_cache
from typing import Dict, Literal, Optional, Union

import requests

from ._errors import HTTPError, RepositoryNotFoundError, hf_raise_for_status
from ._hf_folder import HfFolder


PERMISSION_T = Literal["read", "write"]


def build_hf_headers(
    endpoint: str,
    token: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    required_permission: PERMISSION_T = "read",
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build headers dictionary to send in a HF Hub API call.

    Authorization token is either given in argument (explicit use) or retrieved from the
    machine (e.g. after a `huggingface-cli login` or using `HUGGING_FACE_HUB_TOKEN` env
    variable). In case of an implicit use of the token, we check if the repo on which
    the call will be made is private/gated. If it is a public repo and only read-access
    is required, the token is not passed to preserve privacy. If the API call is not
    made against a repo, it is considered a public call.

    Returned token is also validated. An error is thrown if the server doesn't recognize
    the token or if a write-access token is required but the token is an organization
    one (starting with "api_org***").

    Args:
        endpoint (`str`):
            Endpoint url to which the API call will be made. Used to check token
            validity.
        token (`str`, `optional`):
            Hugging Face token. Will default to the locally saved token if not provided.
        use_auth_token (`str`, `bool`, *optional*):
            A token to be used for the download. If `True`, the token is read from the
            machine (cache or env variable). If a string, it's used as the Hugging Face
            token. If None, the token is read from the machine only if necessary
            (write-access call or private repo).
        required_permission (`Literal['read', 'write']`):
            The permission required by the API call.
        repo_id (`str`, `optional`):
            Id of the repo if the api call is made on a repo (download a file, create a
            commit, list files,...). Use `None` if the api call is not bound to a repo
            (list models, create a repo,...).
        repo_type (`str`, `optional`):
            Repo type if the api call is made on a repo (download a file, create a
            commit, list files,...). Use `None` if the api call is not bound to a repo
            (list models, create a repo,...).

    Returns:
        A `Dict` of headers to pass in your API call..

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If organization token is passed and "write" access is required.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If token is invalid (not recognized by the server).
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If "write" access is required but token is not passed and not saved locally.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If making a call to a private repo but token is not passed and not saved
            locally.
    """
    # Get auth token to send
    token_to_send = _get_token_to_send(
        endpoint, repo_id, repo_type, token, use_auth_token, required_permission
    )

    # Combine headers
    headers = {}
    if token_to_send is not None:
        headers["authorization"] = f"Bearer {token_to_send}"
    # TODO: add user agent in headers
    return headers


def _get_token_to_send(
    endpoint: str,
    repo_id: str,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    required_permission: PERMISSION_T = "read",
) -> Optional[str]:
    # Case token is explicitly provided via `token`
    if token is not None:
        token_to_send = token

    # Case token is explicitly provided via `use_auth_token`
    elif isinstance(use_auth_token, str):
        token_to_send = use_auth_token

    # Case token is explicitly forbidden
    elif use_auth_token is False:
        token_to_send = None

    # Case token is explicitly required
    elif use_auth_token is True:
        cached_token = HfFolder().get_token()
        if cached_token is None:
            raise EnvironmentError(
                "Token is required (`use_auth_token=True`), but no token found. You"
                " need to provide a token or be logged in to Hugging Face with"
                " `huggingface-cli login` or `notebook_login`."
            )
        token_to_send = cached_token

    # Otherwise: user did not give a preference whether to send a token or not.
    # In order to preserve user privacy, token is sent only for private/gated repos or
    # for if the `write` permission is required:
    elif required_permission == "write" or _is_private(endpoint, repo_id, repo_type):
        cached_token = HfFolder().get_token()
        if cached_token is None:
            if required_permission == "write":
                raise EnvironmentError(
                    "Token is required (write-access action) but no token found. You"
                    " need to provide a token or be logged in to Hugging Face with"
                    " `huggingface-cli login` or `notebook_login`."
                )
            else:
                raise EnvironmentError(
                    "Token is required to perform an action on a private repo but no"
                    " token found. You need to provide a token or be logged in to"
                    " Hugging Face with `huggingface-cli login` or `notebook_login`."
                )
        token_to_send = cached_token
    else:  # "read" permission + repo is not private/gated
        token_to_send = None

    # Validate token and return
    if token_to_send is not None:
        if required_permission == "write" and token_to_send.startswith("api_org"):
            raise ValueError(
                "You must use your personal account token for write-access methods. To"
                " generate a write-access token, go to"
                " https://huggingface.co/settings/tokens"
            )
        if not _is_valid_token(endpoint, token_to_send):
            raise ValueError("Invalid token passed!")
    return token_to_send


@lru_cache
def _is_private(endpoint: str, repo_id: str, repo_type: Optional[str] = None) -> bool:
    """Check if the repo on which the API call will be made is private or not.

    Result is cached to avoid multiple `repo_info` API calls.
    """
    if repo_id is None:
        # Means it is a generic API -> public
        return False

    if repo_type is None:
        repo_type = "model"
    r = requests.get(f"{endpoint}/api/{repo_type}s/{repo_id}")
    try:
        hf_raise_for_status(r)
        return False
    except RepositoryNotFoundError:
        return True


@lru_cache
def _is_valid_token(endpoint: str, token: str) -> None:
    """Check if the given token is valid.

    Result is cached to avoid multiple `whoami` API calls.
    """
    r = requests.get(
        f"{endpoint}/api/whoami-v2", headers={"authorization": f"Bearer {token}"}
    )
    try:
        hf_raise_for_status(r)
        return True
    except HTTPError:
        return False
