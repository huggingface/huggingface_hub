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
from typing import Dict, Optional, Union

import requests

from ._errors import HTTPError, RepositoryNotFoundError, hf_raise_for_status
from ._hf_folder import HfFolder


def build_hf_headers(
    *,
    token: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    is_write_action: bool = False,
    url: Optional[str] = None,
    endpoint: Optional[str] = None,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build headers dictionary to send in a HF Hub call.

    Authorization token is either given in argument (explicit use) or retrieved from the
    machine (e.g. after a `huggingface-cli login` or using `HUGGING_FACE_HUB_TOKEN` env
    variable). In case of an implicit use of the token, we check if the repo on which
    the call will be made is private/gated. If it is a public repo and only read-access
    is required, the token is not passed to preserve privacy. If the API call is not
    made against a repo, it is considered a public call.

    Returned token is also validated. An error is thrown if the server doesn't recognize
    the token or if a write-access token is required but the token is an organization
    one (starting with "api_org***").

    The request that will be made must either be a call to the API or an url pointing to
    the Hub. In case of an API call, use `endpoint` argument (and optionally `repo_id`/
    `repo_type`). In case of a direct URL call, use `url` argument and ignore
    `endpoint`. In this later case, the validity of the token is not tested.

    Args:
        token (`str`, `optional`):
            Hugging Face token. Will default to the locally saved token if not provided.
        use_auth_token (`str`, `bool`, *optional*):
            A token to be used for the download. If `True`, the token is read from the
            machine (cache or env variable). If a string, it's used as the Hugging Face
            token. If None, the token is read from the machine only if necessary
            (write-access call or private repo).
        is_write_action (`bool`, default to `False`):
            Set to True if the API call requires a write access. If `True` and token is
            not provided, it will be read from machine. If no token is found, raises an
            exception.
        url (`str`, *optional*):
            Full url that will be called. For example used when downloading a file for
            which you have the full url. If you are making an API call, use `endpoint`
            instead.
        endpoint (`str`, *optional*):
            API endpoint to which the call will be made. Used to check token validity.
            If making a call to the Hub for which you only have a full url, pass it
            `url` to ignore `endpoint`, `repo_id` and `repo_type`.
        repo_id (`str`, `optional`):
            Id of the repo if the api call is made on a repo (create a commit,
            list files,...). To be used only combined with both `endpoint` and
            `use_auth_token` arguments. Use `None` if the api call is not bound to a
            repo (list models, create a repo,...).
        repo_type (`str`, `optional`):
            Repo type if the api call is made on a repo (create a commit,
            list files,...). To be used only combined with both `endpoint` and
            `use_auth_token` arguments. Use `None` if the api call is not bound to a
            repo (list models, create a repo,...).

    Returns:
        A `Dict` of headers to pass in your API call.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If organization token is passed and "write" access is required.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If token is invalid (not recognized by the server).
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If "write" access is required but token is not passed and not saved locally.
    """
    # Get auth token to send
    token_to_send = _get_token_to_send(
        token=token,
        use_auth_token=use_auth_token,
        is_write_action=is_write_action,
        endpoint=endpoint,
        repo_id=repo_id,
        repo_type=repo_type,
        url=url,
    )

    # Combine headers
    headers = {}
    if token_to_send is not None:
        headers["authorization"] = f"Bearer {token_to_send}"
    # TODO: add user agent in headers
    return headers


def _get_token_to_send(
    token: Optional[str],
    use_auth_token: Optional[Union[bool, str]],
    is_write_action: bool,
    repo_id: Optional[str],
    repo_type: Optional[str],
    endpoint: Optional[str],
    url: Optional[str],
) -> Optional[str]:
    """Determine if the token must be provided or validates it.

    Strategy is made is such a way to preserve as much as possible user's privacy. See
    `build_hf_headers` docstring for more details.
    """
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
                " `huggingface-cli login` or `notebook_login`. See"
                " https://huggingface.co/settings/tokens."
            )
        token_to_send = cached_token

    # Otherwise: user did not give a preference whether to send a token or not.
    # In order to preserve user privacy, token is sent only for private/gated repos or
    # for if the `write` permission is required:
    elif is_write_action or _is_private(
        url=url, endpoint=endpoint, repo_id=repo_id, repo_type=repo_type
    ):
        cached_token = HfFolder().get_token()
        if cached_token is None and is_write_action:
            raise EnvironmentError(
                "Token is required (write-access action) but no token found. You"
                " need to provide a token or be logged in to Hugging Face with"
                " `huggingface-cli login` or `notebook_login`. See"
                " https://huggingface.co/settings/tokens."
            )
        token_to_send = cached_token
    else:  # "read" permission + repo is not private/gated
        token_to_send = None

    # Validate token and return
    if token_to_send is not None:
        if is_write_action and token_to_send.startswith("api_org"):
            raise ValueError(
                "You must use your personal account token for write-access methods. To"
                " generate a write-access token, go to"
                " https://huggingface.co/settings/tokens"
            )
        if endpoint is not None and not _is_valid_token(endpoint, token_to_send):
            raise ValueError(
                "Invalid token passed! Go to https://huggingface.co/settings/tokens to"
                " get one."
            )
    return token_to_send


@lru_cache()
def _is_private(
    url: Optional[str],
    endpoint: Optional[str],
    repo_id: Optional[str],
    repo_type: Optional[str],
) -> bool:
    """Check if the repo on which the API call will be made is private or not.

    Result is cached to avoid multiple `repo_info` API calls.
    """
    # HEAD call to url without headers to check if private or not
    if url is not None:
        try:
            hf_raise_for_status(requests.head(url))
            return False
        except RepositoryNotFoundError:
            return True
        except Exception:  # Anything else: don't assume private repo
            return False

    # Should not happen
    if endpoint is None:
        raise ValueError("Either `url` or `endpoint` must be not `None`!")

    # Means it is a generic action -> public
    if repo_id is None:
        return False

    # If action on a repo, is it a private one ?
    if repo_type is None:
        repo_type = "model"
    try:
        hf_raise_for_status(requests.head(f"{endpoint}/api/{repo_type}s/{repo_id}"))
        return False
    except RepositoryNotFoundError:
        return True
    except Exception:  # Anything else: don't assume private repo
        return False


@lru_cache()
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
