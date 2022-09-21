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
import os
from typing import Dict, Optional, Union

from ._hf_folder import HfFolder


def build_hf_headers(
    *, use_auth_token: Optional[Union[bool, str]] = None, is_write_action: bool = False
) -> Dict[str, str]:
    """
    Build headers dictionary to send in a HF Hub call.

    By default, authorization token is always provided either from argument (explicit
    use) or retrieved from the cache (implicit use). To explicitly avoid sending the
    token to the Hub, set `use_auth_token=False` or set the `DISABLE_IMPLICIT_HF_TOKEN`
    environment variable.

    In case of an API call that requires write access, an error is thrown if token is
    `None` or token is an organization token (starting with `"api_org***"`).

    Args:
        use_auth_token (`str`, `bool`, *optional*):
            The token to be sent in authorization header for the Hub call:
                - if a string, it is used as the Hugging Face token
                - if `True`, the token is read from the machine (cache or env variable)
                - if `False`, authorization header is not set
                - if `None`, the token is read from the machine only except if
                  `DISABLE_IMPLICIT_HF_TOKEN` env variable is set.
        is_write_action (`bool`, default to `False`):
            Set to True if the API call requires a write access. If `True`, the token
            will be validated (cannot be `None`, cannot start by `"api_org***"`).

    Returns:
        A `Dict` of headers to pass in your API call.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If organization token is passed and "write" access is required.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If "write" access is required but token is not passed and not saved locally.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If `use_auth_token=True` but token is not saved locally.
    """
    # Get auth token to send
    token_to_send = _get_token_to_send(use_auth_token)
    _validate_token_to_send(token_to_send, is_write_action=is_write_action)

    # Combine headers
    headers = {}
    if token_to_send is not None:
        headers["authorization"] = f"Bearer {token_to_send}"
    # TODO: add user agent in headers
    return headers


def _get_token_to_send(use_auth_token: Optional[Union[bool, str]]) -> Optional[str]:
    """Select the token to send from either `use_auth_token` or the cache."""
    # Case token is explicitly provided
    if isinstance(use_auth_token, str):
        return use_auth_token

    # Case token is explicitly forbidden
    if use_auth_token is False:
        return None

    # Token is not provided: we get it from local cache
    cached_token = HfFolder().get_token()

    # Case token is explicitly required but not found
    if use_auth_token is True and cached_token is None:
        raise EnvironmentError(
            "Token is required (`use_auth_token=True`), but no token found. You need to"
            " provide a token or be logged in to Hugging Face with `huggingface-cli"
            " login` or `notebook_login`. See https://huggingface.co/settings/tokens."
        )

    # Case implicit use of the token is forbidden by env variable
    if os.environ.get("DISABLE_IMPLICIT_HF_TOKEN"):
        return None

    # Otherwise: we use the cached token as the user has not explicitly forbidden it
    return cached_token


def _validate_token_to_send(token: Optional[str], is_write_action: bool) -> None:
    if is_write_action:
        if token is None:
            raise ValueError(
                "Token is required (write-access action) but no token found. You need"
                " to provide a token or be logged in to Hugging Face with"
                " `huggingface-cli login` or `notebook_login`. See"
                " https://huggingface.co/settings/tokens."
            )
        if token.startswith("api_org"):
            raise ValueError(
                "You must use your personal account token for write-access methods. To"
                " generate a write-access token, go to"
                " https://huggingface.co/settings/tokens"
            )
