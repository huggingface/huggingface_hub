# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Contains a helper to get the token from machine (env variable, secret or config file)."""

import configparser
import logging
import os
from pathlib import Path

from .. import constants


logger = logging.getLogger(__name__)


def get_token() -> str | None:
    """
    Get token if user is logged in.

    Note: in most cases, you should use [`huggingface_hub.utils.build_hf_headers`] instead. This method is only useful
          if you want to retrieve the token for other purposes than sending an HTTP request.

    Token is retrieved in priority from the `HF_TOKEN` environment variable. Otherwise, we read the token file located
    in the Hugging Face home folder. Returns None if user is not logged in. To log in, use [`login`] or
    `hf auth login`.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return _get_token_from_environment() or _get_token_from_file()


def _get_token_from_environment() -> str | None:
    # `HF_TOKEN` has priority (keep `HUGGING_FACE_HUB_TOKEN` for backward compatibility)
    return _clean_token(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _get_token_from_file() -> str | None:
    try:
        return _clean_token(Path(constants.HF_TOKEN_PATH).read_text())
    except FileNotFoundError:
        return None


def get_stored_tokens() -> dict[str, str]:
    """
    Returns the parsed INI file containing the access tokens.
    The file is located at `HF_STORED_TOKENS_PATH`, defaulting to `~/.cache/huggingface/stored_tokens`.
    If the file does not exist, an empty dictionary is returned.

    Returns: `dict[str, str]`
        Key is the token name and value is the token.
    """
    tokens_path = Path(constants.HF_STORED_TOKENS_PATH)
    if not tokens_path.exists():
        stored_tokens = {}
    config = configparser.ConfigParser()
    try:
        config.read(tokens_path)
        stored_tokens = {token_name: config.get(token_name, "hf_token") for token_name in config.sections()}
    except configparser.Error as e:
        logger.error(f"Error parsing stored tokens file: {e}")
        stored_tokens = {}
    return stored_tokens


def _save_stored_tokens(stored_tokens: dict[str, str]) -> None:
    """
    Saves the given configuration to the stored tokens file.

    Args:
        stored_tokens (`dict[str, str]`):
            The stored tokens to save. Key is the token name and value is the token.
    """
    stored_tokens_path = Path(constants.HF_STORED_TOKENS_PATH)

    # Write the stored tokens into an INI file
    config = configparser.ConfigParser()
    for token_name in sorted(stored_tokens.keys()):
        config.add_section(token_name)
        config.set(token_name, "hf_token", stored_tokens[token_name])

    stored_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    with stored_tokens_path.open("w") as config_file:
        config.write(config_file)


def _get_token_by_name(token_name: str) -> str | None:
    """
    Get the token by name.

    Args:
        token_name (`str`):
            The name of the token to get.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.

    """
    stored_tokens = get_stored_tokens()
    if token_name not in stored_tokens:
        return None
    return _clean_token(stored_tokens[token_name])


def _save_token(token: str, token_name: str) -> None:
    """
    Save the given token.

    If the stored tokens file does not exist, it will be created.
    Args:
        token (`str`):
            The token to save.
        token_name (`str`):
            The name of the token.
    """
    tokens_path = Path(constants.HF_STORED_TOKENS_PATH)
    stored_tokens = get_stored_tokens()
    stored_tokens[token_name] = token
    _save_stored_tokens(stored_tokens)
    logger.info(f"The token `{token_name}` has been saved to {tokens_path}")


def _clean_token(token: str | None) -> str | None:
    """Clean token by removing trailing and leading spaces and newlines.

    If token is an empty string, return None.
    """
    if token is None:
        return None
    return token.replace("\r", "").replace("\n", "").strip() or None
