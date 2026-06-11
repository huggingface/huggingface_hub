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
import io
import logging
import os
import time
import warnings
from pathlib import Path
from threading import Lock
from typing import TypedDict

from .. import constants
from ..errors import OIDCError
from ._runtime import is_colab_enterprise, is_google_colab


_SECRET_FILE_MODE = 0o600
_SECRET_DIR_MODE = 0o700


def _write_secret(path: Path, content: str) -> None:
    """Write content to file, restricting both the file and its parent directory to owner-only on POSIX systems."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=_SECRET_DIR_MODE)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECRET_FILE_MODE)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    try:
        path.chmod(_SECRET_FILE_MODE)
        path.parent.chmod(_SECRET_DIR_MODE)
    except (OSError, NotImplementedError):
        # Windows does not support POSIX modes; chmod() will raise. Best-effort.
        pass


_IS_GOOGLE_COLAB_CHECKED = False
_GOOGLE_COLAB_SECRET_LOCK = Lock()
_GOOGLE_COLAB_SECRET: str | None = None

logger = logging.getLogger(__name__)


def get_token() -> str | None:
    """
    Get token if user is logged in.

    Note: in most cases, you should use [`huggingface_hub.utils.build_hf_headers`] instead. This method is only useful
          if you want to retrieve the token for other purposes than sending an HTTP request.

    If `HF_OIDC_RESOURCE` is set (Trusted Publishers, typically in CI), a short-lived token obtained via OIDC token
    exchange takes precedence. Otherwise the token is retrieved from the `HF_TOKEN` environment variable, then from the
    token file in the Hugging Face home folder. Returns None if user is not logged in. To log in, use [`login`] or
    `hf auth login`.

    Note: if `HF_OIDC_RESOURCE` is set but the OIDC token exchange fails, this raises instead of returning `None`,
    opting into OIDC is explicit, so a failure surfaces as a clear error rather than a silent fallback.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return (
        _get_token_from_oidc()
        or _get_token_from_environment()
        or _get_token_from_file()
        or _get_token_from_google_colab()
    )


def _get_token_from_google_colab() -> str | None:
    """Get token from Google Colab secrets vault using `google.colab.userdata.get(...)`.

    Token is read from the vault only once per session and then stored in a global variable to avoid re-requesting
    access to the vault.
    """
    # If it's not a Google Colab or it's Colab Enterprise, fallback to environment variable or token file authentication
    if not is_google_colab() or is_colab_enterprise():
        return None

    # `google.colab.userdata` is not thread-safe
    # This can lead to a deadlock if multiple threads try to access it at the same time
    # (typically when using `snapshot_download`)
    # => use a lock
    # See https://github.com/huggingface/huggingface_hub/issues/1952 for more details.
    with _GOOGLE_COLAB_SECRET_LOCK:
        global _GOOGLE_COLAB_SECRET
        global _IS_GOOGLE_COLAB_CHECKED

        if _IS_GOOGLE_COLAB_CHECKED:  # request access only once
            return _GOOGLE_COLAB_SECRET

        try:
            from google.colab import userdata  # type: ignore
            from google.colab.errors import Error as ColabError  # type: ignore
        except ImportError:
            return None

        try:
            token = userdata.get("HF_TOKEN")
            _GOOGLE_COLAB_SECRET = _clean_token(token)
        except userdata.NotebookAccessError:
            # Means the user has a secret call `HF_TOKEN` and got a popup "please grand access to HF_TOKEN" and refused it
            # => warn user but ignore error => do not re-request access to user
            warnings.warn(
                "\nAccess to the secret `HF_TOKEN` has not been granted on this notebook."
                "\nYou will not be requested again."
                "\nPlease restart the session if you want to be prompted again."
            )
            _GOOGLE_COLAB_SECRET = None
        except userdata.SecretNotFoundError:
            # Means the user did not define a `HF_TOKEN` secret => warn
            warnings.warn(
                "\nThe secret `HF_TOKEN` does not exist in your Colab secrets."
                "\nTo authenticate with the Hugging Face Hub, create a token in your settings tab "
                "(https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session."
                "\nYou will be able to reuse this secret in all of your notebooks."
                "\nPlease note that authentication is recommended but still optional to access public models or datasets."
            )
            _GOOGLE_COLAB_SECRET = None
        except ColabError as e:
            # Something happen but we don't know what => recommend to open a GitHub issue
            warnings.warn(
                f"\nError while fetching `HF_TOKEN` secret value from your vault: '{str(e)}'."
                "\nYou are not authenticated with the Hugging Face Hub in this notebook."
                "\nIf the error persists, please let us know by opening an issue on GitHub "
                "(https://github.com/huggingface/huggingface_hub/issues/new)."
            )
            _GOOGLE_COLAB_SECRET = None

        _IS_GOOGLE_COLAB_CHECKED = True
        return _GOOGLE_COLAB_SECRET


def _get_token_from_environment() -> str | None:
    # `HF_TOKEN` has priority (keep `HUGGING_FACE_HUB_TOKEN` for backward compatibility)
    return _clean_token(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _get_token_from_file() -> str | None:
    try:
        return _clean_token(Path(constants.HF_TOKEN_PATH).read_text())
    except FileNotFoundError:
        return None


class _OidcTokenCache(TypedDict):
    resource: str
    token: str
    expires_at: float  # monotonic clock value after which the cached token must be re-exchanged


# Cache for the OIDC-exchanged token: re-exchanging on every `get_token()` call would be wasteful,
# and re-exchanging shortly before expiry transparently keeps long-running jobs authenticated.
_OIDC_TOKEN_LOCK = Lock()
_OIDC_TOKEN_CACHE: _OidcTokenCache | None = None
_OIDC_REFRESH_MARGIN = 300  # re-exchange this many seconds before the token actually expires


def _get_token_from_oidc() -> str | None:
    """Get a short-lived OIDC token in CI (Trusted Publishers).

    Enabled by setting `HF_OIDC_RESOURCE`, which scopes the token to a repo or user.
    The ID token is read from `HF_OIDC_ID_TOKEN` if available, or minted from a supported CI provider (e.g. GitHub Actions).

    Returns `None` when OIDC is not enabled.
    If enabled, any failure is raised explicitly rather than falling back silently.

    See `huggingface_hub._oidc` and https://huggingface.co/docs/hub/trusted-publishers.
    """
    resource = os.environ.get("HF_OIDC_RESOURCE")
    if not resource:
        return None

    from .._oidc import detect_provider, oidc_login

    global _OIDC_TOKEN_CACHE
    with _OIDC_TOKEN_LOCK:
        now = time.monotonic()
        if (
            _OIDC_TOKEN_CACHE is not None
            and _OIDC_TOKEN_CACHE["resource"] == resource
            and now < _OIDC_TOKEN_CACHE["expires_at"]
        ):
            return _OIDC_TOKEN_CACHE["token"]

        # An explicit id token (any provider) takes precedence; otherwise mint from a detected one.
        subject_token = os.environ.get("HF_OIDC_ID_TOKEN") or None
        if subject_token is None and detect_provider() is None:
            raise OIDCError(
                "HF_OIDC_RESOURCE is set but no OIDC id token is available: not running in a supported "
                "CI provider (github) and HF_OIDC_ID_TOKEN is not set. Set HF_OIDC_ID_TOKEN to the id "
                "token minted by your CI provider, or unset HF_OIDC_RESOURCE."
            )

        result = oidc_login(resource=resource, subject_token=subject_token)
        token = result["access_token"]
        expires_in = int(result.get("expires_in", 3600))
        # A pre-supplied HF_OIDC_ID_TOKEN can't be re-minted, so refreshing early is pointless (the id
        # token is likely already expired by then): cache for the full lifetime. Only the auto-minted
        # path can refresh, so only it gets the safety margin.
        margin = 0 if subject_token is not None else _OIDC_REFRESH_MARGIN
        _OIDC_TOKEN_CACHE = {
            "resource": resource,
            "token": token,
            "expires_at": now + max(expires_in - margin, 0),
        }
        return token


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

    buf = io.StringIO()
    config.write(buf)
    _write_secret(stored_tokens_path, buf.getvalue())


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
