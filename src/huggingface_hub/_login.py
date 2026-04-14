# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Contains methods to log in to the Hub."""

import subprocess
import time
from pathlib import Path

from . import constants
from .errors import DeviceCodeError
from .utils import (
    ANSI,
    capture_output,
    get_token,
    is_google_colab,
    is_notebook,
    list_credential_helpers,
    logging,
    run_subprocess,
    set_git_credential,
    unset_git_credential,
)
from .utils._auth import (
    _get_token_by_name,
    _get_token_from_environment,
    _get_token_from_file,
    _save_stored_tokens,
    _save_token,
    get_stored_tokens,
)
from .utils._http import get_session


logger = logging.get_logger(__name__)

_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


def login(
    token: str | None = None,
    *,
    add_to_git_credential: bool = False,
    skip_if_logged_in: bool = True,
) -> None:
    """Login the machine to access the Hub.

    The `token` is persisted in cache and set as a git credential. Once done, the machine
    is logged in and the access token will be available across all `huggingface_hub`
    components. If `token` is not provided, a browser-based OAuth device code flow is used
    to authenticate. You will be prompted to open a URL and enter a code.

    To log in from outside of a script, one can also use `hf auth login` which is
    a cli command that wraps [`login`].

    > [!TIP]
    > [`login`] is a drop-in replacement method for [`notebook_login`] as it wraps and
    > extends its capabilities.

    > [!TIP]
    > When the token is not passed, [`login`] will automatically detect if the script runs
    > in a notebook or not. However, this detection might not be accurate due to the
    > variety of notebooks that exists nowadays. If that is the case, you can always force
    > the UI by using [`notebook_login`] or [`interpreter_login`].

    Args:
        token (`str`, *optional*):
            User access token to generate from https://huggingface.co/settings/token.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user. If `token` is `None`,
            the value of `add_to_git_credential` is ignored and will be prompted again
            to the end user.
        skip_if_logged_in (`bool`, defaults to `True`):
            If `True`, do not prompt for token if user is already logged in.
            Set to `False` to force re-login. In CLI, use `--force` instead.
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If an organization token is passed. Only personal account tokens are valid
            to log in.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If token is invalid.
        [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
            If running in a notebook but `ipywidgets` is not installed.
    """
    if token is not None:
        if not add_to_git_credential:
            logger.info(
                "The token has not been saved to the git credentials helper. Pass "
                "`add_to_git_credential=True` in this function directly or "
                "`--add-to-git-credential` if using via `hf`CLI if "
                "you want to set the git credential as well."
            )
        _validate_and_save_token(token, add_to_git_credential=add_to_git_credential)
    elif is_notebook():
        notebook_login(skip_if_logged_in=skip_if_logged_in, add_to_git_credential=add_to_git_credential)
    else:
        interpreter_login(skip_if_logged_in=skip_if_logged_in, add_to_git_credential=add_to_git_credential)


def logout(token_name: str | None = None) -> None:
    """Logout the machine from the Hub.

    Token is deleted from the machine and removed from git credential.

    Args:
        token_name (`str`, *optional*):
            Name of the access token to logout from. If `None`, will log out from all saved access tokens.
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError):
            If the access token name is not found.
    """
    if get_token() is None and not get_stored_tokens():  # No active token and no saved access tokens
        logger.warning("Not logged in!")
        return
    if not token_name:
        # Delete all saved access tokens and token
        for file_path in (constants.HF_TOKEN_PATH, constants.HF_STORED_TOKENS_PATH):
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                pass
        logger.info("Successfully logged out from all access tokens.")
    else:
        _logout_from_token(token_name)
        logger.info(f"Successfully logged out from access token: {token_name}.")

    unset_git_credential()

    # Check if still logged in
    if _get_token_from_environment() is not None:
        raise OSError(
            "Token has been deleted from your machine but you are still logged in.\n"
            "To log out, you must clear out both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` environment variables."
        )


def auth_switch(token_name: str, add_to_git_credential: bool = False) -> None:
    """Switch to a different access token.

    Args:
        token_name (`str`):
            Name of the access token to switch to.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user. If `token` is `None`,
            the value of `add_to_git_credential` is ignored and will be prompted again
            to the end user.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError):
            If the access token name is not found.
    """
    token = _get_token_by_name(token_name)
    if not token:
        raise ValueError(f"Access token {token_name} not found in {constants.HF_STORED_TOKENS_PATH}")
    # Write token to HF_TOKEN_PATH
    _set_active_token(token_name, add_to_git_credential)
    logger.info(f"The current active token is: {token_name}")
    token_from_environment = _get_token_from_environment()
    if token_from_environment is not None and token_from_environment != token:
        logger.warning(
            "The environment variable `HF_TOKEN` is set and will override the access token you've just switched to."
        )


def auth_list() -> None:
    """List all stored access tokens."""
    tokens = get_stored_tokens()

    if not tokens:
        if _get_token_from_environment():
            logger.info("No stored access tokens found.")
            logger.warning("Note: Environment variable `HF_TOKEN` is set and is the current active token.")
        else:
            logger.info("No access tokens found.")
        return
    # Find current token
    current_token = get_token()
    current_token_name = None
    for token_name in tokens:
        if tokens.get(token_name) == current_token:
            current_token_name = token_name
    # Print header
    max_offset = max(len("token"), max(len(token) for token in tokens)) + 2
    print(f"  {{:<{max_offset}}}| {{:<15}}".format("name", "token"))
    print("-" * (max_offset + 2) + "|" + "-" * 15)

    # Print saved access tokens
    for token_name in tokens:
        token = tokens.get(token_name, "<not set>")
        masked_token = f"{token[:3]}****{token[-4:]}" if token != "<not set>" else token
        is_current = "*" if token == current_token else " "

        print(f"{is_current} {{:<{max_offset}}}| {{:<15}}".format(token_name, masked_token))

    if _get_token_from_environment():
        logger.warning(
            "\nNote: Environment variable `HF_TOKEN` is set and is the current active token independently from the stored tokens listed above."
        )
    elif current_token_name is None:
        logger.warning(
            "\nNote: No active token is set and no environment variable `HF_TOKEN` is found. Use `hf auth login` to log in."
        )


###
# Device Code OAuth (RFC 8628)
###


def _request_device_code() -> dict:
    """Request a device code from the Hub's OAuth device authorization endpoint.

    Returns a dict with keys: device_code, user_code, verification_uri,
    verification_uri_complete, interval, expires_in.
    """
    response = get_session().post(
        f"{constants.ENDPOINT}/oauth/device",
        data={
            "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
        },
    )
    if response.status_code != 200:
        raise DeviceCodeError(
            f"Failed to request device code from {constants.ENDPOINT}/oauth/device "
            f"(status {response.status_code}): {response.text}"
        )
    return response.json()


def _poll_for_token(device_code: str, interval: int = 5, expires_in: int = 900) -> str:
    """Poll the token endpoint until the user authorizes the device.

    Args:
        device_code: The device code from the device authorization response.
        interval: Minimum polling interval in seconds.
        expires_in: Time in seconds before the device code expires.

    Returns:
        The access token string.

    Raises:
        DeviceCodeError: If authorization is denied or the device code expires.
    """
    start_time = time.monotonic()
    while time.monotonic() - start_time < expires_in:
        time.sleep(interval)
        response = get_session().post(
            f"{constants.ENDPOINT}/oauth/token",
            data={
                "grant_type": _DEVICE_CODE_GRANT_TYPE,
                "device_code": device_code,
                "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
            },
        )
        try:
            data = response.json()
        except Exception:
            raise DeviceCodeError(
                f"Failed to parse response from {constants.ENDPOINT}/oauth/token\n"
                f"  Status: {response.status_code}\n"
                f"  URL: {response.url}\n"
                f"  Headers: {dict(response.headers)}\n"
                f"  Body: {response.text[:500]!r}"
            )

        if "access_token" in data:
            return data["access_token"]

        error = data.get("error")
        if error == "authorization_pending":
            # Print a dot to show we're still waiting
            print(".", end="", flush=True)
            continue
        elif error == "slow_down":
            interval += 5
            continue
        elif error == "expired_token":
            raise DeviceCodeError("Device code expired. Please try again.")
        elif error == "access_denied":
            raise DeviceCodeError("Authorization was denied. Please try again.")
        else:
            error_description = data.get("error_description", "")
            raise DeviceCodeError(f"OAuth error: {error} - {error_description}")

    raise DeviceCodeError("Device code expired (timeout). Please try again.")


def _device_code_login(add_to_git_credential: bool = False) -> None:
    """Run the Device Code OAuth flow: request a code, open browser, poll for token, save it."""
    # Step 1: Request device code
    device_info = _request_device_code()

    verification_uri = device_info["verification_uri"]
    user_code = device_info["user_code"]
    interval = device_info.get("interval", 5)
    expires_in = device_info.get("expires_in", 900)

    # Step 2: Display instructions and open browser
    print(f"\n    Open this URL in your browser:\n        {verification_uri}\n")
    print(f"    And enter code: {user_code}\n")

    # Step 3: Poll for token
    print("    Waiting for authorization", end="", flush=True)
    token = _poll_for_token(
        device_code=device_info["device_code"],
        interval=interval,
        expires_in=expires_in,
    )
    print()  # newline after dots

    # Step 4: Validate and save
    _validate_and_save_token(token, add_to_git_credential=add_to_git_credential)
    logger.info("Note: This token expires in 30 days. Run `hf auth login` again to refresh it.")


###
# Interpreter-based login (text)
###


def interpreter_login(*, skip_if_logged_in: bool = True, add_to_git_credential: bool = False) -> None:
    """
    Displays a prompt to log in to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when not run in a notebook.
    [`interpreter_login`] is useful if you want to force the use of the terminal prompt
    instead of a notebook widget.

    For more details, see [`login`].

    Args:
        skip_if_logged_in (`bool`, defaults to `True`):
            If `True`, do not prompt for token if user is already logged in.
            Set to `False` to force re-login. In CLI, use `--force` instead.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user.
    """
    if skip_if_logged_in and get_token() is not None:
        logger.info("User is already logged in. Use `hf auth login --force` to force re-login.")
        return

    if get_token() is not None:
        logger.info(
            "    A token is already saved on your machine. Run `hf auth whoami`"
            " to get more information or `hf auth logout` if you want"
            " to log out."
        )
        logger.info("    Setting a new token will erase the existing one.")

    _device_code_login(add_to_git_credential=add_to_git_credential)


###
# Notebook-based login
###


def notebook_login(*, skip_if_logged_in: bool = True, add_to_git_credential: bool = False) -> None:
    """
    Displays a prompt to log in to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when run in a notebook.
    [`notebook_login`] is useful if you want to force the use of the notebook widget
    instead of a prompt in the terminal.

    For more details, see [`login`].

    Args:
        skip_if_logged_in (`bool`, defaults to `True`):
            If `True`, do not prompt for token if user is already logged in.
            Set to `False` to force re-login. In CLI, use `--force` instead.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user.
    """
    if skip_if_logged_in and get_token() is not None:
        logger.info("User is already logged in. Use `hf auth login --force` to force re-login.")
        return

    try:
        from IPython.display import HTML, display
    except ImportError:
        # Not in a notebook environment, fallback to interpreter login
        interpreter_login(skip_if_logged_in=False, add_to_git_credential=add_to_git_credential)
        return

    # Step 1: Request device code
    device_info = _request_device_code()

    verification_uri = device_info["verification_uri"]
    verification_uri_complete = device_info.get("verification_uri_complete", verification_uri)
    user_code = device_info["user_code"]
    interval = device_info.get("interval", 5)
    expires_in = device_info.get("expires_in", 900)

    # Step 2: Display HTML with link and code
    display(
        HTML(
            '<center><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"'
            ' width="100" alt="Hugging Face"><br><br>'
            f"<p>To log in, open this URL and enter the code:</p>"
            f'<p><a href="{verification_uri_complete}" target="_blank"><b>{verification_uri}</b></a></p>'
            f'<p style="font-size: 1.6em; letter-spacing: 0.3em; font-family: monospace;">'
            f"<b>{user_code}</b></p></center>"
        )
    )

    # Step 3: Poll for token
    display(HTML("<center><i>Waiting for authorization...</i></center>"))
    try:
        token = _poll_for_token(
            device_code=device_info["device_code"],
            interval=interval,
            expires_in=expires_in,
        )
    except DeviceCodeError as e:
        display(HTML(f"<center><b style='color: red;'>Login failed: {e}</b></center>"))
        return

    # Step 5: Validate and save
    try:
        with capture_output() as captured:
            _validate_and_save_token(token, add_to_git_credential=add_to_git_credential)
        message = captured.getvalue()
        # Add the expiration notice
        message += "\nNote: This token expires in 30 days. Run `hf auth login` again to refresh it."
        display(HTML("<center>" + "<br>".join(line for line in message.split("\n") if line.strip()) + "</center>"))
    except Exception as error:
        display(HTML(f"<center><b style='color: red;'>{error}</b></center>"))


###
# Login private helpers
###


def _validate_and_save_token(
    token: str,
    add_to_git_credential: bool,
    token_name: str | None = None,
) -> None:
    """Validate a token via whoami, save it to stored tokens, and set it as active.

    Args:
        token: The access token string.
        add_to_git_credential: Whether to save the token to git credential helpers.
        token_name: Optional override for the token name. If not provided, extracted
            from the whoami response or generated from the username.
    """
    from .hf_api import whoami  # avoid circular import

    if token.startswith("api_org"):
        raise ValueError("You must use your personal account token, not an organization token.")

    token_info = whoami(token)

    # Extract permission info if available
    try:
        permission = token_info["auth"]["accessToken"]["role"]
        logger.info(f"Token is valid (permission: {permission}).")
    except (KeyError, TypeError):
        logger.info("Token is valid.")

    # Determine token name
    if token_name is None:
        try:
            token_name = token_info["auth"]["accessToken"]["displayName"]
        except (KeyError, TypeError):
            username = token_info.get("name", "unknown")
            token_name = f"oauth-{username}"

    # Store token locally
    _save_token(token=token, token_name=token_name)
    # Set active token
    _set_active_token(token_name=token_name, add_to_git_credential=add_to_git_credential)
    logger.info("Login successful.")
    if _get_token_from_environment():
        logger.warning(
            "Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured."
        )
    else:
        logger.info(f"The current active token is: `{token_name}`")


def _logout_from_token(token_name: str) -> None:
    """Logout from a specific access token.

    Args:
        token_name (`str`):
            The name of the access token to logout from.
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError):
            If the access token name is not found.
    """
    stored_tokens = get_stored_tokens()
    # If there is no access tokens saved or the access token name is not found, do nothing
    if not stored_tokens or token_name not in stored_tokens:
        return

    token = stored_tokens.pop(token_name)
    _save_stored_tokens(stored_tokens)

    if token == _get_token_from_file():
        logger.warning(f"Active token '{token_name}' has been deleted.")
        Path(constants.HF_TOKEN_PATH).unlink(missing_ok=True)


def _set_active_token(
    token_name: str,
    add_to_git_credential: bool,
) -> None:
    """Set the active access token.

    Args:
        token_name (`str`):
            The name of the token to set as active.
    """
    token = _get_token_by_name(token_name)
    if not token:
        raise ValueError(f"Token {token_name} not found in {constants.HF_STORED_TOKENS_PATH}")
    if add_to_git_credential:
        if _is_git_credential_helper_configured():
            set_git_credential(token)
            logger.info(
                "Your token has been saved in your configured git credential helpers"
                + f" ({','.join(list_credential_helpers())})."
            )
        else:
            logger.warning("Token has not been saved to git credential helper.")
    # Write token to HF_TOKEN_PATH
    path = Path(constants.HF_TOKEN_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    logger.info(f"Your token has been saved to {constants.HF_TOKEN_PATH}")


def _is_git_credential_helper_configured() -> bool:
    """Check if a git credential helper is configured.

    Warns user if not the case (except for Google Colab where "store" is set by default
    by `huggingface_hub`).
    """
    helpers = list_credential_helpers()
    if len(helpers) > 0:
        return True  # Do not warn: at least 1 helper is set

    # Only in Google Colab to avoid the warning message
    # See https://github.com/huggingface/huggingface_hub/issues/1043#issuecomment-1247010710
    if is_google_colab():
        _set_store_as_git_credential_helper_globally()
        return True  # Do not warn: "store" is used by default in Google Colab

    # Otherwise, warn user
    print(
        ANSI.red(
            "Cannot authenticate through git-credential as no helper is defined on your"
            " machine.\nYou might have to re-authenticate when pushing to the Hugging"
            " Face Hub.\nRun the following command in your terminal in case you want to"
            " set the 'store' credential helper as default.\n\ngit config --global"
            " credential.helper store\n\nRead"
            " https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more"
            " details."
        )
    )
    return False


def _set_store_as_git_credential_helper_globally() -> None:
    """Set globally the credential.helper to `store`.

    To be used only in Google Colab as we assume the user doesn't care about the git
    credential config. It is the only particular case where we don't want to display the
    warning message in [`notebook_login()`].

    Related:
    - https://github.com/huggingface/huggingface_hub/issues/1043
    - https://github.com/huggingface/huggingface_hub/issues/1051
    - https://git-scm.com/docs/git-credential-store
    """
    try:
        run_subprocess("git config --global credential.helper store")
    except subprocess.CalledProcessError as exc:
        raise OSError(exc.stderr)
