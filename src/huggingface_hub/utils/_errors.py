from typing import Optional

from requests import HTTPError, Response

from ._fixes import JSONDecodeError


class HfHubHTTPError(HTTPError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in HF Hub.

    Any HTTPError is converted at least into a `HfHubHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request id from "X-Request-Id" header if exists.
    - Server error message if we can found one in the response body.
    """

    def __init__(self, message: str, response: Optional[Response]):
        if response is not None:
            message = _add_information_to_error_message(message, response)
        super().__init__(message, response=response)

    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]


class RepositoryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existent_repository>")
    huggingface_hub.utils._errors.RepositoryNotFoundError: 404 Client Error: Repository Not Found for url: <url>
    ```
    """


class RevisionNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository but an invalid
    revision.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', 'config.json', revision='<non-existent-revision>')
    huggingface_hub.utils._errors.RevisionNotFoundError: 404 Client Error: Revision Not Found for url: <url>
    ```
    """


class EntryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository and revision
    but an invalid filename.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-existent-file>')
    huggingface_hub.utils._errors.EntryNotFoundError: 404 Client Error: Entry Not Found for url: <url>
    ```
    """


class LocalEntryNotFoundError(EntryNotFoundError, FileNotFoundError, ValueError):
    """
    Raised when trying to access a file that is not on the disk when network is
    disabled or unavailable (connection issue). The entry may exist on the Hub.

    Note: `ValueError` type is to ensure backward compatibility.
    Note: `LocalEntryNotFoundError` derives from `HTTPError` because of `EntryNotFoundError`
          even when it is not a network issue.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-cached-file>',  local_files_only=True)
    huggingface_hub.utils._errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.
    ```
    """

    def __init__(self, message: str):
        super().__init__(message, response=None)


class BadRequestError(HfHubHTTPError, ValueError):
    """
    Raised by `_raise_convert_bad_request` when the server returns HTTP 400 error.

    Example:

    ```py
    >>> resp = requests.post("hf.co/api/check", ...)
    >>> _raise_convert_bad_request(resp, endpoint_name="check")
    huggingface_hub.utils._errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
    ```
    """


def _raise_for_status(response: Response, endpoint_name: Optional[str] = None) -> None:
    """
    Internal version of `response.raise_for_status()` that will refine a
    potential HTTPError.
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        error_code = response.headers.get("X-Error-Code")

        if error_code == "RevisionNotFound":
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Revision Not Found for url: {response.url}."
            )
            raise RevisionNotFoundError(message, response) from e

        elif error_code == "EntryNotFound":
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Entry Not Found for url: {response.url}."
            )
            raise EntryNotFoundError(message, response) from e

        elif error_code == "RepoNotFound" or response.status_code == 401:
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Repository Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct `repo_id` and"
                " `repo_type`."
                + "\nIf the repo is private, make sure you are authenticated."
            )
            raise RepositoryNotFoundError(message, response) from e

        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:"
                if endpoint_name is not None
                else "\n\nBad request:"
            )
            raise BadRequestError(message, response=response) from e

        # Convert `HTTPError` into a `HfHubHTTPError` to display request information
        # as well (request id and/or server error message)
        raise HfHubHTTPError(str(HTTPError), response=response) from e


def _raise_with_request_id(response):
    """Keep alias for now ?"""
    _raise_for_status(response)


def _raise_convert_bad_request(response: Response, endpoint_name: str):
    """
    Calls _raise_for_status on resp and converts HTTP 400 errors into ValueError.

    Keep alias for now ?
    """
    _raise_for_status(response, endpoint_name)


def _add_information_to_error_message(message: str, response: Response) -> str:
    """
    Add information to the error message based on response from the server.
    Used when initializing `HfHubHTTPError`.
    """
    # Add message from response body
    try:
        server_message = response.json().get("error", None)
        if (
            server_message is not None
            and len(server_message) > 0
            and server_message not in message
        ):
            if "\n\n" in message:
                message += "\n\n" + message
            else:
                message += "\n" + message
    except JSONDecodeError:
        pass

    # Add Request ID
    request_id = response.headers.get("X-Request-Id")
    if request_id is not None and request_id not in message:
        request_id_message = f" (Request ID: {request_id})"
        if "\n" in message:
            newline_index = message.index("\n")
            message = (
                message[:newline_index] + request_id_message + message[newline_index:]
            )
        else:
            message += request_id_message

    return message
