from requests import HTTPError, Response

from ._fixes import JSONDecodeError


class RepositoryNotFoundError(HTTPError):
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

    def __init__(self, message, response):
        super().__init__(message, response=response)


class RevisionNotFoundError(HTTPError):
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

    def __init__(self, message, response):
        super().__init__(message, response=response)


class EntryNotFoundError(HTTPError):
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

    def __init__(self, message, response):
        super().__init__(message, response=response)


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

    def __init__(self, message):
        super().__init__(message, response=None)


class BadRequestError(ValueError, HTTPError):
    """
    Raised by `_raise_convert_bad_request` when the server returns HTTP 400 error

    Example:

    ```py
    >>> resp = request.post("hf.co/api/check", ...)
    >>> _raise_convert_bad_request(resp, endpoint_name="check")
    huggingface_hub.utils._errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
    ```
    """

    def __init__(self, message, response):
        super().__init__(message, response=response)


def _add_information_to_error_args(error: HTTPError, response: Response) -> None:
    """
    If the server response raises an HTTPError, add information from the server Response
    to the HTTPError message.

    NOTE: input error is mutated ! 

    Added details:
    - Request id from "X-Request-Id" header if exists.
    - Server error message if we can found one in the response body.
    """
    # Safety check that the HTTP error already has a message.
    if len(error.args) == 0 or not isinstance(error.args[0], str):
        return

    error_message = error.args[0]

    # Add message from response body
    try:
        server_message = response.json().get("error", None)
        if (
            server_message is not None
            and len(server_message) > 0
            and server_message not in error_message
        ):
            error_message += "\n\n" + server_message
    except JSONDecodeError:
        pass

    # Add Request ID
    request_id = response.headers.get("X-Request-Id")
    if request_id is not None and request_id not in error_message:
        request_id_message = f" (Request ID: {request_id})"
        if "\n" in error_message:
            newline_index = error_message.index("\n")
            error_message = error_message[:newline_index] + request_id_message + error_message[newline_index:]
        else:
            error_message += request_id_message

    import pdb;pdb.set_trace()

    # Mutate HTTPError
    error.args = (error_message,) + error.args[1:]


# def _add_request_id_to_error_args(e, request_id)
# def _add_server_message_to_error_args(e: HTTPError, response: Response)


def _raise_for_status(response):
    """
    Internal version of `response.raise_for_status()` that will refine a
    potential HTTPError.
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        if "X-Error-Code" in response.headers:
            error_code = response.headers["X-Error-Code"]
            if error_code == "RepoNotFound":
                message = (
                    f"{response.status_code} Client Error.\n\nRepository Not Found for"
                    f" url: {response.url}.\nIf the repo is private, make sure you are"
                    " authenticated."
                )
                e = RepositoryNotFoundError(message, response)
            elif error_code == "RevisionNotFound":
                message = (
                    f"{response.status_code} Client Error.\n\nRevision Not Found for url:"
                    f" {response.url}."
                )
                e = RevisionNotFoundError(message, response)
            if error_code == "EntryNotFound":
                message = (
                    f"{response.status_code} Client Error.\n\nEntry Not Found for url:"
                    f" {response.url}."
                )
                e = EntryNotFoundError(message, response)

        elif response.status_code == 401:
            # The repo was not found and the user is not Authenticated
            message = (
                f"{response.status_code} Client Error.\n\nRepository Not Found for url:"
                f" {response.url}. If the repo is private, make sure you are"
                " authenticated."
            )
            e = RepositoryNotFoundError(message, response)

        _add_information_to_error_args(e, response=response)
        raise e


def _raise_with_request_id(response):
    _raise_for_status(response)


def _raise_convert_bad_request(response: Response, endpoint_name: str):
    """
    Calls _raise_for_status on resp and converts HTTP 400 errors into ValueError.
    """
    try:
        _raise_for_status(response)
    except HTTPError as exc:
        request_id = response.headers.get("X-Request-Id")
        try:
            details = response.json().get("error", None)
        except JSONDecodeError:
            raise exc
        if response.status_code == 400 and details:
            raise BadRequestError(
                f"Bad request for {endpoint_name} endpoint: {details} (Request ID:"
                f" {request_id})",
                response=response,
            ) from exc
        _add_information_to_error_args(exc, response=response)
        raise
