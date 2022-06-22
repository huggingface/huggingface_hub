from requests import HTTPError


class RepositoryNotFoundError(HTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existant_repository>")
    huggingface_hub.utils._errors.RepositoryNotFoundError: 404 Client Error: Repository Not Found for url: <url>
    ```
    """

    def __init__(self, message, response):
        super(RepositoryNotFoundError, self).__init__(message, response=response)


class RevisionNotFoundError(HTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository but an invalid
    revision.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', 'config.json', revision='<non-existant-revision>')
    huggingface_hub.utils._errors.RevisionNotFoundError: 404 Client Error: Revision Not Found for url: <url>
    ```
    """

    def __init__(self, message, response):
        super(RevisionNotFoundError, self).__init__(message, response=response)


class EntryNotFoundError(HTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository and revision
    but an invalid filename.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-existant-file>')
    huggingface_hub.utils._errors.EntryNotFoundError: 404 Client Error: Entry Not Found for url: <url>
    ```
    """

    def __init__(self, message, response):
        super(EntryNotFoundError, self).__init__(message, response=response)


def _raise_for_status(response):
    """
    Internal version of `request.raise_for_status()` that will refine a
    potential HTTPError.
    """
    request_id = response.headers.get("X-Request-Id")
    try:
        response.raise_for_status()
    except Exception as e:
        if "X-Error-Code" in response.headers:
            error_code = response.headers["X-Error-Code"]
            if error_code == "RepoNotFound":
                message = (
                    f"{response.status_code} Client Error: Repository Not Found for"
                    f" url: {response.url}. If the repo is private, make sure you are"
                    f" authenticated. (Request ID: {request_id})"
                )
                raise RepositoryNotFoundError(message, response)
            elif error_code == "RevisionNotFound":
                message = (
                    f"{response.status_code} Client Error: Revision Not Found for url:"
                    f" {response.url}. (Request ID: {request_id})"
                )
                raise RevisionNotFoundError(message, response)
            elif error_code == "EntryNotFound":
                message = (
                    f"{response.status_code} Client Error: Entry Not Found for url:"
                    f" {response.url}. (Request ID: {request_id})"
                )
                raise EntryNotFoundError(message, response)

        if response.status_code == 401:
            # The repo was not found and the user is not Authenticated
            message = (
                f"{response.status_code} Client Error: Repository Not Found for url:"
                f" {response.url}. If the repo is private, make sure you are"
                f" authenticated. (Request ID: {request_id})"
            )
            raise RepositoryNotFoundError(message, response)

        if request_id is not None and len(e.args) > 0 and isinstance(e.args[0], str):
            e.args = (e.args[0] + f" (Request ID: {request_id})",) + e.args[1:]

        raise e


def _raise_with_request_id(response):
    request_id = response.headers.get("X-Request-Id")
    try:
        response.raise_for_status()
    except Exception as e:
        if request_id is not None and len(e.args) > 0 and isinstance(e.args[0], str):
            e.args = (e.args[0] + f" (Request ID: {request_id})",) + e.args[1:]

        raise e
