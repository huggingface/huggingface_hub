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


def _raise_for_status(request):
    """
    Internal version of `request.raise_for_status()` that will refine a
    potential HTTPError.
    """
    request_id = request.headers.get("X-Request-Id")

    if "X-Error-Code" in request.headers:
        error_code = request.headers["X-Error-Code"]
        if error_code == "RepoNotFound":
            raise RepositoryNotFoundError(
                f"404 Client Error: Repository Not Found for url: {request.url}. If the"
                " repo is private, make sure you are authenticated. (Request ID:"
                f" {request_id})"
            )
        elif error_code == "RevisionNotFound":
            raise RevisionNotFoundError(
                f"404 Client Error: Revision Not Found for url: {request.url}. (Request"
                f" ID: {request_id})"
            )
        elif error_code == "EntryNotFound":
            raise EntryNotFoundError(
                f"404 Client Error: Entry Not Found for url: {request.url}. (Request"
                f" ID: {request_id})"
            )

    if request.status_code == 401:
        # The repo was not found and the user is not Authenticated
        raise RepositoryNotFoundError(
            f"401 Client Error: Repository Not Found for url: {request.url}. If the"
            " repo is private, make sure you are authenticated. (Request ID:"
            f" {request_id})"
        )

    _raise_with_request_id(request)


def _raise_with_request_id(request):
    request_id = request.headers.get("X-Request-Id")
    try:
        request.raise_for_status()
    except Exception as e:
        if request_id is not None and len(e.args) > 0 and isinstance(e.args[0], str):
            e.args = (e.args[0] + f" (Request ID: {request_id})",) + e.args[1:]

        raise e
