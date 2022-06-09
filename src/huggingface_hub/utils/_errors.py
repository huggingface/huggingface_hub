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


class EntryNotFoundError(HTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository and revision
    but an invalid filename.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existant_repository>")
    huggingface_hub.utils._errors.RepositoryNotFoundError: 404 Client Error: Repository Not Found for url: <url>
    ```
    """


class RevisionNotFoundError(HTTPError):
    """Raised when trying to access a hf.co URL with a valid repository but an invalid
    revision."""


def _raise_for_status(request):
    """
    Internal version of `request.raise_for_status()` that will refine a
    potential HTTPError.
    """
    if "X-Error-Code" in request.headers:
        error_code = request.headers["X-Error-Code"]
        if error_code == "RepoNotFound":
            raise RepositoryNotFoundError(
                f"404 Client Error: Repository Not Found for url: {request.url}"
            )
        elif error_code == "RevisionNotFound":
            raise RevisionNotFoundError(
                f"404 Client Error: Revision Not Found for url: {request.url}"
            )
        elif error_code == "EntryNotFound":
            raise EntryNotFoundError(
                f"404 Client Error: Entry Not Found for url: {request.url}"
            )

    request.raise_for_status()
