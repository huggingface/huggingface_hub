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
    if "X-Error-Code" in request.headers:
        error_code = request.headers["X-Error-Code"]
        if error_code == "RepoNotFound":
            raise RepositoryNotFoundError(
                f"404 Client Error: Repository Not Found for url: {request.url}. "
                "If the repo is private, make sure you are authenticated."
            )
        elif error_code == "RevisionNotFound":
            raise RevisionNotFoundError(
                f"404 Client Error: Revision Not Found for url: {request.url}"
            )
        elif error_code == "EntryNotFound":
            raise EntryNotFoundError(
                f"404 Client Error: Entry Not Found for url: {request.url}"
            )

    if request.status_code == 401:
        # The repo was not found and the user is not Authenticated
        raise RepositoryNotFoundError(
            f"401 Client Error: Repository Not Found for url: {request.url}. "
            "If the repo is private, make sure you are authenticated."
        )

    request.raise_for_status()
