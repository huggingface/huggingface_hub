import re
from typing import Optional

from requests import HTTPError, Response

from ..errors import (
    BadRequestError,
    DisabledRepoError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


REPO_API_REGEX = re.compile(
    r"""
        # staging or production endpoint
        ^https://[^/]+
        (
            # on /api/repo_type/repo_id
            /api/(models|datasets|spaces)/(.+)
            |
            # or /repo_id/resolve/revision/...
            /(.+)/resolve/(.+)
        )
    """,
    flags=re.VERBOSE,
)


def hf_raise_for_status(response: Response, endpoint_name: Optional[str] = None) -> None:
    """
    Internal version of `response.raise_for_status()` that will refine a
    potential HTTPError. Raised exception will be an instance of `HfHubHTTPError`.

    This helper is meant to be the unique method to raise_for_status when making a call
    to the Hugging Face Hub.


    Example:
    ```py
        import requests
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("\n`create_commit` expects the repository to exist.")
            raise
    ```

    Args:
        response (`Response`):
            Response from the server.
        endpoint_name (`str`, *optional*):
            Name of the endpoint that has been called. If provided, the error message
            will be more complete.

    <Tip warning={true}>

    Raises when the request has failed:

        - [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it
            doesn't exist, because `repo_type` is not set correctly, or because the repo
            is `private` and you do not have access.
        - [`~utils.GatedRepoError`]
            If the repository exists but is gated and the user is not on the authorized
            list.
        - [`~utils.RevisionNotFoundError`]
            If the repository exists but the revision couldn't be find.
        - [`~utils.EntryNotFoundError`]
            If the repository exists but the entry (e.g. the requested file) couldn't be
            find.
        - [`~utils.BadRequestError`]
            If request failed with a HTTP 400 BadRequest error.
        - [`~utils.HfHubHTTPError`]
            If request failed for a reason not listed above.

    </Tip>
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        error_code = response.headers.get("X-Error-Code")
        error_message = response.headers.get("X-Error-Message")

        if error_code == "RevisionNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Revision Not Found for url: {response.url}."
            raise RevisionNotFoundError(message, response) from e

        elif error_code == "EntryNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Entry Not Found for url: {response.url}."
            raise EntryNotFoundError(message, response) from e

        elif error_code == "GatedRepo":
            message = (
                f"{response.status_code} Client Error." + "\n\n" + f"Cannot access gated repo for url {response.url}."
            )
            raise GatedRepoError(message, response) from e

        elif error_message == "Access to this resource is disabled.":
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Cannot access repository for url {response.url}."
                + "\n"
                + "Access to this resource is disabled."
            )
            raise DisabledRepoError(message, response) from e

        elif error_code == "RepoNotFound" or (
            response.status_code == 401
            and response.request is not None
            and response.request.url is not None
            and REPO_API_REGEX.search(response.request.url) is not None
        ):
            # 401 is misleading as it is returned for:
            #    - private and gated repos if user is not authenticated
            #    - missing repos
            # => for now, we process them as `RepoNotFound` anyway.
            # See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Repository Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct `repo_id` and"
                " `repo_type`.\nIf you are trying to access a private or gated repo,"
                " make sure you are authenticated."
            )
            raise RepositoryNotFoundError(message, response) from e

        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
            )
            raise BadRequestError(message, response=response) from e

        elif response.status_code == 403:
            message = (
                f"\n\n{response.status_code} Forbidden: {error_message}."
                + f"\nCannot access content at: {response.url}."
                + "\nIf you are trying to create or update content, "
                + "make sure you have a token with the `write` role."
            )
            raise HfHubHTTPError(message, response=response) from e

        elif response.status_code == 416:
            range_header = response.request.headers.get("Range")
            message = f"{e}. Requested range: {range_header}. Content-Range: {response.headers.get('Content-Range')}."
            raise HfHubHTTPError(message, response=response) from e

        # Convert `HTTPError` into a `HfHubHTTPError` to display request information
        # as well (request id and/or server error message)
        raise HfHubHTTPError(str(e), response=response) from e
