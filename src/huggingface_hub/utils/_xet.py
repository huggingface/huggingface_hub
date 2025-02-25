from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .. import constants
from . import get_session, hf_raise_for_status, validate_hf_hub_args


class XetTokenType(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class XetMetadata:
    endpoint: str
    access_token: str
    expiration_unix_epoch: int
    refresh_route: Optional[str] = None
    file_hash: Optional[str] = None


def parse_xet_headers(headers: Dict[str, str]) -> Optional[XetMetadata]:
    """
    Parse XET metadata from the HTTP headers or return None if not found.
    Args:
        headers (`Dict`):
           HTTP headers to extract the XET metadata from.
    Returns:
        `XetMetadata` or `None`:
            The metadata needed to make the request to the xet storage service.
            Returns `None` if the headers do not contain the XET metadata.
    """
    # endpoint, access_token and expiration are required
    try:
        endpoint = headers[constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT]
        access_token = headers[constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN]
        expiration_unix_epoch = int(headers[constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION])
    except (KeyError, ValueError, TypeError):
        return None

    return XetMetadata(
        endpoint=endpoint,
        access_token=access_token,
        expiration_unix_epoch=expiration_unix_epoch,
        refresh_route=headers.get(constants.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE),
        file_hash=headers.get(constants.HUGGINGFACE_HEADER_X_XET_HASH),
    )


@validate_hf_hub_args
def fetch_xet_metadata_from_repo_info(
    *,
    token_type: XetTokenType,
    repo_id: str,
    repo_type: str,
    revision: Optional[str] = None,
    headers: Dict[str, str],
    endpoint: Optional[str] = None,
    params: Optional[Dict[str, str]] = None,
) -> XetMetadata:
    """
    Uses the repo info to request a xet access token from Hub.
    Args:
        token_type (`XetTokenType`):
            Type of the token to request: `"read"` or `"write"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        revision (`str`, `optional`):
            The revision of the repo to get the token for.
        headers (`Dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        endpoint (`str`, `optional`):
            The endpoint to use for the request. Defaults to the Hub endpoint.
        params (`Dict[str, str]`, `optional`):
            Additional parameters to pass with the request.
    Returns:
        `XetMetadata`: The metadata needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    url = f"{endpoint}/api/{repo_type}s/{repo_id}/xet-{token_type.value}-token/{revision}"
    return _fetch_xet_metadata_with_url(url, headers, params)


@validate_hf_hub_args
def _fetch_xet_metadata_with_url(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, str]] = None,
) -> XetMetadata:
    """
    Requests the xet access token from the supplied URL.
    Args:
        url: (`str`):
            The access token endpoint URL.
        headers (`Dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
    Returns:
        `XetMetadata`:
            The metadata needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    resp = get_session().get(headers=headers, url=url, params=params)
    hf_raise_for_status(resp)

    metadata = parse_xet_headers(resp.headers)  # type: ignore
    if metadata is None:
        raise ValueError("Xet headers have not been correctly set by the server.")
    return metadata
