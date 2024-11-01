from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

from requests.structures import CaseInsensitiveDict

from .. import constants
from . import get_session, validate_hf_hub_args


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


def xet_metadata_or_none(headers: Union[Dict[str, str], CaseInsensitiveDict[str]]) -> Optional[XetMetadata]:
    """
    Extract XET metadata from the HTTP headers or return None if not found.

    Args:
        headers (`Dict`):
           HTTP headers to extract the XET metadata from.
    """
    xet_endpoint = headers.get(constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT)
    file_hash = headers.get(constants.HUGGINGFACE_HEADER_X_XET_HASH)
    access_token = headers.get(constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN)
    expiration = headers.get(constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION)
    refresh_route = headers.get(constants.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE)

    if xet_endpoint is None or access_token is None or expiration is None:
        return None

    try:
        expiration_unix_epoch = int(expiration)
    except (ValueError, TypeError):
        return None

    return XetMetadata(
        endpoint=xet_endpoint,
        access_token=access_token,
        expiration_unix_epoch=expiration_unix_epoch,
        refresh_route=refresh_route,
        file_hash=file_hash,
    )


@validate_hf_hub_args
def refresh_xet_metadata(
    *,
    xet_metadata: XetMetadata,
    headers: Dict[str, str],
    endpoint: Optional[str] = None,
) -> XetMetadata:
    """
    Utilizes the information in the parsed metadata to request the Hub xet access token.

    Args:
        xet_metadata: (`XetMetadata`):
            The xet metadata provided by the Hub API.
        headers (`Dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
        endpoint (`str`, `optional`):
            The endpoint to use for the request. Defaults to the Hub endpoint.
    Returns:
        `XetMetadata`: The metadata needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    if xet_metadata.refresh_route is None:
        raise ValueError("The provided xet metadata does not contain a refresh endpoint.")
    url = f"{endpoint}{xet_metadata.refresh_route}"
    return _fetch_xet_metadata_with_url(url, headers)


@validate_hf_hub_args
def fetch_xet_metadata_from_repo_info(
    *,
    token_type: XetTokenType,
    repo_id: str,
    repo_type: str,
    revision: Optional[str] = None,
    headers: Dict[str, str],
    endpoint: Optional[str] = None,
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
    return _fetch_xet_metadata_with_url(url, headers)


@validate_hf_hub_args
def _fetch_xet_metadata_with_url(
    url: str,
    headers: Dict[str, str],
) -> XetMetadata:
    """
    Requests the xet access token from the supplied URL.

    Args:
        url: (`str`):
            The access token endpoint URL.
        headers (`Dict[str, str]`):
            Headers to use for the request, including authorization headers and user agent.
    Returns:
        `XetMetadata`: The metadata needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    resp = get_session().get(headers=headers, url=url)
    metadata = xet_metadata_or_none(resp.headers)
    if metadata is None:
        raise ValueError("Xet headers have not been correctly set by the server.")
    return metadata
