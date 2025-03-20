from dataclasses import dataclass
from typing import Dict, Optional

from .. import constants
from . import get_session, validate_hf_hub_args


@dataclass(frozen=True)
class XetMetadata:
    endpoint: str
    access_token: str
    expiration_unix_epoch: int
    refresh_route: Optional[str] = None
    file_hash: Optional[str] = None


def parse_xet_json(json: Dict[str, str]) -> Optional[XetMetadata]:
    """
    Parse XET metadata from a JSON object or return None if not found.

    Args:
        json (`Dict`):
            JSON object to extract the XET metadata from.
    Returns:
        `XetMetadata` or `None`:
            The metadata needed to make the request to the xet storage service.
            Returns `None` if the JSON object does not contain the XET metadata.
    """
    # endpoint, access_token and expiration are required
    try:
        endpoint = json[constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT]
        access_token = json[constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN]
        expiration_unix_epoch = int(json[constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION])
    except (KeyError, ValueError, TypeError):
        return None

    return XetMetadata(
        endpoint=endpoint,
        access_token=access_token,
        expiration_unix_epoch=expiration_unix_epoch,
        refresh_route=json.get(constants.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE),
        file_hash=json.get(constants.HUGGINGFACE_HEADER_X_XET_HASH),
    )

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
    if xet_metadata.refresh_route is None:
        raise ValueError("The provided xet metadata does not contain a refresh endpoint.")
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    url = f"{endpoint}{xet_metadata.refresh_route}"
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
        `XetMetadata`:
            The metadata needed to make the request to the xet storage service.
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    resp = get_session().get(headers=headers, url=url)
    metadata = parse_xet_headers(resp.headers)  # type: ignore
    if metadata is None:
        raise ValueError("Xet headers have not been correctly set by the server.")
    return metadata
