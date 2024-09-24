from dataclasses import dataclass
from typing import Dict, Optional, Union

from requests.structures import CaseInsensitiveDict

from .. import constants


@dataclass(frozen=True)
class HfXetMetadata:
    endpoint: str
    access_token: str


def xet_metadata_or_none(headers: Union[Dict[str, str], CaseInsensitiveDict[str]]) -> Optional[HfXetMetadata]:
    """
    Extract XET metadata from the HTTP headers or return None if not found.

    Args:
        headers (`Dict`):
           HTTP headers to extract the XET metadata from.
    """
    xet_endpoint = headers.get(constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT)
    access_token = headers.get(constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN)

    if xet_endpoint is None or access_token is None:
        return None
    return HfXetMetadata(endpoint=xet_endpoint, access_token=access_token)
