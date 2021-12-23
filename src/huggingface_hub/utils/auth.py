import sys
from base64 import b64encode
from typing import Optional


if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class AuthHeaders(TypedDict):
    authorization: str


def auth_headers(
    token: Optional[str] = None, use_basic: Optional[bool] = False
) -> Optional[AuthHeaders]:
    return (
        {
            "authorization": basic_auth_header("user", token)
            if use_basic
            else f"Bearer {token}"
        }
        if token
        else None
    )


def basic_auth_header(username: str, password: str) -> str:
    """
    Same behaviour as requests.auth.HttpBasicAuth

    see https://github.com/psf/requests/blob/main/requests/auth.py#L79
    """
    auth_str = b":".join((username.encode("latin1"), password.encode("latin1")))
    return f'Basic {b64encode(auth_str).decode("ascii")}'
