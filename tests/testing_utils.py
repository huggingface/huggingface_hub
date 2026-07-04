import os
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from unittest.mock import Mock, patch

import httpx


def repo_name(id: Optional[str] = None, prefix: str = "repo") -> str:
    """
    Return a readable pseudo-unique repository name for tests.

    Example:
    ```py
    >>> repo_name()
    repo-2fe93f-16599646671840

    >>> repo_name("my-space", prefix='space')
    space-my-space-16599481979701
    """
    if id is None:
        id = uuid.uuid4().hex[:6]
    ts = int(time.time() * 10e3)
    return f"{prefix}-{id}-{ts}"


class RequestWouldHangIndefinitelyError(Exception):
    pass


class OfflineSimulationMode(Enum):
    CONNECTION_FAILS = 0
    CONNECTION_TIMES_OUT = 1
    HF_HUB_OFFLINE_SET_TO_1 = 2


@contextmanager
def offline(mode=OfflineSimulationMode.CONNECTION_FAILS, timeout=1e-16):
    """
    Simulate offline mode.

    There are three offline simulation modes:

    CONNECTION_FAILS (default mode): a ConnectionError is raised for each network call.
        Connection errors are created by mocking socket.socket
    CONNECTION_TIMES_OUT: the connection hangs until it times out.
        The default timeout value is low (1e-16) to speed up the tests.
        Timeout errors are created by mocking httpx.request
    HF_HUB_OFFLINE_SET_TO_1: the HF_HUB_OFFLINE_SET_TO_1 environment variable is set to 1.
        This makes the http/ftp calls of the library instantly fail and raise an OfflineModeEnabled error.
    """
    import socket

    # Store the original httpx.request to avoid recursion
    original_httpx_request = httpx.request

    def timeout_request(method, url, **kwargs):
        # Change the url to an invalid url so that the connection hangs
        invalid_url = "https://10.255.255.1"
        if kwargs.get("timeout") is None:
            raise RequestWouldHangIndefinitelyError(
                f"Tried a call to {url} in offline mode with no timeout set. Please set a timeout."
            )
        kwargs["timeout"] = timeout
        try:
            return original_httpx_request(method, invalid_url, **kwargs)
        except Exception as e:
            # The following changes in the error are just here to make the offline timeout error prettier
            if hasattr(e, "request"):
                e.request.url = url
            if hasattr(e, "args") and e.args:
                max_retry_error = e.args[0]
                if hasattr(max_retry_error, "args"):
                    max_retry_error.args = (max_retry_error.args[0].replace("10.255.255.1", f"OfflineMock[{url}]"),)
                e.args = (max_retry_error,)
            raise

    def offline_socket(*args, **kwargs):
        raise socket.error("Offline mode is enabled.")

    def build_offline_client(exc_factory):
        # Build a fake `httpx.Client` whose every HTTP method fails. We patch the cached `_GLOBAL_CLIENT`
        # so that EVERY caller of `get_session()` is offline,
        client = Mock()

        def fail(*args, **kwargs):
            raise exc_factory()

        for method in ("request", "stream", "send", "get", "post", "head", "put", "patch", "delete"):
            setattr(client, method, fail)
        return client

    if mode is OfflineSimulationMode.CONNECTION_FAILS:
        # inspired from https://stackoverflow.com/a/18601897
        offline_client = build_offline_client(lambda: httpx.ConnectError("Connection failed"))
        with patch("socket.socket", offline_socket):
            with patch("huggingface_hub.utils._http._GLOBAL_CLIENT", offline_client):
                yield
    elif mode is OfflineSimulationMode.CONNECTION_TIMES_OUT:
        # inspired from https://stackoverflow.com/a/904609
        offline_client = build_offline_client(lambda: httpx.ConnectTimeout("Connection timed out"))
        # `.request` keeps the "hangs until timeout" behavior so the no-timeout guard is still exercised.
        offline_client.request = timeout_request
        with patch("httpx.request", timeout_request):
            with patch("huggingface_hub.utils._http._GLOBAL_CLIENT", offline_client):
                yield
    elif mode is OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1:
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            yield
    else:
        raise ValueError("Please use a value from the OfflineSimulationMode enum.")


def set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree_with_retry(path: Union[str, Path]) -> None:
    shutil.rmtree(path, onerror=set_write_permission_and_retry)
