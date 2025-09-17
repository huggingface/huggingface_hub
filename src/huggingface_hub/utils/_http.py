# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains utilities to handle HTTP requests in huggingface_hub."""

import atexit
import io
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from shlex import quote
from typing import Any, Callable, Generator, Optional, Union

import httpx

from huggingface_hub.errors import OfflineModeIsEnabled

from .. import constants
from ..errors import (
    BadRequestError,
    DisabledRepoError,
    GatedRepoError,
    HfHubHTTPError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from . import logging
from ._lfs import SliceFileObj
from ._typing import HTTP_METHOD_T


logger = logging.get_logger(__name__)

# Both headers are used by the Hub to debug failed requests.
# `X_AMZN_TRACE_ID` is better as it also works to debug on Cloudfront and ALB.
# If `X_AMZN_TRACE_ID` is set, the Hub will use it as well.
X_AMZN_TRACE_ID = "X-Amzn-Trace-Id"
X_REQUEST_ID = "x-request-id"

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


class HfHubTransport(httpx.HTTPTransport):
    """
    Transport that will be used to make HTTP requests to the Hugging Face Hub.

    What it does:
    - Block requests if offline mode is enabled
    - Add a request ID to the request headers
    - Log the request if debug mode is enabled
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if constants.HF_HUB_OFFLINE:
            raise OfflineModeIsEnabled(
                f"Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."
            )
        request_id = _add_request_id(request)
        try:
            return super().handle_request(request)
        except httpx.RequestError as e:
            if request_id is not None:
                # Taken from https://stackoverflow.com/a/58270258
                e.args = (*e.args, f"(Request ID: {request_id})")
            raise


class HfHubAsyncTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if constants.HF_HUB_OFFLINE:
            raise OfflineModeIsEnabled(
                f"Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."
            )
        request_id = _add_request_id(request)
        try:
            return await super().handle_async_request(request)
        except httpx.RequestError as e:
            if request_id is not None:
                # Taken from https://stackoverflow.com/a/58270258
                e.args = (*e.args, f"(Request ID: {request_id})")
            raise


def _add_request_id(request: httpx.Request) -> Optional[str]:
    # Add random request ID => easier for server-side debug
    if X_AMZN_TRACE_ID not in request.headers:
        request.headers[X_AMZN_TRACE_ID] = request.headers.get(X_REQUEST_ID) or str(uuid.uuid4())
    request_id = request.headers.get(X_AMZN_TRACE_ID)

    # Debug log
    logger.debug(
        "Request %s: %s %s (authenticated: %s)",
        request_id,
        request.method,
        request.url,
        request.headers.get("authorization") is not None,
    )
    if constants.HF_DEBUG:
        logger.debug("Send: %s", _curlify(request))

    return request_id


def default_client_factory() -> httpx.Client:
    """
    Factory function to create a `httpx.Client` with the default transport.
    """
    return httpx.Client(
        transport=HfHubTransport(),
        follow_redirects=True,
        timeout=httpx.Timeout(constants.DEFAULT_REQUEST_TIMEOUT, write=60.0),
    )


def default_async_client_factory() -> httpx.AsyncClient:
    """
    Factory function to create a `httpx.AsyncClient` with the default transport.
    """
    return httpx.AsyncClient(
        transport=HfHubAsyncTransport(),
        follow_redirects=True,
        timeout=httpx.Timeout(constants.DEFAULT_REQUEST_TIMEOUT, write=60.0),
    )


CLIENT_FACTORY_T = Callable[[], httpx.Client]
ASYNC_CLIENT_FACTORY_T = Callable[[], httpx.AsyncClient]

_CLIENT_LOCK = threading.Lock()
_GLOBAL_CLIENT_FACTORY: CLIENT_FACTORY_T = default_client_factory
_GLOBAL_ASYNC_CLIENT_FACTORY: ASYNC_CLIENT_FACTORY_T = default_async_client_factory
_GLOBAL_CLIENT: Optional[httpx.Client] = None


def set_client_factory(client_factory: CLIENT_FACTORY_T) -> None:
    """
    Set the HTTP client factory to be used by `huggingface_hub`.

    The client factory is a method that returns a `httpx.Client` object. On the first call to [`get_client`] the client factory
    will be used to create a new `httpx.Client` object that will be shared between all calls made by `huggingface_hub`.

    This can be useful if you are running your scripts in a specific environment requiring custom configuration (e.g. custom proxy or certifications).

    Use [`get_client`] to get a correctly configured `httpx.Client`.
    """
    global _GLOBAL_CLIENT_FACTORY
    with _CLIENT_LOCK:
        close_session()
        _GLOBAL_CLIENT_FACTORY = client_factory


def set_async_client_factory(async_client_factory: ASYNC_CLIENT_FACTORY_T) -> None:
    """
    Set the HTTP async client factory to be used by `huggingface_hub`.

    The async client factory is a method that returns a `httpx.AsyncClient` object.
    This can be useful if you are running your scripts in a specific environment requiring custom configuration (e.g. custom proxy or certifications).
    Use [`get_async_client`] to get a correctly configured `httpx.AsyncClient`.

    <Tip warning={true}>

    Contrary to the `httpx.Client` that is shared between all calls made by `huggingface_hub`, the `httpx.AsyncClient` is not shared.
    It is recommended to use an async context manager to ensure the client is properly closed when the context is exited.

    </Tip>
    """
    global _GLOBAL_ASYNC_CLIENT_FACTORY
    _GLOBAL_ASYNC_CLIENT_FACTORY = async_client_factory


def get_session() -> httpx.Client:
    """
    Get a `httpx.Client` object, using the transport factory from the user.

    This client is shared between all calls made by `huggingface_hub`. Therefore you should not close it manually.

    Use [`set_client_factory`] to customize the `httpx.Client`.
    """
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        with _CLIENT_LOCK:
            _GLOBAL_CLIENT = _GLOBAL_CLIENT_FACTORY()
    return _GLOBAL_CLIENT


def get_async_session() -> httpx.AsyncClient:
    """
    Return a `httpx.AsyncClient` object, using the transport factory from the user.

    Use [`set_async_client_factory`] to customize the `httpx.AsyncClient`.

    <Tip warning={true}>

    Contrary to the `httpx.Client` that is shared between all calls made by `huggingface_hub`, the `httpx.AsyncClient` is not shared.
    It is recommended to use an async context manager to ensure the client is properly closed when the context is exited.

    </Tip>
    """
    return _GLOBAL_ASYNC_CLIENT_FACTORY()


def close_session() -> None:
    """
    Close the global `httpx.Client` used by `huggingface_hub`.

    If a Client is closed, it will be recreated on the next call to [`get_client`].

    Can be useful if e.g. an SSL certificate has been updated.
    """
    global _GLOBAL_CLIENT
    client = _GLOBAL_CLIENT

    # First, set global client to None
    _GLOBAL_CLIENT = None

    # Then, close the clients
    if client is not None:
        try:
            client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")


atexit.register(close_session)


def _http_backoff_base(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: Union[type[Exception], tuple[type[Exception], ...]] = (
        httpx.TimeoutException,
        httpx.NetworkError,
    ),
    retry_on_status_codes: Union[int, tuple[int, ...]] = HTTPStatus.SERVICE_UNAVAILABLE,
    stream: bool = False,
    **kwargs,
) -> Generator[httpx.Response, None, None]:
    """Internal implementation of HTTP backoff logic shared between `http_backoff` and `http_stream_backoff`."""
    if isinstance(retry_on_exceptions, type):  # Tuple from single exception type
        retry_on_exceptions = (retry_on_exceptions,)

    if isinstance(retry_on_status_codes, int):  # Tuple from single status code
        retry_on_status_codes = (retry_on_status_codes,)

    nb_tries = 0
    sleep_time = base_wait_time

    # If `data` is used and is a file object (or any IO), it will be consumed on the
    # first HTTP request. We need to save the initial position so that the full content
    # of the file is re-sent on http backoff. See warning tip in docstring.
    io_obj_initial_pos = None
    if "data" in kwargs and isinstance(kwargs["data"], (io.IOBase, SliceFileObj)):
        io_obj_initial_pos = kwargs["data"].tell()

    client = get_session()
    while True:
        nb_tries += 1
        try:
            # If `data` is used and is a file object (or any IO), set back cursor to
            # initial position.
            if io_obj_initial_pos is not None:
                kwargs["data"].seek(io_obj_initial_pos)

            # Perform request and handle response
            def _should_retry(response: httpx.Response) -> bool:
                """Handle response and return True if should retry, False if should return/yield."""
                if response.status_code not in retry_on_status_codes:
                    return False  # Success, don't retry

                # Wrong status code returned (HTTP 503 for instance)
                logger.warning(f"HTTP Error {response.status_code} thrown while requesting {method} {url}")
                if nb_tries > max_retries:
                    hf_raise_for_status(response)  # Will raise uncaught exception
                    # Return/yield response to avoid infinite loop in the corner case where the
                    # user ask for retry on a status code that doesn't raise_for_status.
                    return False  # Don't retry, return/yield response

                return True  # Should retry

            if stream:
                with client.stream(method=method, url=url, **kwargs) as response:
                    if not _should_retry(response):
                        yield response
                        return
            else:
                response = client.request(method=method, url=url, **kwargs)
                if not _should_retry(response):
                    yield response
                    return

        except retry_on_exceptions as err:
            logger.warning(f"'{err}' thrown while requesting {method} {url}")

            if isinstance(err, httpx.ConnectError):
                close_session()  # In case of SSLError it's best to close the shared httpx.Client objects

            if nb_tries > max_retries:
                raise err

        # Sleep for X seconds
        logger.warning(f"Retrying in {sleep_time}s [Retry {nb_tries}/{max_retries}].")
        time.sleep(sleep_time)

        # Update sleep time for next retry
        sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff


def http_backoff(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: Union[type[Exception], tuple[type[Exception], ...]] = (
        httpx.TimeoutException,
        httpx.NetworkError,
    ),
    retry_on_status_codes: Union[int, tuple[int, ...]] = HTTPStatus.SERVICE_UNAVAILABLE,
    **kwargs,
) -> httpx.Response:
    """Wrapper around httpx to retry calls on an endpoint, with exponential backoff.

    Endpoint call is retried on exceptions (ex: connection timeout, proxy error,...)
    and/or on specific status codes (ex: service unavailable). If the call failed more
    than `max_retries`, the exception is thrown or `raise_for_status` is called on the
    response object.

    Re-implement mechanisms from the `backoff` library to avoid adding an external
    dependencies to `hugging_face_hub`. See https://github.com/litl/backoff.

    Args:
        method (`Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]`):
            HTTP method to perform.
        url (`str`):
            The URL of the resource to fetch.
        max_retries (`int`, *optional*, defaults to `5`):
            Maximum number of retries, defaults to 5 (no retries).
        base_wait_time (`float`, *optional*, defaults to `1`):
            Duration (in seconds) to wait before retrying the first time.
            Wait time between retries then grows exponentially, capped by
            `max_wait_time`.
        max_wait_time (`float`, *optional*, defaults to `8`):
            Maximum duration (in seconds) to wait before retrying.
        retry_on_exceptions (`type[Exception]` or `tuple[type[Exception]]`, *optional*):
            Define which exceptions must be caught to retry the request. Can be a single type or a tuple of types.
            By default, retry on `httpx.TimeoutException` and `httpx.NetworkError`.
        retry_on_status_codes (`int` or `tuple[int]`, *optional*, defaults to `503`):
            Define on which status codes the request must be retried. By default, only
            HTTP 503 Service Unavailable is retried.
        **kwargs (`dict`, *optional*):
            kwargs to pass to `httpx.request`.

    Example:
    ```
    >>> from huggingface_hub.utils import http_backoff

    # Same usage as "httpx.request".
    >>> response = http_backoff("GET", "https://www.google.com")
    >>> response.raise_for_status()

    # If you expect a Gateway Timeout from time to time
    >>> http_backoff("PUT", upload_url, data=data, retry_on_status_codes=504)
    >>> response.raise_for_status()
    ```

    <Tip warning={true}>

    When using `httpx` it is possible to stream data by passing an iterator to the
    `data` argument. On http backoff this is a problem as the iterator is not reset
    after a failed call. This issue is mitigated for file objects or any IO streams
    by saving the initial position of the cursor (with `data.tell()`) and resetting the
    cursor between each call (with `data.seek()`). For arbitrary iterators, http backoff
    will fail. If this is a hard constraint for you, please let us know by opening an
    issue on [Github](https://github.com/huggingface/huggingface_hub).

    </Tip>
    """
    return next(
        _http_backoff_base(
            method=method,
            url=url,
            max_retries=max_retries,
            base_wait_time=base_wait_time,
            max_wait_time=max_wait_time,
            retry_on_exceptions=retry_on_exceptions,
            retry_on_status_codes=retry_on_status_codes,
            stream=False,
            **kwargs,
        )
    )


@contextmanager
def http_stream_backoff(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: Union[type[Exception], tuple[type[Exception], ...]] = (
        httpx.TimeoutException,
        httpx.NetworkError,
    ),
    retry_on_status_codes: Union[int, tuple[int, ...]] = HTTPStatus.SERVICE_UNAVAILABLE,
    **kwargs,
) -> Generator[httpx.Response, None, None]:
    """Wrapper around httpx to retry calls on an endpoint, with exponential backoff.

    Endpoint call is retried on exceptions (ex: connection timeout, proxy error,...)
    and/or on specific status codes (ex: service unavailable). If the call failed more
    than `max_retries`, the exception is thrown or `raise_for_status` is called on the
    response object.

    Re-implement mechanisms from the `backoff` library to avoid adding an external
    dependencies to `hugging_face_hub`. See https://github.com/litl/backoff.

    Args:
        method (`Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]`):
            HTTP method to perform.
        url (`str`):
            The URL of the resource to fetch.
        max_retries (`int`, *optional*, defaults to `5`):
            Maximum number of retries, defaults to 5 (no retries).
        base_wait_time (`float`, *optional*, defaults to `1`):
            Duration (in seconds) to wait before retrying the first time.
            Wait time between retries then grows exponentially, capped by
            `max_wait_time`.
        max_wait_time (`float`, *optional*, defaults to `8`):
            Maximum duration (in seconds) to wait before retrying.
        retry_on_exceptions (`type[Exception]` or `tuple[type[Exception]]`, *optional*):
            Define which exceptions must be caught to retry the request. Can be a single type or a tuple of types.
            By default, retry on `httpx.Timeout` and `httpx.NetworkError`.
        retry_on_status_codes (`int` or `tuple[int]`, *optional*, defaults to `503`):
            Define on which status codes the request must be retried. By default, only
            HTTP 503 Service Unavailable is retried.
        **kwargs (`dict`, *optional*):
            kwargs to pass to `httpx.request`.

    Example:
    ```
    >>> from huggingface_hub.utils import http_stream_backoff

    # Same usage as "httpx.stream".
    >>> with http_stream_backoff("GET", "https://www.google.com") as response:
    ...     for chunk in response.iter_bytes():
    ...         print(chunk)

    # If you expect a Gateway Timeout from time to time
    >>> with http_stream_backoff("PUT", upload_url, data=data, retry_on_status_codes=504) as response:
    ...     response.raise_for_status()
    ```

    <Tip warning={true}>

    When using `httpx` it is possible to stream data by passing an iterator to the
    `data` argument. On http backoff this is a problem as the iterator is not reset
    after a failed call. This issue is mitigated for file objects or any IO streams
    by saving the initial position of the cursor (with `data.tell()`) and resetting the
    cursor between each call (with `data.seek()`). For arbitrary iterators, http backoff
    will fail. If this is a hard constraint for you, please let us know by opening an
    issue on [Github](https://github.com/huggingface/huggingface_hub).

    </Tip>
    """
    yield from _http_backoff_base(
        method=method,
        url=url,
        max_retries=max_retries,
        base_wait_time=base_wait_time,
        max_wait_time=max_wait_time,
        retry_on_exceptions=retry_on_exceptions,
        retry_on_status_codes=retry_on_status_codes,
        stream=True,
        **kwargs,
    )


def fix_hf_endpoint_in_url(url: str, endpoint: Optional[str]) -> str:
    """Replace the default endpoint in a URL by a custom one.

    This is useful when using a proxy and the Hugging Face Hub returns a URL with the default endpoint.
    """
    endpoint = endpoint.rstrip("/") if endpoint else constants.ENDPOINT
    # check if a proxy has been set => if yes, update the returned URL to use the proxy
    if endpoint not in (constants._HF_DEFAULT_ENDPOINT, constants._HF_DEFAULT_STAGING_ENDPOINT):
        url = url.replace(constants._HF_DEFAULT_ENDPOINT, endpoint)
        url = url.replace(constants._HF_DEFAULT_STAGING_ENDPOINT, endpoint)
    return url


def hf_raise_for_status(response: httpx.Response, endpoint_name: Optional[str] = None) -> None:
    """
    Internal version of `response.raise_for_status()` that will refine a potential HTTPError.
    Raised exception will be an instance of [`~errors.HfHubHTTPError`].

    This helper is meant to be the unique method to raise_for_status when making a call to the Hugging Face Hub.

    Args:
        response (`Response`):
            Response from the server.
        endpoint_name (`str`, *optional*):
            Name of the endpoint that has been called. If provided, the error message will be more complete.

    <Tip warning={true}>

    Raises when the request has failed:

        - [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it doesn't exist, because `repo_type`
            is not set correctly, or because the repo is `private` and you do not have access.
        - [`~utils.GatedRepoError`]
            If the repository exists but is gated and the user is not on the authorized list.
        - [`~utils.RevisionNotFoundError`]
            If the repository exists but the revision couldn't be find.
        - [`~utils.RemoteEntryNotFoundError`]
            If the repository exists but the entry (e.g. the requested file) couldn't be find.
        - [`~utils.BadRequestError`]
            If request failed with a HTTP 400 BadRequest error.
        - [`~utils.HfHubHTTPError`]
            If request failed for a reason not listed above.

    </Tip>
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if response.status_code // 100 == 3:
            return  # Do not raise on redirects to stay consistent with `requests`

        error_code = response.headers.get("X-Error-Code")
        error_message = response.headers.get("X-Error-Message")

        if error_code == "RevisionNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Revision Not Found for url: {response.url}."
            raise _format(RevisionNotFoundError, message, response) from e

        elif error_code == "EntryNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Entry Not Found for url: {response.url}."
            raise _format(RemoteEntryNotFoundError, message, response) from e

        elif error_code == "GatedRepo":
            message = (
                f"{response.status_code} Client Error." + "\n\n" + f"Cannot access gated repo for url {response.url}."
            )
            raise _format(GatedRepoError, message, response) from e

        elif error_message == "Access to this resource is disabled.":
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Cannot access repository for url {response.url}."
                + "\n"
                + "Access to this resource is disabled."
            )
            raise _format(DisabledRepoError, message, response) from e

        elif error_code == "RepoNotFound" or (
            response.status_code == 401
            and error_message != "Invalid credentials in Authorization header"
            and response.request is not None
            and response.request.url is not None
            and REPO_API_REGEX.search(str(response.request.url)) is not None
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
                " make sure you are authenticated. For more details, see"
                " https://huggingface.co/docs/huggingface_hub/authentication"
            )
            raise _format(RepositoryNotFoundError, message, response) from e

        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
            )
            raise _format(BadRequestError, message, response) from e

        elif response.status_code == 403:
            message = (
                f"\n\n{response.status_code} Forbidden: {error_message}."
                + f"\nCannot access content at: {response.url}."
                + "\nMake sure your token has the correct permissions."
            )
            raise _format(HfHubHTTPError, message, response) from e

        elif response.status_code == 416:
            range_header = response.request.headers.get("Range")
            message = f"{e}. Requested range: {range_header}. Content-Range: {response.headers.get('Content-Range')}."
            raise _format(HfHubHTTPError, message, response) from e

        # Convert `HTTPError` into a `HfHubHTTPError` to display request information
        # as well (request id and/or server error message)
        raise _format(HfHubHTTPError, str(e), response) from e


def _format(error_type: type[HfHubHTTPError], custom_message: str, response: httpx.Response) -> HfHubHTTPError:
    server_errors = []

    # Retrieve server error from header
    from_headers = response.headers.get("X-Error-Message")
    if from_headers is not None:
        server_errors.append(from_headers)

    # Retrieve server error from body
    try:
        # Case errors are returned in a JSON format
        try:
            data = response.json()
        except httpx.ResponseNotRead:
            response.read()  # In case of streaming response, we need to read the response first
            data = response.json()

        error = data.get("error")
        if error is not None:
            if isinstance(error, list):
                # Case {'error': ['my error 1', 'my error 2']}
                server_errors.extend(error)
            else:
                # Case {'error': 'my error'}
                server_errors.append(error)

        errors = data.get("errors")
        if errors is not None:
            # Case {'errors': [{'message': 'my error 1'}, {'message': 'my error 2'}]}
            for error in errors:
                if "message" in error:
                    server_errors.append(error["message"])

    except json.JSONDecodeError:
        # If content is not JSON and not HTML, append the text
        content_type = response.headers.get("Content-Type", "")
        if response.text and "html" not in content_type.lower():
            server_errors.append(response.text)

    # Strip all server messages
    server_errors = [str(line).strip() for line in server_errors if str(line).strip()]

    # Deduplicate server messages (keep order)
    # taken from https://stackoverflow.com/a/17016257
    server_errors = list(dict.fromkeys(server_errors))

    # Format server error
    server_message = "\n".join(server_errors)

    # Add server error to custom message
    final_error_message = custom_message
    if server_message and server_message.lower() not in custom_message.lower():
        if "\n\n" in custom_message:
            final_error_message += "\n" + server_message
        else:
            final_error_message += "\n\n" + server_message
    # Add Request ID
    request_id = str(response.headers.get(X_REQUEST_ID, ""))
    if request_id:
        request_id_message = f" (Request ID: {request_id})"
    else:
        # Fallback to X-Amzn-Trace-Id
        request_id = str(response.headers.get(X_AMZN_TRACE_ID, ""))
        if request_id:
            request_id_message = f" (Amzn Trace ID: {request_id})"
    if request_id and request_id.lower() not in final_error_message.lower():
        if "\n" in final_error_message:
            newline_index = final_error_message.index("\n")
            final_error_message = (
                final_error_message[:newline_index] + request_id_message + final_error_message[newline_index:]
            )
        else:
            final_error_message += request_id_message

    # Return
    return error_type(final_error_message.strip(), response=response, server_message=server_message or None)


def _curlify(request: httpx.Request) -> str:
    """Convert a `httpx.Request` into a curl command (str).

    Used for debug purposes only.

    Implementation vendored from https://github.com/ofw/curlify/blob/master/curlify.py.
    MIT License Copyright (c) 2016 Egor.
    """
    parts: list[tuple[Any, Any]] = [
        ("curl", None),
        ("-X", request.method),
    ]

    for k, v in sorted(request.headers.items()):
        if k.lower() == "authorization":
            v = "<TOKEN>"  # Hide authorization header, no matter its value (can be Bearer, Key, etc.)
        parts += [("-H", f"{k}: {v}")]

    body: Optional[str] = None
    if request.content is not None:
        body = request.content.decode("utf-8", errors="ignore")
        if len(body) > 1000:
            body = f"{body[:1000]} ... [truncated]"
    elif request.stream is not None:
        body = "<streaming body>"
    if body is not None:
        parts += [("-d", body.replace("\n", ""))]

    parts += [(None, request.url)]

    flat_parts = []
    for k, v in parts:
        if k:
            flat_parts.append(quote(k))
        if v:
            flat_parts.append(quote(v))

    return " ".join(flat_parts)


# Regex to parse HTTP Range header
RANGE_REGEX = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.IGNORECASE)


def _adjust_range_header(original_range: Optional[str], resume_size: int) -> Optional[str]:
    """
    Adjust HTTP Range header to account for resume position.
    """
    if not original_range:
        return f"bytes={resume_size}-"

    if "," in original_range:
        raise ValueError(f"Multiple ranges detected - {original_range!r}, not supported yet.")

    match = RANGE_REGEX.match(original_range)
    if not match:
        raise RuntimeError(f"Invalid range format - {original_range!r}.")
    start, end = match.groups()

    if not start:
        if not end:
            raise RuntimeError(f"Invalid range format - {original_range!r}.")

        new_suffix = int(end) - resume_size
        new_range = f"bytes=-{new_suffix}"
        if new_suffix <= 0:
            raise RuntimeError(f"Empty new range - {new_range!r}.")
        return new_range

    start = int(start)
    new_start = start + resume_size
    if end:
        end = int(end)
        new_range = f"bytes={new_start}-{end}"
        if new_start > end:
            raise RuntimeError(f"Empty new range - {new_range!r}.")
        return new_range

    return f"bytes={new_start}-"
